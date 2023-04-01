import numpy as np
import torch
import pytorch_lightning
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from mmcv import Config
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from models.registry import MODELS
from models.utils import ImagePool, GANLoss, StructureLoss, LuminanceLoss, robust_l1, get_smooth_loss, freeze_model, unfreeze_model
from models.rnwStarLayers import StyleNet, ResUetGenerator, Discriminator


@MODELS.register_module(name='rnw_star')
class RNW_StarModel(LightningModule):
    """
    The training process
    """

    def __init__(self, opt):
        super(RNW_StarModel, self).__init__()

        self.opt = opt.model

        # components
        self.gan_loss = GANLoss('vanilla')
        self.image_pool = ImagePool(50)
        self.criterionStructure = StructureLoss(
            channel=3, window_size=self.opt.Structure_size, crop_size=self.opt.roi_size)
        self.criterionLuminance1 = LuminanceLoss(
            self.opt.Luminance_size1, self.opt.Luminance_size1, self.opt.roi_size)
        self.criterionLuminance2 = LuminanceLoss(
            self.opt.Luminance_size2, self.opt.Luminance_size2, self.opt.roi_size)

        # networks
        self.GNet = ResUetGenerator(self.opt)
        self.DNet = Discriminator(self.opt)
        self.styleNet = StyleNet(self.opt)

        # manual optimization
        self.automatic_optimization = False

    def forward(self, inputs, ref, y):
        s = self.styleNet(ref, y)
        out = self.GNet(inputs, s)
        return out

    def compute_G_loss(self, x_org, y_org, x_ref, y_trg):
        loss_dict = {}
        unfreeze_model(self.GNet)
        unfreeze_model(self.styleNet)

        s_trg = self.styleNet(x_ref, y_trg)
        x_org2trg = self.GNet(x_org, s_trg)

        # adversarial loss
        predict_fake = self.DNet(x_org2trg, y_trg)
        loss_fake = self.gan_loss(predict_fake, True)

        # style reconstruction loss
        s_pred = self.styleNet(x_org2trg, y_trg)
        loss_style = torch.mean(torch.abs(s_pred - s_trg))

        # cycle-consistency loss
        s_org = self.styleNet(x_org, y_org)
        x_rec = self.GNet(x_org2trg, s_org)
        loss_rec = nn.L1Loss()(x_rec, x_org)

        # disp_loss
        loss_struc = self.criterionStructure(x_org2trg, x_org).mean()
        loss_lum = (self.criterionLuminance1(x_org2trg) + self.criterionLuminance2(x_org2trg)) / 2.0
        loss_disp = (loss_struc * self.opt.lambda_Structure + loss_lum * self.opt.lambda_Luminance) / 2.0

        # loss record
        loss_dict['Adv_loss'] = loss_fake.detach()
        loss_dict['style_loss'] = loss_style.detach()
        loss_dict['reconstruct_loss'] = loss_rec.detach()
        loss_dict['Luminance_loss'] = loss_lum.detach()
        loss_dict['structure_loss'] = loss_struc.detach()

        gan_loss = loss_fake + loss_style * self.opt.lambda_style + loss_rec * self.opt.lambda_recon + loss_disp
        return gan_loss, loss_dict

    def compute_D_loss(self, x_org, y_org, x_ref, y_trg):
        loss_dict = {}
        x_org.requires_grad_()
        predict_true = self.DNet(x_org, y_org)
        loss_reg = self.r1_reg(predict_true, x_org)
        loss_true = self.gan_loss(predict_true, True)

        freeze_model(self.GNet)
        freeze_model(self.styleNet)
        s_trg = self.styleNet(x_ref, y_trg)
        x_org2trg = self.GNet(x_org, s_trg)
        predict_fake = self.DNet(x_org2trg, y_trg)
        loss_fake = self.gan_loss(predict_fake, False)

        loss_dict['Adv_loss'] = (loss_true.detach() + loss_fake.detach()) / 2
        loss_dict['R1_reg'] = loss_reg.detach()

        return loss_true + loss_fake + self.opt.lambda_reg * loss_reg, loss_dict

    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim_G, optim_D, optim_style = self.optimizers()

        # tensorboard logger
        logger = self.logger.experiment

        # y_low设置为0, y_high设置为1
        x_low, y_low = batch_data["low_q"]
        x_high, y_high = batch_data["high_q"]

        x_org, y_org, x_ref, y_trg = x_low, y_low, x_high, y_high
        if self.opt.reverse_train:
            if random.random() > 0.8:
                x_org, y_org, x_ref, y_trg = x_high, y_high, x_low, y_low

        # train D
        d_loss, d_loss_dict = self.compute_D_loss(x_org, y_org, x_ref, y_trg)
        optim_D.zero_grad()
        self.manual_backward(d_loss, retain_graph=True)
        optim_D.step()

        # train G
        g_loss, g_loss_dict = self.compute_G_loss(x_org, y_org, x_ref, y_trg)
        optim_G.zero_grad()
        optim_style.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        optim_G.step()
        optim_style.step()

        logger.add_scalar('train/D_loss', d_loss_dict['Adv_loss'], self.global_step)
        logger.add_scalar('train/D_R1reg', d_loss_dict['R1_reg'], self.global_step)
        logger.add_scalar('train/G_loss', g_loss_dict['Adv_loss'], self.global_step)
        logger.add_scalar('train/G_style_loss', g_loss_dict['style_loss'], self.global_step)
        logger.add_scalar('train/G_reconstruct_loss', g_loss_dict['reconstruct_loss'], self.global_step)
        logger.add_scalar('train/G_luminance_loss', g_loss_dict['Luminance_loss'], self.global_step)
        logger.add_scalar('train/G_structure_loss', g_loss_dict['structure_loss'], self.global_step)

    def training_epoch_end(self, outputs):
        """
        Step lr scheduler
        :param outputs:
        :return:
        """
        sch_G, sch_D, sch_style = self.lr_schedulers()

        sch_G.step()
        sch_D.step()
        sch_style.step()

    def configure_optimizers(self):
        optim_G = Adam(self.GNet.parameters(), lr=self.opt.learning_rate)
        optim_D = Adam(self.DNet.parameters(), lr=self.opt.learning_rate)
        optim_style = Adam(itertools.chain(self.styleNet.style_en_part1.encoder.parameters(),
                                           self.styleNet.style_en_part2.parameters()), lr=self.opt.learning_rate)

        sch_G = MultiStepLR(optim_G, milestones=[15], gamma=0.5)
        sch_D = MultiStepLR(optim_D, milestones=[15], gamma=0.5)
        sch_style = MultiStepLR(optim_style, milestones=[15], gamma=0.5)

        return [optim_G, optim_D, optim_style], [sch_G, sch_D, sch_style]

    def r1_reg(self, d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg

