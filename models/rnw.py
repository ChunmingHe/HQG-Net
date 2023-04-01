import numpy as np
import pytorch_lightning
import torch.nn.functional as F
from mmcv import Config
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from utils import EWMA
from models.rnwLayers import DispNet
from models.rnwLayers import NLayerDiscriminator
from models.registry import MODELS
from models.utils import *
from transforms import EqualizeHist


@MODELS.register_module(name='rnw')
class RNWModel(LightningModule):
    """
    The training process
    """

    def __init__(self, opt):
        super(RNWModel, self).__init__()

        self.opt = opt.model

        # components
        self.gan_loss = GANLoss('lsgan')
        self.image_pool = ImagePool(50)
        self.criterionStructure = StructureLoss(
            channel=3, window_size=self.opt.Structure_size, crop_size=self.opt.roi_size)
        self.criterionLuminance1 = LuminanceLoss(
            self.opt.Luminance_size1, self.opt.Luminance_size1, self.opt.roi_size)
        self.criterionLuminance2 = LuminanceLoss(
            self.opt.Luminance_size2, self.opt.Luminance_size2, self.opt.roi_size)
        self.ego_diff = EWMA(momentum=0.98)
        self._equ_limit = 0.008

        # networks
        self.G = DispNet(self.opt)
        in_chs_D = 5 if self.opt.use_position_map else 3
        self.D = NLayerDiscriminator(in_chs_D, n_layers=3)

        # register image coordinates
        if self.opt.use_position_map:
            h, w = self.opt.height, self.opt.width
            height_map = torch.arange(h).view(
                1, 1, h, 1).repeat(1, 1, 1, w) // (h - 1)
            width_map = torch.arange(w).view(
                1, 1, 1, w).repeat(1, 1, h, 1) // (w - 1)

            self.register_buffer('height_map', height_map, persistent=False)
            self.register_buffer('width_map', width_map, persistent=False)

        # link to dataset
        self.data_link = opt.data_link

        # manual optimization
        self.automatic_optimization = False

    def forward(self, inputs):
        return self.G(inputs)

    def generate_gan_outputs(self, high2high_out, low2high_out):
        # remove scale
        # low2high_out = low2high_out / low2high_out.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        # high2high_out = high2high_out / high2high_out.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        # image coordinates
        if self.opt.use_position_map:
            n = low2high_out.shape[0]
            height_map = self.height_map.repeat(n, 1, 1, 1)
            width_map = self.width_map.repeat(n, 1, 1, 1)
        else:
            height_map = None
            width_map = None

        # return
        return high2high_out, low2high_out, height_map, width_map

    def compute_G_loss(self, low2high_out, height_map, width_map):
        G_loss = 0.0
        #
        # Compute G loss
        #
        freeze_model(self.D)
        if self.opt.use_position_map:
            fake_out = torch.cat([height_map, width_map, low2high_out], dim=1)

        else:
            fake_out = low2high_out
        G_loss += self.gan_loss(self.D(fake_out), True)
        return G_loss

    def compute_D_loss(self, high_in, low2high_out, height_map, width_map):
        D_loss = 0.0
        #
        # Compute D loss
        #
        unfreeze_model(self.D)
        if self.opt.use_position_map:
            real_in = torch.cat([height_map, width_map, high_in], dim=1)
            fake_out = torch.cat(
                [height_map, width_map, low2high_out.detach()], dim=1)
        else:
            real_in = high_in
            fake_out = low2high_out.detach()
        # query
        fake_out = self.image_pool.query(fake_out)
        # compute loss
        D_loss += self.gan_loss(self.D(real_in), True)
        D_loss += self.gan_loss(self.D(fake_out), False)

        return D_loss / 2.0

    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim_G, optim_D = self.optimizers()

        # tensorboard logger
        logger = self.logger.experiment

        # get input data
        high_inputs = batch_data['high_q']
        low_inputs = batch_data['low_q']

        # outputs of G
        low2high_out_list = self.G(low_inputs)
        high2high_out_list = self.G(high_inputs)

        # loss for ego-motion
        disp_loss_dict = self.compute_disp_losses(
            low_inputs, low2high_out_list['disp0'])

        # generate outputs for gan
        high2high_out, low2high_out, height_map, width_map = self.generate_gan_outputs(high2high_out_list['disp0'],
                                                                                       low2high_out_list['disp0'])

        #
        # optimize G
        #
        # compute loss
        G_loss = self.compute_G_loss(low2high_out, height_map, width_map)
        disp_loss = sum(disp_loss_dict.values())
        idt_loss = torch.nn.L1Loss()(high2high_out, high_inputs)
        # log
        logger.add_scalar('train/disp_loss', disp_loss, self.current_epoch)
        logger.add_scalar('train/G_loss', G_loss, self.current_epoch)
        logger.add_scalar('train/idt_loss', idt_loss, self.current_epoch)

        # optimize G
        G_loss_weight = G_loss * self.opt.G_weight + \
            disp_loss + idt_loss * self.opt.lambda_idt

        optim_G.zero_grad()
        self.manual_backward(G_loss_weight, retain_graph=True)
        optim_G.step()

        #
        # optimize D
        #
        # compute loss
        D_loss = self.compute_D_loss(
            high_inputs, low2high_out, height_map, width_map)

        # log
        logger.add_scalar('train/D_loss', D_loss, self.global_step)

        D_loss_weight = D_loss * self.opt.D_weight

        # optimize D
        optim_D.zero_grad()
        self.manual_backward(D_loss_weight, retain_graph=True)
        optim_D.step()

        print("G_loss:{:.4f}, D_loss:{:.4f}, idt_loss:{:.4f}, disp_loss:{:.4f}".format(
            G_loss, D_loss, idt_loss, disp_loss))
        # # return
        # return G_loss + D_loss

    def training_epoch_end(self, outputs):
        """
        Step lr scheduler
        :param outputs:
        :return:
        """
        sch_G, sch_D = self.lr_schedulers()

        sch_G.step()
        sch_D.step()

        # self.data_link.when_epoch_over()

    def configure_optimizers(self):
        optim_G = Adam(self.G.parameters(), lr=self.opt.learning_rate)
        optim_D = Adam(self.D.parameters(), lr=self.opt.learning_rate)

        sch_G = MultiStepLR(optim_G, milestones=[15], gamma=0.5)
        sch_D = MultiStepLR(optim_D, milestones=[15], gamma=0.5)

        return [optim_G, optim_D], [sch_G, sch_D]

    def compute_recons_loss(self, pred, target):
        photometric_loss = robust_l1(pred, target).mean(1, True)
        structure_loss = self.criterionStructure(pred, target)
        reconstruction_loss = (1 * structure_loss + 0 * photometric_loss)
        return reconstruction_loss

    def get_static_mask(self, pred, target):
        # compute threshold
        mask_threshold = self.ego_diff.running_val
        # compute diff
        diff = (pred - target).abs().mean(dim=1, keepdim=True)
        # compute mask
        static_mask = (diff > mask_threshold).float()
        # return
        return static_mask

    def compute_disp_losses(self, inputs, outputs):
        loss_dict = {}
        """
        automask
        """
        use_static_mask = self.opt.use_static_mask
        # update ego diff
        if use_static_mask:
            with torch.no_grad():
                # get diff of two frames
                diff = (outputs - inputs).abs().mean(dim=1)
                diff = torch.flatten(diff, 1)

                # compute quantile
                quantile = np.quantile(
                    diff.cpu().detach(), self.opt.static_mask_quantile, axis=1)
                mean_quantile = quantile.mean()

                # update
                self.ego_diff.update(mean_quantile)

        # compute mask
        color_diff = self.compute_recons_loss(outputs, inputs)
        identity_recons_loss = color_diff + \
            torch.randn(color_diff.shape).type_as(color_diff) * 1e-5

        # static mask
        if use_static_mask:
            static_mask = self.get_static_mask(outputs, inputs)
            identity_recons_loss *= static_mask

        loss_dict['reconstruct_loss'] = self.opt.lambda_Structure * \
            identity_recons_loss.mean()

        loss_dict['Luminance_loss'] = (self.criterionLuminance1(
            outputs) + self.criterionLuminance2(outputs)) * self.opt.lambda_Luminance / 2.0

        """
        disp mean normalization
        """
        if self.opt.disp_norm:
            mean_out = outputs.mean(2, True).mean(3, True)
            outputs = outputs / (mean_out + 1e-7)

        """
        smooth loss
        """
        smooth_loss = get_smooth_loss(outputs, inputs)
        loss_dict['smooth_loss'] = self.opt.disparity_smoothness * smooth_loss

        return loss_dict
