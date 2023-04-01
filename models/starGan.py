import os
import copy
import numpy as np
import pytorch_lightning
import torch
import torch.nn.functional as F
from mmcv import Config
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from models.registry import MODELS
from models.starGanLayers import Generator, MappingNetwork, StyleEncoder, Discriminator
from models.utils import he_init


@MODELS.register_module(name='StarGAN')
class StarGAN(LightningModule):
    """
    The training process
    """

    def __init__(self, opt):
        super(StarGAN, self).__init__()
        self.opt = opt.model
        self.GNet = Generator(self.opt.img_size, self.opt.style_dim, w_hpf=self.opt.w_hpf)
        self.DNet = Discriminator(self.opt.img_size, self.opt.num_domains)
        self.MapNet = MappingNetwork(self.opt.latent_dim, self.opt.style_dim, self.opt.num_domains)
        self.StyleNet = StyleEncoder(self.opt.img_size, self.opt.style_dim, self.opt.num_domains)

        self.GNet.apply(he_init)
        self.DNet.apply(he_init)
        self.MapNet.apply(he_init)
        self.StyleNet.apply(he_init)

        # moving average networks
        self.GNet_ema = copy.deepcopy(self.GNet)
        self.MapNet_ema = copy.deepcopy(self.MapNet)
        self.StyleNet_ema = copy.deepcopy(self.StyleNet)

        self.automatic_optimization = False

    def forward(self, inputs):
        pass
        # return self.G(inputs)

    def compute_G_loss(self, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
        assert (z_trgs is None) != (x_refs is None)
        if z_trgs is not None:
            z_trg, z_trg2 = z_trgs
        if x_refs is not None:
            x_ref, x_ref2 = x_refs

        # adversarial loss
        if z_trgs is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
        out = nets.discriminator(x_fake, y_trg)
        loss_adv = adv_loss(out, 1)

        # style reconstruction loss
        s_pred = nets.style_encoder(x_fake, y_trg)
        loss_sty = torch.mean(torch.abs(s_pred - s_trg))

        # diversity sensitive loss
        if z_trgs is not None:
            s_trg2 = nets.mapping_network(z_trg2, y_trg)
        else:
            s_trg2 = nets.style_encoder(x_ref2, y_trg)
        x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
        x_fake2 = x_fake2.detach()
        loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

        # cycle-consistency loss
        masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
        s_org = nets.style_encoder(x_real, y_org)
        x_rec = nets.generator(x_fake, s_org, masks=masks)
        loss_cyc = torch.mean(torch.abs(x_rec - x_real))

        loss = loss_adv + args.lambda_sty * loss_sty \
               - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
        return loss, Munch(adv=loss_adv.item(),
                           sty=loss_sty.item(),
                           ds=loss_ds.item(),
                           cyc=loss_cyc.item())

    def compute_D_loss(self, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
        assert (z_trg is None) != (x_ref is None)
        # with real images
        x_real.requires_grad_()
        out = nets.discriminator(x_real, y_org)
        loss_real = adv_loss(out, 1)
        loss_reg = r1_reg(out, x_real)

        # with fake images
        with torch.no_grad():
            if z_trg is not None:
                s_trg = nets.mapping_network(z_trg, y_trg)
            else:  # x_ref is not None
                s_trg = nets.style_encoder(x_ref, y_trg)

            x_fake = nets.generator(x_real, s_trg, masks=masks)
        out = nets.discriminator(x_fake, y_trg)
        loss_fake = adv_loss(out, 0)

        loss = loss_real + loss_fake + args.lambda_reg * loss_reg
        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item())

    def training_step(self, batch_data, batch_idx):
        optim_G, optim_D, optim_Style, optim_Map = self.optimizers()
        # tensorboard logger
        logger = self.logger.experiment

        x_real, y_org = batch_data["x_src"], batch_data["y_src"]
        x_ref, x_ref2, y_trg = batch_data["x_ref"], batch_data["x_ref2"], batch_data["y_ref"]
        z_trg, z_trg2 = batch_data["z_trg"], batch_data["z_trg2"]

        # train the discriminator
        d_loss, d_losses_latent = self.compute_D_loss(x_real, y_org, y_trg, z_trg=z_trg, masks=masks)

        optim_D.zero_grad()
        self.manual_backward(d_loss, retain_graph=True)
        optim_D.step()

        d_loss, d_losses_ref = self.compute_D_loss(x_real, y_org, y_trg, x_ref=x_ref, masks=masks)

        optim_D.zero_grad()
        self.manual_backward(d_loss, retain_graph=True)
        optim_D.step()

        # train the generator
        g_loss, g_losses_latent = self.compute_G_loss(x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)

        optim_G.zero_grad()
        optim_Map.zero_grad()
        optim_Style.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        optim_G.step()
        optim_Map.step()
        optim_Style.step()

        g_loss, g_losses_ref = self.compute_G_loss(x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)

        optim_G.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        optim_G.step()

        # compute moving average of network parameters
        self.moving_average(self.GNet, self.GNet_ema, beta=0.999)
        self.moving_average(self.MapNet, self.MapNet_ema, beta=0.999)
        self.moving_average(self.StyleNet, self.MapNet_ema, beta=0.999)

        logger.add_scalar('train/D_loss/d_losses_latent', d_losses_latent, self.current_epoch)
        logger.add_scalar('train/D_loss/d_losses_ref', d_losses_ref, self.current_epoch)
        logger.add_scalar('train/G_loss/g_losses_latent', g_losses_latent, self.current_epoch)
        logger.add_scalar('train/G_loss/g_losses_ref', g_losses_ref, self.current_epoch)

        # generate images for debugging
        if (i + 1) % args.sample_every == 0:
            os.makedirs(args.sample_dir, exist_ok=True)
            utils.debug_image(nets_ema, args, inputs=inputs_val, step=i + 1)

        # compute FID and LPIPS if necessary
        if (i + 1) % args.eval_every == 0:
            calculate_metrics(nets_ema, args, i + 1, mode='latent')
            calculate_metrics(nets_ema, args, i + 1, mode='reference')

    def training_epoch_end(self, outputs):
        """
        Step lr scheduler
        :param outputs:
        :return:
        """
        sch_G, sch_D, sch_Style, sch_Map = self.lr_schedulers()

        sch_G.step()
        sch_D.step()
        sch_Style.step()
        sch_Map.step()

    def validation_step(self, batch, batch_idx):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')

    def configure_optimizers(self):
        optim_G = Adam(self.GNet.parameters(), lr=self.opt.learning_rate)
        optim_D = Adam(self.DNet.parameters(), lr=self.opt.learning_rate)
        optim_Style = Adam(self.StyleNet.parameters(), lr=self.opt.learning_rate)
        optim_Map = Adam(self.MapNet.parameters(), lr=self.opt.learning_rate)

        sch_G = MultiStepLR(optim_G, milestones=[15], gamma=0.5)
        sch_D = MultiStepLR(optim_D, milestones=[15], gamma=0.5)
        sch_Style = MultiStepLR(optim_Style, milestones=[15], gamma=0.5)
        sch_Map = MultiStepLR(optim_Map, milestones=[15], gamma=0.5)

        return [optim_G, optim_D, optim_Style, optim_Map], [sch_G, sch_D, sch_Style, sch_Map]

    def moving_average(self, model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def adv_loss(self, logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

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