import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import LightningModule
from .starGanLayers import ResBlk
from .rnwLayers import Conv3x3, UpConv3x3
from .utils import SSIM
import itertools


class upconvBlk(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconvBlk, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.reduce = nn.Conv2d(dim_in, int(dim_in/2), 1, bias=False)
        self.conv = Conv3x3(dim_in, int(dim_in/2))
        self.upconv = UpConv3x3(self.dim_in, self.dim_out)
        self.leakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x, s):
        s = self.reduce(s)
        x = self.conv(x)
        x = torch.cat([x, s], dim=1)
        x = self.leakyReLU(x)
        x = self.upconv(x)

        return x


class AutoEncoder(LightningModule):
    def __init__(self, opt):
        super(AutoEncoder, self).__init__()

        self.opt = opt
        self.dim_in = self.opt.dim_in  # 64
        img_size = self.opt.img_size   # 384
        max_conv_dim = self.opt.max_conv_dim  # 512

        repeat_num = int(np.log2(img_size)) - 2
        fin_ker = int(img_size / (2 ** repeat_num))
        repeat_max_dim = 0

        # -------------------------------------------
        en_blocks = []

        en_blocks += [nn.Conv2d(3, self.dim_in, 3, 1, 1)]

        for _ in range(repeat_num):
            if self.dim_in == max_conv_dim:
                repeat_max_dim += 1
            self.dim_out = min(self.dim_in * 2, max_conv_dim)
            en_blocks += [ResBlk(self.dim_in, self.dim_out, downsample=True)]
            self.dim_in = self.dim_out

        en_blocks += [nn.Sequential(nn.LeakyReLU(0.2),
                                    nn.Conv2d(self.dim_out, self.dim_out, fin_ker, 1, 0),
                                    nn.LeakyReLU(0.2))]

        self.encoder = nn.ModuleList(en_blocks)
        # -------------------------------------------

        # -------------------------------------------
        self.dim_in = self.dim_out
        de_blocks = []
        de_blocks += [nn.Sequential(nn.LeakyReLU(0.2),
                                    nn.ConvTranspose2d(self.dim_in, self.dim_in, fin_ker, 1, 0),
                                    nn.LeakyReLU(0.2))]

        for _ in range(repeat_num):
            if repeat_max_dim > 0:
                de_blocks += [upconvBlk(self.dim_in, self.dim_in)]
                repeat_max_dim -= 1
                continue
            de_blocks += [upconvBlk(self.dim_in, int(self.dim_in / 2))]
            self.dim_in = int(self.dim_in / 2)

        de_blocks += [nn.Conv2d(self.dim_in, 3, 3, 1, 1)]
        self.decoder = nn.ModuleList(de_blocks)
        # -------------------------------------------
        self.automatic_optimization = False

    def autoencoder_forward(self, inputs):
        features_en = [inputs]
        for blocks in self.encoder:
            features_en.append(blocks(features_en[-1]))

        en_sz = len(features_en)
        features_de = []
        for blocks2 in self.decoder:
            if len(features_en) == en_sz:
                y_de = blocks2(features_en.pop(-1))
                features_de.append(y_de)
            elif len(features_en) > 2:
                y_de = blocks2(features_de[-1], features_en.pop(-1))
                features_de.append(y_de)
            else:
                y_de = blocks2(features_de[-1])
                features_de.append(y_de)

        return features_de[-1]

    def forward(self, inputs):
        x = inputs
        for blocks in self.encoder:
            x = blocks(x)
        return x

    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim = self.optimizers()

        # tensorboard logger
        logger = self.logger.experiment

        # get input data
        inputs = batch_data
        labels = batch_data.detach()

        recon_img = self.autoencoder_forward(inputs)
        MseLoss = nn.MSELoss()(recon_img, labels)
        SSIMLoss = SSIM()(recon_img, labels).mean(1, True).mean()

        loss = MseLoss + SSIMLoss
        logger.add_scalar('train/MSE', MseLoss, self.current_epoch)
        logger.add_scalar('train/SSIM', SSIMLoss, self.current_epoch)

        optim.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        optim.step()

    def training_epoch_end(self, outputs):
        sch_opt = self.lr_schedulers()
        sch_opt.step()

    def configure_optimizers(self):
        optim = Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.opt.learning_rate)

        sch_opt = MultiStepLR(optim, milestones=[15], gamma=0.5)

        return [optim], [sch_opt]
