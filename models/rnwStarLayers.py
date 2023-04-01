import torch
import torch.nn as nn
import numpy as np

from models.rnwLayers import DispEncoder, Conv3x3
from models.autoencoder import AutoEncoder
from models.starGanLayers import ResBlk, AdainResBlk


class UAdaDecoder(nn.Module):
    def __init__(self, num_ch_enc, style_dim):
        super(UAdaDecoder, self).__init__()
        self.decoder = nn.ModuleList()

        # set parameters
        self.num_ch_enc = num_ch_enc
        # components
        # 4
        self.reduce4 = nn.Conv2d(self.num_ch_enc[4], 512, 1, bias=False)
        self.conv4 = Conv3x3(512, 512)
        self.adaRes4 = AdainResBlk(512, 512, style_dim, w_hpf=1, upsample=True)
        # 3
        self.reduce3 = nn.Conv2d(self.num_ch_enc[3], 256, 1, bias=False)
        self.conv3 = Conv3x3(512, 256)
        self.adaRes3 = AdainResBlk(512, 256, style_dim, w_hpf=1, upsample=True)
        # 2
        self.reduce2 = nn.Conv2d(self.num_ch_enc[2], 128, 1, bias=False)
        self.conv2 = Conv3x3(256, 128)
        self.adaRes2 = AdainResBlk(256, 128, style_dim, w_hpf=1, upsample=True)
        # 1
        self.reduce1 = nn.Conv2d(self.num_ch_enc[1], 64, 1, bias=False)
        self.conv1 = Conv3x3(128, 64)
        self.adaRes1 = AdainResBlk(128, 64, style_dim, w_hpf=1, upsample=True)
        # 0
        self.reduce0 = nn.Conv2d(self.num_ch_enc[0], 64, 1, bias=False)
        self.conv0 = Conv3x3(64, 64)
        self.adaRes0 = AdainResBlk(128, 64, style_dim, w_hpf=1, upsample=True)
        self.disp_conv0 = nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect')

    def forward(self, features, s):
        # get features
        f0, f1, f2, f3, f4 = features
        # forward
        # 4
        x4 = self.reduce4(f4)
        x4 = self.conv4(x4)
        x4 = self.adaRes4(x4, s)

        # 3
        f3 = self.reduce3(f3)
        x3 = self.conv3(x4)
        x3 = torch.cat([x3, f3], dim=1)
        x3 = self.adaRes3(x3, s)

        # 2
        f2 = self.reduce2(f2)
        x2 = self.conv2(x3)
        x2 = torch.cat([x2, f2], dim=1)
        x2 = self.adaRes2(x2, s)

        # 1
        f1 = self.reduce1(f1)
        x1 = self.conv1(x2)
        x1 = torch.cat([x1, f1], dim=1)
        x1 = self.adaRes1(x1, s)

        # 0
        f0 = self.reduce0(f0)
        x0 = self.conv0(x1)
        x0 = torch.cat([x0, f0], dim=1)
        x0 = self.adaRes0(x0, s)
        disp0 = torch.sigmoid(self.disp_conv0(x0))

        outputs = disp0
        return outputs


class ResUetGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.style_dim = self.opt.style_dim  # 64

        self.G_en = DispEncoder(num_layers=self.opt.resnet_layers, pre_trained=False)
        self.G_de = UAdaDecoder(num_ch_enc=self.G_en.num_ch_enc, style_dim=self.style_dim)

    def forward(self, x, s):
        features = self.G_en(x)
        out = self.G_de(features, s)
        return out


class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.max_conv_dim = opt.max_conv_dim
        self.num_domains = opt.num_domains
        self.dim_in = opt.dim_in

        blocks = []
        blocks += [nn.Conv2d(3, self.dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(opt.img_size)) - 2
        fin_ker = int(opt.img_size / (2 ** repeat_num))
        repeat_num = 6
        for _ in range(repeat_num):
            self.dim_out = min(self.dim_in*2, self.max_conv_dim)
            blocks += [ResBlk(self.dim_in, self.dim_out, downsample=True)]
            self.dim_in = self.dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(self.dim_out, self.dim_out, fin_ker, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(self.dim_out, self.num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0)))
        out = out[idx, y]  # (batch)
        return out


class StyleNet(nn.Module):
    def __init__(self, opt):
        super(StyleNet, self).__init__()
        self.opt = opt
        self.style_dim = self.opt.style_dim
        self.num_domains = self.opt.num_domains
        self.ae_dim_out = self.opt.ae_dim_out

        self.style_en_part1 = AutoEncoder(self.opt)
        self.style_en_part1.load_state_dict(torch.load("pre_train/pre_train/checkpoint_epoch=299.ckpt", map_location='cpu')['state_dict'])

        self.style_en_part2 = nn.ModuleList()
        for _ in range(self.num_domains):
            self.style_en_part2 += [nn.Linear(self.ae_dim_out, self.style_dim)]

    def forward(self, x, y):
        h = self.style_en_part1.forward(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.style_en_part2:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0)))
        s = out[idx, y]  # (batch, style_dim)
        return s
