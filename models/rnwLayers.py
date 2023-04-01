from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==========================================basic layers================================================================
def get_norm_layer(norm_type: str, num_features: int):
    if norm_type == 'batch_norm':
        return nn.BatchNorm2d(num_features, affine=True)
    elif norm_type == 'instance_norm':
        return nn.InstanceNorm2d(num_features, affine=True)
    else:
        raise ValueError('Unsupported norm layer type: {}.'.format(norm_type))


class DeConv3x3(nn.Module):
    """
    Use transposed convolution to up sample (scale_factor = 2.0)
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(DeConv3x3, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                          output_padding=0, bias=bias)
        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))
        self.non_linear = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.up_conv(x)
        out = self.pad(out)
        out = self.non_linear(out)
        return out


class UpConv3x3(nn.Module):
    """
    Use bilinear followed by conv
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(UpConv3x3, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels, bias=bias)
        self.non_linear = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0, mode='nearest')
        out = self.conv(out)
        out = self.non_linear(out)
        return out


class Conv3x3(nn.Module):
    """
    Convolution layer with 3 kernel size, followed by non_linear layer
    """
    def __init__(self, in_channels, out_channels, padding_mode='reflect', bias=True):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return out

# ==========================================basic layers================================================================


# ==========================================main layers=================================================================
class LeakyReluBottleneck(models.resnet.Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(LeakyReluBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation,
                                                  norm_layer)
        self.relu = nn.LeakyReLU(inplace=True)


class LeakyReluBasicBlock(models.resnet.BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(LeakyReluBasicBlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation,
                                                  norm_layer)
        self.relu = nn.LeakyReLU(inplace=True)


class ResNetWithoutPool(models.ResNet):
    def __init__(self, block, layers):
        super(ResNetWithoutPool, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def build_backbone(num_layers, pretrained=False):
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: LeakyReluBasicBlock, 50: LeakyReluBottleneck}[num_layers]
    model = ResNetWithoutPool(block_type, blocks)
    return model


class DispEncoder(nn.Module):
    """
    Resnet without maxpool
    """
    def __init__(self, num_layers: int, pre_trained=True):
        super(DispEncoder, self).__init__()
        # make backbone
        backbone = build_backbone(num_layers, pre_trained)
        # blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        ])
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, x):
        # from shallow to deep
        features = [(x - 0.45) / 0.225]
        # features = [x]
        for block in self.blocks:
            features.append(block(features[-1]))
        return features[1:]


class DispDecoder(nn.Module):
    def __init__(self, num_ch_enc):
        """
        Initialize a disp decoder which have four output scales
        :param num_ch_enc: number of channels of encoder
        """
        super(DispDecoder, self).__init__()
        # set parameters
        self.num_ch_enc = num_ch_enc
        # components
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        # 4
        self.reduce4 = nn.Conv2d(self.num_ch_enc[4], 512, 1, bias=False)
        self.conv4 = Conv3x3(512, 512)
        self.up_conv4 = UpConv3x3(512, 512)
        self.disp_conv4 = nn.Conv2d(512, 3, 3, padding=1, padding_mode='reflect')
        # 3
        self.reduce3 = nn.Conv2d(self.num_ch_enc[3], 256, 1, bias=False)
        self.conv3 = Conv3x3(512, 256)
        self.up_conv3 = UpConv3x3(256 * 2, 256)
        self.disp_conv3 = nn.Conv2d(256, 3, 3, padding=1, padding_mode='reflect')
        # 2
        self.reduce2 = nn.Conv2d(self.num_ch_enc[2], 128, 1, bias=False)
        self.conv2 = Conv3x3(256, 128)
        self.up_conv2 = UpConv3x3(128 * 2, 128)
        self.disp_conv2 = nn.Conv2d(128, 3, 3, padding=1, padding_mode='reflect')
        # 1
        self.reduce1 = nn.Conv2d(self.num_ch_enc[1], 64, 1, bias=False)
        self.conv1 = Conv3x3(128, 64)
        self.up_conv1 = UpConv3x3(64 * 2, 64)
        self.disp_conv1 = nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect')
        # 0
        self.reduce0 = nn.Conv2d(self.num_ch_enc[0], 64, 1, bias=False)
        self.conv0 = Conv3x3(64, 64)
        self.up_conv0 = UpConv3x3(64 * 2, 64)
        self.disp_conv0 = nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect')

    def forward(self, in_features: list, frame_idx: int = 0):
        """
        Forward step
        :param in_features: features from shallow to deep
        :param frame_idx: index of frame
        :return:
        """
        assert isinstance(in_features, list)
        # get features
        f0, f1, f2, f3, f4 = in_features
        # forward
        # 4
        x4 = self.reduce4(f4)
        x4 = self.conv4(x4)
        x4 = self.leaky_relu(x4)
        x4 = self.up_conv4(x4)
        disp4 = torch.sigmoid(self.disp_conv4(x4))
        # 3
        s3 = self.reduce3(f3)
        x3 = self.conv3(x4)
        x3 = torch.cat([x3, s3], dim=1)
        x3 = self.leaky_relu(x3)
        x3 = self.up_conv3(x3)
        disp3 = torch.sigmoid(self.disp_conv3(x3))
        # 2
        s2 = self.reduce2(f2)
        x2 = self.conv2(x3)
        x2 = torch.cat([x2, s2], dim=1)
        x2 = self.leaky_relu(x2)
        x2 = self.up_conv2(x2)
        disp2 = torch.sigmoid(self.disp_conv2(x2))
        # 1
        s1 = self.reduce1(f1)
        x1 = self.conv1(x2)
        x1 = torch.cat([x1, s1], dim=1)
        x1 = self.leaky_relu(x1)
        x1 = self.up_conv1(x1)
        disp1 = torch.sigmoid(self.disp_conv1(x1))
        # 0
        s0 = self.reduce0(f0)
        x0 = self.conv0(x1)
        x0 = torch.cat([x0, s0], dim=1)
        x0 = self.leaky_relu(x0)
        x0 = self.up_conv0(x0)
        disp0 = torch.sigmoid(self.disp_conv0(x0))
        # pack and return
        outputs = {
            'disp0': disp0,
            'disp1': disp1,
            'disp2': disp2,
            'disp3': disp3,
            'disp4': disp4
        }
        return outputs


class DispNet(nn.Module):
    def __init__(self, opt):
        super(DispNet, self).__init__()

        self.opt = opt

        # networks
        self.DepthEncoder = DispEncoder(self.opt.depth_num_layers, pre_trained=False)
        self.DepthDecoder = DispDecoder(self.DepthEncoder.num_ch_enc)

    def forward(self, inputs):
        outputs = self.DepthDecoder(self.DepthEncoder(inputs))
        return outputs


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer='instance_norm'):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        use_bias = False

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        # nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                get_norm_layer(norm_layer, ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            get_norm_layer(norm_layer, ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)
# ==========================================main layers=================================================================
