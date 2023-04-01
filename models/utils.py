import torch
import torch.nn as nn
import random
import numpy as np
from math import exp
import torch.nn.functional as F
from typing import List


# ===========================================   utils  =================================================================
def _set_requires_grad(models: [List[nn.Module], nn.Module], requires_grad: bool):
    """
    Freeze model's parameters
    :param models: model or list of models
    :param requires_grad: if requires grad
    :return:
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        if m is not None:
            for param in m.parameters():
                param.requires_grad = requires_grad


def freeze_model(models):
    _set_requires_grad(models, False)


def unfreeze_model(models):
    _set_requires_grad(models, True)


class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
# ===========================================   utils  =================================================================


# ===========================================basic loss=================================================================
def robust_l1(pred, target):
    eps = 1e-3
    return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = nn.Parameter(data=_2D_window.expand(channel, 1, window_size, window_size).contiguous(), requires_grad=False)
    return window


def _ssim(img1, img2, window, window_size, channel, K1=0.01, K2=0.03, L=1.0, size_average=True):
    assert img1.size() == img2.size()
    noise = torch.Tensor(np.random.normal(0, 0.01, img1.size())).cuda(img1.get_device())
    new_img1 = clip_by_tensor(img1 + noise, torch.Tensor(np.zeros(img1.size())).cuda(img1.get_device()), torch.Tensor(np.ones(img1.size())).cuda(img1.get_device()))
    new_img2 = clip_by_tensor(img2 + noise, torch.Tensor(np.zeros(img2.size())).cuda(img2.get_device()), torch.Tensor(np.ones(img2.size())).cuda(img2.get_device()))
    mu1 = F.conv2d(new_img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(new_img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(new_img1 * new_img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(new_img2 * new_img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(new_img1 * new_img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2.0

    ssim_map = (sigma12 + C3) / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C3)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class StructureLoss(nn.Module):
    """Define Structure Loss.

    Structure Loss reflects the structural difference between inputs and outputs to some extent.
    """

    def __init__(self, channel=1, window_size=11, crop_size=384, size_average=True):
        """Initialize the StructureLoss class.

        Parameters:
            channel (int) - - number of channels
            window_size (int) - - size of window
            size_average (bool) - - average of batch or not
        """
        super(StructureLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.crop_size = crop_size
        self.window = create_window(window_size, channel)

    def forward(self, img1, img2):
        (_, channel, height, width) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window.data = window
            self.channel = channel

        inputs1 = (img1[:, :, (height-self.crop_size)//2:(height+self.crop_size)//2, (width-self.crop_size)//2:(width+self.crop_size)//2] + 1.0) / 2.0
        inputs2 = (img2[:, :, (height-self.crop_size)//2:(height+self.crop_size)//2, (width-self.crop_size)//2:(width+self.crop_size)//2] + 1.0) / 2.0

        return 1.0 - _ssim(inputs1, inputs2, window, self.window_size, channel, self.size_average)


class LuminanceLoss(nn.Module):
    """Define Illumination Regularization.

    Illumination Regularization reflects the degree of inputs' uneven illumination to some extent.
    """

    def __init__(self, patch_height, patch_width, crop_size=384):
        """Initialize the LuminanceLoss class.

        Parameters:
            patch_height (int) - - height of patch
            patch_width (int) - - width of patch
        """
        super(LuminanceLoss, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.crop_size = crop_size
        self.avgpool = nn.AvgPool2d((patch_height, patch_width), stride=(patch_height, patch_width))

    def forward(self, inputs):
        height = inputs.size()[2]
        width = inputs.size()[3]
        assert height >= self.crop_size and width >= self.crop_size
        assert self.crop_size % self.patch_height == 0 and self.crop_size % self.patch_width == 0, "Patch size Error."

        crop_inputs = (inputs[:, :, (height-self.crop_size)//2:(height+self.crop_size)//2, (width-self.crop_size)//2:(width+self.crop_size)//2] + 1.0) / 2.0
        # [batch_size, channels, self.crop_size, self.crop_size] --> [batch_size, channels, 1, 1]
        global_mean = torch.mean(crop_inputs, [2, 3], True)
        # [batch_size, channels, self.crop_size, self.crop_size] --> [batch_size, channels, N, M]
        D = self.avgpool(crop_inputs)
        E = D - global_mean.expand_as(D)  # [batch_size, channels, N, M]
        upsample = nn.Upsample(size=[self.crop_size, self.crop_size], mode='bicubic', align_corners=False)
        R = upsample(E)  # [batch_size, channels, self.crop_size, self.crop_size]

        return torch.abs(R).mean()


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise ValueError('Unknown gan mode: {}.'.format(self.gan_mode))
        return loss


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
# ==========================================basic loss==================================================================
