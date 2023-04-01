# -*- coding: utf-8 -*-

import os
from datasets.base_dataset import BaseDataset
import cv2
import random
from transforms import Resize, RandomHorizontalFlip, CenterCrop
from torchvision.transforms import ToTensor

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(directory):
    images = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory

    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class pre_train_dataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'train' + 'A')               # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'train' + 'B')               # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B))  # load images from '/path/to/data/trainB'

        self.path = self.A_paths + self.B_paths

        self._need_augment = opt.augment

        self._width, self._height = opt.width, opt.height

        # transforms
        self._to_tensor = ToTensor()
        if self._need_augment:
            self._flip = RandomHorizontalFlip(0.5)
        self._crop = CenterCrop(self._width, self._height)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        img_path = self.path[index]
        img = cv2.imread(img_path, 1)

        img = self._crop(img, inplace=False, unpack=True)

        # augment
        if self._need_augment:
            img = self._flip(img)

        img = self._to_tensor(img)

        return img

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.path)
