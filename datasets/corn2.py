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

# resize size
_RESIZE_SIZE = (512, 512)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(directory, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory

    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class UnalignedDataset(BaseDataset):
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
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')               # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')               # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)                                        # get the size of dataset A
        self.B_size = len(self.B_paths)                                        # get the size of dataset B

        self._need_augment = opt.augment

        self._width, self._height = opt.width, opt.height
        self._need_resize = opt.resize
        # transforms
        self._to_tensor = ToTensor()
        if self._need_augment:
            self._flip = RandomHorizontalFlip(0.5)
        if self._need_resize:
            self._crop = CenterCrop(self._width, self._width // 2)
            self._resize = Resize(*_RESIZE_SIZE)
        else:
            self._crop = CenterCrop(self._width, self._height)
            self._resize = None
        # print message
        print('Total items: {}.'.format(max(self.A_size, self.B_size)))

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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A = cv2.imread(A_path, 1)
        B = cv2.imread(B_path, 1)

        A = self._crop(A, inplace=False, unpack=True)
        B = self._crop(B, inplace=False, unpack=True)

        if self._resize is not None:
            A = self._resize(A)
            B = self._resize(B)
        # augment
        if self._need_augment:
            A = self._flip(A)
            B = self._flip(B)

        A = self._to_tensor(A)
        B = self._to_tensor(B)

        return {'low_q': A, 'high_q': B}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
