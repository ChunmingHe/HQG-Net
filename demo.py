import argparse
import os
import os.path as osp

import cv2
import numpy as np
import pytorch_lightning
import torch
from mmcv import Config
from torchvision.transforms import ToTensor
from tqdm import tqdm
from models import MODELS


# output dir
_OUT_DIR = 'evaluation/star_result/'


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data/CORN_2/testA', help='Tested dataset.')
    parser.add_argument('--config', type=str, default='rnw_star')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/rnw_star/checkpoint_epoch=179.ckpt')
    parser.add_argument('--ref_img_path', type=str, default='data/CORN_2/trainB/aug_1_batch_1_54.tif')
    return parser.parse_args()


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


if __name__ == '__main__':
    # parse args
    args = parse_args()
    # config
    cfg = Config.fromfile(osp.join('configs/', f'{args.config}.yaml'))
    # device
    device = torch.device('cuda')
    # read list file
    image_list = make_dataset(args.root_dir)
    # model
    net: pytorch_lightning.LightningModule = MODELS.build(name='rnw_star', option=cfg)
    net.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['state_dict'])
    net.to(device)
    net.eval()
    print('Successfully load weights from check point {}.'.format(args.checkpoint))
    # transform
    to_tensor = ToTensor()
    # visualization
    visualization_dir = os.path.join(_OUT_DIR, 'visualization/')
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # no grad
    with torch.no_grad():
        # predict
        for idx, item in enumerate(tqdm(image_list)):
            # read image
            rgb = cv2.imread(item, 1)
            ref_img = cv2.imread(args.ref_img_path, 1)
            # to tensor
            t_rgb = to_tensor(rgb).unsqueeze(0).to(device)
            ref_img = to_tensor(ref_img).unsqueeze(0).to(device)
            # feed into net
            y_trg = torch.LongTensor([1]).to(device)
            outputs = net(t_rgb, ref_img, y_trg)
            disp = outputs

            # visualization
            disps = disp.cpu()[0, :, :, :].numpy() * 255
            disps = disps.transpose([1, 2, 0])
            out_fn = os.path.join(visualization_dir, 'aug_{}.png'.format((item.split("/")[-1]).split('.')[0]))
            cv2.imwrite(out_fn, disps)

    # show message
    tqdm.write('Done.')
