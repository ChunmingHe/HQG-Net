import os.path as osp
from argparse import ArgumentParser

from mmcv import Config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets import newCORNDataset
from models import MODELS


def parse_args():
    parser = ArgumentParser(description='Training with DDP.')
    parser.add_argument('--config',
                        type=str,
                        default='rnw_star')
    parser.add_argument('--gpus',
                        type=int,
                        default=4)
    parser.add_argument('--work_dir',
                        type=str,
                        default='checkpoints')
    args = parser.parse_args()
    return args


def main():
    # parse args
    args = parse_args()

    # parse cfg
    cfg = Config.fromfile(osp.join(f'configs/{args.config}.yaml'))

    # show information
    print(f'Now training with {args.config}...')

    # prepare data loader
    dataset = newCORNDataset(cfg.dataset)
    loader = DataLoader(dataset, cfg.imgs_per_gpu, shuffle=True, num_workers=cfg.workers_per_gpu, drop_last=True)

    # define model
    model = MODELS.build(name="rnw_star", option=cfg)

    # define trainer
    work_dir = osp.join(args.work_dir, args.config)
    # save checkpoint every 'cfg.checkpoint_epoch_interval' epochs
    checkpoint_callback = ModelCheckpoint(dirpath=work_dir,
                                          save_weights_only=True,
                                          save_top_k=-1,
                                          filename='checkpoint__{epoch}',
                                          every_n_epochs=cfg.checkpoint_epoch_interval)
    trainer = Trainer(
                      # accelerator="ddp",
                      default_root_dir=work_dir,
                      # gpus=args.gpus,
                      gpus=[5],
                      num_nodes=1,
                      max_epochs=cfg.total_epochs,
                      callbacks=[checkpoint_callback]
                      )

    # training
    trainer.fit(model, loader)


if __name__ == '__main__':
    main()
