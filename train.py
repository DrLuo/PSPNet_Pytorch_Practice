import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse

import numpy as np
import os
import sys
import time

from data import pascal_voc
from models import pspnet
from util import augmentations as aug
from config.config import cfg

from tensorboardX import SummaryWriter


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Pyramid Scene Parsing Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'ADE20K'],
                    type=str, help='VOC or ADE20K')
# TODO: Confirm the data dir
parser.add_argument('--data_dir', default='VOC2012',
                    help='the Root Dir of your dataset')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--power', default=0.9, type=float,
                    help='power for poly learing policy')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():

    value_scale = 255
    mean = cfg['mean']
    std = cfg['cfg']
    mean = [item * value_scale for item in mean]
    std = [item * value_scale for item in std]

    if args.dataset == 'VOC':
        transform = aug.Compose([
            aug.RandomResize([cfg['resize_min'], cfg['resize_max']]),
            aug.RandomRotate([cfg['rotate_min'], cfg['rotate_max']]),
            aug.RandomGaussianBlur(),
            aug.RandomHorizontalFlip(),
            aug.Crop([cfg['train_h'], cfg['train_w']], crop_type='rand', padding=mean),
            aug.ToTensor(),
            aug.Normalize()
        ])
        dataset = pascal_voc.VOCDataset(split='train', data_dir=args.data_dir, transform=transform)
    elif args.dataset == 'ADE20K':
        print("Currently only support VOC")
        return 0

    net = pspnet.PSPNet(layers=cfg['layers'], nclass=cfg['num_class'], bins=cfg['bins'], zoom_factor=cfg['zoom_factor'])
    nn.CrossEntropyLoss(ignore_index=cfg['ignore_label'])

    net.train()



    #criterion =


'''
def validation():
    if
'''


if __name__ == "__main__":

    print('resnet50')
    print(args.data_dir)