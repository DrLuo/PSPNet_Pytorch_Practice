import time
import os
import logging
import argparse

import cv2
import numpy as np
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.functional as F

from models import pspnet

from data import pascal_voc

from config.config import cfg
from util.utils import intersectionAndUnion, check_mkdir
from util import augmentations as aug


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Pyramid Scene Parsing Testing on VOC With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'ADE20K'],
                    type=str, help='VOC or ADE20K')
# TODO: Confirm the data dir
parser.add_argument('--data_dir', default='VOC2012',
                    help='the Root Dir of your dataset')
parser.add_argument('--weight', default='weights/PSPNet_VOC.pth',
                    help='Pretrained base model')
parser.add_argument('--color_path', default='data/voc2012/voc2012_colors.txt',
                    help='path of dataset colors')
parser.add_argument('--name_path', default='data/voc2012/voc2012_names.txt',
                    help='path of dataset category names')
parser.add_argument('--index_start', default=0, type=int,
                    help='evaluation start index in list')
parser.add_argument('--index_step', default=0, type=int,
                    help='evaluation step index in list, 0 means to end')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='eval/',
                    help='the save folder of test result')
args = parser.parse_args()

check_mkdir()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def test_VOC():

    value_scale = 255
    mean = cfg['mean']
    std = cfg['cfg']
    mean = [item * value_scale for item in mean]
    std = [item * value_scale for item in std]


    test_transform = aug.Compose([aug.ToTensor()])
    test_data = pascal_voc.VOCDataset(split='test', data_dir=args.data_dir, transform=test_transform)
    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(test_data.ids)
    else:
        index_end = min(index_start + args.index_step, len(test_data.ids))
    ids = test_data.ids[index_start : index_end]
    test_loader = data.DataLoader(test_data, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    colors = np.loadtxt(args.color_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.name_path)]
    print("Loading files finished!")

    # build model
    model = pspnet.PSPNet(layers=cfg['layers'], nclass=cfg['num_class'], bins=cfg['bins'], zoom_factor=cfg['zoom_factor'])

    # logger.info(model)
    model = torch.nn.DataParallel(model)
    if args.cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    print('loading pretrained model')
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    print("Finished loading model!")

    test_net(test_loader, model, cfg['num_class'], mean)


def test_net(test_loader, net, nclass, mean, std, save_floder ):
    print(">>>>>>>> Start Test <<<<<<<<")
    end = time.time()
    for i, (input, _) in enumerate(test_loader):
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1,2,0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, nclass), dtype=float)









if __name__ == '__main__':
    test_VOC()