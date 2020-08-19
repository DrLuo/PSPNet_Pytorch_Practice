import time
import os
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.functional as F


from util.utils import intersectionAndUnion, check_mkdir


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
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
args = parser.parse_args()


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def test_VOC():




