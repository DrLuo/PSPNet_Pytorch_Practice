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
from util.utils import intersectionAndUnion, check_mkdir, AverageMeter
from util import augmentations as aug


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Pyramid Scene Parsing Testing on VOC With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'ADE20K'],
                    type=str, help='VOC or ADE20K')
# TODO: Confirm the data dir
parser.add_argument('--data_dir', default='E:\Code\VOC2012',
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

check_mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def test_VOC():

    value_scale = 255
    mean = cfg['mean']
    std = cfg['std']
    mean = [item * value_scale for item in mean]
    std = [item * value_scale for item in std]


    test_transform = aug.Compose([aug.ToTensor(), aug.Normalize(mean=mean, std=std)])
    test_data = pascal_voc.VOCDataset(split='val', data_dir=args.data_dir, transform=test_transform)
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
    if args.cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    print('loading pretrained model')
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    print("Finished loading model!")

    save_file_path = os.path.join(args.save_folder, 'test_result.txt')
    test_net(test_loader, model, cfg['num_class'], cfg['ignore_label'], save_file_path)


def test_net(test_loader, net, nclass, ignore_label, save_file ):
    print(">>>>>>>> Start Test <<<<<<<<")
    net.eval()
    t0 = time.time()
    '''
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    '''
    pa = []
    iou_memory = Cal_IoU(nclass)
    # count = np.zeros(nclass)
    # print(count)
    for i, (input, label) in enumerate(test_loader):
        # input = np.squeeze(input.numpy(), axis=0)
        # image = np.transpose(input, (1,2,0))
        label = np.squeeze(label.numpy(), axis=0)
        # h, w = label.shape
        # prediction = np.zeros((h, w, nclass), dtype=float)

        if args.cuda is True:
            input = input.cuda()
        prediction = net(input)
        map = prediction.max(1)[1]
        map = map.cpu()

        map = ToNumpy(map)
        map = np.asarray(map)
        map = np.squeeze(map, axis=2)

        # evaluate pixal-wise acc
        # way 1:
        # acc = cal_acc(seg=map, target=label, ignore_label=ignore_label)

        # evaluate IOU
        intersection, union, target = intersectionAndUnion(map, label, nclass, ignore_label)
        iou_memory.add(intersection, union, nclass)

        # evaluate pixal-wise acc
        # way 2（faster):
        acc = sum(intersection) / sum(target) * 100


        # evaluate IOU way 2
        # way official: add first then div
        # intersection_meter.update(intersection)
        # union_meter.update(union)

        pa.append(acc)
        print('timer: %.4f sec.' % (time.time() - t0), end='')
        print(' || image {} || pixel acc: {} %'.format(i + 1, acc))

        t0 = time.time()

    print(">>>>>> Testing Result <<<<<<")
    f = open(save_file, mode='w')
    f.write("PSPNet evaluated on VOC2012 test dataset\n\n")
    # calculate global benchmark
    mPA = np.mean(pa)
    print("mean Pixel Acc is {} %".format(mPA))
    f.write("mean Pixel Acc is {} %".format(mPA))

    iou_class = iou_memory.sum / iou_memory.count
    print("IOU of each class:")
    print(iou_class)
    f.write("\nIOU of each class:")
    f.write(repr(iou_class))

    mIoU = np.mean(iou_class)
    mIoU = mIoU * 100
    print("mean IoU is {} %".format(mIoU))
    f.write("\nmean IoU is {} %".format(mIoU))
    f.close()
    # print(iou_memory.sum / iou_memory.count)
    # print(np.mean(IoU_class, axis=0))
    '''
    # IOU way 2 (official)
    iou_class = intersection_meter.sum / union_meter.sum
    print("Meter:")
    print(iou_class)
    '''



def cal_acc(seg, target, ignore_label):
    '''

    :param seg: the predicted segmentation result
    :param target: the label of test dataset
    :param ignore_label:
    :return: pixel accuracy sum(TP+TN)/sum(all) %
    '''
    h, w = seg.shape
    assert seg.shape == target.shape
    all_pixel = h * w
    i = 0
    hit = 0
    miss = 0
    while i < h:
        j = 0
        while j < w:
            if target[i, j] == ignore_label:
                all_pixel = all_pixel - 1
            elif seg[i,j] == target[i,j]:
                hit = hit + 1
            else:
                miss = miss + 1
            j = j + 1
        i = i + 1

    # calculate pixel-wise accuracy of one sample
    assert hit + miss == all_pixel
    print(all_pixel)
    acc = hit / all_pixel * 100
    return acc


class Cal_IoU(object):
    def __init__(self, nclass):
        self.reset(nclass)

    def reset(self, nclass):
        self.count = np.zeros(nclass)
        self.sum = np.zeros(nclass)

    def add(self, intersection, union, nclass):
        i = 0
        iou = np.zeros(nclass)

        while i < nclass:
            if union[i] != 0:
                iou[i] = intersection[i] / union[i]
                self.count[i] = self.count[i] + 1
            i = i + 1
        self.sum += iou




def ToNumpy(image):
    # Convert torch.FloatTensor (C x H x W) to numpy.ndarray (H x W x C)
    image = image.numpy()
    image = image.transpose((1,2,0))
    return image



def Normalize(image, mean, std):
    if std is None:
        assert len(mean) > 0
    else:
        assert len(mean) == len(std)

    if std is None:
        for t, m in zip(image, mean):
            t.sub_(m)   # 减去均值
    else:
        for t, m, s in zip(image, mean, std):
            t.sub_(m).div_(s)
    return image





if __name__ == '__main__':
    test_VOC()
