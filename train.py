import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
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
    max_iter = cfg['max_iter']

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

    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {} ...'.format(args.resume))
        net.load_weights(args.resume)
    else:
        base_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        net.vgg.load_state_dict(base_weights)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg['ignore_label'])

    net.train()

    #loss counters
    main_loss = 0
    aux_loss = 0
    epoch = 0
    print("Loading the dataset...")

    epoch_size = len(dataset) // args.batch_size
    print('Train PSPnet on : ', args.dataset)
    print('Epoch size is ', epoch_size)

    step_index = 0

    # 可视化


    # load data
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True,
                                  drop_last=True)
    t0 = time.time()

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, max_iter):
        if iteration != 0 and (iteration % epoch_size == 0):
                        # reset epoch loss counters
            main_loss = 0
            aux_loss = 0
            epoch += 1
            batch_iterator = iter(data_loader)

        poly_learning_rate(optimizer, args.power, iteration, max_iter)

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, label = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            label = label.cuda()

        # forward
        out, aux = net(images)
        # backprop
        optimizer.zero_grad()
        main_loss = criterion(out, label)
        aux_loss = criterion(aux, label)
        loss = main_loss + cfg['aux_weight'] * aux_loss
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (time.time() - t0))
            print('iter' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
            t0 = time.time()

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), os.path.join(args.save_folder, 'PSPNet_' + repr(iteration // 1000) + 'K.pth' ))


    torch.save(net.state_dict(), os.path.join(args.save_folder, 'PSPNet_' + args.dataset + '.pth'))




'''
def validation():
    if
'''

def poly_learning_rate(optimizer, power, iter, max_iter):
    lr = args.lr * ((1 - (iter/max_iter)) ** power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #return lr


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == "__main__":

    train()
    '''
    print(args.lr)
    print(args.data_dir)
    print(args.basenet)
    print(args.dataset)
    iteration = 30000
    print(os.path.join(args.save_folder, 'PSPNet_' + repr(iteration // 1000) + 'K.pth' ))
    print(os.path.join(args.save_folder, 'PSPNet_' + args.dataset + '.pth'))
    '''