import torch
import torchvision
import argparse

import numpy as np
import cv2

from models import pspnet

from data import pascal_voc

from config.config import cfg
import collections
import numbers
# from util import augmentations as aug
from util.utils import intersectionAndUnion, check_mkdir


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Pyramid Scene Parsing Application Demo')
demo_set = parser.add_mutually_exclusive_group()
parser.add_argument('--img_path', default='E:\Code\PSPnet\image\demo.jpg',
                    help='The raw image for demo')
parser.add_argument('--weight', default='weights/PSPNet_VOC.pth',
                    help='Pretrained base model')
parser.add_argument('--color_path', default='data/voc2012/voc2012_colors.txt',
                    help='path of dataset colors')
parser.add_argument('--name_path', default='data/voc2012/voc2012_names.txt',
                    help='path of dataset category names')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='eval/',
                    help='the save folder of test result')
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

def main():
    img_path = args.img_path

    value_scale = 255
    mean = cfg['mean']
    std = cfg['std']
    mean = [item * value_scale for item in mean]
    std = [item * value_scale for item in std]
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.name_path)]
    print("Loading files finished!")

    model = pspnet.PSPNet(layers=cfg['layers'], nclass=cfg['num_class'], bins=cfg['bins'], zoom_factor=cfg['zoom_factor'])

    image = cv2.imread(img_path, cv2.IMREAD_COLOR) # BGR 3 channel ndarray wiht shape H * W * 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop = Crop((cfg['test_h'], cfg['test_w']), padding=mean)
    image = crop(image)
    # cv2.imshow('raw', image)
    # cv2.waitKey(0)

    h, w, _ = image.shape
    image = ToTensor(image)
    input = Normalize(image, mean, std)
    input = input.unsqueeze(0)
    print(input.size())

    if torch.cuda.is_available():
        model = model.cuda()
        input = input.cuda()
    # model.train()
    print('loading pretrained model from %s' % args.weight)
    model.load_state_dict(torch.load(args.weight), strict=False)
    print('loading pretrained model finished!')

    # start inference
    model.eval()
    output = model(input)
    prediction = output.max(1)[1]
    #prediction = prediction.squeeze()
    print(prediction.size())
    map = prediction.cpu().numpy()
    print(map)
    #print(prediction)




def ToTensor(image):
    if not isinstance(image, np.ndarray):
        raise (RuntimeError("demo.ToTensor() only handle np.ndarray"
                            "[eg: data readed by cv2.imread()].\n"))

    if len(image.shape) > 3 or len(image.shape) < 2:
        raise (RuntimeError("ndarray image must have 3 dims or 2 dims.\n"))
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    image = torch.from_numpy(image.transpose((2, 0, 1)))
    if not isinstance(image, torch.FloatTensor):
        image = image.float()
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


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
        Args:
            size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is made.
        """
    def __init__(self, size, padding=None):
        if isinstance(size, int):   # size:裁剪输出图像尺寸
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
            and isinstance(size[0], int) and isinstance(size[1], int) \
            and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))


        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))

    def __call__(self, image):
        h, w,_ = image.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        # 边界扩充
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("augmentation.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h-pad_h_half, pad_w_half, pad_w-pad_w_half, borderType=cv2.BORDER_CONSTANT, value=self.padding)
        # 裁剪
        #print(label.shape)
        h, w, _ = image.shape  #扩充后的图像尺寸

        h_off = (h - self.crop_h) // 2
        w_off = (w - self.crop_w) // 2
        image = image[h_off:(h_off + self.crop_h), w_off:(w_off + self.crop_w), :]

        return image


if __name__ == '__main__':
    main()

