import random
import math
import numpy as np
import numbers
import collections
import cv2

import torch

'''
图像数据增强：
1. random mirror
2. random resize between 0.5, 2

(for VOC and ImageNet)
3. random rotate between -10°, 10°
4. random Gaussian blur
'''


class Compose(object):
    # input: transform: 数据增强的变换的操作集合
    # eg: augmentation.Compose([transform.RandomResize([0.5, 2.0]), transform.ToTensor()])
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, label):
        for t in self.transform:
            image, label = t(image, label)
        return  image, label


class ToTensor(object):
    # Convert numpy.ndarray (H x W x C) to torch.FloatTensor (C x H x W)
    def __call__(self, image, label):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))

        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("ndarray image must have 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("ndarray label must be with 2 dims./n"))

        image = torch.from_numpy(image.transpose((2,0,1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image, label


class Normalize(object):

    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)   # 减去均值
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label


class Resize(object):
    # 输入给定的size (h, w)， 修改size和label，输出image size为(h, w)
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        # 判断输入size可迭代且是2-element tuple or list in the order of (h, w)
        self.size = size

    def __call__(self, image, label):
        # cv2.resize(image, (w,h), interpolation)
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)  #线性插值
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST) #临近插值
        return image, label


class RandomResize(object):
    # randomly resize image & label
    # resize factor [resize_min, resize_max]
    # aspect_ratio [min, max]
    def __init__(self, size, aspect_ratio = None):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], numbers.Number) and isinstance(size[1], numbers.Number) \
                and 0 < size[0] < size[1]:
            self.size = size
        else:
            raise (RuntimeError("Augmentation.RandomResize() size param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("Augmentation.RandomResize() aspect_ratio param error.\n"))

    def __call__(self, image, label):
        temp_size = self.size[0] + (self.size[1] - self.size[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_w = temp_size * temp_aspect_ratio
        scale_factor_h = temp_size / temp_aspect_ratio
        image = cv2.resize(image, fx=scale_factor_w, fy=scale_factor_h, interpolation='INTER_LINEAR')
        label = cv2.resize(label, fx=scale_factor_w, fy=scale_factor_h, interpolation='INTER_NEAREST')
        return image, label


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
        Args:
            size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is made.
        """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
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

        assert crop_type in ['center', 'random']
        self.crop_type = crop_type

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
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        # 边界扩充
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("augmentation.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h-pad_h_half, pad_w_half, pad_w-pad_w_half, borderType=cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(image, pad_h_half, pad_h-pad_h_half, pad_w_half, pad_w-pad_w_half, borderType=cv2.BORDER_CONSTANT, value=self.ignore_label)
        # 裁剪
        h, w = label.shape  #扩充后的图像尺寸
        if self.crop_type == 'random':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:   # 中心裁剪
            h_off = (h - self.crop_h) // 2
            w_off = (w - self.crop_w) // 2
        image = image[h_off:(h_off + self.crop_h), w_off:(w_off + self.crop_w), :]
        label = label[h_off:(h_off + self.crop_h), w_off:(w_off + self.crop_w)]
        return image, label



class RandomRotate(object):
    # Randomly rotate image & label
    # degree from [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):       # ignore_label:忽略对某类label的学习，255表示不忽略
        assert (isinstance(rotate, collections.Iterable) and len(rotate)==2)
        if isinstance (rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("Augmentation.RandomRotate() rotate param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandomRotate() should be a number list with length of 3. \n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)    # 获得旋转矩阵
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label



# 随机水平镜像
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label

# 随机垂直镜像
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    # 随机高斯模糊，滤波器尺寸5*5，标准差为0
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class RGB2BGR(object):

    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label

class BGR2RGB(object):

    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label