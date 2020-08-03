import os
import torch.utils.data
import numpy as np
import  xml.etree.cElementTree as ET
from PIL import Image

VOC_CLASSES = ( # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
NUM_CLASS = 21

class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, is_train=True, keep_difficult=False):
        """VOC格式数据集
                Args:
                    data_dir: VOC格式数据集根目录，该目录下包含：
                        Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
                    split： train、test 或者 eval， 对应于 ImageSets/Main/train.txt,eval.txt
                """
        # 类别
        self.class_name = cfg
        self.is_train = is_train

        self.keep_difficult = keep_difficult


    def get_img_label(self, index):
        img_id = self



    def __getitem__(self, item):
        img = Image.open(self).convert('RGB')