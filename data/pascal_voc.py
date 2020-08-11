import os
from torch.utils import data
import numpy as np
import cv2
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
DATA_PATH = ""

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']


def read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids


def extract_image_label(file_name, split='train', data_dir=None):
    #assert split in ['train', 'val', 'test']
    image_name = os.path.join(data_dir, "JPEGImages", "{}.jpg", file_name)
    label_name = os.path.join(data_dir, "SegmentationClass", "{}.png", file_name)
    return image_name, label_name






class VOCDataset(data.Dataset):

    def __init__(self, split='train', data_dir = None, data_list=None, transform=None, mean=(128, 128, 128), ignore_label=255):
        """VOC格式数据集
                Args:
                    data_dir: VOC格式数据集根目录，该目录下包含：
                        Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
                    split： train、test 或者 eval， 对应于 ImageSets/Main/train.txt,eval.txt
                """
        # 类别
        assert split in ['train', 'val', 'test']
        self.split = split
        self.data_list = data_list
        self.data_dir = data_dir
        self.transform = transform
        self.mean = mean

        self.ignore_label = ignore_label

        # 从train.txt或val.txt文件中读取图片id
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Segmentation", "{}.txt".format(self.split))
        self.ids = read_image_ids(image_sets_file)
        print("Totally {} samples in {} set.".format(len(self.ids), split))


    def __getitem__(self, index):
        file_name = self.ids[index]
        image_path = os.path.join(self.data_dir, "JPEGImages", "{}.jpg", file_name)
        label_path = os.path.join(self.data_dir, "SegmentationClass", "{}.png", file_name)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)    # BGR ndarray H x W x C
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)    # 1 channel Gray ndarray H x W

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label


    def __len__(self):
        return len(self.ids)

