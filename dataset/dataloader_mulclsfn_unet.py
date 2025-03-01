# @Time: 2023/7/28 13:59
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 构建数据集读取
# -------------------------------


import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tools.config import args
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# 数据增强转换
data_transform = transforms.Compose([
    transforms.ToPILImage(),  # 将数组转换为PIL图像
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.RandomVerticalFlip(),  # 随机垂直翻转图像
        transforms.RandomRotation(45),  # 随机旋转图像，最多45度
    ], p=0.5),  # p为应用数据增强操作的概率，此处设为0.5
    transforms.ToTensor(),  # 将PIL图像转换为张量
])

class FNDataset(Dataset):

    def __init__(self, imgs_path, label_path, num_classes=args.num_classes, transform=None, split=None):
        self.transform = transform
        self.imgs_path = imgs_path
        self.label_path = label_path
        self.num_classes = num_classes
        self.split = split

    def __len__(self):
        return len(os.listdir(self.imgs_path))


    def __getitem__(self, idx):
        img_name = os.listdir(self.imgs_path)[idx]
        image_path = os.path.join(self.imgs_path, img_name)
        imgA = cv2.imread(self.imgs_path + img_name)
        imgA = cv2.resize(imgA, (128,128))
        imgA = Image.fromarray(imgA)  # 将NumPy数组转换为PIL Image

        imgB = cv2.imread(self.label_path + img_name.split('.')[0] + '.png', 0)
        imgB = cv2.resize(imgB, (128,128))


        # 对图像进行亮度阈值化处理
        imgA = np.array(imgA)
        threshold_value = 100
        imgA[imgA < threshold_value] = 0
        imgA = Image.fromarray(imgA)


        img_label = imgB

        img_label_onehot = onehot(img_label, self.num_classes)   # w * H * n_class

        img_label_onehot = img_label_onehot.transpose(2, 0, 1)  # n_class * w * H

        onehot_label = torch.FloatTensor(img_label_onehot)
        if self.split == 'train':
          if self.transform == 'data_transform':
              imgA, img_label, onehot_label = self.transform(imgA), self.transform(img_label), self.transform(onehot_label)
          else:
              imgA = self.transform(imgA)
        elif self.split == 'test':
          if self.transform:
            imgA = self.transform(imgA)

        return imgA, img_label, onehot_label, img_name, image_path

def onehot(data, n):
    """ onehot ecoder """
    buf = np.zeros(data.shape + (n,))
    nmsk = np.arange(data.size) * n + data.ravel()

    nmsk = np.clip(nmsk, 0, buf.size - 1)  # 将索引值限制在合法范围内
    buf.ravel()[nmsk] = 1
    return buf


class FNDataset_V1(Dataset):

    def __init__(self, imgs_path, num_classes=args.num_classes, transform=None):
        self.transform = transform
        self.imgs_path = imgs_path
        self.num_classes = num_classes

    def __len__(self):
        return len(os.listdir(self.imgs_path))


    def __getitem__(self, idx):
        img_name = os.listdir(self.imgs_path)[idx]
        image_path = os.path.join(self.imgs_path, img_name)
        imgA = cv2.imread(self.imgs_path + img_name)
        imgA = cv2.resize(imgA, (128,128))
        imgA = Image.fromarray(imgA)  # 将NumPy数组转换为PIL Image


        # 对图像进行亮度阈值化处理
        imgA = np.array(imgA)
        threshold_value = 100
        imgA[imgA < threshold_value] = 0
        imgA = Image.fromarray(imgA)

        if self.transform:
            imgA = self.transform(imgA)

        return imgA, img_name, image_path


