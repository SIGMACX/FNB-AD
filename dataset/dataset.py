# @Time: 2023/7/26 21:56
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  1. 分割任务数据集读取
# -------------------------------

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_folder, mask_folder, transform=None):
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.transform = transform

        self.image_files = [f for f in os.listdir(data_folder) if f.endswith('.png')]
        self.num_samples = len(self.image_files)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.image_files[idx])

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
