# @Time: 2023/9/3 15:13
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 分类网络数据加载
# -------------------------------
import cv2
import torch
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tools.config import args


class classDataset(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform
        self.classes = sorted(os.listdir(self.dataset_root))
        self.classes_to_idx = {cls : idx for idx, cls in enumerate(self.classes)}
        self.data = self._load_dataset()


    def _load_dataset(self):
        data = []
        for cls in self.classes:
            class_dir = os.path.join(self.dataset_root, cls)
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.png')):
                    img_path = os.path.join(class_dir, filename)
                    label = self.classes_to_idx[cls]
                    data.append((img_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

'''
# test load_dataset
image_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/trainingset\class_mul\dataset_class/'

# define transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_data = classDataset(image_folder, transform=transform)
data_loder = DataLoader(class_data, batch_size=1, shuffle=False)

# 打印数据集信息
print(f"数据集包含 {len(class_data)} 个样本")
print(f"类别数目: {len(class_data.classes)}")
print(f"类别列表: {class_data.classes}")

for image, label in data_loder:
    pass
'''
