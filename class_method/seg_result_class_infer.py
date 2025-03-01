# @Time: 2023/9/5 9:40
# @Author: ChenXi
# -*- coding: utf-8 -*-

# ------------------------------------------------------
# class inference task of FNBU-Net segmentation result
# ------------------------------------------------------

import os
import time

import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score
from PIL import Image

from class_method.models.mobilenet import mobilenet
from class_method.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from class_method.models.densenet import densenet121, densenet161, densenet169, densenet201
from class_method.models.resnext import resnext50, resnext101, resnext152
from class_method.models.vggnet import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from class_method.models.googlenet import googlenet

from class_method.load_dataset.class_load_dataset import classDataset
from tools.config import args
from class_method.train_class import test

def seg_result_class_infer(dataset_root, num_classes, batch_size, snapshot_dir):
    seg_result = classDataset(dataset_root=dataset_root,
                              transform = transforms.Compose([
                                  transforms.Resize([128, 128]),
                                  transforms.RandomRotation(10),
                                  transforms.ToTensor()
                              ]))
    seg_result_loader = DataLoader(seg_result, batch_size=batch_size, shuffle=True, num_workers=0)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet34(num_classes)

    model = nn.DataParallel(model)
    model.to(device)

    model.load_state_dict(torch.load(snapshot_dir))
    total_start_time = time.time()

    test_acc, test_aver_accuracy, class_accuracy, test_duration, recalls, \
    f1_scores, incorrect_image_paths  = test(model, seg_result_loader, num_classes)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print("Incorrectly classified images:")
    print(len(incorrect_image_paths))
    for path in incorrect_image_paths:
        print(path)

    print("*" * 80)
    print("Test accuracy: ", test_acc)
    print("Total average Accuracy:", test_aver_accuracy)
    print("Recalls: ", recalls)
    print("F1_scores: ", f1_scores)
    print("Class-wise accuracy:")
    for i in range(num_classes):
        if i == 0:
            print(f"Class abnormal: {class_accuracy[i]}")
        else:
            print(f"Class normal: {class_accuracy[i]}")
    print(f"Total Testing Time: {total_duration:.4f} seconds; And single image :{test_duration:.4f} second!")
    print("*" * 80, "\n")


def class_inference(image_path, model_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet34(num_classes=2)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    # model.eval()
    count_none_fnb = 0
    count_exist_fnb = 0
    for filename in os.listdir(image_path):
        image_path_1 = os.path.join(image_path, filename)

        image = Image.open(image_path_1).convert("L")
        preprocess = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

        # 对图像进行预处理
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # 添加批次维度

        # 使用模型进行预测
        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.softmax(output, dim=1)
        # 获取预测结果
        _, predicted_class = output.max(1)
        print(f"Image: {filename}, 概率: {probabilities.squeeze().tolist()}")

        '''
        # 打印预测结果
        if predicted_class.item() == 0:
            print(f"Image: {filename}, 预测类别: 无 FNB!, 概率: {probabilities.squeeze().tolist()}")
            # print(f"Image: {filename}, Predicted class: None FNB!, {predicted_class} Class probabilities:{probabilities.squeeze().tolist()[0]}")
            count_none_fnb += 1
        elif predicted_class.item() == 1:
            print(f"Image: {filename}, 预测类别: 存在 FNB!, 概率: {probabilities.squeeze().tolist()}")
            # print(f"Image: {filename}, Predicted class: Exist FNB!, {predicted_class} Class probabilities:{probabilities.squeeze().tolist()[1]}")
            count_exist_fnb += 1
        # print(f"Predicted class: {predicted_class.item()}")
        # print(f"Class probabilities: {probabilities.squeeze().tolist()}")  # 将概率转换为Python列表
    print(f"Total None FNB images: {count_none_fnb}")
    print(f"Total Exist FNB images: {count_exist_fnb}")
'''

# 将分割后的结果整理使normal和abnormal分开
def move_file_abnormal_normal(image_folder_path):

    # 创建存放异常图片的文件夹
    abnormal_folder = os.path.join(image_folder_path, "abnormal")
    os.makedirs(abnormal_folder, exist_ok=True)

    # 创建存放正常图片的文件夹
    normal_folder = os.path.join(image_folder_path, "normal")
    os.makedirs(normal_folder, exist_ok=True)

    # 遍历原始图片文件夹中的文件
    for filename in os.listdir(image_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 仅处理.jpg文件，根据实际文件类型修改
            first_digit = filename[0]  # 获取文件名的第一个字符
            source_path = os.path.join(image_folder_path, filename)

            if first_digit == '0':
                # 移动异常图片到异常文件夹
                target_path = os.path.join(abnormal_folder, filename)
            else:
                # 移动正常图片到正常文件夹
                target_path = os.path.join(normal_folder, filename)

            # 移动文件
            shutil.move(source_path, target_path)
            # print(f"Moved {filename} to {abnormal_folder if first_digit == '0' else normal_folder}")


if __name__ == "__main__":
    # np.random.seed(0)
    # torch.manual_seed(0)
    seg_result_class_infer(dataset_root=args.class_test,
                           num_classes=args.class_num_classes,
                           batch_size=args.class_batch_size,
                           snapshot_dir = f"{args.class_snapshots_path}/mobilenet/best_model.pt")
