# @Time: 2023/7/13 19:42
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  将标注的分割图像转为mask图
# -------------------------------

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os

img_path = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\cropped_images/'
label_json_path = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\cropped_images/'
output_floader = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local\\trainingset\mask_148\\'

# 将数据集保存为多标签类别
def labelme2mask(img_folder, json_folder,output_folder):
    for filename in os.listdir(img_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(img_folder, filename)
            json_path = os.path.join(json_folder, filename.replace('.png', '.json'))

            img_bgr = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            img_mask = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)  # 创建空白掩码图像

            with open(json_path, 'r', encoding='utf-8') as f:
                labelme = json.load(f)

            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 为每个标记指定颜色，示例中有3个标记
            for i, each in enumerate(labelme['shapes']):
                if each['shape_type'] == 'polygon':
                    points = np.array(each['points'], dtype=np.int32)
                    color = colors[i % len(colors)]  # 循环使用颜色列表中的颜色
                    cv2.fillPoly(img_mask, [points], color=color)
                else:
                    print('Error shape type!', each['shape_type'])

            mask_filename = os.path.join(output_folder, os.path.basename(img_path).replace('.jpg', '.png'))
            print(mask_filename)
            cv2.imwrite(mask_filename, img_mask)

# 测试多类别图像
labelme2mask(img_path, label_json_path, output_floader)

# 将标签保存为灰度
def labelme2mask_arg(img_folder, json_folder,output_folder):
    for filename in os.listdir(img_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(img_folder, filename)
            json_path = os.path.join(json_folder, filename.replace('.png', '.json'))

            img_bgr = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            img_mask = np.zeros(img_gray.shape, dtype=np.uint8)  # 创建空白掩码图像

            with open(json_path, 'r', encoding='utf-8') as f:
                labelme = json.load(f)

            # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 为每个标记指定颜色，示例中有3个标记
            for i, each in enumerate(labelme['shapes']):
                if each['shape_type'] == 'polygon':
                    points = np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(img_mask, [points], color=255)
                else:
                    print('Error shape type!', each['shape_type'])

            mask_filename = os.path.join(output_folder, os.path.basename(img_path).replace('.jpg', '.png'))
            cv2.imwrite(mask_filename, img_mask)

# labelme2mask_arg(img_path, label_json_path, output_floader)

# 四个通道，将背景保存为一类
def labelme2mask(img_folder, json_folder, output_folder):
    for filename in os.listdir(img_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(img_folder, filename)
            json_path = os.path.join(json_folder, filename.replace('.png', '.json'))

            img_bgr = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            img_mask = np.zeros((img_gray.shape[0], img_gray.shape[1], 4), dtype=np.uint8)  # 创建四通道的空白掩码图像

            with open(json_path, 'r', encoding='utf-8') as f:
                labelme = json.load(f)

            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 红色、绿色、蓝色
            for i, each in enumerate(labelme['shapes']):
                if each['shape_type'] == 'polygon':
                    points = np.array(each['points'], dtype=np.int32)
                    color = colors[i % len(colors)]  # 循环使用颜色列表中的颜色
                    if i == 0:  # 第一个标记作为背景
                        cv2.fillPoly(img_mask, [points], color=(0, 0, 0, 255))  # 将背景填充为黑色(0,0,0)，且alpha通道为255
                    else:
                        cv2.fillPoly(img_mask, [points], color + (255,))  # 将前景填充为指定颜色，且alpha通道为255
                else:
                    print('Error shape type!', each['shape_type'])

            mask_filename = os.path.join(output_folder, os.path.basename(img_path).replace('.png', '.png'))
            print(mask_filename)
            cv2.imwrite(mask_filename, img_mask)

labelme2mask(img_path, label_json_path, output_floader)
