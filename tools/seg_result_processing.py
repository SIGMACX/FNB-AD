# @Time: 2023/11/17 8:58
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 对比分割结果和标签图像，与标签图像不同的位置使用不同颜色表示
# -------------------------------

import cv2
import os
import numpy as np

def compare_and_save_images(segmentation_folder, label_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历标签文件夹中的图像
    for label_filename in os.listdir(label_folder):
        print(label_filename)
        # 构造标签图像和分割结果图像的文件路径
        label_path = os.path.join(label_folder, label_filename)
        segmentation_path = os.path.join(segmentation_folder, label_filename)

        # 读取图像
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # 创建一个空白的图像，用于保存比较结果
        comparison = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)

        # 找到相同的区域并将其设为白色
        white_areas = cv2.bitwise_and(segmentation, label)
        comparison[white_areas > 0] = [255, 255, 255]  # 白色

        # 找到分割结果中多出来的区域并将其设为红色
        diff_segmentation = cv2.absdiff(segmentation, label)
        comparison[diff_segmentation > 0] = [255, 0, 0]  # 红色

        # 找到标签图像中多出来的区域并将其设为蓝色
        diff_label = cv2.absdiff(label, segmentation)
        comparison[diff_label > 0] = [0, 0, 255]  # 蓝色

        # 构造输出图像的文件路径
        output_path = os.path.join(output_folder, f"{label_filename}")

        # 保存比较结果图像
        cv2.imwrite(output_path, comparison)

# 示例用法
segmentation_folder = "../Result/0921/"
label_folder = "../data/FNBS_Dataset/segmentation/testing_dataset/label_gt"
output_folder = "../Result/0921/"

compare_and_save_images(segmentation_folder, label_folder, output_folder)
