# @Time: 2023/7/13 21:21
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  1. 超声波中值滤波
#  2. 直方图均衡化
#  3. 锐化
#  4.
# -------------------------------

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure
import numpy as np
import pywt

img_path = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/trainingset\images/0_00003_1.png'

# 中值滤波函数
def median(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    filtered_image = cv2.medianBlur(image, 3)
    return filtered_image

# 直方图均衡化(细节模糊了)
def equalizeHist(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 应用直方图均衡化
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# 锐化方法
def Lapla(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 应用拉普拉斯滤波器
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # 将锐化图像和原始图像叠加
    sharpened_image = cv2.subtract(image.astype(np.float64), laplacian)
    # sharpened_image = cv2.subtract(laplacian, sharpened_image)
    # sharpened_image = cv2.subtract(laplacian, sharpened_image)

    # 将叠加后的图像进行数据范围限制和类型转换
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return sharpened_image


# 增加对比度
def enhance(image_path):
    filtered_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    alpha = 1.5
    beta = 50
    enhanced_image = np.clip(alpha * filtered_image + beta, 0, 255).astype(np.uint8)
    return enhanced_image



# 找到图中最亮位置的坐标
def max_light(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 找到最亮位置的坐标
    max_pixel = np.unravel_index(np.argmax(image), image.shape)
    max_x, max_y = max_pixel

    # 在图像上进行标记
    marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(marked_image, (max_y, max_x), 5, (0, 0, 255), -1)

    # 打印最亮位置的坐标
    print("最亮位置的坐标：({}, {})".format(max_x, max_y))

    '''
    # 找到亮度排名前10的位置
    flatten_image = image.flatten()  # 将图像转为一维数组
    top_10_indices = np.argsort(flatten_image)[-10:]  # 获取亮度排名前10的索引
    top_10_positions = np.unravel_index(top_10_indices, image.shape)  # 将索引转换为二维坐标
    
    # 在图像上进行标记
    marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for position in zip(*top_10_positions):
        cv2.circle(marked_image, (position[1], position[0]), 5, (0, 0, 255), -1)
    
    # 打印亮度排名前10的位置
    print("亮度排名前10的位置：")
    for i, position in enumerate(zip(*top_10_positions)):
        print("位置 {}: ({}, {})".format(i+1, position[0], position[1]))
    '''
    return marked_image

# 设定一个阈值，将小于阈值的部分全部致为0
def set_value(image_path, threshold):
    image = cv2.imread(image_path)
    # 将彩色图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = np.where(gray_image < threshold, 0, gray_image)
    return filtered_image

result = set_value(img_path, 110)
cv2.imshow('Enhanced Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

