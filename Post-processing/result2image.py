# @Time: 2023/7/29 20:45
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 1.在原始图片中展示分割结果，使用轮廓
# -------------------------------

import cv2
import numpy as np

# 定义类别对应的颜色映射（不包括背景）
class_colors = {
    1: (0, 0, 255),    # 类别1（红色）
    2: (0, 255, 0),    # 类别2（绿色）
    3: (255, 0, 0),    # 类别3（蓝色）
}

def visualize_segmentation(image_path, segmentation_path, save_path):
    # 加载原始图像和分割结果图像
    img = cv2.imread(image_path)
    seg = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)

    # 查找分割区域的轮廓
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 使用CHAIN_APPROX_NONE绘制更精细的轮廓

    # 创建空白图像，与原始图像尺寸相同
    result_img = np.copy(img)

    # 在空白图像上绘制不同类别的轮廓
    for contour in contours:
        # 从颜色映射中获取类别标签
        class_label = seg[contour[0][0][1], contour[0][0][0]] + 1

        # 获取该类别对应的颜色
        color = class_colors.get(class_label, (0, 0, 0))

        # 将轮廓点坐标转换为整数类型
        contour = contour.astype(np.int32)

        # 在空白图像上绘制多边形轮廓
        cv2.polylines(result_img, [contour], isClosed=True, color=color, thickness=2)

    # 将带有轮廓的结果图像与原始图像叠加，以可视化标记效果
    alpha = 0.7  # 调整叠加的透明度，0表示完全透明，1表示完全不透明
    result_img = cv2.addWeighted(result_img, alpha, img, 1 - alpha, 0)

    # 保存结果图像
    cv2.imwrite(save_path, result_img)

# 示例用法
image_path = "../data/FN_local/trainingset/img_148/0_00001_1.png"
segmentation_path = "../Result/mul_fn_seg/class_3/0_00001_1.png"
save_path = "../post_processing_result/1.jpg"

visualize_segmentation(image_path, segmentation_path, save_path)

