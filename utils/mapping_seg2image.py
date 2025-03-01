# -*- coding: utf-8 -*-
# @Time : 2023/8/11 下午8:40
# @Author : ChenXi

import os
from PIL import Image

def overlay_segmentation_on_image(original_image_path, segmentation_image_path, output_folder):
    # 载入原始样本图像和分割结果图像

    original_image = Image.open(original_image_path)
    segmentation_image = Image.open(segmentation_image_path)

    # 将分割结果图像调整到与原始样本图像相同大小
    segmentation_image = segmentation_image.resize(original_image.size, Image.ANTIALIAS)

    # 创建一个新的图像，将原始样本图像作为底图
    overlay = original_image.copy()

    # 创建一个透明度掩码，将分割结果中的白色区域设置为透明部分，透明度设为0.7
    mask = segmentation_image.convert("L")
    mask = mask.point(lambda p: 255 if p == 255 else 0)
    mask = mask.point(lambda p: int(p * 0.3))
    mask = mask.convert("L")

    # 将分割结果的白色区域映射到原始样本图像上，设置颜色为红色
    overlay.paste((255, 0, 0), (0, 0, overlay.width, overlay.height), mask=mask)

    # 保存叠加后的图像到指定输出文件夹
    # output_path = os.path.join(output_folder, os.path.basename(original_image_path))
    overlay.save(output_folder)