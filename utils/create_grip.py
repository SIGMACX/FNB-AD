# @Time: 2023/8/2 10:15
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  创建掩码
# -------------------------------
from PIL import Image
import os
from utils.compute_center import *

def count_light(image_path, value):
    image = Image.open(image_path).convert("L")
    image_array = np.array(image)
    sorted_pixels = np.sort(image_array.flatten())[::-1]
    top_pixels = sorted_pixels[:value]
    average_brightness = np.mean(top_pixels)
    return average_brightness


def process_images(input_folder, filename, output_folder):
    # 判断输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 用于存储每张图像的检测结果
    mean_light_list = []
    count_bn, count_no_bn = 0, 0

    # 遍历输入文件夹中的图片文件

    input_path = input_folder
    output_path = os.path.join(output_folder, filename)
    # 打开图片
    image = Image.open(input_path)
    # 用于存储每张图像灰度值大于150的像素，并进行排序
    top_pixels = []

    # 判断像素值是否大于150，并进行排序
    for x in range(image.width):
        for y in range(image.height):
            pixel_value = image.getpixel((x, y))[0]  # 获取灰度值
            if pixel_value > 150:
                top_pixels.append(pixel_value)
            new_pixel_value = (255, 255, 255) if pixel_value > 150 else (0,0,0)
            image.putpixel((x, y), new_pixel_value)

    # 保存修改后的图片到输出文件夹
    image.save(output_path)

    mean_light = count_light(input_path, value=5000)
    mean_light_list.append(mean_light)
    # print(f"{filename}, mean_light: {mean_light}")
    first_part = filename.split("_")[0]
    if mean_light > 121 and first_part == "1":
        count_bn += 1
    elif mean_light < 121 and first_part == "0":
        count_no_bn += 1
    return count_bn, count_no_bn, mean_light_list


def image_process_images_v2(image_path, detection = False):
    # 打开图片
    image = Image.open(image_path)
    top_pixels = []

    # 判断像素值是否大于150，并进行排序
    for x in range(image.width):
        for y in range(image.height):
            pixel_value = image.getpixel((x, y))[0]  # 获取灰度值
            if pixel_value > 150:
                top_pixels.append(pixel_value)

    # 保存修改后的图片到输出文件夹
    # image.save(output_path)

    # 对top_pixels列表进行排序，找出排名前700的像素
    top_pixels.sort(reverse=True)
    top_pixels = top_pixels[:700]

    # 计算排名前700像素的灰度值均值
    average_value = sum(top_pixels) / len(top_pixels)
    if average_value < 201:
        detection = False
        print('detection: ', detection)
    return detection




if __name__ == "__main__":
    input_folder = "D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\img_128/"  # 输入文件夹路径
    output_folder = "D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\grip/"  # 输出文件夹路径
    detection_results = process_images(input_folder, output_folder)
    # print(f"Detection Results: {detection_results}")

