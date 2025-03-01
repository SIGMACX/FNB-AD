# @Time: 2023/8/22 11:11
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# mapping seg result to image
# -------------------------------
from PIL import Image
import os

def map_segmentation_result(original_image_path, segmentation_result_path, output_path):
    # 打开原始图像和分割结果图像
    original_image = Image.open(original_image_path).convert('L')
    segmentation_result = Image.open(segmentation_result_path).convert('L')

    # 获取原始图像和分割结果图像的宽度和高度
    width, height = original_image.size

    # 创建一个新的图像，用于保存处理后的结果
    result_image = Image.new('L', (width, height), 0)

    # 遍历分割结果图像中的每个像素
    for x in range(width):
        for y in range(height):
            pixel_value = segmentation_result.getpixel((x, y))
            if pixel_value != 0:
                # 如果分割结果图像中的像素值不为0，则在原始图像对应位置保留原始灰度值
                result_image.putpixel((x, y), original_image.getpixel((x, y)))

    # 将处理后的原始图像保存为单独文件
    result_image.save(output_path)



if __name__ == "__main__":
    input_folder = "D:\code_pycharm\Fetal_nasal_bone_detection\data\FNBS_Dataset\segmentation/testing_dataset\image/"
    segmentation_result_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FNBS_Dataset\segmentation/testing_dataset\label_gt/'
    output_folder = "D:\code_pycharm\Fetal_nasal_bone_detection\data\FNBS_Dataset\classification/testingdataset/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for out_filename in os.listdir(output_folder):
        if out_filename == 'abnormal':
            output_path_1 = os.path.join(output_folder, out_filename)
        elif out_filename == 'normal':
            output_path_2 = os.path.join(output_folder, out_filename)
        for filename in os.listdir(input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(input_folder, filename)
                seg_result_path = os.path.join(segmentation_result_folder, filename)
                first_name = filename.split('_')[0]
                if first_name == '0' and out_filename == 'abnormal':
                    output_path = os.path.join(output_path_1, filename)
                elif first_name == '1' and out_filename == 'normal':
                    output_path = os.path.join(output_path_2, filename)
                map_segmentation_result(image_path, seg_result_path, output_path)
