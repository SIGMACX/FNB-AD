# @Time: 2023/8/3 16:26
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  将分割结果映射到原始图像中
# -------------------------------


from PIL import Image
import numpy as np

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


# if __name__ == "__main__":
#     original_image_path = "D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\img_128/0_0727_000_1.png"  # 原始图像路径
#     segmentation_result_path = "D:\code_pycharm\Fetal_nasal_bone_detection\Result\mul_fn_seg\class_3/" \
#                                "2023-08-02_21-07-08\seg_result\seg_result/0_0727_000_1.png"  # 分割结果图像路径
#     output_path = "D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\islands/segforimage.png"  # 输出文件路径
#
#     map_segmentation_result(original_image_path, segmentation_result_path, output_path)

