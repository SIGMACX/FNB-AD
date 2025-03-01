# @Time: 2023/10/9 21:04
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 1. delete files
# -------------------------------

import os

import shutil


# 移动指定文件到指定文件夹
def move_files(source_folder, destination_folder):
    # 要移动的文件名列表
    target_filenames = ['572432']

    # 如果目标文件夹不存在，创建它
    # if not os.path.exists(destination_folder):
    #     os.makedirs(destination_folder)
    count = 0
    # 获取文件夹中的所有文件
    files = os.listdir(source_folder)
    for file_name in files:
        file_path = os.path.join(source_folder, file_name)

        # 检查文件是否是PNG格式
        if file_name.lower().endswith(".jpg") and file_name.split(".")[0] in target_filenames:
            # 构建目标文件的路径
            destination_path = os.path.join(destination_folder, file_name)

            # 移动文件到目标文件夹
            shutil.move(file_path, destination_path)
            print(f"已移动文件: {file_path} 到 {destination_path}")
            count += 1

    print(f"移动指定图片完成,共{count}张；")



# 统计文件个数
def count_files(folder_path):
    files = os.listdir(folder_path)

    # 使用列表推导式筛选出文件（排除子文件夹）
    file_count = len([f for f in files if os.path.isfile(os.path.join(folder_path, f))])

    print(f"文件夹 {folder_path} 中的文件个数为: {file_count}")


# 检查相同文件
def check_same_files(folder_A, folder_B):
    # 获取文件夹A中的所有图片文件
    image_files_A = [f for f in os.listdir(folder_A) if f.lower().endswith('.jpg')]

    # 获取文件夹B中的所有文件名
    files_B = os.listdir(folder_B)

    # 遍历文件夹A中的图片文件
    for image_file_A in image_files_A:
        # 检查文件是否存在于文件夹B中
        if image_file_A in files_B:
            print(f"图片文件 {image_file_A} 存在于文件夹B中")



def compare_same_file(image_folder, text_folder):
    count = 0
    # 获取图片文件夹和文本文件夹中的文件名（不含扩展名）
    image_files = set([os.path.splitext(f)[0] for f in os.listdir(image_folder)])
    text_files = set([os.path.splitext(f)[0] for f in os.listdir(text_folder)])

    # 检查两个文件名集合是否有交集
    common_files = image_files.intersection(text_files)

    # 打印共同文件名（不含扩展名）
    print("共同文件名（不含扩展名）：")
    for file_name in common_files:
        count += 1
        print(file_name)
    print(count)


if __name__ == "__main__":
    folder_path = "./detection_method/models/yolo/datasets/FNB/images/val/"
    count_files(folder_path)

    # folder_path = "./detection_method/models/yolo/datasets/FNB/images/val/"
    # destination_folder = "./detection_method/models/yolo/datasets/FNB/images/train/"
    # move_files(folder_path, destination_folder)

    # folder_A = "/data1/chenxi/code/medical_segmentation/Fetal_nasal_bone_detection/data/鼻骨0727/缺如/"
    # folder_B = "/data1/chenxi/code/medical_segmentation/Fetal_nasal_bone_detection/detection_method/models/yolo/datasets/FNB/images/val/"
    # check_same_files(folder_A, folder_B)

    # same_image_folder_A = "./detection_method/models/yolo/datasets/FNB/images/val/"
    # same_txt_folder_B = "./detection_method/models/yolo/datasets/FNB/labels/val/"
    # compare_same_file(same_image_folder_A, same_txt_folder_B)