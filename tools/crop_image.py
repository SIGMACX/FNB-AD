# @Time: 2023/7/13 15:27
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  1. 图像裁剪（裁剪掉标题信息）
#  2. 将tif格式转为jpg格式
#  3. 重名名文件（0_0000.jpg：表示缺失， 1_0000.jpg：表示存在）
#  4. 删除没有对应json文件的jpg图像
# -------------------------------

from PIL import Image
import os
import cv2
import json
import numpy as np

def crop_image(img_path, save_path):
    left = 80
    top = 80
    right = 1100
    bottom = 852
    for filename in os.listdir(img_path):
        if filename.endswith('.jpg'):
            img_path_1 = os.path.join(img_path, filename)
            image = Image.open(img_path_1)
            # image = cv2.imread(img_path_1)
            # if image.shape != (852, 1136, 3):
            #     print(img_path_1, image.shape)
            cropped_image = image.crop((left, top, right, bottom))
            cropped_image_path = os.path.join(save_path, f'{filename}')
            cropped_image.save(cropped_image_path)

'''
# 测试裁剪代码
img_path = 'D:/code_pycharm/Fetal_nasal_bone_detection/data/images/'
save_path = 'D:/code_pycharm/Fetal_nasal_bone_detection/data/image_cropped/'
crop_image(img_path, save_path)
'''

# 将tif格式转换为jpg格式
def tif2jpg(image_path, save_path):
    for filename in os.listdir(image_path):
        if filename.endswith('.tif') or filename.endswith('.jpg'):
            image_path_1 = os.path.join(image_path, filename)
            image = Image.open(image_path_1)
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            new_save_path = os.path.join(save_path, new_filename)
            image.convert('RGB').save(new_save_path)

'''
# 测试tif格式转为jpg格式代码
img_path = '../data/FNBS_Dataset/detection/'
save_path = '../data/FNBS_Dataset/detection/'
tif2jpg(img_path, save_path)
'''

def rename(folder_path):
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # 遍历图片文件并重命名
    for i, file_name in enumerate(image_files):
        # 获取文件扩展名
        extension = os.path.splitext(file_name)[1]
        # 新的文件名为数字加上扩展名
        new_file_name =  '0_' + f"{i:08}{extension}"
        # 重命名文件
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))

'''
# 重命名文件
img_path = '../data/1/'
rename(img_path)
'''

# 截取矩形框中的图像并保存
def crop_and_save_images(labelme_folder):
    # 获取标注文件夹中的所有文件名
    file_names = os.listdir(labelme_folder)

    for file_name in file_names:
        if file_name.endswith(".json"):
            # 读取json标注文件
            json_path = os.path.join(labelme_folder, file_name)
            with open(json_path, "r") as f:
                data = json.load(f)

            # 读取原始图像
            image_path = os.path.join(labelme_folder, data["imagePath"])
            image = cv2.imread(image_path)

            # 创建保存截取图像的文件夹（在与输入文件夹相同路径下）
            save_folder = os.path.join(labelme_folder, "cropped_images")
            os.makedirs(save_folder, exist_ok=True)

            # 遍历每个标注区域并截取保存
            for shape in data["shapes"]:
                label = shape["label"]
                points = np.array(shape["points"], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)
                cropped_image = image[y:y + h, x:x + w]

                # 创建标签对应的文件夹
                label_folder = os.path.join(save_folder, label)
                os.makedirs(label_folder, exist_ok=True)

                # 保存截取图像
                save_path = os.path.join(label_folder, f"{file_name.split('.')[0]}_{label}.png")
                cv2.imwrite(save_path, cropped_image)

'''
image_path = 'D:/code_pycharm/Fetal_nasal_bone_detection/data/images/'
crop_and_save_images(image_path)
'''

# 读取文件中图像的大小
def get_image_sizes(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                width, height = img.size
                if width != 148 and height != 89:
                    print(f"Image '{filename}' - Width: {width}, Height: {height}")

'''
# 输入图片文件夹路径
folder_path = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/trainingset\mask/'
get_image_sizes(folder_path)
'''

# 裁剪图像，将图像裁剪为统一的尺寸
def resize_images(input_folder, output_folder, target_size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                img_resized = img.resize(target_size)
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)
'''
# 输入图片文件夹路径和目标大小
input_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/trainingset\mask_148/'
output_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/trainingset\mask_148/'
target_size = (148, 148)
resize_images(input_folder, output_folder, target_size)
'''

def remove_jpg_no_json(folder_path):
    # 获取文件夹中的所有jpg文件
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    deleted_count = 0
    count = 0

    # 遍历每个jpg文件
    for jpg_file in jpg_files:
        # 构建对应的json文件路径
        json_file = os.path.join(folder_path, jpg_file.replace('.jpg', '.json'))

        # 检查json文件是否存在
        if not os.path.exists(json_file):
            # 如果没有对应的json文件，就删除jpg文件
            jpg_file_path = os.path.join(folder_path, jpg_file)
            os.remove(jpg_file_path)
            print(f"已删除没有对应json文件的图片：{jpg_file}")
            deleted_count +=  1
        else:
            print(f"保留有对应json文件的图片：{jpg_file}")
            count += 1

    print(f"完成删除操作，共删除了 {deleted_count} 张图片； 保留了{count}张图片；")

'''
# 测试代码
folder_path = "/data1/chenxi/code/medical_segmentation/Fetal_nasal_bone_detection/data/FNBS_Dataset/detection_all/"
remove_jpg_no_json(folder_path)
'''


# 将图片的jpg格式转换为png格式
def jpg2png(image_path, save_path):
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg'):
            image_path_1 = os.path.join(image_path, filename)
            image = Image.open(image_path_1)
            new_filename = os.path.splitext(filename)[0] + '.png'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            new_save_path = os.path.join(save_path, new_filename)
            image.convert('RGB').save(new_save_path)


# 删除检测任务中检测出来的多个FNB
def delete_over_crop_image(folder_path):
    # 获取文件夹中的所有文件名
    files = os.listdir(folder_path)

    # 用于跟踪已删除的文件
    deleted_files = set()

    # 遍历文件夹中的文件
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否是文件而不是目录
        if os.path.isfile(file_path):
            # 获取文件的基本名称（不含扩展名）
            base_name, _ = os.path.splitext(file_name)

            # 如果已经删除过具有相同基本名称的文件，则删除当前文件
            if base_name in deleted_files:
                os.remove(file_path)
                print(f"已删除文件: {file_path}")
            else:
                # 否则，将当前文件的基本名称添加到已删除文件的集合中
                deleted_files.add(base_name)

    print("删除重复文件完成")

# folder_path = "../detection_method/models/yolo/runs/detect/predict13/crops/FNB/png/"
# delete_over_crop_image(folder_path)



import os

# 文件名
file_name = "0000012.png"

# 提取最后一个数字
last_digit = None

# 从右向左遍历文件名字符
for char in reversed(file_name):
    if char.isdigit():
        last_digit = char
        break

# 如果找到最后一个数字，打印它
if last_digit is not None:
    print(f"文件名 {file_name} 中的最后一个数字是: {last_digit}")
else:
    print(f"文件名 {file_name} 中没有数字")

