# @Time: 2023/7/26 15:43
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  创建train_dataset and val_datasets
# -------------------------------

import os
import random
import shutil

def split_dataset(input_folder, train_ratio=0.8):
    file_names = os.listdir(input_folder)
    random.shuffle(file_names)  # 随机打乱文件名列表

    total_files = len(file_names)
    train_files = file_names[:int(total_files * train_ratio)]
    test_files = file_names[int(total_files * train_ratio):]

    with open('train.txt', 'w') as f_train:
        for file_name in train_files:
            f_train.write(os.path.splitext(file_name)[0] + '\n')

    with open('test.txt', 'w') as f_test:
        for file_name in test_files:
            f_test.write(os.path.splitext(file_name)[0] + '\n')

# 输入图片文件夹路径和切分比例
input_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\mask/'
train_ratio = 0.8  # 80%用于训练，20%用于测试

# split_dataset(input_folder, train_ratio)


# 随机选取20%的图像另存为test
import os
import random
import shutil

def random_split_images_with_masks(image_folder, masks_folder, output_train_image_folder, output_train_masks_folder,
                                   output_test_image_folder, output_test_masks_folder, select_ratio=0.2):
    image_files = os.listdir(image_folder)
    masks_files = os.listdir(masks_folder)

    common_files = list(set(image_files) & set(masks_files))  # 找到image和masks中共有的文件名
    random.shuffle(common_files)

    num_files = int(len(common_files) * select_ratio)
    selected_files = common_files[:num_files]
    remaining_files = common_files[num_files:]

    os.makedirs(output_train_image_folder, exist_ok=True)
    os.makedirs(output_train_masks_folder, exist_ok=True)
    os.makedirs(output_test_image_folder, exist_ok=True)
    os.makedirs(output_test_masks_folder, exist_ok=True)

    for file_name in selected_files:
        img_src_path = os.path.join(image_folder, file_name)
        mask_file = file_name.replace('.jpg', '.png')
        mask_src_path = os.path.join(masks_folder, mask_file)

        img_dst_path = os.path.join(output_test_image_folder, file_name)
        mask_dst_path = os.path.join(output_test_masks_folder, mask_file)

        shutil.copyfile(img_src_path, img_dst_path)
        shutil.copyfile(mask_src_path, mask_dst_path)

    for file_name in remaining_files:
        img_src_path = os.path.join(image_folder, file_name)
        mask_file = file_name.replace('.jpg', '.png')
        mask_src_path = os.path.join(masks_folder, mask_file)

        img_dst_path = os.path.join(output_train_image_folder, file_name)
        mask_dst_path = os.path.join(output_train_masks_folder, mask_file)

        shutil.copyfile(img_src_path, img_dst_path)
        shutil.copyfile(mask_src_path, mask_dst_path)

'''
# 输入image图像文件夹路径和masks标签图像文件夹路径以及选择比例
image_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/trainingset\images/'
masks_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/trainingset\masks/'
output_train_image_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/train\img/'
output_train_masks_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/train\masks/'
output_test_image_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test/img/'
output_test_masks_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\mask/'
select_ratio = 0.2  # 选择比例为20%

random_split_images_with_masks(image_folder, masks_folder, output_train_image_folder, output_train_masks_folder,
                               output_test_image_folder, output_test_masks_folder, select_ratio)
'''

# 判断两个文件夹中的图片名字是否相同，如果不同则打印
def compare_filenames(folder1, folder2):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    different_files = files1.symmetric_difference(files2)

    if different_files:
        print("Files with different names:")
        for file in different_files:
            print(file)
    else:
        print("All files have the same names.")

'''
# 输入两个文件夹的路径
folder1 = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test/img/'
folder2 = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\labelcol/'
compare_filenames(folder1, folder2)
'''

# 按照名称区分数据集，0为abnormal，1为normal
def class_dataset(ori_path, out_path):
    for filename in os.listdir(ori_path):
        source_path = os.path.join(ori_path, filename)
        first_name = filename.split('_')[0]

        # 确定目标文件夹名称
        if first_name == '0':
            target_folder = 'abnormal'
        elif first_name == '1':
            target_folder = 'normal'
        else:
            continue  # 忽略其他文件

        # 构建目标文件夹路径
        target_path = os.path.join(out_path, target_folder)

        # 如果目标文件夹不存在，则创建
        if not os.path.isdir(target_path):
            os.makedirs(target_path)

        # 复制文件到目标文件夹
        shutil.copyfile(source_path, os.path.join(target_path, filename))
'''
ori_path = '../data/FNBS_Dataset/segmentation/training_dataset/image/'
out_path = '../data/FNBS_Dataset/classification_ori/testing_dataset/'
class_dataset(ori_path, out_path)
'''