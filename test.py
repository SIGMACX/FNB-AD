# @Time: 2023/7/28 8:49
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  测试代码
# -------------------------------

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

from dataset.dataset import CustomDataset
from model.networks import U_Net


test_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\img/'
test_mask_folder = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\labelcol/'

# 定义数据加载器和数据转换
transform = transforms.Compose([
    transforms.Grayscale(),  # 将RGB图像转换为灰度图像
    transforms.ToTensor(),
])

batch_size = 32

test_dataset = CustomDataset(test_folder, test_mask_folder, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 初始化模型
in_channels = 1  # 输入通道数，对应单通道的超声波图像
out_channels = 3  # 输出通道数，对应三个类别的分割结果
model = U_Net(in_channels, out_channels)

# 加载已训练的模型参数
model_path = "unet_model.pth"
model.load_state_dict(torch.load(model_path))

# 设置模型为评估模式，关闭Dropout等
model.eval()

# 将模型移动到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 创建空的列表用于保存测试结果
results = []

criterion = nn.CrossEntropyLoss()

# 测试模型并保存分割结果
with torch.no_grad():
    for test_inputs, _ in test_dataloader:
        test_inputs = test_inputs.to(device)  # 将输入数据移动到GPU上

        # 运行模型得到分割结果
        outputs = model(test_inputs)
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别

        # 将预测结果添加到results列表
        for i in range(predicted.size(0)):
            result = predicted[i].cpu().numpy()  # 将张量转换为numpy数组
            results.append(result)
print(results)
# 自定义保存路径
save_dir = "custom_results"  # 自定义保存文件夹名

# 确保保存文件夹存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 将results中的分割结果保存为图像文件
for i, result in enumerate(results):
    result_img = Image.fromarray(result.astype('uint8'))  # 创建PIL图像对象

    # 将预测的类别映射回原始的像素值（根据具体任务和类别映射关系）
    result_img = result_img.convert('L')  # 转为灰度图像
    result_img.putpalette([0, 0, 255,  # 类别0对应的颜色（黑色）
                           255, 0, 0,  # 类别1对应的颜色（红色）
                           0, 255, 0])  # 类别2对应的颜色（绿色）

    file_path = os.path.join(save_dir, f'result_{i + 1}.png')  # 自定义文件路径
    result_img.save(file_path)  # 保存图像文件

# 打印测试结果
test_loss = 0.0
total_samples = 0
correct_samples = 0

with torch.no_grad():
    for test_inputs, test_labels in test_dataloader:
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        test_outputs = model(test_inputs)
        test_loss += criterion(test_outputs, test_labels.argmax(dim=1)).item()

        _, predicted = torch.max(test_outputs, 1)
        total_samples += test_labels.size(0)
        correct_samples += (predicted == test_labels).sum().item()

    accuracy = correct_samples / total_samples
    test_loss /= len(test_dataloader)

print(f'Final Test - Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
