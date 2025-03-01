# @Time: 2023/11/02 14:05
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 输出特征图
# -------------------------------

import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from model.models.FNB_UNet_DSConv import FNB_UNet_DSConv_MLP, FNB_UNet_DSConv
import cv2
import numpy as np
from tools.config import args

from model.models.FNB_UNet_SingConv import FNB_UNet_SingConv
from model.models.FNB_UNet_SingConv_MLP import FNB_UNet_SingConv_MLP
from model.models.FNB_UNet_DouConv import FNB_UNet_DouConv
from model.models.FNB_UNet_DouConv_MLP import FNB_UNet_DouConv_MLP
from model.models.FNB_UNet_MLP_Add import FNB_UNet_DSConv_MLP_add

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# 检查是否有可用的GPU，如果有，将其用于计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例并将其移动到适当的设备
# model = FNB_UNet_DSConv_MLP(n_channels=3, n_classes=2).to(device)
# model = FNB_UNet_DSConv(n_channels=3, n_classes=2).to(device)
# model = FNB_UNet_SingConv(n_channels=3, n_classes=2).to(device)
# model = FNB_UNet_SingConv_MLP(n_channels=3, n_classes=2).to(device)
# model = FNB_UNet_DouConv(n_channels=3, n_classes=2).to(device)
# model = FNB_UNet_DouConv_MLP(n_channels=3, n_classes=2).to(device)
model = FNB_UNet_DSConv_MLP_add(n_channels=3, n_classes=2).to(device)

model_name = "FNB_UNet_DSConv_MLP_add_1"

# 加载预训练的模型权重文件
# model_weights_path = "../snapshots/1105/FNB_UNet_DSConv_MLP/best_model.pkl"  # 模型权重文件的路径
model_weights_path = f"../snapshots/1109/{model_name}/best_model.pt"

# 加载模型的权重并将其移动到适当的设备
model_weights = torch.load(model_weights_path, map_location=device)
model.load_state_dict(model_weights)

# 设置模型为评估模式
model.eval()
print(model)

# 加载输入图像
image_folder_path = "../data/FNBS_Dataset/segmentation/testing_dataset/image/"  # 输入图像的路径
image_filenames = os.listdir(image_folder_path)
image_filenames.sort()  # 按字母顺序排序，你也可以根据需要使用其他排序方法

for idx, filename in enumerate(image_filenames):
    image_path = os.path.join(image_folder_path, filename)
    image = Image.open(image_path)

    # 数据预处理和转换
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 注册钩子以保存特征图
    save_folder = f"../heatmap_figures/{model_name}/{os.path.splitext(filename)[0]}"

    # 设置透明度
    alpha = 0.5  # 设置透明度为0.5


    def save_feature_maps_hook(module, input, output, save_folder, layer_name):
        feature_map = output[0].detach().cpu().numpy()[0]

        # Normalize the feature_map
        min_value = np.min(feature_map)
        max_value = np.max(feature_map)
        normalized_feature_map = ((feature_map - min_value) / (max_value - min_value) * 255).astype(np.uint8)

        # Apply color map
        rgba_feature_map = cv2.applyColorMap(normalized_feature_map, cv2.COLORMAP_JET)

        # Resize the feature map to match the input image size
        rgba_feature_map = cv2.resize(rgba_feature_map, (image.width, image.height))

        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Combine the feature map and the input image with alpha blending
        result_image = cv2.addWeighted(image_cv, 1 - alpha, rgba_feature_map, alpha, 0)

        file_path = os.path.join(save_folder, f"{layer_name}.png")
        if os.path.exists(file_path):
            file_path = os.path.join(save_folder, f"{layer_name}_1.png")

        # If the directory does not exist, create it
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the result image
        cv2.imwrite(file_path, result_image)


    # 用于保存每一层的特征图
    feature_maps = []

    for layer in model.children():
        layer_name = layer.__class__.__name__
        hook = layer.register_forward_hook(
            lambda module, input, output, save_folder=save_folder, layer_name=layer_name:
            save_feature_maps_hook(module, input, output, save_folder, layer_name)
        )
        feature_maps.append(hook)

    # 进行模型的前向传播
    with torch.no_grad():
        output = model(input_tensor)

    # 移除之前注册的钩子，以避免下一次循环时覆盖之前的结果
    for hook in feature_maps:
        hook.remove()

    print(f"Saved feature heatmaps for {filename} in {save_folder}")

print("Finished Save!")

