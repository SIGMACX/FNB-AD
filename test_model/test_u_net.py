# @Time: 2023/7/31 21:46
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 1. 测试u_net代码
# -------------------------------

import datetime
import os

import cv2
import numpy as np
import shutil
import torch
import imageio

from dataset.dataloader_mulclsfn_unet import FNDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy import misc
from tools.config import args
from model.models.U_net_selfattention import U_Net, AttU_Net


# 定义颜色编码表，这里使用红、绿、蓝三种颜色
label_colors = {
    0: [0, 0, 0],       # 背景（黑色）
    1: [255, 255, 255],   # 第一个类别（红色）
}

def colorize_mask(mask, label_colors):
    # 根据颜色编码表，将标签图像转换成彩色图像
    h, w = mask.shape
    mask_colorized = np.zeros((h, w, 3), dtype=np.uint8)

    # 将背景（类别0）的颜色改为黑色
    mask_colorized[mask == 0] = [0, 0, 0]

    for label, color in label_colors.items():
        # 标签从0开始，因此需要将标签值加0
        mask_colorized[mask == label] = color
    return mask_colorized


def calculate_iou(pred_mask, gt_mask, num_classes):
    iou_list = []
    for class_id in range(num_classes):
        pred_class = pred_mask == class_id
        gt_class = gt_mask == class_id

        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()

        iou = intersection / (union + 1e-10)  # Add a small epsilon to avoid division by zero
        iou_list.append(iou)
    return iou_list


def inference(num_classes, input_channels, snapshot_dir, save_path):
    test_dataset = FNDataset(
        imgs_path='./data/FN_local/testingset/test/img/',
        label_path='./data/FN_local/testingset/test/labelcol/',
        split='test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]))

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fn_model = AttU_Net(input_channels, num_classes).to(device)
    # print(fn_model)
    fn_model.load_state_dict(torch.load(snapshot_dir))
    fn_model.eval()

    class_ious = [[] for _ in range(num_classes)]

    for index, (img, img_mask, _, img_name) in enumerate(test_dataloader):
        img = img.to(device)
        img_mask = img_mask.to(device)

        output = fn_model(img)
        output = torch.sigmoid(output)
        b, _, w, h = output.size()
        _, w_gt, h_gt = img_mask.size()

        pred = output.cpu().permute(0, 2, 3, 1).contiguous(). \
            view(-1, num_classes).max(1)[1].view(b, w, h).numpy().squeeze()

        os.makedirs(save_path, exist_ok=True)
        result_path = os.path.join(save_path, img_name[0].replace('.jpg', '.png'))

        # 将标签图像转换为彩色图像
        pred_colorized = colorize_mask(pred, label_colors)
        imageio.imwrite(result_path, pred_colorized)

        # Convert ground truth mask to numpy array
        gt_mask = img_mask.cpu().numpy().squeeze()

        # Calculate IoU for each class and store the results
        iou_list = calculate_iou(pred, gt_mask, num_classes)
        for class_id, iou in enumerate(iou_list):
            class_ious[class_id].append(iou)

    # Calculate the mean IoU for each class
    mean_ious = [np.mean(ious) for ious in class_ious]

    # Print the results
    for class_id, mean_iou in enumerate(mean_ious):
        print(f"Class {class_id}: Mean IoU = {mean_iou:.4f}")

    print('Test done!')



if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    inference(num_classes=args.num_classes,
              input_channels=3,
              snapshot_dir='./snapshots/Fn_unet/unet_model_2000.pkl',
              save_path=f'/data1/chenxi/code/medical_segmentation/Fetal_nasal_bone_detection/Result/u_net_fn_seg/class_3/{current_time}')
