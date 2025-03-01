# @Time: 2023/7/31 14:07
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 1. 测试双分类的代码
# -------------------------------

import datetime
import os
import time

import cv2
import numpy as np
import shutil
import torch
import imageio
import torch.nn as nn
from thop import profile

from dataset.dataloader_mulclsfn_unet import FNDataset, FNDataset_V1
from torchvision import transforms
from torch.utils.data import DataLoader
from model.unet import Fn_Net_UNet_V1, FNB_UNet
from scipy import misc
from tools.config import args
from model.models.U_net_selfattention import U_Net, AttU_Net, R2U_Net, R2AttU_Net, NestedUNet

from utils.post_processing import *
from model.models.U_Next import UNext, UNext_S
from model.models.ege_unet import EGEUNet
from model.models.swin_unet import SwinUnet


from model.models.FNBU_Net_MLP import FNB_UNet_MLP

from metrics import evaluate_images_in_folder


# FNB_UNet_XX  Ablation
from model.models.FNB_UNet_DouConv import FNB_UNet_DouConv
from model.models.FNB_UNet_SingConv import FNB_UNet_SingConv
from model.models.FNB_UNet_SingConv_MLP import FNB_UNet_SingConv_MLP
from model.models.FNB_UNet_DouConv_MLP import FNB_UNet_DouConv_MLP
from model.models.FNB_UNet_DSConv import FNB_UNet_DSConv, FNB_UNet_DSConv_MLP
from model.models.FNB_UNet_MLP_Add import FNB_UNet_DSConv_MLP_add

# 定义颜色编码表，这里使用红、绿、蓝三种颜色
label_colors = {
    0: [0, 0, 0],       # 背景（黑色）
    1: [255, 255, 255],   # 第一个类别（白色）
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

def evaluate(model, dataloader, device):
    model.eval()
    num_samples = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for img, _, img_mask, _, _, in dataloader:
            img = img.to(device)
            img_mask = img_mask.to(device)

            output = model(img)
            output = torch.sigmoid(output)
            num_samples += img.size(0)

            # compute acc
            predicted = (output > 0.5).float()
            correct += (predicted == img_mask).sum().item()
            total += img_mask.numel()

    acc = correct / total
    return acc


def seg_inference(test_image_path, test_label_path, num_classes, gpu,
                  input_channels, snapshot_dir, save_path, is_ference=False, is_gpu=True):
    if is_ference:
        test_dataset = FNDataset_V1(
            imgs_path=test_image_path,
            transform=transforms.Compose([
                transforms.Resize([128, 128]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
    else:
        test_dataset = FNDataset(
            imgs_path=test_image_path,
            label_path=test_label_path,
            split='test',
            transform=transforms.Compose([
                transforms.Resize([128, 128]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if is_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fn_model = SwinUnet(embed_dim=96, patch_height=4, patch_width=4, class_num=2)  #  test swinunet
    # fn_model = EGEUNet(num_classes, input_channels)
    # fn_model = UNext(num_classes, input_channels)  # UNext, UNext_S
    # fn_model = FNB_UNet_MLP(input_channels, num_classes)
    # fn_model = NestedUNet(input_channels, num_classes)  # test U_Net, AttU_Net, R2U_Net, NestedUNet, R2AttU_Net

    # ablation FNB_UNet_XX
    # FNB_UNet_DouConv, FNB_UNet_SingConv, FNB_UNet_SingConv_MLP， FNB_UNet_DouConv_MLP
    # fn_model = FNB_UNet_DouConv_MLP(input_channels, num_classes)
    # fn_model = FNB_UNet_DSConv_MLP(input_channels, num_classes)  # FNB_UNet_DSConv_MLP, FNB_UNet_DSConv
    fn_model = FNB_UNet_DSConv_MLP_add(input_channels, num_classes)

    # fn_model = nn.DataParallel(fn_model).cuda()
    fn_model = fn_model.to(device)
    # print(fn_model)
    fn_model.load_state_dict(torch.load(snapshot_dir))

    fn_model.eval()

    class_ious = [[] for _ in range(num_classes)]

    total_start_time = time.time()
    if is_ference:
        for index, (img, img_name, image_path) in enumerate(test_dataloader):
            single_start_time = time.time()
            img = img.to(device)

            output = fn_model(img)
            output = torch.sigmoid(output)
            b, _, w, h = output.size()

            pred = output.cpu().permute(0, 2, 3, 1).contiguous(). \
                view(-1, num_classes).max(1)[1].view(b, w, h).numpy().squeeze()

            seg_reult = os.path.join(save_path, 'seg_result')
            os.makedirs(seg_reult, exist_ok=True)
            result_path = os.path.join(seg_reult, img_name[0].replace('.jpg', '.png'))

            # 将标签图像转换为彩色图像
            pred_colorized = colorize_mask(pred, label_colors)
            imageio.imwrite(result_path, pred_colorized)
            # Convert ground truth mask to numpy array

    else:
        for index, (img, img_mask, _, img_name, image_path) in enumerate(test_dataloader):
            single_start_time = time.time()
            img = img.to(device)
            img_mask = img_mask.to(device)

            output = fn_model(img)
            output = torch.sigmoid(output)
            b, _, w, h = output.size()
            _, w_gt, h_gt = img_mask.size()

            pred = output.cpu().permute(0, 2, 3, 1).contiguous(). \
                view(-1, num_classes).max(1)[1].view(b, w, h).numpy().squeeze()

            seg_reult = os.path.join(save_path, 'seg_result')
            os.makedirs(seg_reult, exist_ok=True)
            result_path = os.path.join(seg_reult, img_name[0].replace('.jpg', '.png'))

            # 将标签图像转换为彩色图像
            pred_colorized = colorize_mask(pred, label_colors)
            imageio.imwrite(result_path, pred_colorized)
            # Convert ground truth mask to numpy array
            gt_mask = img_mask.cpu().numpy().squeeze()

            # Calculate IoU for each class and store the results
            iou_list = calculate_iou(pred, gt_mask, num_classes)
            for class_id, iou in enumerate(iou_list):
                class_ious[class_id].append(iou)


        total_end_time = time.time()  # 记录总的测试时间结束点
        total_elapsed_time = total_end_time - total_start_time
        print(f"Total Testing Time: {total_elapsed_time:.4f} seconds; Single Testing Time: "
              f"{total_elapsed_time/len(test_dataloader):.4f}")

        flops, params = profile(fn_model, inputs=(img,))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')

        acc = evaluate(model=fn_model, dataloader=test_dataloader, device=device)
        print('Acc is: %f' % acc)


if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    seg_inference(test_image_path=args.test_image_path,
                  test_label_path=args.test_label_path,
                  num_classes=args.num_classes,
                  input_channels=3,
                  snapshot_dir='./snapshots/1109/FNB_UNet_DSConv_MLP_add/best_model.pt',
                  save_path = f'./Result/1109/FNB_UNet_DSConv_MLP_add/',
                  gpu=args.gpu)

    evaluate_images_in_folder('./Result/0921/FNB_UNet_DSConv_MLP_add/seg_result/',
                              args.test_label_path)
    print('Test done!')
