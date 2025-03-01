# @Time: 2023/7/27 9:22
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  segmentation task (training)
#  lr scheduler(09-05)
# -------------------------------

import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

import sys
from tools.config import args
from dataset.dataloader_mulclsfn_unet import FNDataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
# from model.unet import Fn_Net_UNet
# from model.models.U_net_selfattention import *
from torch.nn.modules.loss import CrossEntropyLoss
from model.loss.diceloss import *

from model.models.U_net_selfattention import U_Net, AttU_Net, R2U_Net, R2AttU_Net, NestedUNet
from model.models.U_Next import UNext, UNext_S
from model.models.ege_unet import EGEUNet
from model.models.FNBU_Net_MLP import FNB_UNet_MLP
from model.models.swin_unet import SwinUnet

# FNB_UNet_XX  Ablation
from model.models.FNB_UNet_DouConv import FNB_UNet_DouConv
from model.models.FNB_UNet_SingConv import FNB_UNet_SingConv
from model.models.FNB_UNet_SingConv_MLP import FNB_UNet_SingConv_MLP
from model.models.FNB_UNet_DouConv_MLP import FNB_UNet_DouConv_MLP

from model.models.FNB_UNet_DSConv import FNB_UNet_DSConv, FNB_UNet_DSConv_MLP
from model.models.FNB_UNet_MLP_Add import FNB_UNet_DSConv_MLP_add


def train(opoch_num, num_classes, input_channels, batch_size, lr, momentum, patience, save_path, train_interval, TD):
    train_dataset = FNDataset(imgs_path = args.train_image_path,
                              label_path = args.train_label_path,
                              split = 'train',
                              transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                              )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = FNDataset(imgs_path=args.test_image_path,
                             label_path=args.test_label_path,
                             split='test',
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])]))

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fn_model = R2AttU_Net(input_channels, num_classes)  # UNet, AttU_Net, R2U_Net, NestedUNet, R2AttU_Net
    # fn_model = EGEUNet(num_classes, input_channels)
    # fn_model = UNext(num_classes, input_channels)   # UNeXt, UMeXt_S
    # fn_model = FNB_UNet_MLP(input_channels, num_classes)
    # fn_model = SwinUnet(embed_dim=96, patch_height=4, patch_width=4, class_num=2)  # origin patch_w, *_h=4

    # FNB_UNet_XX ablation
    # FNB_UNet_DouConv, FNB_UNet_SingConv, FNB_UNet_SingConv_MLP, FNB_UNet_DouConv_MLP
    # fn_model = FNB_UNet_DouConv(input_channels, num_classes)
    # fn_model = FNB_UNet_DSConv_MLP(input_channels, num_classes)  #FNB_UNet_DSConv, FNB_UNet_DSConv_MLP
    fn_model = FNB_UNet_DSConv_MLP_add(input_channels, num_classes)

    # fn_model = nn.DataParallel(fn_model)   # parallel train
    # print(fn_model)
    if torch.cuda.is_available():
        fn_model = fn_model.to('cuda')
    # fn_model = fn_model.to(device)

    # 损失函数
    criterion = nn.BCELoss().to(device)
    # criterion = DiceLoss().to(device)   # 收敛慢
    # criterion = nn.CrossEntropyLoss().to(device)  # 收敛不明显
    criterion_1 = CrossEntropyLoss().to(device)
    dice_loss = DiceLoss(num_classes)

    # 优化器
    optimizer = optim.SGD(fn_model.parameters(), lr = lr, momentum=momentum)
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, min_lr=0.0001)

    print("\n", "#" * 50, f"\n Start Training {save_path}! \n", "#" * 50, "\n")

    best_acc = 0.0
    no_improvement_counter = 0
    max_no_improvement = 10

    for epo in range(opoch_num):
        train_loss = 0
        fn_model.train()
        for index, (img, _, img_mask, _, _,) in enumerate(train_dataloader):
            img = img.to(device)
            img_mask = img_mask.to(device)

            optimizer.zero_grad()
            output = fn_model(img)

            output = torch.sigmoid(output)

            loss = criterion(output, img_mask)
            loss_1 = criterion_1(output, img_mask)
            loss_dice = dice_loss(output, img_mask, softmax=True)
            # loss = 0.2 * loss + 0.8 * loss_1

            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()
            # scheduler.step()
            lr_scheduler.step(iter_loss)   # lr 衰减 0.1~0.0001

            if np.mod(index, 20) == 0:
                print('Epoch: {}/{}, Step: {}/{}, Train loss: {}'.format(
                    epo, opoch_num, index, len(train_dataloader), iter_loss))

        os.makedirs(f'./checkpoints/{save_path}', exist_ok=True)
        os.makedirs(f'./snapshots/{TD}/{save_path}', exist_ok=True)

        if np.mod(epo+1, train_interval) == 0:
            '''# 保存模型
            torch.save(fn_model.state_dict(),
                './snapshots/0921/{}/model_{}.pt'.format(save_path, epo+1))
            print('Saveing Checkpoints: model_{}.pt'.format(epo+1))'''

            avg_test_loss, accuracy = evaluate(model=fn_model,
                                               dataloader=test_dataloader,
                                               criterion=criterion,
                                               device=device)
            print(f'\nAverage Testing Loss: {avg_test_loss}, Accuracy: {accuracy:.4f}')
            if accuracy > best_acc:
                best_acc = accuracy
                # save the best model
                torch.save(fn_model.state_dict(), './snapshots/{}/{}/best_model.pt'.format(TD,save_path))
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

        if no_improvement_counter >= max_no_improvement:
            print(f"No improvement in validation accuracy for {epo+1}. Training stopped.")
            break


    print("\n Training Finished. \n")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_samples = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for img, _, img_mask, _, _, in dataloader:
            img = img.to(device)
            img_mask = img_mask.to(device)

            output = model(img)
            output = torch.sigmoid(output)
            loss =  criterion(output, img_mask)

            total_loss += loss.item()
            num_samples += img.size(0)

            # compute acc
            predicted = (output > 0.5).float()
            correct += (predicted == img_mask).sum().item()
            total += img_mask.numel()

    avg_loss = total_loss / num_samples
    accuracy = correct / total
    return avg_loss, accuracy


# 自定义学习率调度器类
class CustomStepLR(StepLR):
    def __init__(self, optimizer, step_size, gamma, min_lr):
        self.min_lr = min_lr
        self.gamma = gamma
        super(CustomStepLR, self).__init__(optimizer, step_size, gamma)

    def get_lr(self):
        # 获取当前学习率
        current_lr = super(CustomStepLR, self).get_lr()[0]

        # 如果当前学习率小于最小学习率，则停止更新
        if current_lr < self.min_lr:
            current_lr = self.min_lr

        return [current_lr * self.gamma]


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    train(opoch_num=args.epoch,
          num_classes=args.num_classes,
          input_channels=3,
          batch_size=args.batch_size,
          lr=args.lr,
          momentum=args.momentum,
          patience=args.patience,
          save_path='FNB_UNet_DSConv_MLP_add_1',
          TD='1109',
          train_interval=args.train_interval)