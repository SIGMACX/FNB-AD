# -*- coding: utf-8 -*-
# @Time : 2023/8/13 下午9:04
# @Author : ChenXi

import torch
import torch.nn.functional as F

def dice_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    union = torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3])
    return torch.mean((2. * intersection + smooth) / (union + smooth))

def dice_p_bce(in_gt, in_pred):
    return -dice_coef(in_gt, in_pred)