# -*- coding: utf-8 -*-
# @Time : 2023/8/14 上午10:26
# @Author : ChenXi

from model.module.unet_parts import *
from model.models.U_Next import shiftedBlock, OverlapPatchEmbed

import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        original_size = x.size()[2:]  # Save the original height and width
        x = x.mean(dim=(2, 3))  # Global average pooling along spatial dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), -1, 1, 1)  # Reshape to (batch_size, channels, 1, 1)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)  # Resize back to original size
        return x


class FNB_UNet_DouConv(nn.Module):
    """ Full assembly of the parts to form the complete network """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FNB_UNet_DouConv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)

        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



