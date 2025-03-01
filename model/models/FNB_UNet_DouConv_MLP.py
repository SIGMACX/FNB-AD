# @Time: 2023/9/24 21:47
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# FNB_UNet_DouConv_MLP
# -------------------------------

from model.module.unet_parts import *
from model.models.U_Next import shiftedBlock, OverlapPatchEmbed

import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, in_chan, out_chan, drop_prob=0.0, mlp_1 = True):
        super(mlp, self).__init__()

        # Multi-Layer Perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(in_chan, in_chan),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(in_chan, out_chan)
        )

        # Depthwise Separable Convolution
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1, groups=in_chan),
            nn.BatchNorm2d(in_chan),
            nn.GELU(),
            nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_chan)
        )
        self.mlp_1 = mlp_1

    def forward(self, x):
        # MLP
        if self.mlp_1:
            mlp_output = self.mlp(x.view(-1, 128))
        else:
            mlp_output = self.mlp(x.view(-1, 256))


        # Depthwise Separable Convolution
        dw_conv_output = self.dw_conv(x)
        # Add MLP output to DWConv output
        out = mlp_output.view(x.size(0), -1, 8, 8) + dw_conv_output

        return out




class FNB_UNet_DouConv_MLP(nn.Module):
    """ Full assembly of the parts to form the complete network """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FNB_UNet_DouConv_MLP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)

        self.up1 = Up(384, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        self.mlp_1 = mlp(in_chan=128, out_chan=256, mlp_1=True)
        self.mlp_2 = mlp(in_chan=256, out_chan=256, mlp_1=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.mlp_1(x5)
        x5 = self.mlp_2(x5)

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
