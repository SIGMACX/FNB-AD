# @Time: 2023/8/4 21:48
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  模型参数量计算
# -------------------------------

import torch
from thop import profile
from model.models.U_Next import UNext
from tools.config import args
from torchsummary import summary
from model.unet import Fn_Net_UNet_V1, FNB_UNet, FNB_UNet_V2
from model.models.U_net_selfattention import U_Net, AttU_Net, R2U_Net, R2AttU_Net, NestedUNet
from PIL import Image
import torchvision.transforms as transforms
from model.models.U_Next import UNext
from model.models.ege_unet import EGEUNet
from model.models.swin_unet import SwinUnet

# classification
from class_method.models.mobilenet import mobilenet
from class_method.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from class_method.models.densenet import densenet121, densenet161, densenet169, densenet201
from class_method.models.resnext import resnext50, resnext101, resnext152
from class_method.models.vggnet import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from class_method.models.googlenet import googlenet

# 定义网络
n_channels = 3  # 输入图像的通道数
num_classes = args.num_classes   # 输出图像的类别数
bilinear = True
# model = NestedUNet(n_channels, num_classes)
# model = UNext(num_classes, n_channels)
# model = SwinUnet(embed_dim=96, patch_height=4, patch_width=4, class_num=2)  # origin patch_w, *_h=4

# classification
# model = mobilenet(num_classes)      # define MobileNet model
# model = resnet152(num_classes)      # resnet18, 34, 50, 101, 152
# model = densenet161(num_classes)     # densenet121, 161, 169, 201
# model = resnext152(num_classes)     # resnext50, 101, 152
# model = vgg19_bn(num_classes)     # vgg11, 13, 16, 19
model = googlenet(num_classes)

# 将模型移动到GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 读取一张图像作为输入数据
image_path = '../data/FN_local/trainingset/img_148/0_00001_1.png'
image = Image.open(image_path).convert('L')
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
# input_data = transform(image).to(device)
input_data = torch.randn(1, 1, 128, 128).to(device)



# 使用torchsummary计算FLOPs和参数数量
# summary(model, (3, 128, 128))  # 输入图像大小为148x148

# 计算FLOPS和参数数量
flops, params = profile(model, inputs=(input_data,))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')



