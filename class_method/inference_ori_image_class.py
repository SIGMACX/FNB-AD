# @Time: 2023/9/11 15:54
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------------------------
# inference class model based on origin_crop image
# -------------------------------------------------


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from class_method.models.mobilenet import mobilenet
from class_method.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from class_method.models.densenet import densenet121, densenet161, densenet169, densenet201
from class_method.models.resnext import resnext50, resnext101, resnext152
from class_method.models.vggnet import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from class_method.models.googlenet import googlenet


from class_method.load_dataset.class_load_dataset import classDataset
from tools.config import args
from class_method.train_class import test

def seg_result_class_infer(dataset_root, num_classes, batch_size, snapshot_dir):
    seg_result = classDataset(dataset_root=dataset_root,
                              transform = transforms.Compose([
                                  transforms.Resize([128, 128]),
                                  transforms.RandomRotation(10),
                                  transforms.ToTensor()
                              ]))
    seg_result_loader = DataLoader(seg_result, batch_size=batch_size, shuffle=True, num_workers=0)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = mobilenet(num_classes)
    # model = resnet101(num_classes)    # resnet18, 34, 50, 101, 152
    # model = densenet201(num_classes)   # densenet121, 161, 169, 201
    # model = resnext152(num_classes)    # resnext50, renext101, 152
    # model = vgg13_bn(num_classes)    #vgg11_bn, 13, 16, 19

    model = nn.DataParallel(model)
    model.to(device)

    model.load_state_dict(torch.load(snapshot_dir))

    total_start_time = time.time()

    test_acc, test_aver_accuracy, class_accuracy, test_duration, recalls, f1_scores = test(model, seg_result_loader, num_classes)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print("*" * 80)
    print("Test accuracy: ", test_acc)
    print("Total average Accuracy:", test_aver_accuracy)
    print("Recalls: ", recalls)
    print("F1_scores: ", f1_scores)
    print("Class-wise accuracy:")
    for i in range(num_classes):
        if i == 0:
            print(f"Class abnormal: {class_accuracy[i]}")
        else:
            print(f"Class normal: {class_accuracy[i]}")
    print(f"Total Testing Time: {total_duration:.4f} seconds; And single image :{test_duration:.4f} second!")
    print("*" * 80, "\n")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(0)
    torch.manual_seed(0)

    seg_result_class_infer(dataset_root=args.all_image_test_path,
                           num_classes=args.class_num_classes,
                           batch_size=args.class_test_batch_size,
                           snapshot_dir = f"{args.class_snapshots_all_path}/mobilenet/best_model.pt")
