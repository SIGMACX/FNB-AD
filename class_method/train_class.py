# @Time: 2023/9/4 10:02
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# training model
# -------------------------------
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from class_method.models.mobilenet import mobilenet
from class_method.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from class_method.models.densenet import densenet121, densenet161, densenet169, densenet201
from class_method.models.resnext import resnext50, resnext101, resnext152
from class_method.models.vggnet import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from class_method.models.googlenet import googlenet

# transformer
from model.models.swin_unet import SwinUnet

from class_method.load_dataset.class_load_dataset import classDataset
from tools.config import args


def train(image_train_path, image_test_path, num_epoch, num_classes, batch_size,
          lr, save_path, test_interval, snapshots_path, gpu):
    train_dataset = classDataset(dataset_root = image_train_path,
                                 transform = transforms.Compose([
                                     transforms.Resize([224, 224]),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     # transforms.Normalize(
                                     #     mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ]))

    test_dataset = classDataset(dataset_root=image_test_path,
                                transform=transforms.Compose([
                                    transforms.Resize([224, 224]),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(
                                    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = mobilenet(num_classes)      # define MobileNet model
    # model = resnet152(num_classes)      # resnet18, 34, 50, 101, 152
    # model = densenet201(num_classes)     # densenet121, 161, 169, 201
    # model = resnext152(num_classes)     # resnext50, 101, 152
    # model = vgg11_bn(num_classes)     # vgg11, 13, 16, 19
    model = googlenet(num_classes)

    model = nn.DataParallel(model)
    model.to(device)

    # define loss and optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    print("\n", "#" * 50, f"\n Start Training {save_path}!\n", "#" * 50, "\n")
    start_time = time.time()
    best_accuracy = 0.0
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        predictions = []
        ground_truths = []

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

        accuracy = accuracy_score(ground_truths, predictions)
        print(f"Epoch {epoch+1}/{num_epoch}, "
              f"Loss: {running_loss/len(train_loader)}, "
              f"Accuracy: {accuracy}")

        os.makedirs(f'./{snapshots_path}/{save_path}', exist_ok = True)
        os.makedirs(f'./{snapshots_path}/{save_path}', exist_ok = True)

        if np.mod(epoch+1, test_interval) == 0:
            # torch.save(model.state_dict(),
            #            './{snapshots_path}/{}/class_model_{}.pt'.format(save_path, epoch+1))
            # print('Saving Class Checkpoints(all): class_model_{}.pt'.format(epoch+1))

            test_accuracy, test_aver_accuracy, class_accuracy, \
            test_duration, recalls, f1_scores = test(model, test_loader, num_classes, gpu)

            print(f'Accuracy on test dataset after {epoch+1} epochs: {test_accuracy}')
            print("*" * 50)
            print("Test_aver_accuracy: ", test_aver_accuracy)
            print("Recalls: ", recalls)
            print("F1_scores: ", f1_scores)
            print("Class-wise accuracy: ")
            print("Test accuracy: ", test_accuracy)
            for i in range(num_classes):
                if i == 0:
                    print(f"Class abnormal: {class_accuracy[i]}")
                else:
                    print(f"Class normal: {class_accuracy[i]}")
            print("*" * 50)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                # save the best model
                torch.save(model.state_dict(),
                           './{snapshots_path}/{}/best_model.pt'.format(save_path))
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration:.2f} seconds")
    print(f"Finished Training {save_path}!")
    print("*" * 50)

    # final_test_accuracy = test(model, test_loader, num_classes)
    # print(f'Final accuracy on test dataset : {final_test_accuracy}')

def test(model, test_loader, num_classes, gpu):
    model.eval()
    correct = 0
    total = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)

    incorrect_image_paths = []  # 存储错误分类的图像路径

    with torch.no_grad():
        for test_input, test_labels, img_path in test_loader:
            test_start = time.time()
            test_inputs, test_labels = test_input.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)
            _, predicted = torch.max(test_outputs, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

            for i in range(num_classes):
                class_total[i] += (test_labels == i).sum().item()
                class_correct[i] += (predicted[test_labels == i] == i).sum().item()

                # 计算每个类别的真正例、假正例和假负例
                for i in range(num_classes):
                    true_positives[i] += ((predicted == i) & (test_labels == i)).sum().item()
                    false_positives[i] += ((predicted == i) & (test_labels != i)).sum().item()
                    false_negatives[i] += ((predicted != i) & (test_labels == i)).sum().item()

            # 检查哪些样本被错误分类并记录其路径
            for i in range(len(img_path)):
                if predicted[i] != test_labels[i]:
                    incorrect_image_paths.append(img_path[i])

            test_end = time.time()
            test_duration = test_end - test_start

    accuracy = correct / total
    class_accuracy = class_correct / class_total

    recalls = true_positives / (true_positives + false_negatives)
    precisions = true_positives / (true_positives + false_positives)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    recalls = np.mean(recalls)
    f1_scores = np.mean(f1_scores)

    average_accuracy = np.mean(class_accuracy)
    print(total)

    return accuracy, average_accuracy, class_accuracy, test_duration, recalls, f1_scores, incorrect_image_paths



if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    train(image_train_path=args.ori_image_train_path,
          image_test_path=args.ori_image_test_path,
          num_epoch = args.class_epoch,
          num_classes = args.class_num_classes,
          batch_size = args.class_batch_size,
          lr = args.class_lr,
          test_interval = 100,
          save_path = 'googlenet',
          snapshots_path=args.class_snapshots_ori_path,
          gpu=args.gpu)





