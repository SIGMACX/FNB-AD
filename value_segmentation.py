# @Time: 2023/9/20 20:30
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# value segmentation
# -------------------------------
import os
import sys

import cv2
import numpy as np
# from test_dou_class_u_net import calculate_iou

def load_img_label_from_folder(images_folder, label_folder):
    image_data = []
    label_data = []

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(images_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image_data.append(img)

            label_filename = os.path.splitext(filename)[0] + ".png"
            label_path = os.path.join(label_folder, label_filename)
            label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label_data.append(label_img)

    return image_data, label_data

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))  # 混淆矩阵
        self.pixel_accuracy = 0
        self.class_pixel_accuracy = np.zeros(self.numClass)
        self.mean_pixel_accuracy = 0
        self.intersection_over_union = np.zeros(self.numClass)
        self.mean_intersection_over_union = 0
        self.class_recall = np.zeros(self.numClass)
        self.class_precision = np.zeros(self.numClass)
        self.class_fnr = np.zeros(self.numClass)
        self.class_fpr = np.zeros(self.numClass)
        self.dice_coefficient = np.zeros(self.numClass)

    def genConfusionMatrix(self, imgPredict, imgLabel):
        """
        Generate the confusion matrix.
        :param imgPredict: Predicted segmentation image (numpy array)
        :param imgLabel: Ground truth segmentation image (numpy array)
        :return: Confusion matrix (numpy array)
        """
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def pixelAccuracy(self):
        # Pixel Accuracy (PA) = (TP + TN) / (TP + TN + FP + FN)
        self.pixel_accuracy = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return self.pixel_accuracy

    def classPixelAccuracy(self):
        # Class Pixel Accuracy (CPA) = TP / (TP + FP)
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + np.finfo(float).eps)
        self.class_pixel_accuracy = classAcc
        return classAcc

    def meanPixelAccuracy(self):
        # Mean Pixel Accuracy (MPA) = (CPA_1 + CPA_2 + ... + CPA_N) / N
        self.mean_pixel_accuracy = np.nanmean(self.class_pixel_accuracy)
        return self.mean_pixel_accuracy

    def IntersectionOverUnion(self):
        # Intersection over Union (IoU) = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = (self.confusionMatrix.sum(axis=1) + self.confusionMatrix.sum(axis=0)) - intersection
        self.intersection_over_union = intersection / (union + np.finfo(float).eps)
        return self.intersection_over_union

    def meanIntersectionOverUnion(self):
        # Mean Intersection over Union (MIoU) = (IoU_1 + IoU_2 + ... + IoU_N) / N
        self.mean_intersection_over_union = np.nanmean(self.intersection_over_union)
        return self.mean_intersection_over_union

    def classRecall(self):
        # Recall = TP / (TP + FN)
        self.class_recall = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + np.finfo(float).eps)
        return self.class_recall

    def classPrecision(self):
        # Precision = TP / (TP + FP)
        self.class_precision = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + np.finfo(float).eps)
        return self.class_precision

    def classFNR(self):
        # False Negative Rate (FNR) = FN / (FN + TP)
        self.class_fnr = 1 - self.class_recall
        return self.class_fnr

    def classFPR(self):
        # False Positive Rate (FPR) = FP / (FP + TN)
        self.class_fpr = 1 - self.class_precision
        return self.class_fpr

    def diceCoefficient(self):
        # Dice Coefficient = 2 * TP / (2 * TP + FP + FN)
        self.dice_coefficient = (2 * np.diag(self.confusionMatrix)) / (
                2 * np.diag(self.confusionMatrix) + self.confusionMatrix.sum(axis=1) + self.confusionMatrix.sum(axis=0))
        return self.dice_coefficient

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def dice_coefficient(predicted, target, smooth=1e-5):
        # 将预测结果和目标转换为numpy数组
        predicted = predicted.flatten()
        target = target.flatten()

        # 计算交集和并集
        intersection = np.sum(predicted * target)
        union = np.sum(predicted) + np.sum(target)

        # 计算Dice系数
        dice = (2.0 * intersection + smooth) / (union + smooth)

        return dice

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


if __name__ == '__main__':
    images_folder = "./Result/0921/NestedUNet/seg_result/"
    label_folder = './data/FNBS_Dataset/segmentation/testing_dataset/label_gt/'

    # 检查 images_folder 是否存在
    if not os.path.exists(images_folder):
        print(f"Error: The specified 'images_folder' does not exist: {images_folder}")
        sys.exit(1)  # 退出程序，返回错误代码 1

    # 检查 label_folder 是否存在
    if not os.path.exists(label_folder):
        print(f"Error: The specified 'label_folder' does not exist: {label_folder}")
        sys.exit(1)  # 退出程序，返回错误代码 1

    image_data, label_data = load_img_label_from_folder(images_folder, label_folder)

    metric = SegmentationMetric(2)

    pa_list = []  # List to store Pixel Accuracy (PA) values
    cpa_list = []  # List to store Class Pixel Accuracy (CPA) values
    mpa_list = []  # List to store Mean Pixel Accuracy (MPA) values
    iou_list = []  # List to store Intersection over Union (IoU) values
    miou_list = []  # List to store Mean Intersection over Union (MIoU) values
    recall_list = []  # List to store Recall values
    precision_list = []  # List to store Precision values
    fnr_list = []  # List to store False Negative Rate (FNR) values
    fpr_list = []  # List to store False Positive Rate (FPR) values
    dice_list = []  # List to store Dice Coefficient values

    for imgPredict, imgLabel in zip(image_data, label_data):
        # M_ioU = calculate_iou(imgPredict, imgLabel, num_classes=2)
        # Convert images and labels to 0 and 1 binary images
        imgPredict = np.array(imgPredict / 255., dtype=np.uint8)
        imgLabel = np.array(imgLabel / 255., dtype=np.uint8)

        # Calculate evaluation metrics and accumulate confusion matrix
        metric.addBatch(imgPredict, imgLabel)

        # Calculate and store individual metrics for this image
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        iou = metric.IntersectionOverUnion()
        miou = metric.meanIntersectionOverUnion()
        recall = metric.classRecall()
        precision = metric.classPrecision()
        fnr = metric.classFNR()
        fpr = metric.classFPR()
        dice = metric.diceCoefficient()

        pa_list.append(pa)
        cpa_list.append(cpa)
        mpa_list.append(mpa)
        iou_list.append(iou)
        miou_list.append(miou)
        recall_list.append(recall)
        precision_list.append(precision)
        fnr_list.append(fnr)
        fpr_list.append(fpr)
        dice_list.append(dice)

    # Calculate the mean of the metrics
    mean_pa = np.mean(pa_list)
    mean_cpa = np.mean(cpa_list)
    mean_mpa = np.mean(mpa_list)
    mean_iou = np.mean(iou_list)
    mean_miou = np.mean(miou_list)
    mean_recall = np.mean(recall_list)
    mean_precision = np.mean(precision_list)
    mean_fnr = np.mean(fnr_list)
    mean_fpr = np.mean(fpr_list)
    mean_dice = np.mean(dice_list)

    # Print the mean values of the metrics
    print('Mean Pixel Accuracy (PA) is : %f' % mean_pa)
    print('Mean Class Pixel Accuracy (CPA) is :', mean_cpa)
    print('Mean Mean Pixel Accuracy (MPA) is : %f' % mean_mpa)
    print('Mean Intersection over Union (IoU) is :', mean_iou)
    print('Mean Mean Intersection over Union (MIoU) is : %f' % mean_miou)
    print('Mean Class Recall is :', mean_recall)
    print('Mean Class Precision is :', mean_precision)
    print('Mean Class False Negative Rate (FNR) is :', mean_fnr)
    print('Mean Class False Positive Rate (FPR) is :', mean_fpr)
    print('Mean Dice Coefficient is :', mean_dice)



