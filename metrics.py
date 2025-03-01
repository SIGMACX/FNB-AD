# @Time: 2023/8/2 9:25
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 评价指标
# -------------------------------

"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import os

import numpy as np
import cv2
# from test_dou_class_u_net import calculate_iou

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""
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

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        # m_IoU = calculate_iou()
        MIOU = max(self.IntersectionOverUnion())
        return mIoU, MIOU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

import os
import cv2
import numpy as np

# 在此处放置 SegmentationMetric 类的定义和方法

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

def calculate_iou(prediction, target):
    intersection = np.logical_and(prediction, target)
    union = np.logical_or(prediction, target)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice(prediction, target):
    intersection = np.logical_and(prediction, target)
    dice = (2.0 * np.sum(intersection)) / (np.sum(prediction) + np.sum(target))
    return dice


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def Dice(inp, target, eps=1e-5):
	# 抹平了，弄成一维的
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    # 计算交集中的数量
    overlap = np.sum(input_flatten * target_flatten)
    # 返回值，让值在0和1之间波动
    return np.clip(((2 * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)


def evaluate_images_in_folder(images_folder, label_folder):
    image_data, label_data = load_img_label_from_folder(images_folder, label_folder)

    metric = SegmentationMetric(2)  # 2表示有2个分类，有几个分类就填几

    total_iou = 0
    total_mIoU = 0
    total_dice = 0


    for imgPredict, imgLabel in zip(image_data, label_data):
        # 将图像和标签转换为0和1的二值图像
        imgPredict = np.array(imgPredict / 255., dtype=np.uint8)
        imgLabel = np.array(imgLabel / 255., dtype=np.uint8)

        # 计算评价指标并累加混淆矩阵
        metric.addBatch(imgPredict, imgLabel)

        # 计算IoU和Dice
        iou = calculate_iou(imgPredict, imgLabel)
        # dice = calculate_dice(imgPredict, imgLabel)

        dc = Dice(imgPredict, imgLabel)

        total_iou += iou
        total_mIoU += metric.meanIntersectionOverUnion()[0]
        total_dice += dc

        # print('IoU is: %f' % iou)
        # print('Dice is: %f' % dice)

    # 计算各项评价指标
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = total_mIoU / len(image_data)  # Calculate the mean mIoU
    _, MIOU = metric.meanIntersectionOverUnion()

    print('PA is: %f' % pa)
    print('cPA is:', cpa)
    print('mPA is: %f' % mpa)
    print("MIOU is: %f " % MIOU)
    print('Mean Dice is: %f' % (total_dice / len(image_data)))


'''
# 测试内容
if __name__ == '__main__':
    images_folder = "./Result/0921/FNB_UNet_MLP/seg_result/"
    label_folder = './data/FNBS_Dataset/segmentation/testing_dataset/label_gt/'
    evaluate_images_in_folder(images_folder, label_folder)
'''

