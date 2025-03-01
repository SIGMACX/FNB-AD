# @Time: 2023/7/29 17:42
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 1. 设置参数
# -------------------------------

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_epoch', type=int, help='epoch', dest='epoch', default=2000)
parser.add_argument('--num_classes', type=int, help='num of classes', dest='num_classes', default=2)
parser.add_argument('--batch_size', type=int, help='batch size', dest='batch_size', default=128)
parser.add_argument('--lr', type=int, help='learning rate', dest='lr', default=5e-1)
parser.add_argument('--train_interval', type=int, help='train interval', dest='train_interval', default=20)
parser.add_argument('--momentum', type=float, help='momentum of optimization', dest='momentum', default=0.9)
parser.add_argument('--patience', type=int, help='Number of learning rate decay times', dest='patience', default=100)
parser.add_argument('--gpu', type=str, choices=["0","1","2",'3'], dest='gpu', default='3')

# =============================== detection parameters ======================================
parser.add_argument('--detection_num_epoch', type=int, help='epoch of detection', dest='detection_num_epoch', default=1000)
parser.add_argument('--train_image_root_detection', help='train image dir of detection task', dest='train_image_root_detection',
                    default='../data/FNBS_Dataset/D_part/test/image/')
parser.add_argument('--train_label_root_detection', help='train_label dir of detection task', dest='train_label_root_detection',
                    default='../data/FNBS_Dataset/D_part/test/label/')
parser.add_argument('--train_label_txt_detection', help='train_label dir of detection task', dest='train_label_txt_detection',
                    default='../data/FNBS_Dataset/D_part/test/label_txt/')

parser.add_argument('--test_image_root_detection', help='test image dir of detection task', dest='test_image_root_detection',
                    default='../data/FNBS_Dataset/D_part/test/image/')
parser.add_argument('--test_label_root_detection', help='test_label dir of detection task', dest='test_label_root_detection',
                    default='../data/FNBS_Dataset/D_part/train/train_coco.json')

parser.add_argument('--detection_batch_size', type=int, help='batch size of detection task',
                    dest='detection_batch_size', default=1)
parser.add_argument('--detection_lr', type=int, help='learning rate of detection', dest='detection_lr', default=1e-3)
parser.add_argument('--detection_num_classes', help = 'number of detection classes',
                    dest = 'detection_num_classes', default=2)





# =============================== segmentation parameters ======================================
# determine f_nb path
parser.add_argument('--train_image_path', help='train image path', dest='train_image_path',
                    default='./data/FN_local/trainingset/img_148/'
                    )
parser.add_argument('--train_label_path', help='train label path', dest='train_label_path',
                    default='./data/FN_local/trainingset/mask_bw/'
                    )
parser.add_argument('--test_image_path', help='test image path', dest='test_image_path',
                    default='./data/FNBS_Dataset/segmentation/testing_dataset/image/'
                    # default='./data/FN_local/testingset/test/img_324_128/'
                    )
parser.add_argument('--test_label_path', help='test label path', dest='test_label_path',
                    default='./data/FNBS_Dataset/segmentation/testing_dataset/label_gt/'
                    # default='./data/FN_local/testingset/test/mask_324_128/'
                    )

parser.add_argument('--seg_result_path', help='segmentation result path', dest='seg_result_path',
                    default='./Result/mul_fn_seg/class_3/EGEUNet_0905/train_dataset/seg_result/')

parser.add_argument('--seg_for_image_output_folder', help='output_path of segmentation for image result',
                    dest='seg_for_image_result',
                    default='./Result/mul_fn_seg/class_3/EGEUNet_0905/train_dataset/seg_for_image_result')

parser.add_argument('--center_point_result', help='output_path of center point for image result',
                    dest = 'center_points_result',
                    default='./Result/mul_fn_seg/class_3/EGEUNet_0905/train_dataset/center_points_result')

parser.add_argument('--seg_filter_result', help='filtering result of segmentations', dest = 'seg_filter_result',
                    default='./Result/mul_fn_seg/class_3/EGEUNet_0905/train_dataset/seg_filter_result')

parser.add_argument('--mapping_seg2image_folder', help='mapping result of segmentations',
                    dest = 'mapping_seg2image_folder',
                    default='./Result/mul_fn_seg/class_3/EGEUNet_0905/train_dataset/mapping_seg2image_result')

# =============================== classification parameters (result of FNB_UNet) ======================================

parser.add_argument('--class_epoch', type=int, help='class_epoch', dest='class_epoch', default=300)
parser.add_argument('--class_batch_size', type=int, help='class batch size', dest='class_batch_size', default=2)
parser.add_argument('--class_test_batch_size', type=int, help='class test batch size', dest='class_test_batch_size', default=1024)
parser.add_argument('--class_lr', type=int, help='learning rate for class task', dest='class_lr', default=1e-3)
parser.add_argument('--class_num_classes', type=int, help='class task num of classes', dest='class_num_classes', default=2)
parser.add_argument('--test_interval', type=int, help='test_interval', dest='test_interval', default=100)

parser.add_argument('--class_train_path', help='train image path of classification', dest='class_train',
                    default='../data/FNBS_Dataset/classification/trainingdataset/'
                    )
parser.add_argument('--class_test_path', help='test image path of classification', dest='class_test',
                    default='../data/FNBS_Dataset/classification/testingdataset/'
                    )
parser.add_argument('--class_snapshots_path', help='class snapshots path', dest='class_snapshots_path',
                    default='../class_method/class_snapshots/')
parser.add_argument('--seg_filter_result_class', help='filtering result of segmentations',
                    dest='seg_filter_result_class',
                    default="../Result/mul_fn_seg/class_3/FNB_UNet_MLP/best/seg_filter_result/")
parser.add_argument('--seg_filter_class', help='filtering result of segmentations for classification train',
                    dest = 'seg_filter_class',
                    default='../Result/mul_fn_seg/class_3/EGEUNet_0905/train_dataset/seg_filter_result/')
parser.add_argument('--seg_filter_class_test', help='filtering result of segmentations for classification test',
                    dest='seg_filter_class_test',
                    default='../Result/mul_fn_seg/class_3/EGEUNet_0905/testing_dataset/seg_filter_result/')

# =============================== classification parameters (ori_crop_image) ======================================
parser.add_argument('--ori_image_train_path', help='train image path of origin sample', dest='ori_image_train_path',
                    default='../data/FNBS_Dataset/classification_ori/training_dataset/')
parser.add_argument('--ori_image_test_path', help='test image path of origin sample', dest='ori_image_test_path',
                    default='../data/FNBS_Dataset/classification_ori/testing_dataset/')
parser.add_argument('--class_snapshots_ori_path', help='class snapshots_ori path', dest='class_snapshots_ori_path',
                    default='../class_method/class_snapshots_ori/')


# =============================== classification parameters (all_image) ======================================
parser.add_argument('--all_image_train_path', help='train image path of all sample', dest='all_image_train_path',
                    default='../data/FNBS_Dataset/all_image/train/')
parser.add_argument('--all_image_test_path', help='test image path of all sample', dest='all_image_test_path',
                    default='../data/FNBS_Dataset/all_image/test/')
parser.add_argument('--class_snapshots_all_path', help='class snapshots_all path', dest='class_snapshots_all_path',
                    default='../class_method/class_snapshots_all/')


# transformer diet
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')


#===============================main parameters (main.py)==========================================
# model path
parser.add_argument('--detection_model_pt', help='trained model path of detection', dest='detection_model_pt',
                    default="./detection_method/models/yolo/FNB_train_files/n_scratch_v8x/weights/best.pt")
parser.add_argument('--segmentation_model_pt', help='trained model path of segmentation', dest='segmentation_model_pt',
                    default='./snapshots/1109/FNB_UNet_DSConv_MLP_add_1/best_model.pt')
parser.add_argument('--classification_model_pt', help='trained model path of classification', dest='classification_model_pt',
                    default="./class_method/class_snapshots/resnet34/best_model.pt")

# save path
# detection result save path default
parser.add_argument('--detection_default_save_path', help='detection model output default save path',
                    dest='detection_default_save_path', default="./runs/detect/")
parser.add_argument('--detection_new_save_path', help='detection model output new save path', dest='detection_new_save_path',
                    default="./runs/20231109/v8x_FNB_UNet_DSConv_MLP_add_1_resnet34/")


parser.add_argument('--segmentation_save_path', help='segmentation model output save path', dest='segmentation_save_path',
                    default="./runs/20231109/v8x_FNB_UNet_DSConv_MLP_add_1_resnet34/DSC/")
parser.add_argument('--segmentation_mask_path', help='segmentation model output mask path', dest='segmentation_mask_path',
                    default="./runs/20231109/v8x_FNB_UNet_DSConv_MLP_add_1_resnet34/DSC/seg_result/")
parser.add_argument('--seg2img_path', help='segmentation to image output save path', dest='seg2img_path',
                    default="./runs/20231109/v8x_FNB_UNet_DSConv_MLP_add_1_resnet34/DSC/seg2img_path/")

# image path
parser.add_argument('--detection_image_path', help='image path of detection model', dest='detection_image_path',
                    default="./detection_method/models/yolo/datasets/FNB/images/val/")
parser.add_argument('--segmentation_image_path', help='image path of segmentation model', dest='segmentation_image_path',
                    default="./runs/20231109/v8x_FNB_UNet_DSConv_MLP_add_1_resnet34/predict/crops/FNB/")
                    # default="./detection_method/models/yolo/runs/detect/predict/crops/FNB/")
parser.add_argument('--segmentation_image_png_path', help='image path of segmentation model', dest='segmentation_image_png_path',
                    default="./runs/20231109/v8x_FNB_UNet_DSConv_MLP_add_1_resnet34/predict/crops/FNB/png/")
parser.add_argument('--classification_image_path', help='image path of classification model', dest='classification_image_path',
                    default='')

# path for saving classification metrics.
parser.add_argument('--class_metrics_txt', help='path for saving classification metrics', dest='class_metrics_txt',
                    default="./runs/20231109/v8x_FNB_UNet_DSConv_MLP_add_1_resnet34/v8x_FNB_UNet_DSConv_MLP_add_1_resnet34.txt")

args = parser.parse_args()
