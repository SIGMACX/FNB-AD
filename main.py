# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import cv2
import torch
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tools.config import args

from test_dou_class_u_net import seg_inference
from metrics import evaluate_images_in_folder
from tools.crop_image import jpg2png
from class_method.seg_result_class_infer import class_inference, move_file_abnormal_normal, seg_result_class_infer
from utils.seg2image import map_segmentation_result
from tools.crop_image import resize_images



# 加载模型
# detection_model = args.detection_model_pt
# segmentation_model = ''
# classification_model = ''

# 数据路径
# detection_path = "./data/FNBS_Dataset/detection_all/image/train/1_00103.jpg"
# jpg2png(detection_path, detection_path)

# D_model = YOLO(args.detect_yolov8n)
# D_model.load(detection_model)
# D_model = YOLO(detection_model)
# D_model = D_model.load(detection_model)
# D_model = D_model.load(detection_model)
# results = D_model.predict(source=detection_path, save=True, save_txt=True, save_crop=True)




def main(detection_model_path, detection_image_path, segmentation_image_path, segmentation_image_png_path, num_classes,
         segmentation_model_pt, segmentation_save_path, segmentation_mask_path, seg2img_path, classification_model_pt):
    start_time = time.time()
    # D_model = YOLO(args.detect_yolov8n)
    # D_model = D_model.load(detection_model_path)
    # D_model.predict(source=detection_image_path, save=True, save_crop=True, device='0')
    jpg2png(segmentation_image_path, segmentation_image_png_path)
    seg_inference(test_image_path=segmentation_image_png_path,
                  num_classes=num_classes,
                  test_label_path=None,
                  input_channels=3,
                  snapshot_dir=segmentation_model_pt,
                  save_path = segmentation_save_path,
                  is_ference=True)

    print("1 - Finished Segmentation inference!")
    segmentation_mask_path = segmentation_mask_path
    segmentation_image_path = segmentation_image_png_path
    seg2img_path = seg2img_path
    if not os.path.exists(seg2img_path):
        os.makedirs(seg2img_path)
    resize_images(segmentation_image_png_path, segmentation_image_png_path, (128,128))
    for filename in os.listdir(segmentation_mask_path):
        if filename.endswith(".png"):
            origin_image_path = os.path.join(segmentation_image_png_path, filename)
            seg_result_path = os.path.join(segmentation_mask_path, filename)
            seg_for_image_output_path = os.path.join(seg2img_path, filename)
            map_segmentation_result(original_image_path=origin_image_path,
                                    segmentation_result_path=seg_result_path, output_path=seg_for_image_output_path)
    print("2 - Finished mapping of segmentation to image!")
    move_file_abnormal_normal(args.seg2img_path)
    print("3 - Classification inference Result: ")
    # class_inference(image_path=seg2img_path, model_path=classification_model_pt)
    seg_result_class_infer(dataset_root=seg2img_path,
                           num_classes=args.class_num_classes,
                           batch_size=args.class_batch_size,
                           snapshot_dir=classification_model_pt)
    print('Finished Classification inference!')
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Inference completed in {total_time:.6f} seconds")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    main(detection_model_path=args.detection_model_pt,
         detection_image_path=args.detection_image_path,
         segmentation_image_path=args.segmentation_image_path,
         segmentation_image_png_path = args.segmentation_image_png_path,
         num_classes=args.num_classes,
         segmentation_model_pt=args.segmentation_model_pt,
         segmentation_save_path=args.segmentation_save_path,
         segmentation_mask_path = args.segmentation_mask_path,
         seg2img_path = args.seg2img_path,
         classification_model_pt=args.classification_model_pt)

