# -*- coding: utf-8 -*-
# @Time : 2023/8/6 下午7:17
# @Author : ChenXi

import os
from utils.compute_center import *
from utils.post_processing import *
from utils.create_grip import *
from utils.seg2image import map_segmentation_result
from utils.compute_center import process_image_islands_v2
from utils.post_processing import calculate_first_pixels_average_grayscale


class FetalNasalBoneDetector:
    def __init__(self, image_folder, seg_folder, seg_for_image_output_folder, lv_seg_result_folder, center_points_result_folder):
        self.image_folder = image_folder
        self.seg_folder = seg_folder
        self.seg_for_image_output_folder = seg_for_image_output_folder
        self.lv_seg_result_folder = lv_seg_result_folder
        self.center_points_result_folder = center_points_result_folder

    def determine_bn(self):
        if not os.path.exists(seg_for_image_output_folder):
            os.makedirs(seg_for_image_output_folder)
        nb_count = 0
        no_nb_count = 0
        for filename in os.listdir(self.image_folder):
            if filename.endswith(".png"):
                origin_image_path = os.path.join(self.image_folder, filename)
                seg_result_path = os.path.join(self.seg_folder, filename)
                seg_for_image_output_path = os.path.join(seg_for_image_output_folder, filename)
                # center_points_result_path = os.path.join(center_points_result_folder, filename)
                # lv_seg_result_path = os.path.join(lv_seg_result_folder, filename)


                map_segmentation_result(original_image_path=origin_image_path,
                                        segmentation_result_path=seg_result_path, output_path=seg_for_image_output_path)

                island_areas, island_centers, island_grayscale_means, island_count, input_path_name = \
                    process_image_islands_v2(input_path=seg_for_image_output_path,
                                             lv_output_path = lv_seg_result_folder,
                                             center_point_result_path = self.center_points_result_folder)


                first_pixels_average_light = calculate_first_pixels_average_grayscale(seg_result_path, value=500)
                # print(int(island_grayscale_means[0]))
                if island_count == 1:
                    if first_pixels_average_light < 150 or island_areas[0] < 1500:
                        print(f"{filename} is no nb!")
                        no_nb_count += 1
                    else:
                        print(f"{filename} have nb!")
                        nb_count += 1
                elif island_count == 2:
                    x1, y1 = island_centers[0]
                    x2, y2 = island_centers[1]
                    island_means = (int(island_grayscale_means[0])+int(island_grayscale_means[1])) / 2
                    if first_pixels_average_light < 150 or island_means < 150:
                        print(f"{filename} is no nb!")
                        no_nb_count += 1
                    else:
                        print(f"{filename} have nb!")
                        nb_count += 1

                elif island_count == 3:
                    x3, y3 = island_centers[2]
                    island_means = (int(island_grayscale_means[0])+int(island_grayscale_means[1])) / 2
                    if first_pixels_average_light > 150 or int(island_grayscale_means[2]) > island_means:
                        print(f"{filename} have nb!")
                        nb_count += 1
                    else:
                        print(f"{filename} have nb!")
                        nb_count += 1
                else:
                    print(f"================= Error ===============! {filename}")

        print(f"no nb count is {no_nb_count}; nb count is {nb_count}")