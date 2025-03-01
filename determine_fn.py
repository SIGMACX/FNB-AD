# @Time: 2023/8/1 19:20
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
#  判断鼻骨存在
#  |----未发育（无鼻骨第三块）
#  |----发育不完整（第三块灰度均值小于前两个）
#  |----完整发育（第三块灰度均值大于前两个）
# -------------------------------
import os
from utils.compute_center import *
from utils.post_processing import *
from utils.create_grip import *
from utils.seg2image import map_segmentation_result
from utils.compute_center import process_image_islands_v2
from utils.post_processing import calculate_first_pixels_average_grayscale, within_5_percent
from tools.config import args
from utils.mapping_seg2image import overlay_segmentation_on_image


class FetalNasalBoneDetector:
    def __init__(self, image_folder, seg_folder, seg_for_image_output_folder, filter_seg_result_folder,
                 center_points_result_folder, mapping_seg2image_folder, first=False):
        self.image_folder = image_folder
        self.seg_folder = seg_folder
        self.seg_for_image_output_folder = seg_for_image_output_folder
        self.filter_seg_result_folder = filter_seg_result_folder
        self.center_points_result_folder = center_points_result_folder
        self.first = first
        self.mapping_seg2image_folder = mapping_seg2image_folder

    def determine_bn(self):
        if not os.path.exists(self.seg_for_image_output_folder):
            os.makedirs(self.seg_for_image_output_folder)

        if not os.path.exists(self.mapping_seg2image_folder):
            os.makedirs(self.mapping_seg2image_folder)
        nb_count = 0
        no_nb_count = 0
        error_nb_count = 0

        count_5 = 0
        island_means_sum = 0
        island_means_1_sum = 0
        island_means_2_sum = 0
        island_means_3_sum = 0
        for filename in os.listdir(self.image_folder):
            if filename.endswith(".png"):
                origin_image_path = os.path.join(self.image_folder, filename)
                seg_result_path = os.path.join(self.seg_folder, filename)
                seg_for_image_output_path = os.path.join(self.seg_for_image_output_folder, filename)
                mapping_seg2image_path = os.path.join(self.mapping_seg2image_folder, filename)
                # center_points_result_path = os.path.join(center_points_result_folder, filename)
                # lv_seg_result_path = os.path.join(lv_seg_result_folder, filename)


                map_segmentation_result(original_image_path=origin_image_path,
                                        segmentation_result_path=seg_result_path, output_path=seg_for_image_output_path)

                island_areas, island_centers, island_grayscale_means, island_count, input_path_name = \
                    process_image_islands_v2(input_path=seg_for_image_output_path,
                                             lv_output_path = self.filter_seg_result_folder,
                                             center_point_result_path = self.center_points_result_folder)
                # print(f"{filename}, {island_grayscale_means}")

                overlay_segmentation_on_image(original_image_path = origin_image_path,
                                              segmentation_image_path = seg_result_path,
                                              output_folder = mapping_seg2image_path)


                if self.first == True:
                    first_pixels_average_light = calculate_first_pixels_average_grayscale(seg_result_path, value=500)
                    # print(int(island_grayscale_means[0]))
                    first_name = filename.split("_")
                    if island_count == 1:
                        if first_pixels_average_light < 150 or island_areas[0] < 1500:
                            if first_name[0] == "0":
                                print(f"{filename} is no nb!")
                                no_nb_count += 1
                            else:
                                # print(f"{filename} have nb!")
                                # nb_count += 1
                                error_nb_count += 1
                                print(f"================= Error ===============! {filename}")
                        else:
                            if first_name[0] == "1":
                                print(f"{filename} have nb!")
                                nb_count += 1
                            else:
                                # print(f"{filename} is no nb!")
                                no_nb_count += 1
                    elif island_count == 2:
                        x1, y1 = island_centers[0]
                        x2, y2 = island_centers[1]
                        island_mean_1 = int(island_grayscale_means[0])
                        island_mean_2 = int(island_grayscale_means[1])
                        island_means = (int(island_grayscale_means[0])+int(island_grayscale_means[1])) / 2
                        if first_pixels_average_light < 150 or island_means < 150 \
                                or within_5_percent(island_mean_1, island_mean_2):
                            if first_name[0] == "0":
                                print(f"{filename} is no nb!")
                                no_nb_count += 1
                            else:
                                print(f"{filename} have nb!")
                                nb_count += 1
                                # error_nb_count += 1
                                # print(f"================= Error ===============! {filename}")
                        else:
                            if first_name[0] == "1":
                                print(f"{filename} have nb!")
                                nb_count += 1
                            else:
                                # print(f"{filename} have no nb!")
                                no_nb_count += 1
                                # error_nb_count += 1
                                # print(f"================= Error ===============! {filename}")

                    elif island_count == 3:
                        x3, y3 = island_centers[2]
                        island_means = (int(island_grayscale_means[0])+int(island_grayscale_means[1])) / 2
                        if first_pixels_average_light > 150 or int(island_grayscale_means[2]) > island_means \
                            or within_5_percent(island_means, island_mean_3):
                            if first_name[0] == "1":
                                print(f"{filename} have nb!")
                                nb_count += 1
                            else:
                                # print(f"{filename} have no nb!")
                                # no_nb_count += 1
                                error_nb_count += 1
                                print(f"================= Error ===============! {filename}")
                        else:
                            if first_name[0] == "0":
                                print(f"{filename} have no nb!")
                                no_nb_count += 1
                            else:
                                print(f"{filename} have nb!")
                                nb_count += 1
                    else:
                        nb_count += 1


            if self.first == False:
                first_pixels_average_light = calculate_first_pixels_average_grayscale(seg_result_path, value=1500)
                # print(int(island_grayscale_means[0]))

                first_name = filename.split("_")[0]

                if island_count == 1:
                    island_mean_1 = int(island_grayscale_means[0])
                    if first_pixels_average_light < 150 or island_areas[0] < 1500:
                        # print(f"{filename} is no nb!")
                        if first_name == "0":
                            no_nb_count += 1
                        elif first_name == "1":
                            error_nb_count += 1
                    else:
                        # print(f"{filename} have nb!")
                        if first_name == "1":
                            nb_count += 1
                        elif first_name == "0":
                            error_nb_count += 1
                elif island_count == 2:
                    island_mean_1 = int(island_grayscale_means[0])
                    island_mean_2 = int(island_grayscale_means[1])

                    if int(island_areas[0]) > 800 or int(island_areas[1]) > 800:
                        if within_5_percent(island_mean_1, island_mean_2) or within_5_percent(first_pixels_average_light, island_mean_1) \
                            or within_5_percent(first_pixels_average_light, island_mean_2):
                            if first_name == '1':
                                nb_count += 1
                            elif first_name == "0":
                                error_nb_count += 1
                        else:
                            if first_name == "0":
                                no_nb_count += 1
                            elif first_name == "1":
                                error_nb_count += 1
                    else:
                        if first_name == "0":
                            no_nb_count += 1

                        elif first_name == "1":
                            error_nb_count += 1

                elif island_count == 3:
                    island_mean_1 = int(island_grayscale_means[0])
                    island_mean_2 = int(island_grayscale_means[1])
                    island_mean_3 = int(island_grayscale_means[2])

                    # x3, y3 = island_centers[2]
                    island_means = (int(island_grayscale_means[0]) + int(island_grayscale_means[1])) / 2
                    # nb_count += 1

                    if within_5_percent(island_mean_1, island_mean_3) or within_5_percent(island_mean_2, island_mean_3)  or\
                            island_mean_3 > island_mean_1 or island_mean_3 > island_mean_2:
                        if first_name == "1":
                            nb_count += 1
                            '''count_5 += 1
                            island_means_1_sum += island_mean_1
                            island_means_2_sum += island_mean_2
                            island_means_3_sum += island_mean_3'''
                        else:
                            error_nb_count += 1
                    else:
                        if first_name == "0":
                            no_nb_count += 1
                            count_5 += 1
                            island_means_1_sum += island_mean_1
                            island_means_2_sum += island_mean_2
                            island_means_3_sum += island_mean_3
                        else:
                            error_nb_count += 1
                # elif island_count == 4:
                #     island_mean_1 = int(island_grayscale_means[0])
                #     island_mean_2 = int(island_grayscale_means[1])
                #     island_mean_3 = int(island_grayscale_means[2])
                #     island_mean_4 = int(island_grayscale_means[3])
                #     island_4_mean = (island_mean_1+island_mean_2+island_mean_3+island_mean_4)/4
                #     if within_5_percent(island_4_mean, first_pixels_average_light):
                #         if first_name == "1":
                #             nb_count += 1
                #         else:
                #             error_nb_count += 1
                else:
                    error_nb_count += 1

            else:
                pass

        print(f"count is {no_nb_count + nb_count + error_nb_count}, no nb count is {no_nb_count}; "
              f"nb count is {nb_count}; error nb is {error_nb_count}")
        # print(f" {count_5}, {island_means_1_sum / count_5}, {island_means_2_sum / count_5}, {island_means_3_sum / count_5}")
        # print result of normal
        # print(f"count: {count_5}, island_means_1_sum: {island_means_1_sum/count_5}, island_means_2_sum: {island_means_2_sum/count_5} \
        #        island_means_3_sum: {island_means_3_sum / count_5}, island_mean_sum: {island_means_sum / count_5}")


if __name__ == '__main__':
    detector = FetalNasalBoneDetector(
        image_folder = args.test_image_path,
        seg_folder = args.seg_result_path,
        seg_for_image_output_folder = args.seg_for_image_result,
        filter_seg_result_folder = args.seg_filter_result,
        center_points_result_folder = args.center_points_result,
        mapping_seg2image_folder = args.mapping_seg2image_folder,
        first=False
        )
    detector.determine_bn()


















