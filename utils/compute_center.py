# @Time: 2023/8/1 15:09
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# 计算中心点坐标
# -------------------------------
import os
import sys

from PIL import Image, ImageDraw
import numpy as np
from utils.post_processing import calculate_first_pixels_average_grayscale

def numIslands(image_path, min_area=50, output_folder=None):
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image file not found at: {}".format(image_path))

    image = Image.open(image_path).convert('L')
    pixels = image.load()
    row, col = image.size

    grid = [["0" for _ in range(col)] for _ in range(row)]
    image_light_average = calculate_first_pixels_average_grayscale(image_path, value=1000)
    print(image_light_average)
    for y in range(row):
        for x in range(col):
            gray_value = pixels[x, y]
            if image_light_average < 150:
                if gray_value > 100:
                    grid[y][x] = "1"
            elif image_light_average > 150:
                if gray_value > 140:
                    grid[y][x] = "1"

    row, col, ret = len(grid), len(grid[0]), 0
    centers = {}
    image_area = {}
    image_grayscale_means = {}  # Dictionary to store grayscale means for each region

    def dfs(x, y):
        grid[x][y] = '0'
        for c in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            nx, ny = x + c[0], y + c[1]
            if 0 <= nx < row and 0 <= ny < col and grid[nx][ny] == '1':
                dfs(nx, ny)

    for i in range(row):
        for j in range(col):
            if grid[i][j] == '1':
                # Calculate the area of the current region
                area = 0
                x_sum = 0
                y_sum = 0
                gray_sum = 0  # Sum of grayscale values for the current region
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if 0 <= x < row and 0 <= y < col and grid[x][y] == '1':
                        grid[x][y] = '0'  # Mark as visited
                        area += 1
                        x_sum += x
                        y_sum += y
                        if pixels[x, y] > 100:
                            gray_sum += pixels[x, y]  # Note the order of (y, x) for pixel access
                        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

                if area > min_area:
                    ret += 1
                    center_x = x_sum / area
                    center_y = y_sum / area
                    centers[ret] = (center_x, center_y)
                    image_area[ret] = area

                    # Calculate grayscale mean using binary representation
                    grayscale_mean = gray_sum / area
                    image_grayscale_means[ret] = grayscale_mean

        # 将 grid 中的像素值转换为对应的颜色，然后保存为图像
        marked_image = Image.new('RGB', (col, row))
        marked_pixels = marked_image.load()
        for y in range(row):
            for x in range(col):
                if grid[y][x] == "1":
                    marked_pixels[x, y] = (255, 0, 0)  # 将岛屿部分赋值为红色
                else:
                    marked_pixels[x, y] = (0, 0, 0)  # 其他部分赋值为黑色

        # 保存带有红色标记的原图到指定的输出文件夹中
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_image_path = os.path.join(output_folder, "red_center.png")
            marked_image.save(output_image_path)

    return ret, centers, image_area, image_grayscale_means

def image_to_grid(image_path):
    # Load the image
    image = Image.open(image_path)
    pixels = image.load()
    row, col = image.size

    grid = [["0" for _ in range(col)] for _ in range(row)]
    for y in range(row):
        for x in range(col):
            r, g, b = pixels[x, y]
            if r > 50 and g > 50 and b > 50:
                grid[y][x] = "1"

    return grid


def find_and_modify_points(image_path, points, radius, new_grayscale_value):
    # Load the image
    image = Image.open(image_path)
    pixels = image.load()
    # Create a drawing context to draw on the image
    draw = ImageDraw.Draw(image)
    # Find and modify the points
    for point in points:
        x, y = point
        # Draw a circle centered at (x, y) with the specified radius and color
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=new_grayscale_value)
        # Modify the grayscale value of the pixel at (x, y)
        pixels[x, y] = (new_grayscale_value, new_grayscale_value, new_grayscale_value)
    return image



#*****************************************************************************************************
#                           计算区域中心点坐标，区域面积，区域数量、区域灰度均值
#*****************************************************************************************************

def process_image_islands_v2(input_path, lv_output_path, center_point_result_path, min_area=100):
    image = Image.open(input_path).convert('L')
    width, height = image.size
    image_light_average = calculate_first_pixels_average_grayscale(input_path, value=1000)

    # 遍历图片中的每个像素
    grid = [["0" for _ in range(width)] for _ in range(height)]
    # 遍历图片中的每个像素
    for x in range(width):
        for y in range(height):
            pixel_value = image.getpixel((x, y))
            if image_light_average < 150:
                if pixel_value < 120:
                    # 将灰度值小于150的像素标记为0
                    image.putpixel((x, y), 0)
                else:
                    grid[y][x] = "1"
            elif image_light_average > 150:
                if pixel_value < 140:
                    image.putpixel((x, y), 0)
                else:
                    grid[y][x] = "1"

    # 将处理后的图片保存到新的文件
    if lv_output_path:
        first_filename = os.path.basename(input_path).split("_")[0]
        if first_filename == '0':
             lv_output_path = os.path.join(lv_output_path, 'abnormal')
        elif first_filename == '1':
            lv_output_path = os.path.join(lv_output_path, 'normal')
        os.makedirs(lv_output_path, exist_ok=True)
        output_file_path = os.path.join(lv_output_path, os.path.basename(input_path))
        image.save(output_file_path)

    # 统计处理后的岛屿数量、各个岛屿的面积、中心点坐标和灰度均值
    def dfs(x, y):
        stack = [(x, y)]
        grid[x][y] = '0'
        area = 1
        x_sum = x
        y_sum = y
        gray_sum = image.getpixel((y, x))  # Initialize grayscale sum

        while stack:
            x, y = stack.pop()

            for c in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                nx, ny = x + c[0], y + c[1]
                if 0 <= nx < height and 0 <= ny < width and grid[nx][ny] == '1':
                    stack.append((nx, ny))
                    grid[nx][ny] = '0'
                    area += 1
                    x_sum += nx
                    y_sum += ny
                    gray_sum += image.getpixel((ny, nx))  # Update grayscale sum

        return area, x_sum, y_sum, gray_sum

    island_areas = []
    island_centers = []
    island_grayscale_means = []
    island_count = 0
    for i in range(height):
        for j in range(width):
            if grid[i][j] == '1':
                area, x_sum, y_sum, gray_sum = dfs(i, j)
                if area > 100:
                    island_areas.append(area)
                    island_centers.append((x_sum / area, y_sum / area))
                    island_grayscale_means.append(gray_sum / area)
                    island_count += 1

    # 直接将中心点位置及周围8个点填充为绿色，并在中心点上标出x轴和y轴
    draw = ImageDraw.Draw(image)
    for center in island_centers:
        center_x, center_y = int(center[1]), int(center[0])
        for i in range(center_x - 1, center_x + 2):
            for j in range(center_y - 1, center_y + 2):
                draw.point((i, j), fill='green')

    # 保存标记了中心点坐标的图片
    if center_point_result_path:
        os.makedirs(center_point_result_path, exist_ok=True)
        output_file_path = os.path.join(center_point_result_path, os.path.basename(input_path))
        image.save(output_file_path)

    # 按照中心点的 y 坐标进行排序，并保持 island_grayscale_means 的顺序与排序后的中心点一致
    if island_centers and island_grayscale_means:
        sorted_centers, sorted_grayscale_means = zip(
            *sorted(zip(island_centers, island_grayscale_means), key=lambda item: item[0][1]))
    else:
        sorted_centers, sorted_grayscale_means = [], []
    sorted_areas = [area for _, area in sorted(zip(island_centers, island_areas), key=lambda item: item[0][1])]

    # return island_areas, island_centers, island_grayscale_means, island_count
    return sorted_areas, sorted_centers, sorted_grayscale_means, island_count, input_path


# input_path = 'D:\code_pycharm\Fetal_nasal_bone_detection/data/FN_local/testingset/test/img/'
# # image_grid = image_to_grid(image_path)
# lv_output_path = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\lv_result/'
# center_point_path = 'D:\code_pycharm\Fetal_nasal_bone_detection\data\FN_local/testingset/test\center_point_result/'
#
# for filename in os.listdir(input_path):
#     input_path_1 = os.path.join(input_path, filename)
#     print(input_path_1)
#     island_areas, island_centers, island_grayscale_means, island_count = \
#         process_image_islands_v2(input_path_1, lv_output_path, center_point_path)
#     print("各个岛屿的面积：", island_areas)
#     print("各个岛屿的中心点坐标：", island_centers)
#     print("各个岛屿的灰度均值：", island_grayscale_means)
#     print("岛屿数量：", island_count)
