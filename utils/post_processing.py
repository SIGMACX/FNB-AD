# @Time: 2023/8/1 11:04
# @Author: ChenXi
# -*- coding: utf-8 -*-

# -------------------------------
# -------------------------------


import cv2
import numpy as np
from PIL import Image

def find_nonzero_coordinates(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    nonzero_coords = np.transpose(np.nonzero(image))
    return nonzero_coords

def find_all_coordinates(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    nonzero_coords = np.column_stack(np.where(image > 155))
    return nonzero_coords

def find_highest_gray_values(image_path, num_points):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    sorted_indices = np.argsort(image.ravel())[::-1]

    highest_gray_coords = np.unravel_index(sorted_indices[:num_points], image.shape)

    return highest_gray_coords

def calculate_average_grayscale(image_path, x, y, radius):
    image = Image.open(image_path)
    pixels = image.load()
    row, col = image.size

    sum_grayscale = 0
    count_pixels = 0

    for i in range(max(0, int(y) - radius), min(row, int(y) + radius + 1)):
        for j in range(max(0, int(x) - radius), min(col, int(x) + radius + 1)):
            r, g, b = pixels[j, i]
            grayscale_value = (r + g + b) // 3

            if grayscale_value > 100:
                sum_grayscale += grayscale_value
                count_pixels += 1

    average_grayscale = sum_grayscale / count_pixels if count_pixels > 0 else 0

    return average_grayscale


def calculate_top_30_percent_average_grayscale(image_path, value):
    # Load the image
    image = Image.open(image_path)
    pixels = np.array(image)

    # Flatten the pixel values and sort them in descending order
    sorted_pixels = np.sort(pixels.flatten())[::-1]

    # Calculate the index for the top 50% pixels
    top_50_percent_index = int(value * len(sorted_pixels))

    # Select the top 50% pixels
    top_50_percent_pixels = sorted_pixels[:top_50_percent_index]

    # Calculate the average grayscale value of the top 50% pixels
    average_grayscale = np.mean(top_50_percent_pixels)

    return average_grayscale


def calculate_first_pixels_average_grayscale(image_path, value):
    # Load the image
    image = Image.open(image_path).convert("L")
    pixels = np.array(image)

    sorted_pixels = np.sort(pixels.flatten())[::-1]
    first_pixels = sorted_pixels[:value]

    # Calculate the average grayscale value of the first 1000 pixels
    average_grayscale = np.mean(first_pixels)

    return average_grayscale


def within_5_percent(value1, value2):
    threshold = 0.05 # 5%的阈值
    difference = abs(value1 - value2)
    avg_value = (value1 + value2) / 2
    # print(difference / avg_value)
    if difference / avg_value <= threshold:
        return True
    else:
        return False


within_5_percent(127, 140)