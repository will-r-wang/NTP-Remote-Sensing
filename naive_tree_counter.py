"""
Purpose
-------
Takes in an input source folder and applies a naive tree count with a predefined color thresholding function. Outputs the total pixel count and estimated number of trees.

Inputs
-------
- Input source folder in which images are contained
- Output csv path to which image and tree counts are written to
"""


import argparse
import os
import csv
import cv2
import numpy as np


COLOR_THRESHOLD_MIN = [74, 83, 52]
COLOR_THRESHOLD_MAX = [150, 209, 193]
ESTIMATED_PIXELS_PER_TREE = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='naive tree count for an input source folder with images')
    parser.add_argument('-o', '--output-csv-path', help='Output csv file with image and tree counts', default="output_tree_counts.csv")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input-source-folder', help='Input image file name', required=True)
    args = parser.parse_args()

    if os.path.exists(args.output_csv_path):
        os.remove(args.output_csv_path)

    with open(args.output_csv_path, mode='w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',')
        csv_writer.writerow(['image_file_path', 'num_green_pixels', 'num_trees'])

        for filename in os.listdir(args.input_source_folder):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".tif"):
                image = cv2.imread(args.input_source_folder + '/' + filename)
                height, width, _ = image.shape
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # color thresholding with simple MIN/MAX bounding
                GREEN_MIN = np.array(COLOR_THRESHOLD_MIN, np.uint8)
                GREEN_MAX = np.array(COLOR_THRESHOLD_MAX, np.uint8)

                dst = cv2.inRange(hsv, GREEN_MIN, GREEN_MAX)

                num_green_pixels = cv2.countNonZero(dst)
                print('File name:', filename)
                print('Total # of green pixels:', str(num_green_pixels))
                print('Total estimated # of trees:', str(num_green_pixels//ESTIMATED_PIXELS_PER_TREE))
                print('-------------------------')

                csv_writer.writerow([filename, str(num_green_pixels), str(num_green_pixels//ESTIMATED_PIXELS_PER_TREE)])
