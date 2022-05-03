"""
Purpose
-------
Takes in an input VGG json annotated file and generates the corresponding masks and final masked output image

Inputs
-------
- Input json file name to which objects are read from
- Output folder path to which masks are written to
- Image dimensions
"""


import json
import argparse
import cv2
import numpy as np


OUTPUT_FOLDER_PATH = "images/masks/"
IMAGE_HEIGHT = 2456
IMAGE_WIDTH = 1634

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generates masks from an input vgg json annotation file and corresponding image')
    parser.add_argument('-i', '--input-json-file', help='Input json file name', required=True)
    args = parser.parse_args()

    polygons = {}
    with open(args.input_json_file) as file:
        data = json.load(file)

    for project in data:
        filename = data[project]["filename"]

        for idx, region in enumerate(data[project]["regions"]):
            key = filename[:-4] + "*" + str(idx + 1)

            x_points = data[project]["regions"][idx]["shape_attributes"]["all_points_x"]
            y_points = data[project]["regions"][idx]["shape_attributes"]["all_points_y"]

            polygon_coords = list(zip(x_points, y_points))

            mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)) # dimensions of treefall.png
            cv2.fillPoly(mask, [np.array(polygon_coords)], color=(255, 255, 255))
            cv2.imwrite(OUTPUT_FOLDER_PATH + filename + "_" + str(idx) + ".png" , mask)
