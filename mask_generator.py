import argparse
import cv2
import json
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generates masks from an input vgg json annotation file and corresponding image')
    parser.add_argument('-i', '--input-json-file', help='Input json file name', required=True)
    args = parser.parse_args()

    polygons = {}
    with open(args.input_json_file) as file:
    # with open("./via_region_data.json") as file:
        data = json.load(file)

    for project in data:
        filename = data[project]["filename"]

        for idx, region in enumerate(data[project]["regions"]):
            key = filename[:-4] + "*" + str(idx + 1)

            x_points = data[project]["regions"][idx]["shape_attributes"]["all_points_x"]
            y_points = data[project]["regions"][idx]["shape_attributes"]["all_points_y"]

            polygon_coords = list(zip(x_points, y_points))

            mask = np.zeros((2456, 1634)) # dimensions of treefall.png
            cv2.fillPoly(mask, [np.array(polygon_coords)], color=(255, 255, 255))
            cv2.imwrite("images/masks/" + filename + "_" + str(idx) + ".png" , mask)
