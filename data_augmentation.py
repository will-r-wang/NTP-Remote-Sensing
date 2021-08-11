"""
Purpose
-------
Takes in the via_region_data VGG16 annotated JSON file and augments each data point through applying a counter clockwise rotation.

Inputs
-------
- Input via_region_data VGG16 annotated JSON file
- Original image annotated with VGG16
- Output JSON file to store augmented data points
"""


import json
import cv2


# Takes in as input the image width and json dumped data and outputs the augmented json object
def rotate_counter_clockwise(width, json_data):
    for project in json_data:
        for region in json_data[project]["regions"]:
            x_points = region["shape_attributes"]["all_points_x"]
            region["shape_attributes"]["all_points_x"] = region["shape_attributes"]["all_points_y"]
            region["shape_attributes"]["all_points_y"] = [width - x_point for x_point in x_points]

    return json_data


VIA_REGION_DATA_INPUT = "via_region_data.json"
INPUT_IMAGE_PATH = "images/treefall.png"
AUGMENTED_VIA_REGION_DATA_OUTPUT = "via_region_data_rotated.json"

if __name__ == '__main__':
    with open(VIA_REGION_DATA_INPUT) as file:
        data = json.load(file)

    image = cv2.imread(INPUT_IMAGE_PATH)
    height, width, _ = image.shape

    with open(AUGMENTED_VIA_REGION_DATA_OUTPUT, "w") as outfile:
        json.dump(rotate_counter_clockwise(width, data), outfile, indent=4)
