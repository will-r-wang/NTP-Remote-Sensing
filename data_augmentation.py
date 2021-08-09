import cv2
import json

def rotate_counter_clockwise(width, json_data):
    for project in json_data:
        for region in json_data[project]["regions"]:
            temp = region["shape_attributes"]["all_points_x"]
            region["shape_attributes"]["all_points_x"] = region["shape_attributes"]["all_points_y"]
            region["shape_attributes"]["all_points_y"] = [width - x_point for x_point in temp]

    return json_data

if __name__ == '__main__':
    with open("via_region_data.json") as file:
        data = json.load(file)

    image = cv2.imread("images/treefall.png")
    height, width, _ = image.shape

    with open("via_region_data_rotated.json", "w") as outfile:
        json.dump(rotate_counter_clockwise(width, data), outfile, indent=4)
