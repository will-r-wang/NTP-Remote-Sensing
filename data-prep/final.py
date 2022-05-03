"""
Purpose
-------
Iterates over the resultant binary masked images in a specified subfolder and generates the final output masked image.

Inputs
-------
- Image dimensions
- Root folder containing masks
- Output image file name
"""


import os
import cv2
import numpy as np


IMAGE_HEIGHT = 2456
IMAGE_WIDTH = 1634
IMAGE_CHANNELS = 3
ROOT_FOLDER = "./images/masks"
OUTPUT_IMAGE = "mask_overlay.png"

if __name__ == '__main__':
    base_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    for mask in os.listdir(ROOT_FOLDER):
        mask_image = cv2.imread(ROOT_FOLDER + "/" + mask)
        base_image = np.logical_or(base_image, mask_image)
    cv2.imwrite(OUTPUT_IMAGE, base_image * 1)
