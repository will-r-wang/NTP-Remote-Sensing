import cv2
import numpy as np
import os

if __name__ == '__main__':
    base_image = np.zeros((2456, 1634, 3))
    for mask in os.listdir("./images/masks"):
        mask_image = cv2.imread("./images/masks/" + mask)
        base_image = np.logical_or(base_image, mask_image)
    cv2.imwrite("mask_overlay.png", base_image * 1)
