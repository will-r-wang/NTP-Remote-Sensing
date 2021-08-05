import argparse
import cv2
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='naive tree count for an input image file')
    parser.add_argument('-o', '--output-image-path', help='Output annotated image file name', default="images/sample_trees_annotated.tif")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input-image-path', help='Input image file name', required=True)
    args = parser.parse_args()

    image = cv2.imread(args.input_image_path)
    height, width, _ = image.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # RGB mask filtering
    GREEN_MIN = np.array([74, 83, 52], np.uint8)
    GREEN_MAX = np.array([150, 209, 193], np.uint8)

    dst = cv2.inRange(hsv, GREEN_MIN, GREEN_MAX)

    EST_PIXELS_PER_TREE = 1000
    num_green_pixels = cv2.countNonZero(dst)
    print('Total # of green pixels:', str(num_green_pixels))
    print('Total estimated # of trees:', str(num_green_pixels//EST_PIXELS_PER_TREE))

    mask = dst == 0
    green = np.zeros_like(image, np.uint8)
    green[mask] = image[mask]

    # Image grid-line annotation
    for i in range(1, height//400 + 1):
        cv2.line(green, (0, i * 400), (width, i * 400), (0, 0, 255), 1)

    for j in range(1, width//400 + 1):
        cv2.line(green, (j * 400, 0), (j * 400, height), (0, 0, 255), 1)

    cv2.imwrite(args.output_image_path, green)
