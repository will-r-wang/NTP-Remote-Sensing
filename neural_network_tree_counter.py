"""
Purpose
-------
Simple MLPClassifier backed neural network tree counter that counts the number of trees in a given input image

Inputs
-------
- Input training data folder in which images part of the train dataset
- Input positive samples data folder in which tree images are contained
- Input negative samples data folder in which non-tree images are contained
- Output folder path to which resultant images are written to
"""


import cv2
import glob
import numpy as np

from sklearn.neural_network import MLPClassifier


# takes in an input file path (image) and flattens the RGB values into a single 1-dimensional array
def flatten_image(file_path):
    image = cv2.imread(file_path)
    flattened_array = []
    for x in range(40):
        for y in range(40):
            vals = image[x, y]
            flattened_array += (vals[0]/256.0, vals[1]/256.0, vals[2]/256.0)
    return [flattened_array]


POSITIVE_IMAGES_SOURCE_PATH = "./images/positives/*.png"
NEGATIVE_IMAGES_SOURCE_PATH = "./images/negatives/*.png"
TRAIN_IMAGES_SOURCE_PATH = "./images/train/*.png"
OUTPUT_FOLDER_PATH = "./images/outputs/"

if __name__ == '__main__':
    total = []
    for image in glob.glob(POSITIVE_IMAGES_SOURCE_PATH):
        total += flatten_image(image)

    for image in glob.glob(NEGATIVE_IMAGES_SOURCE_PATH):
        total += flatten_image(image)

    clf = MLPClassifier(solver = 'lbfgs' , alpha = 1e-5, hidden_layer_sizes = (250,100,40), random_state = 1)
    clf.fit(total, [0] * 89 + [1] * 35)

    train = glob.glob(TRAIN_IMAGES_SOURCE_PATH)

    for image in train:
        base = cv2.imread(image)
        output = np.zeros(base.shape)

        for i in range((base.shape[0]-40)):
            for j in range((base.shape[1]-40)):
                if clf.predict([base[i:i+40, j:j+40].flatten()/256.0])[0] == 0 and tuple(output[i,j]) != (255,255,255):
                        output[i,j] =  base[i,j]
                else if tuple(output[i,j]) != (255,255,255):
                        output[i,j] = base[i,j]
            print(str(i) + " of " + str(base.shape[0]-40))
        cv2.imwrite(OUTPUT_FOLDER_PATH + image.split("/")[-1][:-4] + ".png", output)
