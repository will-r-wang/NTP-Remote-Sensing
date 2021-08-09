import cv2
import glob
import numpy as np

from sklearn.neural_network import MLPClassifier

def process(location):
    image = cv2.imread(location)
    temp = []
    for x in range(40):
        for y in range(40):
            vals = image[x,y]
            temp += (vals[0]/256.0,vals[1]/256.0,vals[2]/256.0)
    return [temp]

total = []
for i in glob.glob("./images/positives/*.png"):
    total += process(i)

for i in glob.glob("./images/negatives/*.png"):
    total += process(i)

clf = MLPClassifier(solver = 'lbfgs' , alpha = 1e-5, hidden_layer_sizes = (250,100,40), random_state = 1)
clf.fit(total, [0] * 89 + [1] * 35)

train = glob.glob("./images/train/*.png")

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
    cv2.imwrite("./images/outputs/" + image.split("/")[-1][:-4] + ".png", output)
