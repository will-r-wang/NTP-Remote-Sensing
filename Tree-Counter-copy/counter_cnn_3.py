from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import os
import cv2
import glob 
import tensorflow as tf
from random import shuffle
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

model_path = './model.h5' 
test_dir = "./test/"
resize_shape = (24,24)

# load model
model = load_model(model_path)

# load test items. 
test_items = glob.glob(test_dir + "//*//*.png")

# shuffle(test_items)
#test_items = test_items[6318:9585]

# create empty truth table
truth_table = pd.DataFrame({'filename':[],'pred_proba':[]})
i = 0 

# for each image in test_items, predict classification probability and add to truth_table
for _test_item in test_items:

  img_data = cv2.imread(_test_item)
#  switch blue and red channel because cv2.imread reads images in as BGR (instead of RGB)
  img_RGB = np.zeros((24,24,3))
  img_RGB[:,:,0] = img_data[:,:,2]
  img_RGB[:,:,2] = img_data[:,:,0]
  img_RGB[:,:,1] = img_data[:,:,1]
  img_data = img_RGB
#  end testing
  img_data = img_data/255.
  img_data = cv2.resize(img_data, resize_shape)
  img_data = np.expand_dims(img_data, axis = 0) 

  model_op = model.predict(img_data)
#  model_op = model.predict_classes(img_data)
  model_op = model_op[0][0]
  print(_test_item)
  print(model_op)
  truth_table.loc[i,'filename'] = _test_item
  truth_table.loc[i,'pred_proba'] = model_op
  truth_table.loc[i,'pred'] = int(round(model_op,0))
  i+=1
  
  
  
  
##  delete below
#  img_data = np.squeeze(img_data)
#  plt.imshow(img_data)
#  plt.show()
# 
  
  
  
  
  
  
  
  
