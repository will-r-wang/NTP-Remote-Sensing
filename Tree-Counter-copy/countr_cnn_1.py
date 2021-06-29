"""
This code:
1. extracts image thumbnails based on annotated results file 
2. locates images in correct train/test folders and tree/no_tree subfolders

It uses the pickled lookup_table.pkl file created by 'counter_cnn_lookup_table.py'
It uses the annotation dataframe file selected by user in root directory.  Either df_train_val.pkl or df_test.pkl

Instructions:
    1. select df_train_val.pkl or df_test.pkl to assign to 'df' variable
    2. select option 1 or option 2 to assign cropped images to train/val or test folders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from PIL import Image

counter = 0

# load lookup_table to look up pixel centers of each grid point location
lookup_table = pd.read_pickle("./lookup_table.pkl")

# select location of images
RGB_fileroot = 'C:/Users/smcguire/Desktop/Annotation Data Filtered RGB FOVs 1 and 2/' 


# read in df_test or df_train dataframe with only 1 set of annotations per image 
df = pd.read_pickle('./df_train_val.pkl')

length = 12 # half of width (and height) of cropped images i.e. crop_width = 10 means cropped image is 20 x 20 pixels
n = 0
m = 0
p=0
num_skips = 4 # number of cropped thumbnails with no trees to skip.  this produces a more balanced train and test ratio of trees/no_trees

# extract thumbnails to proper locations by looping through rows in df (annotation data) using i and loop through 
# all grid points using j
for i in range(len(df)): 
#for i in range(19,21):
    im = Image.open(RGB_fileroot + df.loc[i, 'tag'] + '.png') # open image
    counter = counter + 1
#    im.show()
    for j in range(1,50):  # loop through grid points and set crop coordinates      
        x = lookup_table.loc[j, 'x']
        y = lookup_table.loc[j, 'y']
        crop_rectangle = (x - length, y - length, x + length, y + length)
        cropped_im = im.crop(crop_rectangle)
               
        # option 1: images to train and val folders: put half of 'tree' images to train and half to validate
        if df.loc[i,'on_tree_' + str(j)]: # if tree location is true   
            if n==0:                               
                cropped_im.save('./train/tree/' + df.loc[i,'tag'] + ' on_tree'+ str(j) + '.png', 'PNG') # change location to a "tree folder" in train or validate
                n+=1            
            else:
                cropped_im.save('./validate/tree/' + df.loc[i,'tag'] + ' on_tree'+ str(j) + '.png', 'PNG') # change location to a "tree folder" in train or validate
                n=0               
        # put fraction of 'not a tree' images in train and validate in equal portions               
        else: # if 'on_tree_j' not true   (change to elif and then else to catch errors?)                   
            if p == num_skips:
                if m==0:
                    cropped_im.save('./train/no_tree/' + df.loc[i,'tag'] + ' on_tree'+ str(j) + '.png', 'PNG')
                    m+=1            
                else:
                    cropped_im.save('./validate/no_tree/' + df.loc[i,'tag'] + ' on_tree'+ str(j) + '.png', 'PNG') # change location to a "tree folder" in train or validate
                    m=0
                p=0
            else:
                p+=1

#        # option 2: images to test folder
#        if df.loc[i,'on_tree_' + str(j)]: # if tree location is true 
#            cropped_im.save('./test/tree/' + df.loc[i,'tag'] + ' on_tree'+ str(j) + '.png', 'PNG') 
#        else:
#            cropped_im.save('./test/no_tree/' + df.loc[i,'tag'] + ' on_tree'+ str(j) + '.png', 'PNG')
            

