"""
This code:
    1. takes the truth_table from countr_cnn_3 and creates a dataframe 'pred_counts' with image level tree counts
    2. creates dataframe 'test_results' with predicted and actual tree counts at image level

"""

import pandas as pd
import numpy as np

# create dataframe, tt, from truth_table created by countr_cnn_3
tt = pd.read_pickle("./truth_table.pkl")
tt['image_name'] = ''

# add image name to each row
for ind in tt.index:
    tt.loc[ind,'image_name'] = tt.loc[ind,'filename'].split('\\')[2].split(' ')[0]

# select unique image_names
tt_uniq_images = pd.DataFrame(data = tt.drop_duplicates(subset = 'image_name', keep = 'first').image_name)

# create dataframe for unique image names and tree 
pred_counts = pd.DataFrame(columns = ['image_name','pred_count'])

for ind in tt_uniq_images.index:
    # select an image name from tt_uniq_images
    im = tt_uniq_images.loc[ind,'image_name']
    # create dataframe subset of thumbnails with unique image name
    im_thumbs = tt[tt['image_name']==im]
    # get sum (tree_count) of pred column of im_thumbs
    tree_count = int(im_thumbs['pred'].sum())
    # add image name and tree count to image_counts dataframe
    new_row = pd.DataFrame([[im,tree_count]], columns = ['image_name', 'pred_count'])
    pred_counts = pred_counts.append(new_row, ignore_index=True)
 
# load df_test which has image names and the actual annotated tree count   
df_test = pd.read_pickle('./df_test.pkl')

for ind in pred_counts.index:
    # get image name from each row of pred_counts
    image_name = pred_counts.loc[ind,'image_name']
    # get index of row with the same image name ('tag'column) in df_test
    actual_count_index = df_test.index[df_test['tag'] == image_name].tolist()[0]
    # get the actual count 
    actual_count = df_test.loc[actual_count_index, 'tree_count']
    # add it to pred_counts dataframe in the new 'actual count' column
    pred_counts.loc[ind, 'actual_count'] = actual_count

# save dataframe with predicted and actual image level tree counts
test_results = pred_counts
test_results.to_pickle('./test_results.pkl')


    
    
    
    











#im = tt_uniq_images.loc[25,'image_name']  
#image_thumbs = tt[tt['image_name'] == im]
#image_count = len(image_thumbs[image_thumbs['pred'] == 1.0])




#    
