"""
This code creates a lookup table with pixel centers for the x by y grid.  Lookup_table is used in counter_cnn1 script. The lookup_table is used to find the center of the cropped image thumbnail based on the grid data in the file
Note:  users will need to change values in loops based on desired xy grid coordinates.  e.g. code below generates a 7 x 7 grid every 12 pixels
"""

import numpy as np
import pandas as pd

coords_array = np.array([])  # create empty array to fill with calculated x, y pixel centers

# calculate 7 y values.  for each y value, calculate 7 x values.  12 is initial offset value with 28 pixels between
# each location.  images = 194 x 194 pixels.  194/7 = ~28
for j in range(7): 
    y = 12 + 28*j 
    for i in range(7):
        x = 12 + 28*i
        coords = np.array([x,y])
        coords_array = np.append(coords_array, coords) # append current x,y values to coords_array
   
# resize array to two columns      
coords_array = np.resize(coords_array,98).reshape(49,2)

# convert np array to dataframe and start index at 1 instead of 0
lookup_table = pd.DataFrame(coords_array, columns = ['x', 'y'])
lookup_table.index = np.arange(1, len(lookup_table)+1)

        
