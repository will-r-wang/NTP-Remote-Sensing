from PIL import Image
import numpy

startX, startY = 4999, 4999
endX, endY = 10000, 10000

image_name = "2019_Lac Gus_544500_5220800.tif"

image = Image.open(image_name)

for i in range(startX, endX, 500):
    for j in range(startY, endY, 500):
        image.crop((i, j, i + 500, j + 500)).save("image" + str(i) + str(j) + ".png")
