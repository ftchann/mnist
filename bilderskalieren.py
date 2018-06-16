# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:38:48 2018

@author: Yann
"""
import numpy as np
from PIL import Image
import skimage.io
import matplotlib.pyplot as plot
from skimage.filters import threshold_otsu, threshold_adaptive
baseheight = 28
basewidth  = 28

img = Image.open('IMG_20180614_102413.jpg').convert('LA')
w, h = img.size
print(img.size)
if w >= h:
    #quadratisch machen, schwarzweissmachen, auf 28x28 scalieren
    x = (w-h)/2
    img = img.crop((x, 0, w-x, h))
    img = img.resize((basewidth, baseheight)).save('resized_image.png')
else:
    #quadratisch machen, schwarzweissmachen, auf 28x28 scalieren
    x=(h-w)/2
    img = img.crop((0, x, w, h-x))
    img = img.resize((basewidth, baseheight)).save('resized_image.png')
#reshape von 28x28 zu 784
img_array = skimage.io.imread('resized_image.png', 'L').astype(np.float32)  
#Auf 255 erweitern 
image= img_array*255
#Treshholding
global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

##show image
plot.imshow(binary_global)
img_array = 1-np.int32(binary_global)
img_data = img_array.reshape(784)
print(img_array)

#print(img_data)#.save(' resized_image.png')# convert image to black and white
#img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
#img.save('resized_image.jpg')
