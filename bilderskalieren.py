# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:38:48 2018

@author: Yann
"""
import numpy as np
import PIL
from PIL import Image
import skimage.io as ski
import matplotlib.pyplot as plot
baseheight = 28
basewidth  = 28

img = Image.open('2.jpg').convert('LA')
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
img_array = ski.imread('resized_image.png', 'L').astype(np.float32)   
img_data = 1-img_array.reshape(784)
print(img_array)
##show image
plot.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')
#print(img_data)#.save(' resized_image.png')# convert image to black and white
#img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
#img.save('resized_image.jpg')