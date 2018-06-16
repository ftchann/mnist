# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:38:48 2018

@author: Yann
"""
import numpy as np
from PIL import Image
import skimage.io
import matplotlib.pyplot as plot
import skimage.transform
from skimage.filters import threshold_otsu, threshold_adaptive
from scipy import ndimage

baseheight = 28
basewidth  = 28

img = Image.open('IMG_20180614_102413.jpg').convert('LA')
#w, h = img.size
##print(img.size)
#if w >= h:
#    #quadratisch machen, schwarzweissmachen, auf 28x28 scalieren
#    x = (w-h)/2
#    img = img.crop((x, 0, w-x, h)).save('resizedimage.png')
#    w=h
#else:
#    #quadratisch machen, schwarzweissmachen, auf 28x28 scalieren
#    x=(h-w)/2
#    img = img.crop((0, x, w, h-x)).save('resizedimage.png')
#    h=w
##reshape von 28x28 zu 784
img_array = skimage.io.imread('resizedimage.png', 'L').astype(np.float32)  
#Auf 255 erweitern 
image= img_array*255
print(image)
#Treshholding
global_thresh = threshold_otsu(image)
binary_global = image > global_thresh
print(binary_global)



##show image
img_array2 = 1-np.int32(binary_global)
print(img_array2.size)
img_data2 = skimage.transform.rescale(img_array2,1/28.5)
#print(28/w)
#print(w)
#print(h)
img_data = img_data2.reshape(784)
#
plot.imshow(img_data2, cmap='gray')

#print(img_array2)
#print(img_data)

#print(img_data)#.save(' resized_image.png')# convert image to black and white
#img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
#img.save('resized_image.jpg')

def Schwerpunkt(image):
    #Diese Funktion dient zum  bestimmen des Schwerpunkts des Bildes
    indices = np.indices((np.shape(image)))
    print (indices)
    
    x = (np.sum(image * indices[1])) / np.sum(image)
    y = (np.sum(image * indices[0])) / np.sum(image)
    

Schwerpunkt(image)




