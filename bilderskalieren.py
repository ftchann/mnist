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
img_array = skimage.io.imread('5.jpg', 'L').astype(np.float32)  
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
#img_data2 = skimage.transform.rescale(img_array2,1/28.5)
#print(28/w)
#print(w)
#print(h)
img_data = img_array2.reshape(784)
#
plot.imshow(img_array2, cmap='gray')

#print(img_array2)
#print(img_data)

#print(img_data)#.save(' resized_image.png')# convert image to black and white
#img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
#img.save('resized_image.jpg')

def Schwerpunkt(image):
    #Diese Funktion dient zum  bestimmen des Schwerpunkts des Bildes
    indices = np.indices((np.shape(image)))
    #multipliziert die Werte in der Bildmatrix mit ihrem jewweiligen x-Achsenabschnitt und teilt die Summe davon mit der Summe aller Werte der Matrix
    x = (np.sum(image * indices[1])) / np.sum(image)
    y = (np.sum(image * indices[0])) / np.sum(image)
    return x,y

def MaxAbstand(image):
    mabstand = 0
    Sx, Sy = Schwerpunkt(image)
    for j in range (np.shape(image)[1]):
        for i in range (np.shape(image)[0]):
            #Geht durch Zeilen und Spalten der Matrix und berechnet den Abstand, falls der Matrixwert grösser als 0 ist
            if image[i, j] > 0:
                dx = i - Sx
                dy = j - Sy
                abstand = dx ** 2 + dy ** 2
                #Wenn der neue Abstand grösser als der bisherige Abstand ist, wird der alte Abstand überschrieben
                if abstand > mabstand: 
                    mabstand = abstand
            else:
                pass
            
    return mabstand
            
                    
                
                

        
   
print(MaxAbstand(img_array2))




