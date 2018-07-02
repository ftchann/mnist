# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:38:48 2018
@author: Yann
"""
import numpy as np

import skimage.io
import matplotlib.pyplot as plot
import skimage.transform
from skimage.filters import threshold_otsu, threshold_local
from scipy import ndimage


#reshape von 28x28 zu 784
img_array = skimage.io.imread('7.jpg', 'L').astype(np.float32) 

#Auf 255 erweitern 
image= img_array*255


#Treshholding
blocksize = 9555
global_thresh = threshold_local(image, blocksize, offset=70)
binary_global = image > global_thresh
#print(binary_global)

##show image
img_array2 = abs(1-np.int32(binary_global))
#Werte von 0 bis 1
#Shape Image
shape_img = np.shape(img_array2)
#max width lenght
if shape_img[0] > shape_img[1]:
    maxwh = shape_img[0]
else:
    maxwh = shape_img[1]
#Bild um Grösse erweitern, damit man immer um Quadrat um das Objekt schneiden kann.
img_array2 = np.pad(img_array2, maxwh,'constant', constant_values=(0))
def Schwerpunkt(image):
    #Diese Funktion dient zum  bestimmen des Schwerpunkts des Bildes
    indices = np.indices((np.shape(image)))
    #multipliziert die Werte in der Bildmatrix mit ihrem jewweiligen x-Achsenabschnitt und teilt die Summe davon mit der Summe aller Werte der Matrix
    x = (np.sum(image * indices[1])) / np.sum(image)
    y = (np.sum(image * indices[0])) / np.sum(image)
    return x,y


def MaxAbstand(Sxa, Sya, image):
    maxdxy = 0
    Sx = Sxa
    Sy = Sya

    for j in range (np.shape(image)[1]):
        for i in range (np.shape(image)[0]):
            #Geht durch Zeilen und Spalten der Matrix und berechnet den Abstand, falls der Matrixwert grösser als 0 ist
            if image[i, j] > 0.1:
                dx = abs(i - Sx)
                dy = abs(j - Sy)
                #abstand = dx ** 2 + dy ** 2
                #Wenn der neue Abstand grösser als der bisherige Abstand ist, wird der alte Abstand überschrieben
                if dx > maxdxy: 
                    maxdxy = dx
                if dy > maxdxy:
                    maxdxy = dy
            else:
                pass
            
    return maxdxy
            

Sx, Sy = Schwerpunkt(img_array2)
MaxAbstand2 = MaxAbstand(Sx, Sy, img_array2)

OberY = int(Sy - MaxAbstand2 + 0.5)
UnterY = int(Sy + MaxAbstand2 + 0.5)
LinksX = int(Sx - MaxAbstand2 + 0.5)
RechtsX = int(Sx + MaxAbstand2 + 0.5)
Breite= RechtsX-LinksX
Höhe = UnterY - OberY
print(Höhe,Breite)

format_img = img_array2[OberY:UnterY, LinksX:RechtsX]

print(np.shape(format_img))
#print(Sx,Sy)
#print(MaxAbstand2)
shape_format_img = np.shape(format_img)

#print(shape_format_img[0])
print(shape_format_img[0])
format_img_rescale = skimage.transform.rescale(format_img*255,20/shape_format_img[0])

img_0final = np.pad(format_img_rescale, 4,'constant', constant_values=(0))
print(np.shape(img_0final))
img_0final = np.reshape(img_0final, 28*28)
img_0final = (img_0final / img_0final[np.argmax(img_0final)]) * 255
print(img_0final)
img_final = np.reshape(img_0final,(28,28))
plot.imshow(img_final, cmap='gray')
##print(img_final)
