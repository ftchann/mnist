# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:38:48 2018
@author: Yann
"""
import numpy as np

import skimage.io
import matplotlib.pyplot as plot
import skimage.transform
from skimage.filters import threshold_local

mnistbild=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


def readpicture():
    #Daten in Grauwerten einlesen
    img_array = skimage.io.imread('6.jpg',as_grey=True)
    #Auf 255 erweitern 
    image= img_array*255
    #Treshholding
    blocksize=9555
    local_thresh = threshold_local(image, blocksize, method='mean', offset=50)
    binary_local = image > local_thresh
    #werte umkehren
    img_array2 = abs(1-np.int32(binary_local))
	#Die Hälfte der Breite oder Länge bestimmen 
    shape_img = np.shape(img_array2)
    if shape_img[0] > shape_img[1]:
        maxwh = shape_img[0]
    else:
        maxwh = shape_img[1]
        #Bild um maxwh (Die Hälfte der Breite oder Länge) erweitern, damit man immer um Quadrat um das Objekt Cut kann.
    img_array2 = np.pad(img_array2, int(maxwh),'constant', constant_values=0)
    return img_array2

def CenterofMass(image):
    #Diese Funktion dient zum  bestimmen des CenterofMasss des Bildes
    indices = np.indices((np.shape(image)))
    #multipliziert die Werte in der Bildmatrix mit ihrem jewweiligen x-Achsenabschnitt und teilt die Summe davon mit der Summe aller Werte der Matrix
    x = (np.sum(image * indices[0])) / np.sum(image)
    y = (np.sum(image * indices[1])) / np.sum(image)
    return x,y


def MaxDistance(Sxa, Sya, image):
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
            

def Cut(img_array2):
    Sx, Sy = CenterofMass(img_array2)
    MaxDistance2 = MaxDistance(Sx, Sy, img_array2)
    #Scneiden
    OberY = int(Sy - MaxDistance2)
    UnterY = int(Sy + MaxDistance2)
    LinksX = int(Sx - MaxDistance2)
    RechtsX = int(Sx + MaxDistance2)
    format_img = img_array2[LinksX:RechtsX,OberY:UnterY]
    return format_img

def transformMatrix(format_img):
    shape_format_img = np.shape(format_img)
	#Auf 20*20 skalieren
    format_img = skimage.transform.pyramid_reduce(format_img*255, downscale=(shape_format_img[0]/20))
    #format_img = skimage.transform.rescale(format_img*255, 20/shape_format_img[0])
    img_0final = np.pad(format_img, 4,'constant', constant_values=(0))
	#4Pixel breiter Rand hinzufügen
    img_0final = np.reshape(img_0final, 28*28)
	#Werte anpassen zwischen 0-255
    img_0final = (img_0final / img_0final[np.argmax(img_0final)]) * 255
    return(img_0final)



img_array = readpicture()
format_img = Cut(img_array)
img_final = transformMatrix(format_img)
img_0final = np.reshape(img_final,(28,28))
print(CenterofMass(img_0final))


#format_0img = skimage.transform.pyramid_reduce(format_img*255, downscale=(shape_format_img[0]/20))
#format_1img = skimage.transform.rescale(format_img*255, 20/shape_format_img[0])
#print(CenterofMass(format_0img))
#print(CenterofMass(format_1img))
#plot.imshow(format_0img, cmap='gray')



plot.imshow(img_0final, cmap='gray')

