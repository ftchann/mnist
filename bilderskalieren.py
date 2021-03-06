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

path= 'Testfotos/6_1.jpg'

def readpicture(path):
    #Daten in Grauwerten einlesen
    img_array = skimage.io.imread(path,as_grey=True)
    #Auf 255 erweitern 
    image= img_array*255
    #Treshholding, die Werte werden je nach Helligkeit geändert. Helle Bilder Funktionieren besser. Es kann auch eine andere Methode Treshholding benutzt werden.
    blocksize=911
    local_thresh = threshold_local(image, blocksize, method='mean', offset=40)
    binary_local = image > local_thresh
    #werte umkehren
    img_array2 = abs(1-np.int32(binary_local))
	#Die Hälfte der Breite oder Länge bestimmen je nach dem was grösser ist
    shape_img = np.shape(img_array2)
    if shape_img[0] > shape_img[1]:
        maxwh = shape_img[0]
    else:
        maxwh = shape_img[1]
        #Bild um maxwh (Die Hälfte der Breite oder Länge) erweitern, damit man immer um Quadrat um das Objekt schneiden kann.
    img_array2 = np.pad(img_array2, int(maxwh/2),'constant', constant_values=0)
    return img_array, img_array2

def CenterofMass(image):
    #Matrix aus Indices kreieren  
    indices = np.indices((np.shape(image)))
    #Formel für Schwerpunktberechnung auf beiden Achsen anwenden  
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
    #Schneiden
    OberY = int(Sy + MaxDistance2)
    UnterY = int(Sy - MaxDistance2)
    LinksX = int(Sx - MaxDistance2)
    RechtsX = int(Sx + MaxDistance2)
    #Falls nicht quadratisch geschnitten wurde  
    if OberY - UnterY > RechtsX - LinksX:
        OberY -= int((OberY-UnterY)-(RechtsX-LinksX))
    if RechtsX - LinksX > OberY - UnterY:
        RechtsX -= int((RechtsX-LinksX)-(OberY-UnterY))
    #Neue Matrix mit diesen Rändern.
    format_img = img_array2[LinksX:RechtsX,UnterY:OberY]
    plot.imshow(format_img, cmap='gray')
    return format_img

def transformMatrix(format_img):
    shape_format_img = np.shape(format_img)
	#Auf 20*20 skalieren
    format_img = skimage.transform.pyramid_reduce(format_img*255, downscale=(shape_format_img[0]/20))
    #format_img = skimage.transform.rescale(format_img*255, 20/shape_format_img[0])
    #4Pixel breiter Rand hinzufügen
    img_0final = np.pad(format_img, 4,'constant', constant_values=(0))
    #In Vektor umwandeln 1 * 784
    img_0final = np.reshape(img_0final, 28*28)
	#Werte anpassen zwischen 0-255
    img_0final = (img_0final / img_0final[np.argmax(img_0final)]) * 255
    return(img_0final)

def start(path):
    fig, axes = plot.subplots(ncols=4, figsize=(8, 3))
    ax = axes.ravel()
    original, img_array = readpicture(path)
    ax[0].imshow(original, cmap=plot.cm.gray)
    ax[0].set_title('Originales Bild')
    format_img = Cut(img_array)
    ax[1].imshow(img_array, cmap=plot.cm.gray)
    ax[1].set_title('Treshholding')
    img_final = transformMatrix(format_img)
    ax[2].imshow(format_img, cmap=plot.cm.gray)
    ax[2].set_title('Ziffer ausgeschnitten')
    img_0final = np.reshape(img_final,(28,28))
    ax[3].imshow(img_0final, cmap=plot.cm.gray)
    ax[3].set_title('Herunterskaliert')
    plot.show()
    return img_final


