# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 00:11:25 2018

@author: Yann
"""
import numpy as np
import matplotlib.pyplot as plot
def readdata(imgplace, labelplace, n):
    #In binary Modus lesen von https://pjreddie.com/projects/mnist-in-csv/ besucht:11.10.2018
    images = open(imgplace, "rb")
    label = open(labelplace, "rb")
    #Erste 16 bezieungsweise 8 bytes überspringen, da keine Daten drin sind. read() funktion springt immer auf das Nächste.
    images.read(16)
    label.read(8)
    imagedata = []
    #Ganze Datei durchgehen, n = anzahl bilder
    for i in range(1):
        #Label lesen und zwischenspeichern
        image = []
        #784 Pixels lesen
        for j in range(28*28):
            image.append(ord(images.read(1)))
        #bild in Bilddata einfügen
        imagedata.append(image)
    return imagedata
Bild = readdata("Trainingsdaten/train-images.idx3-ubyte", "Trainingsdaten/train-labels.idx1-ubyte",1)
Bild = np.asfarray(Bild)
Bild = np.reshape(Bild,(28,28))
plot.imsave("Bild.png",Bild, cmap='gray')