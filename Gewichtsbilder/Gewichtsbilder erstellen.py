# -*- coding: utf-8 -*-s
"""
Created on Fri Oct  5 22:31:40 2018

@author: Florian
"""
import numpy as np
import matplotlib.pyplot as plot

"""Diese Funktion stellt die Gewichtsmatrizen als Bild auf einem Farbverlauf da"""

def save(images):
    for i in range(10):
        imagevector = weights[i]                            #der Gewichtsvektor eines Neurons wird eingelesen
        image = np.reshape(imagevector, (28, 28))           #aus dem Vektor wird ein Bild erstell
        plot.imshow(image, cmap='inferno')                  #eigentlich nicht nötig; stellt bild in konsole dar
        plot.imsave(str(i)+'.png', image, cmap='inferno')   #speichert Bild als png ab
        
        
weights = np.load('bestweight_0hiddenlayer.npy')
save(weights)