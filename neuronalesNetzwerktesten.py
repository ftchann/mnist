# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 01:37:18 2018

@author: Yann
"""
import numpy as np
import neuronalesNetzwerk as nk
 
eingabeneuronen = nk.eingabeneuronen
versteckteneuronen = nk.versteckteneuronen
ausgabeneuronen = nk.ausgabeneuronen
#learnrate 
learnrate = nk.learnrate
#Neuronales Netzwerk erstellen
na = nk.neuronalesNetzwerk(eingabeneuronen, versteckteneuronen, ausgabeneuronen, learnrate)
#Gewichte laden   
gewichte = np.load("gewicht.npy")
na.ge_ev = gewichte[0]
na.ge_va = gewichte[1]
#Testdatei laden
test_daten_liste = nk.lesen("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000)
performance = na.abfragen2(test_daten_liste)
print(performance)
