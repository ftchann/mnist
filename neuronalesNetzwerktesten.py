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
verstecktelayers = nk.verstecktelayers
#learnrate 
learnrate = nk.learnrate
#Neuronales Netzwerk erstellen
na = nk.neuronalesNetzwerk(eingabeneuronen, versteckteneuronen, ausgabeneuronen, learnrate, verstecktelayers)
#Gewichte laden   
gewichte = np.load("bestgewicht.npy")
na.ge_v1 = gewichte[0]
na.ge_v2 = gewichte[1]
na.ge_v3 = gewichte[2]
na.ge_v4 = gewichte[3]
na.ge_v5 = gewichte[4]
na.ge_va = gewichte[5]
#Testdatei§ laden
test_daten_liste = nk.lesen("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000)
performance = na.abfragen2(test_daten_liste)
print(performance)
