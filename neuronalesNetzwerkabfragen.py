# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 01:37:18 2018

@author: Yann
"""
import numpy as np
import neuronalesNetzwerk as nk
import bilderskalieren as bs
eingabeneuronen = nk.eingabeneuronen
versteckteneuronen = nk.versteckteneuronen
ausgabeneuronen = nk.ausgabeneuronen
verstecktelayer = nk.verstecktelayers
#learnrate 
learnrate = nk.learnrate

na = nk.neuronalesNetzwerk(eingabeneuronen, versteckteneuronen, ausgabeneuronen, learnrate, verstecktelayer)
   
gewichte = np.load("bestgewicht.npy")
na.ge_v1 = gewichte[0]
na.ge_v2 = gewichte[1]
na.ge_v3 = gewichte[2]
na.ge_v4 = gewichte[3]
na.ge_v5 = gewichte[4]
na.ge_va = gewichte[5]
input_liste = bs.img_0final
print(input_liste)
outputs = na.abfragen(input_liste)
print(outputs)
Zahl = np.argmax(outputs)
print(Zahl)