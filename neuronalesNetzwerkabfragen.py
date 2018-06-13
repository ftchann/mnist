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
#learnrate 
learnrate = nk.learnrate

na = nk.neuronalesNetzwerk(eingabeneuronen, versteckteneuronen, ausgabeneuronen, learnrate)
   
gewichte = np.load("gewicht.npy")
na.ge_ev = gewichte[0]
na.ge_va = gewichte[1]
input_liste = bs.img_data
print(input_liste)
outputs = na.abfragen(input_liste)
print(outputs)
Zahl = np.argmax(outputs)
print(Zahl)