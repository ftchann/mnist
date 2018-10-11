# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 01:37:18 2018

@author: Yann
"""
import numpy as np
import neuronalesNetzwerk as nk
import bilderskalieren as bs
numberof_input_neurons = 784
numberof_hidden_neurons = 200
numberof_output_neurons = 10
#Anzahl versteckte Layers definieren
numberof_hidden_layers = 1
#learningrate definieren
learningrate = 0.1
#Aktivationfunktion
activation_function = 'sigmoid'
#Verzerrung (Bias ein- und ausschalten)
bias = True
#Neuronales Netzwerk erstellen
na = nk.neuralNetwork(numberof_input_neurons, numberof_hidden_neurons, numberof_output_neurons, learningrate, numberof_hidden_layers,activation_function,bias)
#Gewichte laden 
weight = np.load("bestweight_1hiddenlayer.npy")
if numberof_hidden_layers == 0:
    na.weight_hidden_1_input = weight[0]
if numberof_hidden_layers == 1:
    na.weight_hidden_1_input = weight[0]
    na.weight_hidden_output = weight[1]
if numberof_hidden_layers == 5:
    na.weight_hidden_1_input = weight[0]
    na.weight_hidden_2_1 = weight[1]
    na.weight_hidden_3_2 = weight[2]
    na.weight_hidden_4_3 = weight[3]
    na.weight_hidden_5_4 = weight[4]
    na.weight_hidden_output = weight[5]
#Bild laden
input_list = bs.img_final
#Ausgabe Matrix
outputs = na.asknetwork(input_list)
print(outputs)
Number = np.argmax(outputs)
#Zahl ausgeben
print(Number)