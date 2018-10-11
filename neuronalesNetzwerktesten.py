# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 01:37:18 2018

@author: Yann
"""
import numpy as np
import neuronalesNetzwerk as nk

numberof_input_neurons = 784
numberof_hidden_neurons = 20
numberof_output_neurons = 10
#Anzahl versteckte Layers definieren
numberof_hidden_layers = 1
#learningrate definieren
learningrate = 0.1
#Aktivationfunktion
activation_function = 'sigmoid'
#Neuronales Netzwerk erstellen
#Schwellwert
bias=True
na = nk.neuralNetwork(numberof_input_neurons, numberof_hidden_neurons, numberof_output_neurons, learningrate, numberof_hidden_layers,activation_function,bias)
#Gewichte laden 
weight = np.load("bestweight.npy")
na.weight_hidden_1_input = weight[0]
na.weight_hidden_2_1 = weight[1]
na.weight_hidden_3_2 = weight[2]
na.weight_hidden_4_3 = weight[3]
na.weight_hidden_5_4 = weight[4]
na.weight_hidden_output = weight[5]

#Testdatei laden
test_data_list = nk.readdata("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000)
#Performance ausgeben
performance = na.testnetwork(test_data_list)
print(performance)
