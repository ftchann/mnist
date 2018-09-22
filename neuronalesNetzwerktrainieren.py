# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 02:19:56 2018

@author: Yann
"""

import neuronalesNetzwerk as nk
 
#Anzahl von Eingabe, Versteckten und Ausgabeneuronen definieren
numberof_input_neurons = 784
numberof_hidden_neurons = 20
numberof_output_neurons = 10
#Anzahl versteckte Layers definieren
numberof_hidden_layers = 1
#Aktivationfunktion
activation_function = 'sigmoid'
#learningrate definieren
learningrate = 0.1

n = nk.neuralNetwork(numberof_input_neurons, numberof_hidden_neurons, numberof_output_neurons, learningrate, numberof_hidden_layers,activation_function)

n.trainnetwork()