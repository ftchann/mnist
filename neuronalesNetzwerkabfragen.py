# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 01:37:18 2018

@author: Yann
"""
import numpy as np
import neuronalesNetzwerk as nk
import bilderskalieren as bs
#Dateipfad
path = 'Testfotos/0_1.jpg'
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
weight = np.load("bestgewicht/bestweight_1hiddenlayer_sigmoid_250neurons..npy")
if bias == True:
    biases = np.load("bestgewicht/bestbias_1hiddenlayer_sigmoid_250neurons.npy")
if numberof_hidden_layers == 0:
    na.weight_hidden_1_input = weight
    if bias==True:
        na.weights_hidden_output_bias = biases
if numberof_hidden_layers == 1:
    na.weight_hidden_1_input = weight[0]
    na.weight_hidden_output = weight[1]
    if bias==True:
        na.weights_hidden_1_input_bias = biases[0]
        na.weights_hidden_output_bias = biases[1]

if numberof_hidden_layers == 5:
    na.weight_hidden_1_input = weight[0]
    na.weight_hidden_2_1 = weight[1]
    na.weight_hidden_3_2 = weight[2]
    na.weight_hidden_4_3 = weight[3]
    na.weight_hidden_5_4 = weight[4]
    na.weight_hidden_output = weight[5]
    if bias==True:
        na.weights_hidden_1_input_bias = bias[0]
        na.weights_hidden_2_1_bias = biases[1]
        na.weights_hidden_3_2_bias = biases[2]
        na.weights_hidden_4_3_bias = biases[3]
        na.weights_hidden_5_4_bias = biases[4]
        na.weights_hidden_output_bias = biases[5]
#Bild laden
output_list = bs.start(path)
#Ausgabe Matrix
if na.hidden_layers == 0:
    outputs = na.forwardprop(output_list)
else:
    outputs = na.forwardprop(output_list)[0]
Number = np.argmax(outputs)
#Zahl ausgeben
print(Number)