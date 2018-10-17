# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 01:37:18 2018

@author: Yann
Fast identisches Programm zu neuronalesNetzwerktesten nur hier zum testen der Genauigkeit mit dem Testset
"""
import numpy as np
import neuronalesNetzwerk as nk

numberof_input_neurons = 784
numberof_hidden_neurons = 100
numberof_output_neurons = 10
#Anzahl versteckte Layers definieren
numberof_hidden_layers = 0
#learningrate definieren
learningrate = 0.1
#Aktivationfunktion
activation_function = 'sigmoid'
#Schwellwert
bias=True
#Neuronales Netzwerk erstellen

na = nk.neuralNetwork(numberof_input_neurons, numberof_hidden_neurons, numberof_output_neurons, learningrate, numberof_hidden_layers,activation_function,bias)
#Gewichte laden 
weight = np.load("gewichte/weight_0hiddenlayer_sigmoid.npy")
if bias == True:
    biases = np.load("gewichte/bias_0hiddenlayer_sigmoid.npy")
if numberof_hidden_layers == 0:
    na.weight_hidden_output = weight
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
        na.weights_hidden_1_input_bias = biases[0]
        na.weights_hidden_2_1_bias = biases[1]
        na.weights_hidden_3_2_bias = biases[2]
        na.weights_hidden_4_3_bias = biases[3]
        na.weights_hidden_5_4_bias = biases[4]
        na.weights_hidden_output_bias = biases[5]

#Testdatei laden
test_data_list = nk.readdata("Trainingsdaten/t10k-images.idx3-ubyte", "Trainingsdaten/t10k-labels.idx1-ubyte", 10000)
#Performance ausgeben
performance = na.testnetwork(test_data_list)
print(performance)
