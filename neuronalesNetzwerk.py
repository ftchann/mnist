# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:16:22 2018
@author: Yann
"""
import numpy as np
import time
import sys

#Anzahl von Eingabe, Versteckten und Ausgabeneuronen definieren
#Für Mnist muss input_neurons = 784 und output_neurons = 10 sein.
numberof_input_neurons = 784
numberof_hidden_neurons = 20
numberof_output_neurons = 10
#Anzahl versteckte Layers definieren
numberof_hidden_layers = 1
#learningrate definieren
learningrate = 0.01
#Aktivationfunktion
activation_function = 'lrelu'

#Neuronales Netzwerk definieren
class neuralNetwork:
    #Aktivierungsfunktionen
	#Sigmoid Funktion
    def sigmoid(self, x): 
        return 1 / (1 + np.exp(-x))
    #Ableitung SigmoidFunktion
    def sigmoid_derivative(self, x):
        return x*(1-x)
    #Tangenshyperbolicus funktion
    def tanh(self, x):
        #np.tanh(x)
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    #Ableitung der Tangenshyperbolicus funktion               
    def tanh_derivative(self, x):
        return (1 - (x ** 2))
    #Relu funktion
    def relu(self, x):
        return x * (x > 0)
    #Ableitung Relu
    def relu_derivative(self, x):
        return 1 * (x > 0)
    #Leaky ReLu
    def lrelu(self, x):
        return x * (x > 0) + (x <= 0) * 0.01*x
    #Ableitung Leaky Relu
    def lrelu_derivative(self, x):
        return 1 * (x > 0) + (x <= 0) * 0.01
    #neuronales Netzwerk inistialisieren
    def __init__(self, numberof_input_neurons, numberof_hidden_neurons, numberof_output_neurons, learningrate, numberof_hidden_layers, activation_function):
        np.random.seed(1)#Seed festlegen
        self.function = activation_function
        self.hidden_layers = numberof_hidden_layers #Anzahl verstecktelayers
        self.input_neurons = numberof_input_neurons # Anzahl Eingabeneuronen
        self.output_neurons= numberof_output_neurons # Anzahl ausgabeneuronen
        self.hidden_neurons_5 = numberof_hidden_neurons # Anzahl hiddenneuronen Layer 5
        self.hidden_neurons_1 = self.hidden_neurons_5 * self.hidden_layers # Anzahl hiddenneuronen Layer 1
        self.hidden_neurons_2 = self.hidden_neurons_5*(self.hidden_layers-1) # Anzahl hiddenneuronen Layer 2
        self.hidden_neurons_3 = self.hidden_neurons_5*(self.hidden_layers-2) # Anzahl hiddenneuronen Layer 3
        self.hidden_neurons_4 = self.hidden_neurons_5*(self.hidden_layers-3) # Anzahl hiddenneuronen Layer 4
		#Alles 0 setzen. Behebt das Problem beim speichern, wenn nicht alle hidden layers genutzt werden.
        self.weight_hidden_output = 0
        self.weight_hidden_5_4 = 0
        self.weight_hidden_4_3 = 0
        self.weight_hidden_3_2 = 0
        self.weight_hidden_2_1 = 0
        self.weight_hidden_1_input = 0
        #Gewichtungsmatrixen definieren
        #Grösse der Gewichtungsmatrix ist hintere Layer mal vordere Layer.
        #Für die Gewichtungsmatrixen gibt man am Anfang Zufallszahlen. Diese sind 0 +- hiddnennodes hoch -0.5 
        #Gewichte Versteckte-Outputlayer
        if self.hidden_layers == 0:
            self.weight_hidden_output = np.random.normal(0.0,pow(self.output_neurons, -0.5),(self.output_neurons, self.input_neurons))
            
        if self.hidden_layers > 0:
            self.weight_hidden_output = np.random.normal(0.0,pow(self.output_neurons, -0.5),(self.output_neurons, self.hidden_neurons_5))
            
        #Gewichte Hiddenlayer 1 - 5
        #Im Moment haben alle Verstecktenlayers die selbe Anzahl Neuronen numberof_hidden_neurons * numberof_hidden_layers
        if self.hidden_layers >= 1:
            self.weight_hidden_1_input = np.random.normal(0.0,pow(self.hidden_neurons_1, -0.5),(self.hidden_neurons_1, self.input_neurons))
        
        if self.hidden_layers >= 2:
            self.weight_hidden_2_1 = np.random.normal(0.0,pow(self.hidden_neurons_2, -0.5),(self.hidden_neurons_2, self.hidden_neurons_1))
            
        if self.hidden_layers >= 3:
            self.weight_hidden_3_2 = np.random.normal(0.0,pow(self.hidden_neurons_3, -0.5),(self.hidden_neurons_3, self.hidden_neurons_2))
            
        if self.hidden_layers >= 4:
            self.weight_hidden_4_3 = np.random.normal(0.0,pow(self.hidden_neurons_4, -0.5),(self.hidden_neurons_4, self.hidden_neurons_3))
            
        if self.hidden_layers >= 5:
            self.weight_hidden_5_4 = np.random.normal(0.0,pow(self.hidden_neurons_5, -0.5),(self.hidden_neurons_5, self.hidden_neurons_4))
        #Learnrate
        self.lr = learningrate
        #Aktivierungsfunktionsliste
        self.activationfunction={'sigmoid':self.sigmoid, 'relu':self.relu, 'tanh':self.tanh, 'lrelu':self.lrelu}
        #Liste der Ableitung der Aktivierungsfunktions
        self.activationfunction_derivative={'sigmoid':self.sigmoid_derivative, 'relu':self.relu_derivative, 'tanh':self.tanh_derivative, 'lrelu':self.lrelu_derivative}
    
    #neuronales Netzwerk trainieren
    def train(self, inputs_list, target_list,):
        #Inputsliste nehmen und transformieren damit sie hoch steht
        inputs = np.array(inputs_list, ndmin=2).T
        #Das Gleiche mit der Zielliste
        targets = np.array(target_list, ndmin=2).T
        
        if self.hidden_layers == 0:
            #Analog zu numberof_hidden_layers = 1 einfach ohne die versteckten Komponenten.
            #Dies ist nur ein Experiment welche genauigkeit sich mit keinen hiddenn Layers erzielen lässt.
            output_inputs = np.dot(self.weight_hidden_output, inputs)
            output_outputs = self.activationfunction[self.function](output_inputs)
            output_error = targets - output_outputs
            self.weight_hidden_output += self.lr * np.dot(output_error * self.activationfunction_derivative[self.function](output_outputs), inputs.T)
            
        elif self.hidden_layers == 1: 
            #Eingabe mal Gewicht
            hidden_inputs = np.dot(self.weight_hidden_1_input, inputs)
            #Das ganze in die Aktivierungsfunktion
            hidden_outputs = self.activationfunction[self.function](hidden_inputs)
            #Die alten Ausgaben (neue Eingaben) mal Gewichtsmatrix
            output_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
            #Das ganze in die Aktivierungsfunktion
            output_outputs = self.activationfunction[self.function](output_inputs)    
        
           
            #ausgabefehler (Ziel-Ausgabe)
            output_error = (targets - output_outputs) * self.activationfunction_derivative[self.function](output_outputs)
            #verstecktefehler (Ausgabe mal Gewicht) Gewichtsmatrix umkehren da wir jetzt zurückrechnen
            hidden_error = np.dot(self.weight_hidden_output.T, output_error) * self.activationfunction_derivative[self.function](hidden_outputs)
            #Gewichte aktuallisieren Versteckt-Ausgabe
            self.weight_hidden_output += self.lr * np.dot(output_error, hidden_outputs.T)
            #Gewichte aktuallisieren Eingabe-Versteckt                            
            self.weight_hidden_1_input += self.lr * np.dot(hidden_error, inputs.T)                     
            
        
        elif self.hidden_layers == 5:
			#Analog zu hidden_layers==1
            #hiddens Layer 1
            hidden_1_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_1_input, inputs))
            #hiddens Layer 2  
            hidden_2_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_2_1, hidden_1_outputs))
            #hiddens Layer 3
            hidden_3_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_3_2, hidden_2_outputs))
            #hiddens Layer 4
            hidden_4_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_4_3, hidden_3_outputs))
            #hiddens Layer 5
            hidden_5_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_5_4, hidden_4_outputs))
            #output Layer
            output_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_output, hidden_5_outputs))
            
            output_error = (targets - output_outputs) * self.activationfunction_derivative[self.function](output_outputs)
            hidden_5_error = np.dot(self.weight_hidden_output.T, output_error) * self.activationfunction_derivative[self.function](hidden_5_outputs)
            hidden_4_error = np.dot(self.weight_hidden_5_4.T, hidden_5_error) * self.activationfunction_derivative[self.function](hidden_4_outputs)
            hidden_3_error = np.dot(self.weight_hidden_4_3.T, hidden_4_error) * self.activationfunction_derivative[self.function](hidden_3_outputs)
            hidden_2_error = np.dot(self.weight_hidden_3_2.T, hidden_3_error) * self.activationfunction_derivative[self.function](hidden_2_outputs)
            hidden_1_error = np.dot(self.weight_hidden_2_1.T, hidden_2_error) * self.activationfunction_derivative[self.function](hidden_1_outputs)
            #Gewichte aktualisieren Versteckt-Ausgabe
            self.weight_hidden_output += self.lr * np.dot(output_error, hidden_5_outputs.T)
            #Gewichte aktualisieren Versteckt_5-4
            self.weight_hidden_5_4 += self.lr * np.dot(hidden_5_error, hidden_4_outputs.T) 
            #Gewichte aktualisieren Versteckt_4-3
            self.weight_hidden_4_3 += self.lr * np.dot(hidden_4_error, hidden_3_outputs.T)
            #Gewichte aktualisieren Versteckt_3-2
            self.weight_hidden_3_2 += self.lr * np.dot(hidden_3_error, hidden_2_outputs.T)
            #Gewichte aktualisieren Versteckt_2-1
            self.weight_hidden_2_1 += self.lr * np.dot(hidden_2_error, hidden_1_outputs.T)
            #Gewichte aktualisieren Versteckt1-Eingabe
            self.weight_hidden_1_input += self.lr * np.dot(hidden_1_error, inputs.T)
            
        else:
            #Wenn eine nicht vorgesehene Anzahl hidden Layers gesetzt wird, beendet sich das Program  
            sys.exit("Error: Anzahl hidden Layers ungültig")
            
            
    
            
    #neuronales Netzwerk abfragen ähnlich wie neuronales Netzwerk trainieren
    def asknetwork(self, inputs_list):
        #Inputsliste nehmen und transformieren damit sie hoch steht
        inputs = np.array(inputs_list, ndmin=2).T
        #Analog zu numberof_hidden_layers = 1 einfach ohne die versteckten Komponenten
        if self.hidden_layers == 0:
            output_inputs = np.dot(self.weight_hidden_output, inputs)
            output_outputs = self.activationfunction[self.function](output_inputs)
            return output_outputs
        if self.hidden_layers == 1:    
           
            #Inputs mal Gewicht
            hidden_inputs = np.dot(self.weight_hidden_1_input, inputs)
            #Das ganze in die Aktivierungsfunktion
            hidden_outputs = self.activationfunction[self.function](hidden_inputs)
            #Die alten Ausgaben (neue Eingaben) mal Gewichtsmatrix
            output_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
            #Das ganze in die Aktivierungsfunktion
            output_outputs = self.activationfunction[self.function](output_inputs)
        
            return output_outputs
        
        if self.hidden_layers == 5:
            #Versteckte Ausgaben berechnen analog wie bei hidden_layers=1
            hidden_1_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_1_input, inputs))
            hidden_2_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_2_1, hidden_1_outputs))
            hidden_3_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_3_2, hidden_2_outputs))
            hidden_4_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_4_3, hidden_3_outputs))
            hidden_5_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_5_4, hidden_4_outputs))
            output_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_output, hidden_5_outputs))

            return output_outputs
            
    def testnetwork(self, testdatalist):
    #Performance Richtige, Versuche
        Right = 0
        Tries = 0
    #data nehmen
        test_data_list = testdatalist
        for i in range(len(test_data_list)):
            data = test_data_list[i]
        #data in matrix umwandeln und normieren auf 1
            inputs = (np.asfarray(data[1:]) / 255.0)
            rightnumber = int(data[0])
        #Zeil kreieren
            outputs = self.asknetwork(inputs)
            Zahl = np.argmax(outputs)
        
            if(Zahl==rightnumber):
                Right += 1
                Tries += 1
            else:
                Tries +=1
                
     
    #performance ausgeben
        performance = Right/Tries
        print("Performance =", performance)
        return performance
        

#trainingsdaten einlesen und trennen 
    def trainnetwork(self):
        bestperformance = 0
        ite_without_imp = 0
        iterations = 0
         #datei öffnen trainingsdaten
        training_data_list = readdata("Trainingsdaten/train-images.idx3-ubyte", "Trainingsdaten/train-labels.idx1-ubyte", 60000)
        #datei öffnen testdaten
        test_data_list = readdata("Trainingsdaten/t10k-images.idx3-ubyte", "Trainingsdaten/t10k-labels.idx1-ubyte", 10000)
        while ite_without_imp < 8:
            start = time.time()
            for i in range(len(training_data_list)):
                data = training_data_list[i]
                #data in matrix umwandeln und normieren auf 1
                inputs = (np.asfarray(data[1:]) / 255.0)
                #Ziel kreieren
                targets = np.zeros(numberof_output_neurons) 
                #Beim Ziel muss die richtige Zahl wert 1 haben. richtige Zahl steht immer vorne
                targets[int(data[0])] = 1
                self.train(inputs, targets)
            performance = self.testnetwork(test_data_list)  
            if performance > bestperformance:
                #Format npy [gewichte1, gewichte2, gewichte3,...]
                if self.hidden_layers == 0:
                    best_weight = self.weight_hidden_output
                    np.save("bestweight_1hiddenlayer.npy", best_weight)
                if self.hidden_layers == 1:
                    best_weight = np.array([self.weight_hidden_1_input, self.weight_hidden_output])
                    np.save("bestweight_1hiddenlayer.npy", best_weight)
                if self.hidden_layers == 5:
                    best_weight = np.array([self.weight_hidden_1_input, self.weight_hidden_2_1, self.weight_hidden_3_2, self.weight_hidden_4_3, self.weight_hidden_5_4, self.weight_hidden_output])
                    np.save("bestweight_5hiddenlayer.npy", best_weight)
                ite_without_imp = 0
                #beste Gewichte     
                #bestperformance neu setzen
                bestperformance = performance
            else:
                ite_without_imp = ite_without_imp + 1
            iterations += 1
            end = time.time()
            print("Durchläufe ohne Verbesserung:",ite_without_imp)
            print("Durchläufe:", iterations)
            print("Zeit in Minuten:", (end-start)/60)
            print("bestperformance:", bestperformance)

        


#Datei lesen
def readdata(imgf, labelf, n):
    #In binary Modus lesen von https://pjreddie.com/projects/mnist-in-csv/
    images = open(imgf, "rb")
    label = open(labelf, "rb")
    #Erste 16 bezieungsweise 8 bytes überspringen, da keine Daten drin sind. read() funktion springt immer auf das Nächste.
    images.read(16)
    label.read(8)
    imagedata = []
    #Ganze Datei durchgehen, n = anzahl bilder
    for i in range(n):
        #lesen und zwischenspeichern
        image = [ord(label.read(1))]
        for j in range(28*28):
            image.append(ord(images.read(1)))
        #bild in Bilddata einfügen
        imagedata.append(image)
    return imagedata

#Neuronales Netzwerk erstellen.
if __name__ == "__main__":
    n = neuralNetwork(numberof_input_neurons, numberof_hidden_neurons, numberof_output_neurons, learningrate, numberof_hidden_layers, activation_function)
    n.trainnetwork()