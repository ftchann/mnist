# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:16:22 2018
@author: Yann
"""
import numpy as np
import time
import sys
import xlwt


#Anzahl von Eingabe, Versteckten und Ausgabeneuronen definieren
#Für Mnist muss input_neurons = 784 und output_neurons = 10 sein.
numberof_input_neurons = 784
numberof_hidden_neurons = 250
numberof_output_neurons = 10
#Anzahl versteckte Layers definieren
numberof_hidden_layers = 1
#learningrate definieren
learningrate = 0.1
#Aktivierungsfunktion definieren (zur Auswahl stehen sigmoid, tanh, relu und lrelu (leaky ReLu)) #In die Outlayer kommt immer Sigmoid, tanh könnte auch verwendet werden. ReLu und LReLu hingegen nicht.
activation_function = 'sigmoid'
#Verzerrung (Bias ein- und ausschalten)
bias = True

#Neuronales Netzwerk definieren
class neuralNetwork:
    #Aktivierungsfunktionen:
	#Sigmoid Funktion
    def sigmoid(self, x): 
        return 1 / (1 + np.exp(-x))
    #Ableitung SigmoidFunktion
    def sigmoid_derivative(self, x):
        return x*(1-x)
    #Tangenshyperbolicus funktion
    def tanh(self, x):
        #Es könnte auch np.tanh(x) verwendet werden
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
    #Ableitung leaky ReLu
    def lrelu_derivative(self, x):
        return 1 * (x > 0) + (x <= 0) * 0.01
    #neuronales Netzwerk inistialisieren
    def __init__(self, numberof_input_neurons, numberof_hidden_neurons, numberof_output_neurons, learningrate, numberof_hidden_layers, activation_function,bias):
        np.random.seed(1)#Seed  für random funktion festlegen (damit gibt die np.random bei jedem Durchgang die gleichen Zahlen aus)
        #In den folgenden Layers werden die Parameter der Definition weitergegeben und die nötigen Variablen erstellt
        self.bias=bias#Bias 
        self.lr = learningrate # Lernrate
        self.function = activation_function#Aktivierungsfunktion
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
        #Gewichtungsmatrixen und Bias definieren
        #Grösse der Verzerrung gleich der Anzahl Neuronen in der Schicht.
        #Grösse der Gewichtungsmatrix ist hintere Layer mal vordere Layer.
        #Für die Gewichtungsmatrixen gibt man am Anfang Zufallszahlen. Diese sind 0 +- hiddnennodes hoch -0.5 
        #Verzerrungsmatrix hat ebenfalls am Anfang Zufalls zahlen. Diese sind 0 +- hiddnennodes hoch -0.5 
        #Gewichte Versteckte-Outputlayer
        if self.hidden_layers == 0:
            self.weight_hidden_output = np.random.normal(0.0,pow(self.output_neurons, -0.5),(self.output_neurons, self.input_neurons))
            if self.bias ==True:
                self.weights_hidden_output_bias = np.random.normal(0.0,pow(self.output_neurons, -0.5),(self.output_neurons, 1))
        if self.hidden_layers > 0:
            self.weight_hidden_output = np.random.normal(0.0,pow(self.output_neurons, -0.5),(self.output_neurons, self.hidden_neurons_5))
            if self.bias ==True:
                self.weights_hidden_output_bias = np.random.normal(0.0,pow(self.output_neurons, -0.5),(self.output_neurons, 1))
        #Gewichte Hiddenlayer 1 - 5
        #Verzerrung Hiddenlayer 1 - 5
        #Im Moment haben alle Verstecktenlayers die selbe Anzahl Neuronen numberof_hidden_neurons * numberof_hidden_layers
        if self.hidden_layers >= 1:
            self.weight_hidden_1_input = np.random.normal(0.0,pow(self.hidden_neurons_1, -0.5),(self.hidden_neurons_1, self.input_neurons))
            if self.bias ==True:
                self.weights_hidden_1_input_bias = np.random.normal(0.0,pow(self.hidden_neurons_1, -0.5),(self.hidden_neurons_1, 1))
        if self.hidden_layers >= 2:
            self.weight_hidden_2_1 = np.random.normal(0.0,pow(self.hidden_neurons_2, -0.5),(self.hidden_neurons_2, self.hidden_neurons_1))
            if self.bias ==True:
                self.weights_hidden_2_1_bias = np.random.normal(0.0,pow(self.hidden_neurons_2, -0.5),(self.hidden_neurons_2, 1))
        if self.hidden_layers >= 3:
            self.weight_hidden_3_2 = np.random.normal(0.0,pow(self.hidden_neurons_3, -0.5),(self.hidden_neurons_3, self.hidden_neurons_2))
            if self.bias ==True:
                self.weights_hidden_3_2_bias = np.random.normal(0.0,pow(self.hidden_neurons_3, -0.5),(self.hidden_neurons_3, 1))
        if self.hidden_layers >= 4:
            self.weight_hidden_4_3 = np.random.normal(0.0,pow(self.hidden_neurons_4, -0.5),(self.hidden_neurons_4, self.hidden_neurons_3))
            if self.bias ==True:
                self.weights_hidden_4_3_bias = np.random.normal(0.0,pow(self.hidden_neurons_4, -0.5),(self.hidden_neurons_4, 1))
        if self.hidden_layers >= 5:
            self.weight_hidden_5_4 = np.random.normal(0.0,pow(self.hidden_neurons_5, -0.5),(self.hidden_neurons_5, self.hidden_neurons_4))
            if self.bias ==True:
                self.weights_hidden_5_4_bias = np.random.normal(0.0,pow(self.hidden_neurons_1, -0.5),(self.hidden_neurons_5, 1))
        #Aktivierungsfunktions dictionary
        self.activationfunction={'sigmoid':self.sigmoid, 'relu':self.relu, 'tanh':self.tanh, 'lrelu':self.lrelu}
        #dictionary der Ableitung der Aktivierungsfunktions
        self.activationfunction_derivative={'sigmoid':self.sigmoid_derivative, 'relu':self.relu_derivative, 'tanh':self.tanh_derivative, 'lrelu':self.lrelu_derivative}
    
    #macht die forward Propagation mit einer input Liste
    def forwardprop(self, inputs_list):
        #Inputsliste nehmen und transformieren damit sie hoch steht
        inputs = np.array(inputs_list, ndmin=2).T
        
        if self.hidden_layers == 0:
            #Analog zu numberof_hidden_layers = 1 einfach ohne die versteckten Komponenten.
            #Dies ist nur ein Experiment welche genauigkeit sich mit keinen hiddenn Layers erzielen lässt.
            #Wenn es einen Verzerrung hat wird er addiert.
            if self.bias == True:
                 output_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_output, inputs)+self.weights_hidden_output_bias)
            else:
                output_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_output, inputs))
            return output_outputs
            
        elif self.hidden_layers == 1:
            #Wenn es einen Verzerrung hat wird er addiert.
            if self.bias == True:
                hidden_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_1_input, inputs)+self.weights_hidden_1_input_bias)
                output_outputs = self.activationfunction['sigmoid'](np.dot(self.weight_hidden_output, hidden_outputs)+self.weights_hidden_output_bias)
            else:
                #Eingabe mal Gewicht und dann das ganze in die Aktivierungsfunktion
                hidden_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_1_input, inputs))
                #Die alten Ausgaben (neue Eingaben) mal Gewichtsmatrix und dann das ganze in die Aktivierungsfunktion
                output_outputs = self.activationfunction['sigmoid'](np.dot(self.weight_hidden_output, hidden_outputs))
            return output_outputs, hidden_outputs
               
        elif self.hidden_layers == 5:
            #Wenn es einen Verzerrung hat wird er addiert.
            if bias == True:
            #Analog zu hidden_layers==1
                hidden_1_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_1_input, inputs)+self.weights_hidden_1_input_bias)
                #hiddens Layer 2  
                hidden_2_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_2_1, hidden_1_outputs)+self.weights_hidden_2_1_bias)
                #hiddens Layer 3
                hidden_3_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_3_2, hidden_2_outputs)+self.weights_hidden_3_2_bias)
                #hiddens Layer 4
                hidden_4_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_4_3, hidden_3_outputs)+self.weights_hidden_4_3_bias)
                #hiddens Layer 5
                hidden_5_outputs = self.activationfunction[self.function](np.dot(self.weight_hidden_5_4, hidden_4_outputs)+self.weights_hidden_5_4_bias)
                #output Layer
                output_outputs = self.activationfunction['sigmoid'](np.dot(self.weight_hidden_output, hidden_5_outputs)+self.weights_hidden_output_bias)
            else:
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
                output_outputs = self.activationfunction['sigmoid'](np.dot(self.weight_hidden_output, hidden_5_outputs))
            return output_outputs, hidden_1_outputs, hidden_2_outputs, hidden_3_outputs, hidden_4_outputs, hidden_5_outputs
        else:
            #Wenn eine nicht vorgesehene Anzahl hidden Layers gesetzt wird, beendet sich das Program  
            sys.exit("Error: Anzahl hidden Layers ungültig")
            
        
    def backprop(self, inputs_list, targets_list):
        #analog zum importieren der input Liste 
        targets = np.array(targets_list, ndmin=2).T
        inputs = np.array(inputs_list, ndmin=2).T
        if self.hidden_layers == 0:
            output_outputs = self.forwardprop(inputs_list)
            #Die abweichung
            output_error = targets - output_outputs
            self.weight_hidden_output += self.lr * np.dot(output_error * self.activationfunction_derivative[self.function](output_outputs), inputs.T)
            #Backpropagation von der Verzerrung
            if bias == True:
                self.weights_hidden_output_bias += self.lr * output_error * self.activationfunction_derivative[self.function](output_outputs)
        if self.hidden_layers == 1:
            output_outputs, hidden_outputs = self.forwardprop(inputs_list)
            #ausgabefehler (Ziel-Ausgabe)
            output_error = (targets - output_outputs) * self.activationfunction_derivative['sigmoid'](output_outputs)
            #verstecktefehler (Ausgabe mal Gewicht) Gewichtsmatrix umkehren da wir jetzt zurückrechnen
            hidden_error = np.dot(self.weight_hidden_output.T, output_error) * self.activationfunction_derivative[self.function](hidden_outputs)
            #Gewichte aktuallisieren Versteckt-Ausgabe
            self.weight_hidden_output += self.lr * np.dot(output_error, hidden_outputs.T)
            #Gewichte aktuallisieren Eingabe-Versteckt                            
            self.weight_hidden_1_input += self.lr * np.dot(hidden_error, inputs.T)
            #Backpropagation von der Verzerrung
            if bias == True:
                self.weights_hidden_output_bias += self.lr * output_error
                self.weights_hidden_1_input_bias += self.lr * hidden_error                                       
        if self.hidden_layers == 5:
            
            output_outputs, hidden_1_outputs, hidden_2_outputs, hidden_3_outputs, hidden_4_outputs, hidden_5_outputs = self.forwardprop(inputs_list)
            #analog zu Hiddenlayers = 1 berechnung des Fehlers
            output_error = (targets - output_outputs) * self.activationfunction_derivative['sigmoid'](output_outputs)
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
            #Backprogattion von der Verzerrung
            if bias == True:
                self.weights_hidden_output_bias += output_error
                self.weights_hidden_5_4_bias +=hidden_5_error
                self.weights_hidden_4_3_bias +=hidden_4_error
                self.weights_hidden_3_2_bias +=hidden_3_error
                self.weights_hidden_2_1_bias +=hidden_2_error
                self.weights_hidden_1_input_bias += hidden_1_error
   
    
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
            #Nimmt nur das erste Element aus dem return der Funktion forwardprop
            #Die Unterteilung zwischen 0 und 1+ Hidden Layers ist nötig,
            #weil das Programm sonst nicht das 0te Element bei nur einem return zu nehmen
            if self.hidden_layers == 0:
                outputs = self.forwardprop(inputs)
            else:
                outputs = self.forwardprop(inputs)[0]
            Zahl = np.argmax(outputs)
            #Vergleich errechnete und soll Zahl
            if(Zahl == rightnumber):
                Right += 1
                Tries += 1
            else:
                Tries += 1
                
     
    #performance ausgeben
        performance = Right/Tries
        print("Performance =", performance)
        return performance
        

#trainingsdaten einlesen und trennen 
    def trainnetwork(self):
        bestperformance = 0
        epoch_without_imp = 0
        epoch = 0
        line = 1
         #datei öffnen trainingsdaten
        training_data_list = readdata("Trainingsdaten/train-images.idx3-ubyte", "Trainingsdaten/train-labels.idx1-ubyte", 60000)
        #datei öffnen testdaten
        test_data_list = readdata("Trainingsdaten/t10k-images.idx3-ubyte", "Trainingsdaten/t10k-labels.idx1-ubyte", 10000)
        while (epoch_without_imp < 8 and line < 100):
            start = time.time()
            for i in range(len(training_data_list)):
                data = training_data_list[i]
                #data in matrix umwandeln und normieren auf 1
                inputs = (np.asfarray(data[1:]) / 255.0)
                #Ziel kreieren
                targets = np.zeros(numberof_output_neurons) 
                #Beim Ziel muss die richtige Zahl wert 1 haben. richtige Zahl steht immer vorne
                targets[int(data[0])] = 1
                self.backprop(inputs, targets)
            performance = self.testnetwork(test_data_list)  
            if performance > bestperformance:
                #Format npy [gewichte1, gewichte2, gewichte3,...]
                if self.hidden_layers == 0:
                    best_weight = self.weight_hidden_output
                    np.save("bestweight_0hiddenlayer.npy", best_weight)
                    if self.bias==True:
                        best_bias = self.weights_hidden_output_bias
                        np.save("bestbias_0hiddenlayer.npy",best_bias)
                if self.hidden_layers == 1:
                    best_weight = np.array([self.weight_hidden_1_input, self.weight_hidden_output])
                    np.save("bestweight_1hiddenlayer.npy", best_weight)
                    if self.bias==True:
                        best_bias = np.array([self.weights_hidden_1_input_bias, self.weights_hidden_output_bias])
                        np.save("bestbias_1hiddenlayer.npy",best_bias)
                if self.hidden_layers == 5:
                    best_weight = np.array([self.weight_hidden_1_input, self.weight_hidden_2_1, self.weight_hidden_3_2, self.weight_hidden_4_3, self.weight_hidden_5_4, self.weight_hidden_output])
                    np.save("bestweight_5hiddenlayer.npy", best_weight)
                    if self.bias==True:
                        best_bias = np.array([self.weights_hidden_1_input_bias, self.weights_hidden_2_1_bias,self.weights_hidden_3_2_bias, self.weights_hidden_4_3_bias,self.weights_hidden_5_4_bias, self.weights_hidden_output_bias])
                        np.save("bestbias_5hiddenlayer.npy",best_bias)
                epoch_without_imp = 0
                #beste Gewichte     
                #bestperformance neu setzen
                bestperformance = performance
            else:
                epoch_without_imp = epoch_without_imp + 1
            epoch += 1
            end = time.time()
            print("Durchläufe ohne Verbesserung:",epoch_without_imp)
            print("Durchläufe:", epoch)
            print("Zeit in Sekunden:", (end-start))
            print("bestperformance:", bestperformance)
            sheet1.write(line, 0, (end-start))
            sheet1.write(line, 1, performance)
            line += 1
            wb.save('tempergebnisse.xls')
        


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
    wb = xlwt.Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'Zeit')
    sheet1.write(0, 1, 'Genauigkeit')
    n = neuralNetwork(numberof_input_neurons, numberof_hidden_neurons, numberof_output_neurons, learningrate, numberof_hidden_layers, activation_function,bias)
    n.trainnetwork()
