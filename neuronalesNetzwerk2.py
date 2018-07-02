 # -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:16:22 2018
@author: Yann
"""
import numpy as np
import time
import sys

#Inputs, Hidden, Outputsnodes

eingabeneuronen = 784
versteckteneuronen = 20
ausgabeneuronen = 10
#VersteckteLayers ändern

#learnrate 
learnrate = 0.1


#Neuronales Netzwerk definieren
class neuronalesNetzwerk:
    #neuronales Netzwerk inistialisieren
    def __init__(self, eingabeneuronen, verstecketeneuronen, ausgabeneuronen, learnrate):
        np.random.seed(1)
        self.eneuron = eingabeneuronen
        self.vneuron = verstecketeneuronen
        self.aneuron= ausgabeneuronen
        self.vneuron1 = 20
        self.vneuron2 = 40
        self.ge_va = 0
        self.ge_v5 = 0
        self.ge_v4 = 0
        self.ge_v3 = 0
        self.ge_v2 = 0
        self.ge_v1 = 0
        #Gewichtungsmatrixen definieren
        #Grösse der Gewichtungsmatrix ist bei Geingave_versteckt versteckteneuron mal eingabeneuron und bei Gversteckt_ausgabe ausgabeneuron mal verstecktenodes.
        #Für die Gewichtungsmatrixen gibt man am Anfang Zufallszahlen. Anfangszahlen zwischen +- hiddnennodes hoch -0.5 
        #Gewichte Hiddenlayer 1 - 5
        #Im Moment haben alle Hiddenlayers die selbe Anzahl Neuronen <- Stimmt nicht
        self.ge_v1 = np.random.normal(0.0,pow(self.vneuron1, -0.5),(self.vneuron1, self.eneuron))
        self.ge_v2 = np.random.normal(0.0,pow(self.vneuron2, -0.5),(self.vneuron2, self.vneuron1))
                #Gewichte Verstekt Ausgabe 
        self.ge_va = np.random.normal(0.0,pow(self.aneuron, -0.5),(self.aneuron, self.vneuron2))
        #Learnrate
        self.lr = learnrate
        #Sigmoid
    def sigmoid(self, x): 
        return 1 / (1 + np.exp(-x))
        #Aktivierungsfunktion ableitung
    def sigmoid_ableitung(self, x):
        return x*(1-x)
    #neuronales Netzwerk tranieren
    def trainieren(self, inputs_list, ziel_liste,):
        #Inputsliste nehmen und transformieren damit sie hoch steht
        inputs = np.array(inputs_list, ndmin=2).T
        #Das Gleiche mit der Zielliste
        ziele = np.array(ziel_liste, ndmin=2).T
        
        v1 = self.sigmoid(np.dot(self.ge_v1, inputs))
        v2 = self.sigmoid(np.dot(self.ge_v2, v1))
        output = self.sigmoid(np.dot(self.ge_va, v2))
            #Gewichte aktualisieren ge_va (Output)
        fehler_output = (ziele - output) * self.sigmoid_ableitung(output) 
        fehler_2 = np.dot(self.ge_va.T, fehler_output) * self.sigmoid_ableitung(v2)
        fehler_1 = np.dot(self.ge_v2.T, fehler_2) * self.sigmoid_ableitung(v1)
        anpassung_output = np.dot(fehler_output, v2.T)
        anpassung_2 = np.dot(fehler_2, v1.T)
        anpassung_1 = np.dot(fehler_1, inputs.T)
        self.ge_va += anpassung_output
        self.ge_v2 += anpassung_2
        self.ge_v1 += anpassung_1
    #neuronales Netzwerk abfragen
    def abfragen(self, inputs_list):
        #Inputsliste nehmen und transformieren damit sie hoch steht
        inputs = np.array(inputs_list, ndmin=2).T
        
        v1 = self.sigmoid(np.dot(self.ge_v1, inputs))
        v2 = self.sigmoid(np.dot(self.ge_v2, v1))
        output = self.sigmoid(np.dot(self.ge_va, v2))
           
        return output
    #abfragen
    def abfragen2(self, testdatenliste):
    #Performance
        anzahlRichtige = 0
        anzahlVersuche = 0
    #daten nehmen
        test_daten_liste = testdatenliste
        for i in range(len(test_daten_liste)):
            daten = test_daten_liste[i]
        #daten in matrix umwandeln und normieren auf 1
            inputs = (np.asfarray(daten[1:]) / 255.0 * 0.999) + 0.001
            richtigeZahl = int(daten[0])
        #Zeil kreieren
            outputs = self.abfragen(inputs)
            Zahl = np.argmax(outputs)
        
            if(Zahl==richtigeZahl):
                anzahlRichtige += 1
                anzahlVersuche += 1
            else:
                anzahlVersuche +=1
                pass
     
    #performance ausgeben
        performance = anzahlRichtige/anzahlVersuche
        print("Performance =", performance)
        return performance
        pass

#trainieren daten einlesen und trennen 
    def trainieren2(self):
        bestperformance = 0
        Durchlaufe = 0
        Durchläufe = 0
         #datei öffnen lesen1
        trainings_daten_liste = lesen("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 60000)
        #datei öffnen lesen1
        test_daten_liste = lesen("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000)
        while Durchlaufe < 8:
            start = time.time()
            for i in range(len(trainings_daten_liste)):
                daten = trainings_daten_liste[i]
                #daten in matrix umwandeln und normieren auf 1
                inputs = (np.asfarray(daten[1:]) / 255.0 * 0.999) + 0.001
                #Zeil kreieren
                ziele = np.zeros(ausgabeneuronen) + 0.001
                #Beim Ziel muss die richtige Zahl wert 1 haben. richtige Zahl steht in der Excel tabelle immer vorne
                ziele[int(daten[0])] = 1
                self.trainieren(inputs, ziele)
            performance = self.abfragen2(test_daten_liste)  
            if performance > bestperformance:
                #Format npy [gewichte1, gewichte2, gewichte3,...]
                best_ge = np.array([self.ge_v1, self.ge_v2, self.ge_v3, self.ge_v4, self.ge_v5, self.ge_va])
                np.save("bestgewicht.npy", best_ge)
                Durchlaufe = 0
                #beste Gewichte     
                #bestperformance neu setzen
                bestperformance = performance
            else:
                Durchlaufe = Durchlaufe + 1
            Durchläufe += 1
            end = time.time()
            print("Durchlaufe ohne Verbesserung:",Durchlaufe)
            print("Durchläufe:", Durchläufe)
            print("Zeit in Minuten:", (end-start)/60)
            print("bestperformance:", bestperformance)
#            print(best_ge)
        pass
    pass


#Datei lesen
def lesen(imgf, labelf, n):
    #In binary Modus lesen von https://pjreddie.com/projects/mnist-in-csv/
    bilder = open(imgf, "rb")
    label = open(labelf, "rb")
    #Erste 16 bezieungsweise 8 bytes überspringen, da keine Daten drin sind. Lesen funktion springt immer auf das Nächste.
    bilder.read(16)
    label.read(8)
    bilddaten = []
    #Ganze Datei durchgehen, n = anzahl bilder
    for i in range(n):
        #Lesen und zwischenspeichern
        bild = [ord(label.read(1))]
        for j in range(28*28):
            bild.append(ord(bilder.read(1)))
        #bild in Bilddaten einfügen
        bilddaten.append(bild)
    return bilddaten


if __name__ == "__main__":
    n = neuronalesNetzwerk(eingabeneuronen, versteckteneuronen, ausgabeneuronen, learnrate)
    n.trainieren2()