# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:16:22 2018

@author: Yann
"""
import numpy as np
import scipy
import time

#Inputs, Hidden, Outputsnodes , bias 

eingabeneuronen = 784
versteckteneuronen = 500
ausgabeneuronen = 10
verstecktelayers = 2
#learnrate 
learnrate = 0.05

#Neuronales Netzwerk definieren
class neuronalesNetzwerk:
    #neuronales Netzwerk inistialisieren
    def __init__(self, eingabeneuronen, verstecketeneuronen, ausgabeneuronen, learnrate, verstecktelayers):
        self.eneuron = eingabeneuronen
        self.vneuron = verstecketeneuronen
        self.aneuron= ausgabeneuronen
        self.vlayer = verstecktelayers
        #Gewichtungsmatrixen definieren
        #Grösse der Gewichtungsmatrix ist bei Geingave_versteckt versteckteneuron mal eingabeneuron und bei Gversteckt_ausgabe ausgabeneuron mal verstecktenodes.
        #Für die Gewichtungsmatrixen gibt man am Anfang Zufallszahlen. Anfangszahlen zwischen +- hiddnennodes hoch -0.5 
        #Gewichte Verstekt Ausgabe
        self.ge_va = np.random.normal(0.0,pow(self.aneuron, -0.5),(self.aneuron, self.vneuron))
        #Gewichte Hiddenlayer 1 - 5
        #Im Moment haben alle Hiddenlayers die selbe Anzahl Neuronen
        self.ge_v1 = np.random.normal(0.0,pow(self.vneuron, -0.5),(self.vneuron, self.eneuron))
        self.ge_v2 = np.random.normal(0.0,pow(self.vneuron, -0.5),(self.vneuron, self.eneuron))
        self.ge_v3 = np.random.normal(0.0,pow(self.vneuron, -0.5),(self.vneuron, self.eneuron))
        self.ge_v4 = np.random.normal(0.0,pow(self.vneuron, -0.5),(self.vneuron, self.eneuron))
        self.ge_v5 = np.random.normal(0.0,pow(self.vneuron, -0.5),(self.vneuron, self.eneuron))
        #Learnrate
        self.lr = learnrate
        #Sigmoid
        def sigmoid(x): 
            return scipy.special.expit(x)
        #Aktivierungsfunktion
        self.aktivierungsfunktion = sigmoid
        
        pass
    #neuronales Netzwerk tranieren
    def trainieren(self, inputs_list, ziel_liste,):
        #Inputsliste nehmen und transformieren damit sie hoch steht
        inputs = np.array(inputs_list, ndmin=2).T
        #Das Gleiche mit der Zielliste
        ziele = np.array(ziel_liste, ndmin=2).T
        
        if self.vlayer == 0:
            #Analog zu verstecktelayers = 1 einfach ohne die versteckte Komponenten
            #Dies ist nur ein Experiment welche genauigkeit sich mit keinen versteckten Layers erzielen lässt
            ausgabe_inputs = np.dot(self.ge_va, inputs)
            ausgabe_outputs = self.aktivierungsfunktion(ausgabe_inputs)
            
            ausgabe_fehler = ziele - ausgabe_outputs
            self.ge_va += self.lr * np.dot(ausgabe_fehler * ausgabe_outputs * (1-ausgabe_outputs), inputs.T)
            pass
        
        if self.vlayer == 1: 
            #Inputs mal Gewicht
            versteckte_inputs = np.dot(self.ge_v1, inputs)
            #Das ganze in die Aktivierungsfunktion
            versteckte_outputs = self.aktivierungsfunktion(versteckte_inputs)
            #versteckte_outputs (neues inputs) mal Gewicht
            ausgabe_inputs = np.dot(self.ge_va, versteckte_outputs)
            #Das ganze in die Aktivierungsfunktion
            ausgabe_outputs = self.aktivierungsfunktion(ausgabe_inputs)    
        
           
            #ausgabefehler (Ziel-Output)
            ausgabe_fehler = ziele - ausgabe_outputs
            #verstecktefehler (Output mal Gewicht) Gewichtsmatrix umkehren da wir jetzt zurückrechnen
            versteckte_fehler = np.dot(self.ge_va.T, ausgabe_fehler)
        
            #Gewichte aktuallisieren hidden output
            self.ge_va += self.lr * np.dot(ausgabe_fehler * ausgabe_outputs * (1-ausgabe_outputs), versteckte_outputs.T)
            #Gewichte aktuallisieren hidden und output                            
            self.ge_v1 += self.lr * np.dot(versteckte_fehler * versteckte_outputs * (1-versteckte_outputs), inputs.T)                     
            pass
        
        if self.vlayer == 5:
            #Verstecktes Layer 1
            versteckte_1_inputs = np.dot(self.ge_v1, inputs)
            versteckte_1_outputs = self.aktivierungsfunktion(versteckte_1_inputs)
            #Verstecktes Layer 2
            versteckte_2_inputs = np.dot(self.ge_v2, versteckte_1_outputs)
            versteckte_2_outputs = self.aktivierungsfunktion(versteckte_2_inputs)
            #Verstecktes Layer 3
            versteckte_3_inputs = np.dot(self.ge_v3, versteckte_2_outputs)
            versteckte_3_outputs = self.aktivierungsfunktion(versteckte_3_inputs)
            #Verstecktes Layer 4
            versteckte_4_inputs = np.dot(self.ge_v4, versteckte_3_outputs)
            versteckte_4_outputs = self.aktivierungsfunktion(versteckte_4_inputs)
            #Verstecktes Layer 5
            versteckte_5_inputs = np.dot(self.ge_v5, versteckte_4_outputs)
            versteckte_5_outputs = self.aktivierungsfunktion(versteckte_5_inputs)  
            #Ausgabe Layer
            ausgabe_inputs = np.dot(self.ge_va, versteckte_5_outputs)
            ausgabe_outputs = self.aktivierungsfunktion(versteckte_1_inputs)
            
            ausgabe_fehler = ziele - ausgabe_outputs
            versteckte_5_fehler = np.dot(self.ge_va.T, ausgabe_fehler)
            versteckte_4_fehler = np.dot(self.ge_va.T, ausgabe_fehler)
            versteckte_3_fehler = np.dot(self.ge_va.T, ausgabe_fehler)
            versteckte_2_fehler = np.dot(self.ge_va.T, ausgabe_fehler)
            versteckte_1_fehler = np.dot(self.ge_va.T, ausgabe_fehler)
            #Gewichte aktualisieren ge_va (Output)
            self.ge_va += self.lr * np.dot(ausgabe_fehler * ausgabe_outputs * (1-ausgabe_outputs), versteckte_5_outputs.T)
            #Gewichte aktualisieren ge_va und ge_v5 (Output und verstecktes Layer 5)
            self.ge_v5 += self.lr * np.dot(versteckte_5_fehler * versteckte_5_outputs * (1-versteckte_5_outputs), versteckte_4_outputs.T) 
            #Gewichte aktualisieren ge_va, ge_v5 und ge_v4 (Output und verstecktes Layer 5 und 4)
            self.ge_v4 += self.lr * np.dot(versteckte_4_fehler * versteckte_4_outputs * (1-versteckte_4_outputs), versteckte_3_outputs.T)
            #Gewichte aktualisieren ge_va, ge_v5, ge_v4 und ge_v3 (Output und verstecktes Layer 5)
            self.ge_v3 += self.lr * np.dot(versteckte_3_fehler * versteckte_3_outputs * (1-versteckte_3_outputs), versteckte_2_outputs.T)
            #Gewichte aktualisieren ge_va und ge_v5 (Output und verstecktes Layer 5)
            self.ge_v2 += self.lr * np.dot(versteckte_2_fehler * versteckte_2_outputs * (1-versteckte_2_outputs), versteckte_1_outputs.T)
            #Gewichte aktualisieren ge_va und ge_v5 (Output und verstecktes Layer 5)
            self.ge_v1 += self.lr * np.dot(versteckte_1_fehler * versteckte_1_outputs * (1-versteckte_1_outputs), inputs.T)
            pass
        
        else:
            print ("Error: Anzahl versteckter Layers ungültig")
            
    
            
            
            
            
    #neuronales Netzwerk abfragen
    def abfragen(self, inputs_list):
        #Inputsliste nehmen und transformieren damit sie hoch steht
        inputs = np.array(inputs_list, ndmin=2).T
        #Inputs mal Gewicht
        versteckte_inputs = np.dot(self.ge_v1, inputs)
        #Das ganze in die Aktivierungsfunktion
        versteckte_outputs = self.aktivierungsfunktion(versteckte_inputs)
        #versteckte_outputs (neues inputs) mal Gewicht
        ausgabe_inputs = np.dot(self.ge_va, versteckte_outputs)
        #Das ganze in die Aktivierungsfunktion
        ausgabe_outputs = self.aktivierungsfunktion(ausgabe_inputs)
        
        return ausgabe_outputs
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
                best_ge = np.array([self.ge_v1, self.ge_va])
                np.save("gewicht.npy", best_ge)
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
            print("Zeit:", end-start)
            print("bestperformance:", bestperformance)
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




