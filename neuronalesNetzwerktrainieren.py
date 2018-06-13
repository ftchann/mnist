# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 02:19:56 2018

@author: Yann
"""

import neuronalesNetzwerk as nk
 
eingabeneuronen = nk.eingabeneuronen
versteckteneuronen = nk.versteckteneuronen
ausgabeneuronen = nk.ausgabeneuronen
#learnrate 
learnrate = nk.learnrate

n = nk.neuronalesNetzwerk(eingabeneuronen, versteckteneuronen, ausgabeneuronen, learnrate)

n.trainieren2()