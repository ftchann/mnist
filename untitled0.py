# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:15:36 2018

@author: Yann
"""

import numpy as np
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))
def relu(x):
    return x * (x > 0)
activationfunctionList={'sigmoid':sigmoid, 'relu':relu}
activationfunction = 'relu'
print(activationfunctionList[function](5))