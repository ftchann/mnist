# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:15:36 2018

@author: Yann
"""

import numpy as np
def relu(x):
    return x * (x > 0)
def relu_derivative(x):
    return 1 * (x > 0)
a=np.arange(-5,5)
print(a)
print(relu(a))
print(relu_derivative(a))