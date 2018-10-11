# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 22:31:40 2018

@author: Florian
"""
import numpy as np
import xlwt

a = np.random.rand(3,1)
print(a)
print(np.shape(a))
def re():
    np.random.seed(1)
    a = np.random.normal(0.0, 1, (2,2))
    b = np.random.rand(2, 2)
    return a, b
    
