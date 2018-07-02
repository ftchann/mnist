# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:06:19 2018

@author: Yann
"""

import numpy as np
def sigmoid(x):
     return 1 / (1 + np.exp(-x))
a=np.arange(10)

a=a.T

b=np.arange(20)


a=sigmoid(b)
print(a)