# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 21:56:01 2018

@author: Yann
"""
import numpy as np
A=np.arange(5)
B=np.arange(5)
B[2]=5
C=np.array_equal(A,B)