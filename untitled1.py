# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:06:19 2018

@author: Yann
"""

import numpy as np

a=np.arange(10)

a=a.T

b=np.arange(200)
b=np.reshape(b,(20,10))
c=np.dot(b,a)

print(a/2)

