# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:06:19 2018

@author: Yann
"""

import numpy as np

a=np.arange(4)
a=a.reshape(2,2)  

print(np.pad(a, 2,'constant', constant_values=(0)))
 