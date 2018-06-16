# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 15:26:11 2018

@author: Florian
"""

import numpy as np


print(np.indices((3,3)))
x = np.zeros((3,3))
print(x)
def t():
    return 2,10

#print (t()[1])
x, y =t()
print (x ,y)
#def schwerpunkt():
#    print(np.indices(3,3))
