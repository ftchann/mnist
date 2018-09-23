# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:15:36 2018

@author: Yann
"""

import numpy as np
import matplotlib.pyplot as plot
np.random.seed(1)
a=(np.random.normal(0.0,pow(20, -0.5),(20,1)))
c=np.random.rand(20,1)
d=np.sort(c)
print(d)