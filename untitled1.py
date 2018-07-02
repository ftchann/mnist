# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:06:19 2018

@author: Yann
"""

import numpy as np
shape = (3,9)
np.random.seed(1)
b=2 * np.random.random((1, shape[1])) - 1
c=2 * np.random.random(shape) - 1
a= np.vstack((c, b))
