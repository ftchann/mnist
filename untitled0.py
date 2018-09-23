# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:15:36 2018

@author: Yann
"""

import numpy as np
import skimage.io
import matplotlib.pyplot as plot
img_array2= skimage.io.imread('6.jpg', flatten=True)
print(img_array2)
plot.imshow(img_array2, cmap='gray')
