# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 22:37:07 2018

@author: Yann_
"""
from skimage import data
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp_coords
def shift_up10_left20(xy):
    return xy - np.array([-20, 10])[None, :]

image = data.astronaut().astype(np.float32)
coords = warp_coords(shift_up10_left20, image.shape)
warped_image = map_coordinates(image, coords)
print(warped_image)