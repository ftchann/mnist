# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 19:39:07 2018

@author: Yann
"""

import cv2
import numpy as np

images = np.zeros((4,784))

gray = cv2.imread("2.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
gray = cv2.resize(255-gray, (28, 28))