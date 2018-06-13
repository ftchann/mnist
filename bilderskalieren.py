# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:38:48 2018

@author: Yann
"""

import PIL
from PIL import Image
baseheight = 300
basewidth  = 300

img = Image.open('grosserpanda.jpg')
w, h = img.size
print(img.size)
if w >= h:
    x = (w-h)/2
    img.crop((x, 0, w-x, h)).save('resized_image.jpg')
else:
    x=(h-w)/2
    img.crop((0, x, w, h-x)).save('resized_image.jpg')
#img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
#img.save('resized_image.jpg')