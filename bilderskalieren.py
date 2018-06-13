# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:38:48 2018

@author: Yann
"""

import PIL
from PIL import Image
baseheight = 28
basewidth  = 28

img = Image.open('1.jpg')
w, h = img.size
print(img.size)
if w >= h:
    x = (w-h)/2
    img = img.crop((x, 0, w-x, h))
    img = img.convert('1')
    img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS).save(' resized_image.png')
else:
    x=(h-w)/2
    img = img.crop((0, x, w, h-x))
    img = img.convert('1')
    img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS).save(' resized_image.png')# convert image to black and white
#img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
#img.save('resized_image.jpg')