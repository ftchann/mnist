import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage.io
from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive

img_array = skimage.io.imread('resized_image.png', flatten=True)
print(img_array*255)

image = img_array*255
print(image)
global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 35
binary_adaptive = threshold_adaptive(image, block_size, offset=10)

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

for ax in axes:
    ax.axis('off')

plt.show()
