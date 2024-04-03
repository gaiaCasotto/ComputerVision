import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.transform import hough_line

#part 7.1
im = cv2.imread("week06_data/Box3.bmp")
plt.imshow(im)
plt.show()

edges = cv2.Canny(im, threshold1=70, threshold2=200) 
plt.imshow(edges)
plt.show()

#part 7.2
hspace, angles, dists = hough_line(edges)
print("Dimensions of Hough space:", hspace.shape)