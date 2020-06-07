# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import cv2 as cv
import os
import glob
from matplotlib import pyplot as plt
import matplotlib.animation as animation


img1 = cv.imread('im1.png')
img2 = cv.imread('im0.png')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

mean1 = np.mean(img1)
var1 = np.var(img1)
var2 = np.var(img2)
mean2 = np.mean(img2)



#plt.figure(figsize=(7,4))
#plt.yticks([])
#plt.title("mean = {:0.2f}, variance = {:0.2f}".format(mean1, var1))

#plt.hist(img1.ravel(), bins=256, range=(0, 256),color="blue")
#plt.savefig("hist_feder2.pdf",bbox_inches="tight")


#plt.figure(figsize=(7,4))
#plt.yticks([])
#plt.title("mean= {:0.2f}, variance = {:0.2f}".format(mean2, var2))

#plt.hist(img2.ravel(), bins=256, range=(0, 256),color="blue")
#plt.savefig("hist_pattern2.pdf",bbox_inches="tight")
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
ret3, thresh = cv.threshold(img1, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
erode = cv.erode(thresh, kernel, iterations=1)
im = cv.subtract(thresh, erode)
#plt.figure()
#plt.imshow(thresh)
cv.imwrite("threshold.png", thresh)
cv.imwrite("edge.png", im)
#plt.figure()
#plt.imshow(im)
ret2,th2 = cv.threshold(img1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imwrite("otsu.png", th2)
print(ret3)
diff = th2-thresh
cv.imwrite("diff.png", diff)