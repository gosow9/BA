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


img1 = cv.imread('saved2.png',0)
img2 = cv.imread('saved5.png',0)
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

mean1 = np.mean(img1)
var1 = np.var(img1)
var2 = np.var(img2)
mean2 = np.mean(img2)



plt.figure(figsize=(7,4))
plt.yticks([])
plt.title("mean = {:0.2f}, variance = {:0.2f}".format(mean1, var1))

plt.hist(img1.ravel(), bins=256, range=(0, 256),color="blue")
plt.savefig("hist_feder2.pdf",bbox_inches="tight")


plt.figure(figsize=(7,4))
plt.yticks([])
plt.title("mean= {:0.2f}, variance = {:0.2f}".format(mean2, var2))

plt.hist(img2.ravel(), bins=256, range=(0, 256),color="blue")
plt.savefig("hist_pattern2.pdf",bbox_inches="tight")
