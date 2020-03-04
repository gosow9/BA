# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('test_schraube.png',0)
kernel = np.ones((3,3),np.uint8)

img = cv.equalizeHist(img) 
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret3,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.hist(blur.ravel(),256,[0,256]); plt.show()
opening = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel)
th1 = th3-th2
plt.figure()
plt.imshow(blur)
plt.figure("opening")
plt.imshow(opening)
plt.figure()
# plt.imshow(th2)
# plt.figure()

plt.hist(img.ravel(),256,[0,256]); plt.show()



# image = cv2.imread('test_schraube.png')

# hist = cv2.calcHist([image],[0],None,[256],[0,256])
# cv2.imshow('img', image)
# #cv2.imshow('img', hist)
# cv2.waitKey(0)
