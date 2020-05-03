# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def opening_by_reconstruction(img, erosion_kernel, dilation_kernel):
     im_eroded = cv.erode(img, erosion_kernel)
     im_opened = reconstruction_by_dilation(im_eroded, img, dilation_kernel)
     return im_opened


def reconstruction_by_dilation(img, mask, dilation_kernel):
     im_old = img.copy()
     while(1):
         img = cv.dilate(img, dilation_kernel)
         img = cv.bitwise_and(img, mask)
         if np.array_equal(im_old, img):
             break
         im_old = img.copy()
     return img


img = cv.imread('Bilder_threshholding/im39.png',0)
kernel = np.ones((3,3),np.uint8)
retval, treshhold = cv.threshold(img,114,255, cv.THRESH_BINARY)

plt.figure("original")
plt.imshow(img)
plt.figure("treshhold")
plt.imshow(treshhold)
# im = cv.equalizeHist(im) 
blur = cv.GaussianBlur(img,(5,5),3)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
ret3,th3 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#histogram equalisation
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(50,50))
cl1 = clahe.apply(img)
equ = cv.equalizeHist(img)

#plotting everthing
plt.figure("otsu")
plt.imshow(th3)
plt.figure("mean")
plt.imshow(th2)
plt.figure("hist")
plt.hist(img.ravel(),256,[0,256]) 
plt.figure("histeq")
plt.imshow(cl1) 
plt.figure("histeqhist")
plt.hist(cl1.ravel(),256,[0,256]) 

plt.figure("eq")
plt.imshow(equ) 
plt.figure("eqhist")
plt.hist(equ.ravel(),256,[0,256]) 
# ret3,th2 = cv.threshold(im,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plt.hist(blur.ravel(),256,[0,256]); plt.show()
# opening = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel)
# th1 = th3-th2

# plt.figure("Blur")
# plt.imshow(blur)
# plt.figure("opening")
# plt.imshow(opening)
# plt.figure()
# # plt.imshow(th2)
# # plt.figure()





# image = cv2.imread('test_schraube.png')

# hist = cv2.calcHist([image],[0],None,[256],[0,256])
# cv2.imshow('img', image)
# #cv2.imshow('img', hist)
# cv2.waitKey(0)
