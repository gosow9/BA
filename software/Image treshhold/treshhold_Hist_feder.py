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

def loadImage():
    #images = [cv.imread(file,0) for file in glob.glob("Backlight/Full*.png")]
    images = [cv.imread("Backlight/Full{}.png".format(file),0) for file in range(0,100)]
    return images
    
images = loadImage()
imag = [cv.cvtColor(file, cv.COLOR_BGR2RGB) for file in images]

img = cv.imread('BacklightFeder/Full80.png',0)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#dim = (1280, 720) 
#img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
#imag = cv.cvtColor(img, cv.COLOR_BGR2RGB)
mean = []
for i in imag:
    mean.append(np.mean(i))
    

im = imag[50]
# for i in range(30,40):
#     im = im + imag[i]
# im= im / 10
print(np.mean(im))    

plt.figure("mean")
plt.plot(mean) 
#img = images[0]
#kernel = np.ones((3,3),np.uint8)
#retval, treshhold = cv.threshold(img,114,255, cv.THRESH_BINARY)

#for nr in range(90, 99, 5):
plt.figure("original{}".format(50))
plt.subplot(211)
#retval, treshhold = cv.threshold(imag,114,255, cv.THRESH_BINARY)
plt.imshow(imag[50])
#plt.hist(imag.ravel(),256,[0,256])
plt.subplot(212)
plt.hist(imag[50].ravel(), bins=256, range=(0, 256), fc='k', ec='k')

sub = im-img 
plt.figure("subtraction")
plt.subplot(211)
plt.imshow(sub)
plt.subplot(212)
plt.hist(sub.ravel(), bins=256, range=(0, 256), fc='k', ec='k')



plt.figure("meanBackground")
plt.subplot(211)
plt.imshow(im)
plt.subplot(212)
plt.hist(im.ravel(), bins=256, range=(0, 256), fc='k', ec='k')

plt.figure("line")
plt.plot(img[1200])
# ims = []
# for add in range(0,50,1):
#     ims.append((plt.hist(imag[add].ravel(), bins=256, range=(0, 256), fc='k', ec='k')))

# im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=10,
#                                     blit=True)
# # To save this second animation with some metadata, use the following command:
# im_ani.save('im.mp4')

# plt.show()


    #plt.figure("treshhold{}".format(nr))
    #plt.imshow(treshhold)
# im = cv.equalizeHist(im) 
# blur = cv.GaussianBlur(img,(5,5),3)
# th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#             cv.THRESH_BINARY,11,2)
# ret3,th3 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#histogram equalisation
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(50,50))
# cl1 = clahe.apply(img)
# equ = cv.equalizeHist(img)

#plotting everthing
# plt.figure("otsu")
# plt.imshow(th3)
# plt.figure("mean")
# plt.imshow(th2)
# plt.figure("hist")
# plt.hist(img.ravel(),256,[0,256]) 
# plt.figure("histeq")
# plt.imshow(cl1) 
# plt.figure("histeqhist")
# plt.hist(cl1.ravel(),256,[0,256]) 

# plt.figure("eq")
# plt.imshow(equ) 
# plt.figure("eqhist")
# plt.hist(equ.ravel(),256,[0,256]) 
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
