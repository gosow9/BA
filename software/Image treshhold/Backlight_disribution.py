# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2 as cv
import os
import glob
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import scipy.misc


#images = [cv.imread(file,0) for file in glob.glob("BacklightFeder/*.png")]
#imag = [cv.cvtColor(file, cv.COLOR_BGR2RGB) for file in images]

img = cv.imread('b_2lines.png',0)
imag = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#img = images[0]
#kernel = np.ones((3,3),np.uint8)
#retval, treshhold = cv.threshold(img,114,255, cv.THRESH_BINARY)
plt.plot(img[400])
plt.plot(img[1400])
plt.plot(img[1])
#for nr in range(90, 99, 5):
#plt.figure("original{}".format(50))
#plt.subplot(211)
#retval, treshhold = cv.threshold(imag,114,255, cv.THRESH_BINARY)
plt.figure()
plt.imshow(img,cmap="gray")



kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
top = tophat
tophat = tophat[::100,::100]
#img = scipy.misc.imresize(img, 0.1, interp='nearest')

plt.figure()
plt.imshow(top,cmap="gray")

img = img[::100,::100]
background = np.ones(np.shape(img))
background = (background * np.max(img)) - img
imgnew = img + tophat*3

xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
xx = xx*10
yy = yy*10
# create the figure

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, imgnew ,rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0)
# plt.figure()
# plt.contour(xx, yy, imgnew , cmap=plt.cm.coolwarm, linewidth=0)


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, tophat ,rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0)
# plt.figure()
# plt.contour(xx, yy, tophat , cmap=plt.cm.coolwarm, linewidth=0)

# # show it
# plt.show()
#plt.hist(imag.ravel(),256,[0,256])
#plt.subplot(212)
#plt.hist(imag[50].ravel(), bins=256, range=(0, 256), fc='k', ec='k')


#fig2 = plt.figure("histos")
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
