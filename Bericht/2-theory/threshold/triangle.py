#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:58:37 2020

@author: crenda
"""



import numpy as np
import cv2 as cv
import os
import glob
from matplotlib import pyplot as plt
import matplotlib.animation as animation


img1 = cv.imread('saved2.png')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

mean1 = np.mean(img1)
var1 = np.var(img1)

h, p = np.histogram(img1, bins=np.arange(256))
print(h,p)
print(np.max(h))



plt.figure(figsize=(7,4))
plt.yticks([])


plt.hist(img1.ravel(), bins=256, range=(0, 256),color="tab:blue")
plt.plot([22,h.argmax()],[0,h.max()], color="tab:red")
plt.savefig("hist_feder2.pdf",bbox_inches="tight")

dist = 0
a = h.max()
b = 22 - h.argmax()
print(h.argmax())
thresh = 0
for i in range(1, h.argmax()):
    tempdist = a*i + b*h[i]
    if tempdist > dist:
        dist = tempdist
        thresh = i
thresh -= 1
print(thresh)

def f(x):
    x=x-22
    return 297181/122*x
y = 62
plt.plot([y,thresh],[f(y),h[thresh]], color= "magenta")
plt.plot([thresh,thresh],[0,h[thresh]],'--', color= "magenta")
plt.plot([thresh],[h[thresh]],'o', color= "magenta")
plt.plot([y],[f(y)],'o', color= "magenta")
plt.text(90,60000,"max. distance", color="magenta", size=12, bbox=dict(boxstyle="square",
                   ec=(1., 1, 1),
                   fc=(1, 1, 1),
                   ))

plt.title("thresh value = {}".format(thresh))
plt.savefig("triangle.pdf")





