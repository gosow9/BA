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

plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

images = [cv.imread(file) for file in glob.glob("b*.png")]
imag = [cv.cvtColor(file, cv.COLOR_BGR2GRAY) for file in images]
plt.ioff()
for i,nr in zip(imag, range(len(imag))):
    img = i[::10,::10]
    val_max = img.max()
    val_min = img.min()
    val_var = np.var(img)
    val_mean = np.mean(img)
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    xx = xx*10
    yy = yy*10
    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0)
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(xx, yy, img , cmap=plt.cm.coolwarm)
    fig.colorbar(surf, shrink=1, aspect=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Max value = {}, Min value = {}, Variance = {:2f}, Mean = {:2f}".format(val_max, val_min, val_var, val_mean)) 
    plt.savefig("3d{}.png".format(nr),dpi=150)

    