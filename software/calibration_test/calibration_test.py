import cv2
import numpy as np
import matplotlib.pyplot as plt


# load intrinsics
mtx = np.loadtxt('matlab/mtx.txt')
dst = np.loadtxt('matlab/dst.txt')

w = 3280
h = 2464

new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx,dst,(w,h),1,(w,h))

img = cv2.imread('snap0.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.undistort(img, mtx, dst, None, new_mtx)




plt.imshow(img)

