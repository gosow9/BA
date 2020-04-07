import cv2
import numpy as np
import matplotlib.pyplot as plt


mtx = np.loadtxt('mtx.txt')
dst = np.loadtxt('dist.txt')

w = 3280
h = 2464

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dst,(w,h),1,(w,h))

img = np.ones((h,w))*255

for i in range(1,16):
    img = cv2.line(img, (0, i*154), (3280, i*154), (0, 0, 0), 10)

for i in range(1,20):
    img = cv2.line(img, (i*164, 0), (i*164, 2464), (0, 0, 0), 10)

img = cv2.undistort(img, mtx, dst, None, newcameramtx)

cv2.imwrite('res_opencv.jpg', img)

plt.imshow(img)