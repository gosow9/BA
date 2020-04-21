import numpy as np
import cv2

# load results
std_c = np.loadtxt('params/std.txt')
rpe_c = np.loadtxt('params/rpe.txt')
sharp_c = np.loadtxt('params/sharp.txt')
fx_c = np.loadtxt('params/fx.txt')
fy_c = np.loadtxt('params/fy.txt')
cx_c = np.loadtxt('params/cx.txt')
cy_c = np.loadtxt('params/cy.txt')
k1_c = np.loadtxt('params/k1.txt')
k2_c = np.loadtxt('params/k2.txt')
k3_c = np.loadtxt('params/k3.txt')
k4_c = np.loadtxt('params/k4.txt')
k5_c = np.loadtxt('params/k5.txt')
k6_c = np.loadtxt('params/k6.txt')
p1_c = np.loadtxt('params/p1.txt')
p2_c = np.loadtxt('params/p2.txt')
s1_c = np.loadtxt('params/s1.txt')
s2_c = np.loadtxt('params/s2.txt')
s3_c = np.loadtxt('params/s3.txt')
s4_c = np.loadtxt('params/s4.txt')

s = 0
fx = fx_c[s]
fy = fy_c[s]
cx = cx_c[s]
cy = cy_c[s]
k1 = k1_c[s]
k2 = k2_c[s]
k3 = k3_c[s]
k4 = k4_c[s]
k5 = k5_c[s]
k6 = k6_c[s]
p1 = p1_c[s]
p2 = p2_c[s]
s1 = s1_c[s]
s2 = s2_c[s]
s3 = s3_c[s]
s4 = s4_c[s]

w = 3280
h = 2464

K = np.array([[fx, 0,  cx],
              [0,  fy, cy],
              [0,  0,  1]])

dst = np.array([k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4])

map_x, map_y = cv2.initUndistortRectifyMap(K, dst, None, K, (w, h), cv2.CV_32FC1)

im_d = cv2.imread('png_images/im2.png')
im_u = cv2.remap(im_d, map_x, map_y, cv2.INTER_LINEAR)

# show result
cv2.namedWindow('im_u', cv2.WINDOW_NORMAL)
cv2.resizeWindow('im_u', 820, 616)
cv2.imshow('im_u', im_u)

cv2.namedWindow('im_d', cv2.WINDOW_NORMAL)
cv2.resizeWindow('im_d', 820, 616)
cv2.imshow('im_d', im_d)

cv2.waitKey()

