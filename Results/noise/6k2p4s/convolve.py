import numpy as np
import cv2
import glob
import time

t_ref = time.time()

# select model
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 10**(-12))
flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
objp = np.zeros((7*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_s = [] # 3d point in real world space
imgpoints_s = [] # 2d points in image plane.
s_s = 0    


images = glob.glob('np_images/*.txt')

for fname in images:
    im = np.loadtxt(fname).astype(np.uint8)

    ret, corners = cv2.findChessboardCornersSB(im, (8,7), flags=cv2.CALIB_CB_ACCURACY)

    if ret == True:
        objpoints_s.append(objp)
        imgpoints_s.append(corners)
            
        retval_h, sharpness_h = cv2.estimateChessboardSharpness(im, (8,7), corners)
        retval_v, sharpness_v = cv2.estimateChessboardSharpness(im, (8,7), corners, vertical=True)
        s_s = (retval_h[0]+retval_h[0])/2 
   
s_c = []    
mask = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
mean_error = []

for m in mask:   
    objp = np.zeros((7*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints_c = [] # 3d point in real world space
    imgpoints_c = [] # 2d points in image plane.
    
    images = glob.glob('np_images/*.txt')
    
    for fname in images:
        im = np.loadtxt(fname).astype(np.uint8)
        im = cv2.GaussianBlur(im, (m,m), 0)

        ret, corners = cv2.findChessboardCornersSB(im, (8,7), flags=cv2.CALIB_CB_ACCURACY)

        if ret == True:
            objpoints_c.append(objp)
            imgpoints_c.append(corners)
            
            retval_h, sharpness_h = cv2.estimateChessboardSharpness(im, (8,7), corners)
            retval_v, sharpness_v = cv2.estimateChessboardSharpness(im, (8,7), corners, vertical=True)
            s_c.append((retval_h[0]+retval_h[0])/2)

    d = np.array(imgpoints_s) - np.array(imgpoints_c)
    mean_error.append(np.mean(np.abs(d)))
    
    print(m)
    
cv2.namedWindow('im', cv2.WINDOW_NORMAL)
cv2.resizeWindow('im', 820, 616)
cv2.imshow('im', conv)
cv2.waitKey()

# print elapsed time
print((time.time()-t_ref)/60)
