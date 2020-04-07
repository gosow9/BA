import cv2
import numpy as np
import random
import glob
import time

t_ref = time.time()
files = glob.glob('../im2/*.png')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

nr = 0
# get 10 sets
for i in range(10):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # get a random set of 20 images   
    images = []
    for i in range(20):
        d=random.choice(files)
        images.append('../im2/' + d)


    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCornersSB(gray, (10,7), flags=cv2.CALIB_CB_ACCURACY)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            for c in corners:
                cv2.drawMarker(img, (c[0][0], c[0][1]), (0, 255, 255), markerSize=1)
                
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('img', 820, 616)
            cv2.imshow('img',img)
        
            cv2.waitKey(500)
        
        
    np.save('checkerboards/imgp{:}'.format(nr), imgpoints)
    np.save('checkerboards/objp{:}'.format(nr), objpoints)
    nr += 1 
    
# print elapsed time
print((time.time()-t_ref)/60)
