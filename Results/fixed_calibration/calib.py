import cv2
import numpy as np
import glob
import time

t_ref = time.time()

images = glob.glob('im/*.png')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
objp = np.zeros((7*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCornersSB(gray, (8,7), flags=cv2.CALIB_CB_ACCURACY)

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

# select model with 6 ks (k4 and k6 = 0), 4s' (s4 = 0)
flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL #+ cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K6 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 10^(-12))
ret, mtx, dist, rvecs, tvecs, newobjp, stdin, stdex, pve, stdnewobjp = cv2.calibrateCameraROExtended(objpoints, imgpoints, (3280, 2464), 1, None, None, flags=flags, criteria=criteria)

# store result to files
np.savetxt('mtx.txt', mtx)
np.savetxt('dist.txt', dist)

with open('params.txt', 'w') as f:
    f.write('RMS reprojection error = {:}\n\n'.format(ret))
    f.write('Camera Matrix:\n')
    f.write('fx = {:} +/- {:}\n'.format(mtx[0][0], stdin[0][0]))
    f.write('fy = {:} +/- {:}\n'.format(mtx[1][1], stdin[1][0]))
    f.write('cx = {:} +/- {:}\n'.format(mtx[0][2], stdin[2][0]))
    f.write('cy = {:} +/- {:}\n\n'.format(mtx[1][2], stdin[3][0]))
    f.write('Radial Distortion:\n')
    f.write('k1 = {:} +/- {:}\n'.format(dist[0][0], stdin[4][0]))
    f.write('k2 = {:} +/- {:}\n'.format(dist[0][1], stdin[5][0]))
    f.write('k3 = {:} +/- {:}\n'.format(dist[0][4], stdin[8][0]))
    f.write('k4 = {:} +/- {:}\n'.format(dist[0][5], stdin[9][0]))
    f.write('k5 = {:} +/- {:}\n'.format(dist[0][6], stdin[10][0]))
    f.write('k6 = {:} +/- {:}\n\n'.format(dist[0][7], stdin[11][0]))
    f.write('Tangential Distortion\n')
    f.write('p1 = {:} /- {:}\n'.format(dist[0][2], stdin[6][0]))
    f.write('p2 = {:} /- {:}\n\n'.format(dist[0][3], stdin[7][0]))
    f.write('Thin Prism Distortion:\n')
    f.write('s1 = {:} +/- {:}\n'.format(dist[0][8], stdin[12][0]))
    f.write('s2 = {:} +/- {:}\n'.format(dist[0][9], stdin[13][0]))
    f.write('s3 = {:} +/- {:}\n'.format(dist[0][10], stdin[14][0]))
    f.write('s4 = {:} +/- {:}\n\n'.format(dist[0][11], stdin[15][0]))
    
    
# print elapsed time
print((time.time()-t_ref)/60)
    