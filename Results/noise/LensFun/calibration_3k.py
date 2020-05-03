import numpy as np
import cv2
import glob
import time

t_ref = time.time()

# lists to store values
sharp = []
rpe = []
fx = []
fy = []
cx = []
cy = []
k1 = []
k2 = []
k3 = []
k4 = []
k5 = []
k6 = []
p1 = []
p2 = []
s1 = []
s2 = []
s3 = []
s4 = []

step = 0.5
std = np.arange(0, 50+step, step)

for sigma in std:
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
    objp = np.zeros((7*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('np_images/*.txt')

    s = []
    for fname in images:
        # load image
        img = np.loadtxt(fname)
         # add gaussian noise
        img_n = img + np.random.normal(0, sigma, np.shape(img))
    
        # covert to uint8
        img_n = np.clip(img_n, 0, 255)       
        img_n = np.round(img_n).astype(np.uint8)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCornersSB(img_n, (8,7), flags=cv2.CALIB_CB_ACCURACY)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            retval_h, sharpness_h = cv2.estimateChessboardSharpness(img_n,(8,7), corners)
            retval_v, sharpness_v = cv2.estimateChessboardSharpness(img_n,(8,7), corners, vertical=True)
            s.append((retval_h[0]+retval_h[0])/2)    
   
    
    # select model
    flags = cv2.CALIB_THIN_PRISM_MODEL
    ret, mtx, dist, rvecs, tvecs, newobjp, stdin, stdex, pve, stdnewobjp = cv2.calibrateCameraROExtended(objpoints, imgpoints, (3280, 2464), 1, None, None, flags=flags)

    # append lists
    sharp.append(np.mean(s))
    rpe.append(ret)
    fx.append(mtx[0][0])
    fy.append(mtx[1][1])
    cx.append(mtx[0][2])
    cy.append(mtx[1][2])
    k1.append(dist[0][0])
    k2.append(dist[0][1])
    k3.append(dist[0][4])
    k4.append(dist[0][5])
    k5.append(dist[0][6])
    k6.append(dist[0][7])
    p1.append(dist[0][2])
    p2.append(dist[0][3])
    s1.append(dist[0][8])
    s2.append(dist[0][9])
    s3.append(dist[0][10])
    s4.append(dist[0][11])

    # save params
    with open('params_3k/complete/params_{:}.txt'.format(sigma), 'w') as f:
        f.write('RMS reprojection error = {:}\n'.format(ret))
        f.write('sharpness = {:}\n\n'.format(np.mean(s)))
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

# save lists
np.savetxt('params_3k/rpe.txt', rpe)
np.savetxt('params_3k/sharp.txt', sharp)
np.savetxt('params_3k/fx.txt', fx)
np.savetxt('params_3k/fy.txt', fy)
np.savetxt('params_3k/cx.txt', cx)
np.savetxt('params_3k/cy.txt', cy)
np.savetxt('params_3k/k1.txt', k1)
np.savetxt('params_3k/k2.txt', k2)
np.savetxt('params_3k/k3.txt', k3)
np.savetxt('params_3k/k4.txt', k4)
np.savetxt('params_3k/k5.txt', k5)
np.savetxt('params_3k/k6.txt', k6)
np.savetxt('params_3k/p1.txt', p1)
np.savetxt('params_3k/p2.txt', p2)
np.savetxt('params_3k/s1.txt', s1)
np.savetxt('params_3k/s2.txt', s2)
np.savetxt('params_3k/s3.txt', s3)
np.savetxt('params_3k/s4.txt', s4)

# print elapsed time
print((time.time()-t_ref)/60)