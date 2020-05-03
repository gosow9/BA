import cv2
import numpy as np
import glob
import os

images = glob.glob('np_images/*.txt')

sharp = {}

for fname in images:
    img = np.loadtxt(fname).astype(np.uint8)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCornersSB(img, (8,7), flags=cv2.CALIB_CB_ACCURACY)
    
    if ret == True:
        retval_h, sharpness_h = cv2.estimateChessboardSharpness(img,(8,7), corners)
        retval_v, sharpness_v = cv2.estimateChessboardSharpness(img,(8,7), corners, vertical=True)
        sharp.update({fname: (retval_h[0]+retval_v[0])/2})
        
    else:
        sharp.update({fname: None})
        
with open('sharpness.txt','w') as f:
    for s in sharp:
        f.write(str(s)+': '+str(sharp[s])+'\n')
