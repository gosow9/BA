import cv2
import numpy as np
import glob
import os

images = glob.glob('im2/*.png')

sharp = {}

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCornersSB(gray, (8,7), flags=cv2.CALIB_CB_ACCURACY)
    
    if ret == True:
        retval, sharpness = cv2.estimateChessboardSharpness(gray,(8,7), corners)
        sharp.update({fname: retval[0]})
        
    else:
        os.rename(fname, 'im2/unsharp'+fname[3:])
        
with open('sharpness.txt','w') as f:
    for s in sharp:
        f.write(str(s)+': '+str(sharp[s])+'\n')
