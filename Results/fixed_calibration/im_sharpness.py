import cv2
import numpy as np
import glob
import os

images = glob.glob('im2/*.png')

sharp = {}

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCornersSB(gray, (8,7), flags=cv2.CALIB_CB_ACCURACY)
    
    if ret == True:
        retval, sharpness = cv2.estimateChessboardSharpness(gray,(8,7), corners)
        retval1, sharpness1 = cv2.estimateChessboardSharpness(gray,(8,7), corners, vertical=True)
        sharp.update({fname: (retval[0]+retval1[0]/2)})
        
    else:
        sharp.update({fname: None})
        os.rename(fname, 'im2/del/'+fname[3:])
        
with open('sharpness2.txt','w') as f:
    for s in sharp:
        f.write(str(s)+': '+str(sharp[s])+'\n')
