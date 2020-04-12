import cv2
import numpy as np

img = cv2.imread('dark.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCornersSB(gray, (8,7), flags=cv2.CALIB_CB_ACCURACY)
    
if ret == True:
   retval, sharpness = cv2.estimateChessboardSharpness(gray,(8,7), corners)
   retval1, sharpness1 = cv2.estimateChessboardSharpness(gray,(8,7), corners, vertical=True)

