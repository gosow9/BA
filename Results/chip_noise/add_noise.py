import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression


img_bright = cv2.imread('bright.png')
gray_bright = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
img_dark = cv2.imread('dark.png')
gray_dark = cv2.cvtColor(img_dark, cv2.COLOR_BGR2GRAY)


# Find the chessboard corners
ret_bright, corners_bright = cv2.findChessboardCornersSB(gray_bright, (10,7), flags=cv2.CALIB_CB_ACCURACY)
ret_dark, corners_dark = cv2.findChessboardCornersSB(gray_dark, (10,7), flags=cv2.CALIB_CB_ACCURACY)    

# estimate sharpness
retval_h_bright, sharpness_h_bright = cv2.estimateChessboardSharpness(gray_bright,(10,7), corners_bright)
retval_v_bright, sharpness_v_bright = cv2.estimateChessboardSharpness(gray_bright,(10,7), corners_bright, vertical=True)
retval_h_dark, sharpness_h_dark = cv2.estimateChessboardSharpness(gray_dark,(10,7), corners_dark)
retval_v_dark, sharpness_v_dark = cv2.estimateChessboardSharpness(gray_dark,(10,7), corners_dark, vertical=True)

# calculate mean
sharpness_bright = (retval_h_bright[0]+retval_v_bright[0])/2
sharpness_dark = (retval_h_dark[0]+retval_v_dark[0])/2

step = 0.25
std = np.arange(0, 25+step, step)

sharpness_noise = []

for s in std:
    # add noise
    gray_noise = gray_bright + np.random.normal(0, s, np.shape(gray_bright))
    gray_noise = gray_noise.astype(np.uint8)

    # find the chessboard corners
    ret_noise, corners_noise = cv2.findChessboardCornersSB(gray_noise, (10,7), flags=cv2.CALIB_CB_ACCURACY)

    if ret_noise == True:
        # estimate sharpness
        retval_h_noise, sharpness_h_noise = cv2.estimateChessboardSharpness(gray_noise,(10,7), corners_noise)
        retval_v_noise, sharpness_v_noise = cv2.estimateChessboardSharpness(gray_noise,(10,7), corners_noise, vertical=True)

        sharpness_noise.append((retval_h_noise[0]+retval_v_noise[0])/2)
    
    else:
        sharpness_noise.append(None)

# linear regression  
reg = LinearRegression().fit(std.reshape(-1,1), sharpness_noise)
a = reg.coef_[0]
b = reg.intercept_

line = std*a+b

# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})   
 
fig, ax = plt.subplots(1,1)
ax.plot(std, sharpness_noise, color='tab:blue', label='Sharpness')
ax.plot(std, line, color='tab:orange', label='Regression\n'+'a = {:.2f}, b = {:.2f}'.format(a, b))
ax.set_xlim([0, np.max(std)])
ax.set_xticks([0, 5, 10, 15, 20, 25])
ax.set_ylim([0, 40])
ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
ax.grid(True)
ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Sharpness')

fig.savefig('add_noise.pdf')