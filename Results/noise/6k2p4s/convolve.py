import numpy as np
import cv2
import glob
import time
import matplotlib.pyplot as plt

def r_dist_model(k1, k2, k3, k4, k5, k6, q, r):
    return q*(1 + k1*r**2 + k2*r**4 + k3*r**6)/(1 + k4*r**2 + k5*r**4 +k6*r**6)

def t_dist_model_x(p1, p2, x, y, r):
    return x + 2*p1*x*y+p2*(r**2+2*x**2)

def t_dist_model_y(p1, p2, x, y, r):
    return y +  2*p2*x*y+p1*(r**2+2*y**2)

def full_dist_model_x(k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, x, y, r):
    r_dist = r_dist_model(k1, k2, k3, k4, k5, k6, x, r)
    t_dist = t_dist_model_x(p1, p1, x, y, r)
    return r_dist + t_dist - x +s1*r**2 + s2*r**4
    
def full_dist_model_y(k1, k2, k3, k4, k5, k6, p1, p2, s3, s4, x, y, r):
    r_dist = r_dist_model(k1, k2, k3, k4, k5, k6, y, r)
    t_dist = t_dist_model_y(p1, p1, x, y, r)
    return r_dist + t_dist - y + s3*r**2 + s4*r**4

t_ref = time.time()

# distortion model
f = 2700
k1 = 4.1
k2 = 36
k3 = 38
k4 = 4
k5 = 34
k6 = 40
p1 = 0.002
p2 = 0.0019
s1 = -0.0014
s2 = -0.001
s3 = -0.0022
s4 = 0.00013

# select model
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 10**(-12))
flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
objp = np.zeros((7*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_s = [] # 3d point in real world space
imgpoints_s = [] # 2d points in image plane.

images = glob.glob('np_images/f/*.txt')

for fname in images:
    im = np.loadtxt(fname).astype(np.uint8)

    ret, corners = cv2.findChessboardCornersSB(im, (8,7), flags=cv2.CALIB_CB_ACCURACY)

    if ret == True:
        objpoints_s.append(objp)
        imgpoints_s.append(corners)
   
mask =71

# Arrays to store object points and image points from all the images.
objpoints_c = [] # 3d point in real world space
imgpoints_c = [] # 2d points in image plane.
    
    
for fname in images:
    im = np.loadtxt(fname).astype(np.uint8)
    im = cv2.GaussianBlur(im, (mask,mask), 0)

    ret, corners = cv2.findChessboardCornersSB(im, (8,7), flags=cv2.CALIB_CB_ACCURACY)

    if ret == True:
        objpoints_c.append(objp)
        imgpoints_c.append(corners)

ret_s, mtx_s, dist_s, rvecs_s, tvecs_s, newobjp_s, stdin_s, stdex_s, pve_s, stdnewobjp_s = cv2.calibrateCameraROExtended(objpoints_s, imgpoints_s, (3280, 2464), 1, None, None, flags=flags, criteria=criteria)
ret_c, mtx_c, dist_c, rvecs_c, tvecs_c, newobjp_c, stdin_c, stdex_c, pve_c, stdnewobjp_c = cv2.calibrateCameraROExtended(objpoints_c, imgpoints_c, (3280, 2464), 1, None, None, flags=flags, criteria=criteria)

# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

# prepare axis
res = 1
x = np.arange(0, 2464/2, res)
y = np.arange(0, 3280/2, res/2464*3280)
r = np.sqrt(x**2+y**2)

# compute model
r_new = np.sqrt((x/f)**2+(y/f)**2)  
x_m = f*full_dist_model_x(k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, x/f, y/f, r_new)
y_m = f*full_dist_model_y(k1, k2, k3, k4, k5, k6, p1, p2, s3, s4, x/f, y/f, r_new)
r_m = np.sqrt(x_m**2+y_m**2)

fx_s = mtx_s[0][0]
fy_s = mtx_s[1][1]
k1_s = dist_s[0][0]
k2_s = dist_s[0][1]
p1_s = dist_s[0][2]
p2_s = dist_s[0][3]
k3_s = dist_s[0][4]
k4_s = dist_s[0][5]
k5_s = dist_s[0][6]
k6_s = dist_s[0][7]
s1_s = dist_s[0][8]
s2_s = dist_s[0][9]
s3_s = dist_s[0][10]
s4_s = dist_s[0][11]

r_new = np.sqrt((x/fx_s)**2+(y/fy_s)**2)  
x_s = fx_s*full_dist_model_x(k1_s, k2_s, k3_s, k4_s, k5_s, k6_s, p1_s, p2_s, s1_s, s2_s, x/fx_s, y/fy_s, r_new)
y_s = fy_s*full_dist_model_y(k1_s, k2_s, k3_s, k4_s, k5_s, k6_s, p1_s, p2_s, s3_s, s4_s, x/fy_s, y/fy_s, r_new)
r_s = np.sqrt(x_s**2+y_s**2)

fx_c = mtx_c[0][0]
fy_c = mtx_c[1][1]
k1_c = dist_c[0][0]
k2_c = dist_c[0][1]
p1_c = dist_c[0][2]
p2_c = dist_c[0][3]
k3_c = dist_c[0][4]
k4_c = dist_c[0][5]
k5_c = dist_c[0][6]
k6_c = dist_c[0][7]
s1_c = dist_c[0][8]
s2_c = dist_c[0][9]
s3_c = dist_c[0][10]
s4_c = dist_c[0][11]

r_new = np.sqrt((x/f)**2+(y/f)**2)  
x_c = fx_c*full_dist_model_x(k1_c, k2_c, k3_c, k4_c, k5_c, k6_c, p1_c, p2_c, s1_c, s2_c, x/fx_c, y/fy_c, r_new)
y_c = fy_c*full_dist_model_y(k1_c, k2_c, k3_c, k4_c, k5_c, k6_c, p1_c, p2_c, s3_c, s4_c, x/fx_c, y/fy_c, r_new)
r_c = np.sqrt(x_c**2+y_c**2)

    
fig, ax = plt.subplots(1,1)
ax.plot(r, r_m-r,label='Model')
label = r'1: $e_{rp}$ = '+'{:.2}'.format(ret_s)
ax.plot(r, r_s-r,label=label)
label = r'2: $e_{rp}$ = '+'{:.2}'.format(ret_c)
ax.plot(r, r_c-r,label=label)
ax.grid(True)
ax.set_xlabel('Radius (pixel)')
ax.set_ylabel('Distortion (pixel)')
ax.set_xlim([0, 2100])
ax.set_xticks([0, 300, 600, 900, 1200, 1500, 1800, 2100])
ax.set_ylim([0, 50])
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.legend()

# print elapsed time
print((time.time()-t_ref)/60)
