import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time

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
cy_m = 2464/2
cx_m = 3280/2
k1_m = 4.1
k2_m = 36
k3_m = 38
k4_m = 4
k5_m = 34
k6_m = 40
p1_m = 0.002
p2_m = 0.0019
s1_m = -0.0014
s2_m = -0.001
s3_m = -0.0022
s4_m = 0.00013

images_p = glob.glob('np_images/p/*.txt')
images_f = glob.glob('np_images/f/*.txt')

# select model
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 10**(-12))
flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
objp = np.zeros((7*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_p = [] # 3d point in real world space
imgpoints_p = [] # 2d points in image plane.
   
for fname in images_p:
    im = np.loadtxt(fname).astype(np.uint8)

    ret, corners = cv2.findChessboardCornersSB(im, (8,7), flags=cv2.CALIB_CB_ACCURACY)

    if ret == True:
        objpoints_p.append(objp)
        imgpoints_p.append(corners)
  

# Arrays to store object points and image points from all the images.
objpoints_f = [] # 3d point in real world space
imgpoints_f = [] # 2d points in image plane.
   
for fname in images_f:
    im = np.loadtxt(fname).astype(np.uint8)

    ret, corners = cv2.findChessboardCornersSB(im, (8,7), flags=cv2.CALIB_CB_ACCURACY)

    if ret == True:
        objpoints_f.append(objp)
        imgpoints_f.append(corners)

ret_p, mtx_p, dist_p, rvecs_p, tvecs_p, newobjp_p, stdin_p, stdex_p, pve_p, stdnewobjp_p = cv2.calibrateCameraROExtended(objpoints_p, imgpoints_p, (3280, 2464), 1, None, None, flags=flags, criteria=criteria)
ret_f, mtx_f, dist_f, rvecs_f, tvecs_f, newobjp_f, stdin_f, stdex_f, pve_f, stdnewobjp_f = cv2.calibrateCameraROExtended(objpoints_f, imgpoints_f, (3280, 2464), 1, None, None, flags=flags, criteria=criteria)

fx_p = mtx_p[0][0]
fy_p = mtx_p[1][1]
k1_p = dist_p[0][0]
k2_p = dist_p[0][1]
k3_p = dist_p[0][4]
k4_p = dist_p[0][5]
k5_p = dist_p[0][6]
k6_p = dist_p[0][7]
p1_p = dist_p[0][2]
p2_p = dist_p[0][3]
s1_p = dist_p[0][8]
s2_p = dist_p[0][9]
s3_p = dist_p[0][10]
s4_p = dist_p[0][11]

fx_f = mtx_f[0][0]
fy_f = mtx_f[1][1]
k1_f = dist_f[0][0]
k2_f = dist_f[0][1]
k3_f = dist_f[0][4]
k4_f = dist_f[0][5]
k5_f = dist_f[0][6]
k6_f = dist_f[0][7]
p1_f = dist_f[0][2]
p2_f = dist_f[0][3]
s1_f = dist_f[0][8]
s2_f = dist_f[0][9]
s3_f = dist_f[0][10]
s4_f = dist_f[0][11]

# prepare axis
res = 1
x = np.arange(0, 2464/2, res)
y = np.arange(0, 3280/2, res/2464*3280)
r = np.sqrt(x**2+y**2)

# compute model
r_new = np.sqrt((x/f)**2+(y/f)**2)  
x_m = f*full_dist_model_x(k1_m, k2_m, k3_m, k4_m, k5_m, k6_m, p1_m, p2_m, s1_m, s2_m, x/f, y/f, r_new)
y_m = f*full_dist_model_y(k1_m, k2_m, k3_m, k4_m, k5_m, k6_m, p1_m, p2_m, s3_m, s4_m, x/f, y/f, r_new)
r_m = np.sqrt(x_m**2+y_m**2)

# compute calibration
r_new = np.sqrt((x/fx_p)**2+(y/fy_p)**2)  
x_p = fx_p*full_dist_model_x(k1_p, k2_p, k3_p, k4_p, k5_p, k6_p, p1_p, p2_p, s1_p, s2_p, x/fx_p, y/fy_p, r_new)
y_p = fy_p*full_dist_model_y(k1_p, k2_p, k3_p, k4_p, k5_p, k6_p, p1_p, p2_p, s3_p, s4_p, x/fx_p, y/fy_p, r_new)
r_p = np.sqrt(x_p**2+y_p**2)

r_new = np.sqrt((x/fx_f)**2+(y/fy_f)**2)  
x_f = fx_f*full_dist_model_x(k1_f, k2_f, k3_f, k4_f, k5_f, k6_f, p1_f, p2_f, s1_f, s2_f, x/fx_f, y/fy_f, r_new)
y_f = fy_f*full_dist_model_y(k1_f, k2_f, k3_f, k4_f, k5_f, k6_f, p1_f, p2_f, s3_f, s4_f, x/fx_f, y/fy_f, r_new)
r_f = np.sqrt(x_f**2+y_f**2)

# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})         

fig, ax = plt.subplots(1,1)
ax.plot(r, r_m-r, label='Model')
label = r'1: $e_{rp}$ = ' + '{:.2}'.format(ret_p)
ax.plot(r, r_p-r, label=label)
label = r'2: $e_{rp}$ = ' + '{:.2}'.format(ret_f)
ax.plot(r, r_f-r, label=label)
ax.set_xlim([0, 2100])
ax.set_xticks([0, 300, 600, 900, 1200, 1500, 1800, 2100])
ax.set_ylim([0, 50])
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel('Radius (pixel)')
ax.set_ylabel('Distortion (pixel)')
ax.legend()
ax.grid(True)

fig.savefig('location.pdf')

print((time.time()-t_ref)/60)