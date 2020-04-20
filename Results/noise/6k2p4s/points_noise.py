import numpy as np
import cv2
import random
import time
import matplotlib.pyplot as plt
from pylab import rcParams


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


def get_imgpoints(r, t):
    # empty image
    imgp = []

    im = np.ones((2464, 3280))*170

    size = 308

    x_check = int(8*size/2 - 1)
    y_check = int(9*size/2 - 1) 
    
    x_im = int(2462/2 - 1)
    y_im = int(3280/2 - 1)

    z = 250
    
    R, _ = cv2.Rodrigues(r)
                          
    for i in np.arange(0, 7*size, size):
        for j in np.arange(0, 8*size, size):
            # index with respect to checkboard center
            x = i - x_check + size
            y = j - y_check + size

            # project to 3D
            x_3D = x/f*z
            y_3D = y/f*z
                        
            vec_3D = np.array([x_3D, y_3D, 0])
                        
            # transformation
            vec_3D_new = R@vec_3D + t
            vec_3D_new[2] += z
                        
            # distortion and project to 2D
            x_2D = vec_3D_new[0]/vec_3D_new[2]
            y_2D = vec_3D_new[1]/vec_3D_new[2]
            r = np.sqrt(x_2D**2+y_2D**2)
                        
            x_2D_dst = f*full_dist_model_x(k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, x_2D, y_2D, r)+x_im
            y_2D_dst = f*full_dist_model_y(k1, k2, k3, k4, k5, k6, p1, p2, s3, s4, x_2D, y_2D, r)+y_im
                        
            imgp.append([y_2D_dst, x_2D_dst])
        
    # flip order
    imgp_flipped = []
    for d in reversed(imgp):
        imgp_flipped.append([d])
        
    return np.array(imgp_flipped)

t_ref = time.time()

# define rotation and translation
r1 = np.array([0.7, 0.7, 0], dtype=np.float64)
t1 = np.array([40, -70, 200], dtype=np.float64)
r2 = np.array([-0.7, 0.7, 0], dtype=np.float64)
t2 = np.array([40, 70, 200], dtype=np.float64)
r3 = np.array([0.7, -0.7, 0], dtype=np.float64)
t3 = np.array([-40, -70, 200], dtype=np.float64)
r4 = np.array([-0.7, -0.7, 0], dtype=np.float64)
t4 = np.array([-40, 70, 200], dtype=np.float64)

r5 = np.array([-0.5, -0.4, 0.3], dtype=np.float64)
t5 = np.array([30, -60, 120], dtype=np.float64)
r6 = np.array([0.5, 0.4, 0.3], dtype=np.float64)
t6 = np.array([-30, 60, 120], dtype=np.float64)
r7 = np.array([-0.5, 0.4, -0.3], dtype=np.float64)
t7 = np.array([-30, -60, 120], dtype=np.float64)
r8 = np.array([0.5, -0.4, -0.3], dtype=np.float64)
t8 = np.array([30, 60, 120], dtype=np.float64)

r9 = np.array([1, 0, 0], dtype=np.float64)
t9 = np.array([0, -150, 300], dtype=np.float64)
r10 = np.array([-0.7, 0, 0], dtype=np.float64)
t10 = np.array([0, -150, 300], dtype=np.float64)
r11 = np.array([-1, 0, 0], dtype=np.float64)
t11 = np.array([0, 150, 300], dtype=np.float64)
r12 = np.array([0.7, 0, 0], dtype=np.float64)
t12 = np.array([0, 150, 300], dtype=np.float64)

r13 = np.array([0, 0.8, 0], dtype=np.float64)
t13 = np.array([-50, 0, 250], dtype=np.float64)
r14 = np.array([0, -0.8, 0], dtype=np.float64)
t14 = np.array([-50, 0, 250], dtype=np.float64)
r15 = np.array([0, 0.8, 0], dtype=np.float64)
t15 = np.array([50, 0, 250], dtype=np.float64)
r16 = np.array([0, -0.8, 0], dtype=np.float64)
t16 = np.array([50, 0, 250], dtype=np.float64)

r17 = np.array([-0.2, -0.2, 0.2], dtype=np.float64)
t17 = np.array([50, -100, 250], dtype=np.float64)
r18 = np.array([0.2, 0.2, 0.2], dtype=np.float64)
t18 = np.array([-50, 100, 250], dtype=np.float64)
r19 = np.array([-0.2, 0.2, -0.2], dtype=np.float64)
t19 = np.array([-50, -100, 250], dtype=np.float64)
r20 = np.array([0.2, -0.2, -0.2], dtype=np.float64)
t20 = np.array([50, 100, 250], dtype=np.float64)

r = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20]
t = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20]

# distortion model
f = 2700
k1 = 4.1
k2 = 36
k3 = 38
k4 = 4
k5 = 34
k6 = 40
p1 = 0#0.002
p2 = 0#0.0019
s1 = 0#-0.0014
s2 = 0#-0.001
s3 = 0#-0.0022
s4 = 0#0.00013

# add noise
for i in range(20):
    t[i][0] += np.random.normal(0, 10)
    t[i][1] += np.random.normal(0, 10)
    t[i][2] += np.random.normal(0, 10)
    r[i][0] += np.random.normal(0, 0.05)
    r[i][1] += np.random.normal(0, 0.05)
    r[i][2] += np.random.normal(0, 0.05)    


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
objp = np.zeros((7*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = []

for i in range(20):
    imgp = get_imgpoints(r[i], t[i])   
    objpoints.append(objp)
    imgpoints.append(imgp.astype(np.float32))
  
# setup calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 10^(-6))
#flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_ZERO_TANGENT_DIST

step = 0.05
std = np.arange(0, 0+step, step)

ret = []
mtx = []
dist = []
mean_error = []

for s in std:
    # add noise   
    imgpoints_n = imgpoints + np.random.normal(0, s, np.shape(imgpoints)).astype(np.float32)
    
    # calibrate camera
    ret_s, mtx_s, dist_s, rvecs_s, tvecs_s, newobjp_s, tdin_s, stdex_s, pve_s, stdnewobjp_s = cv2.calibrateCameraROExtended(objpoints, imgpoints_n, (3280, 2464), 1, None, None, flags=flags, criteria=criteria)

    ret.append(ret_s)
    mtx.append(mtx_s)
    dist.append(dist_s)

    d = np.array(imgpoints) - np.array(imgpoints_n)
    mean_error.append(np.mean(np.abs(d)))

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

fig, ax = plt.subplots(1,1)
ax.plot(r, r_m-r, label='Model')
ax.grid(True)

for i in range(len(std)):
    # compute distortion
    fx_s = mtx[i][0][0]
    fy_s = mtx[i][1][1]
    k1_s = dist[i][0][0]
    k2_s = dist[i][0][1]
    p1_s = dist[i][0][2]
    p2_s = dist[i][0][3]
    k3_s = dist[i][0][4]
    k4_s = dist[i][0][5]
    k5_s = dist[i][0][6]
    k6_s = dist[i][0][7]
    s1_s = dist[i][0][8]
    s2_s = dist[i][0][9]
    s3_s = dist[i][0][10]
    s4_s = dist[i][0][11]
    
    r_new = np.sqrt((x/fx_s)**2+(y/fy_s)**2)  
    x_s = fx_s*full_dist_model_x(k1_s, k2_s, k3_s, k4_s, k5_s, k6_s, p1_s, p2_s, s1_s, s2_s, x/fx_s, y/fy_s, r_new)
    y_s = fy_s*full_dist_model_y(k1_s, k2_s, k3_s, k4_s, k5_s, k6_s, p1_s, p2_s, s3_s, s4_s, x/fx_s, y/fy_s, r_new)
    r_s = np.sqrt(x_s**2+y_s**2)

    ax.plot(r, r_s-r, label=r'$\sigma$ = {:.2}'.format(std[i]))

ax.legend()
# print elapsed time
print((time.time()-t_ref)/60) 