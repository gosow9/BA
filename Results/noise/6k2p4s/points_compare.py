import numpy as np
import cv2
import random
import time
from pylab import rcParams
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
     
    # add transformed checkerboard
    for i in range(8):
        for j in range(9):
            if (i%2==0 and j%2==0) or (i%2!=0 and j%2!=0):
                for a in range(size):
                    for b in range(size):
                        # index with respect to checkboard center
                        x = i*size + a - x_check
                        y = j*size + b - y_check
                        
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
                        
                        x_2D_dst = f*full_dist_model_x(k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, x_2D, y_2D, r)
                        y_2D_dst = f*full_dist_model_y(k1, k2, k3, k4, k5, k6, p1, p2, s3, s4, x_2D, y_2D, r)
                        
                    
                        # new index with respect to image-cooridnates
                        x_new = int(round(x_2D_dst + x_im))
                        y_new = int(round(y_2D_dst + y_im))
                        
                        if x_new < 0 or x_new >= 2464:
                            continue
                        
                        if y_new < 0 or y_new >= 3280:
                            continue
                        
                        im[x_new][y_new] = 40
    
                      
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
      
    imgp_f = []
    for d in imgp:
        imgp_f.append([d])
      
    return np.array(imgp_flipped), np.array(imgp_f), im

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

r21 = np.array([0.7, 0.7, 0], dtype=np.float64)
t21 = np.array([105, -140, 200], dtype=np.float64)
r22 = np.array([-0.7, 0.7, 0], dtype=np.float64)
t22 = np.array([105, 140, 200], dtype=np.float64)
r23 = np.array([0.7, -0.7, 0], dtype=np.float64)
t23 = np.array([-105, -140, 200], dtype=np.float64)
r24 = np.array([-0.7, -0.7, 0], dtype=np.float64)
t24 = np.array([-105, 140, 200], dtype=np.float64)

#r = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24]
#t = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24]

r = [r1, r2, r3, r4, r5, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24]
t = [t1, t2, t3, t4, t5, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24]

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

dst = [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]


# add noise to the fistst 20
for i in range(20):
    t[i][0] += np.random.uniform(-10, 10)
    t[i][1] += np.random.uniform(-10, 10)
    t[i][2] += np.random.uniform(-10, 10)
    r[i][0] += np.random.uniform(-0.1, 0.1)
    r[i][1] += np.random.uniform(-0.1, 0.1)
    r[i][2] += np.random.uniform(-0.1, 0.1)    


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
objp = np.zeros((7*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_s = [] # 3d point in real world space
imgpoints_s = [] # 2d points in image plane.
images = []

imgpoints_f = []

for i in range(len(t)):
    imgp, imgp_f, im = get_imgpoints(r[i], t[i])
    images.append(im)
    
    objpoints_s.append(objp)
    imgpoints_s.append(imgp.astype(np.float32))
    imgpoints_f.append(imgp_f.astype(np.float32))

    print(i)
    
# calibrate camera
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 10^(-12))
flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
ret_s, mtx_s, dist_s, rvecs_s, tvecs_s, newobjp_s, stdin_s, stdex_s, pve_s, stdnewobjp_s = cv2.calibrateCameraROExtended(objpoints_s, imgpoints_s, (3280, 2464), 1, None, None, flags=flags, criteria=criteria)
#ret_s, mtx_s, dist_s, rvecs_s, tvecs_s, tdin_s, stdex_s, pve_s = cv2.calibrateCameraExtended(objpoints_s, imgpoints_s, (3280, 2464), None, None, flags=flags, criteria=criteria)

# Compare
objpoints_c = [] # 3d point in real world space
imgpoints_c = [] # 2d points in image plane.

for img in images:  
    # Find the chess board corners
    ret, corners = cv2.findChessboardCornersSB(img.astype(np.uint8), (8,7), flags=cv2.CALIB_CB_ACCURACY)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints_c.append(objp)
        imgpoints_c.append(corners)
        
    
d1 = np.mean(np.abs(np.array(imgpoints_s)-np.array(imgpoints_c)))

#comparison
imgpoints_comp = []
k = 0

for img in images:  
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img.astype(np.uint8), (8,7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(img.astype(np.uint8), corners, (11,11), (-1,-1), criteria)
        imgpoints_comp.append(corners2)

    else:
        print(k)
    k+=1

d2 = np.mean(np.abs(np.array(imgpoints_f)-np.array(imgpoints_comp)))
           
ret_c, mtx_c, dist_c, rvecs_c, tvecs_c, newobjp_c, stdin_c, stdex_c, pve_c, stdnewobjp_c = cv2.calibrateCameraROExtended(objpoints_c, imgpoints_c, (3280, 2464), 1, None, None, flags=flags, criteria=criteria)

# with blur
objpoints_g = [] # 3d point in real world space
imgpoints_g = [] # 2d points in image plane.

for img in images:
    img = cv2.GaussianBlur(img, (21, 21), 0)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCornersSB(img.astype(np.uint8), (8,7), flags=cv2.CALIB_CB_ACCURACY)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints_g.append(objp)
        imgpoints_g.append(corners)
            
ret_g, mtx_g, dist_g, rvecs_g, tvecs_g, newobjp_g, stdin_g, stdex_g, pve_g, stdnewobjp_g = cv2.calibrateCameraROExtended(objpoints_g, imgpoints_g, (3280, 2464), 1, None, None, flags=flags, criteria=criteria)


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
k3_s = dist_s[0][4]
k4_s = dist_s[0][5]
k5_s = dist_s[0][6]
k6_s = dist_s[0][7]
p1_s = dist_s[0][2]
p2_s = dist_s[0][3]
s1_s = dist_s[0][8]
s2_s = dist_s[0][9]
s3_s = dist_s[0][10]
s4_s = dist_s[0][11]

r_new = np.sqrt((x/fx_s)**2+(y/fy_s)**2)  
x_s = fx_s*full_dist_model_x(k1_s, k2_s, k3_s, k4_s, k5_s, k6_s, p1_s, p2_s, s1_s, s2_s, x/fx_s, y/fy_s, r_new)
y_s = fx_s*full_dist_model_y(k1_s, k2_s, k3_s, k4_s, k5_s, k6_s, p1_s, p2_s, s3_s, s4_s, x/fx_s, y/fy_s, r_new)
r_s = np.sqrt(x_s**2+y_s**2)

fx_c = mtx_c[0][0]
fy_c = mtx_c[1][1]
k1_c = dist_c[0][0]
k2_c = dist_c[0][1]
k3_c = dist_c[0][4]
k4_c = dist_c[0][5]
k5_c = dist_c[0][6]
k6_c = dist_c[0][7]
p1_c = dist_c[0][2]
p2_c = dist_c[0][3]
s1_c = dist_c[0][8]
s2_c = dist_c[0][9]
s3_c = dist_c[0][10]
s4_c = dist_c[0][11]

r_new = np.sqrt((x/fx_c)**2+(y/fy_c)**2)  
x_c = fx_c*full_dist_model_x(k1_c, k2_c, k3_c, k4_c, k5_c, k6_c, p1_c, p2_c, s1_c, s2_c, x/fx_c, y/fy_c, r_new)
y_c = fx_c*full_dist_model_y(k1_c, k2_c, k3_c, k4_c, k5_c, k6_c, p1_c, p2_c, s3_c, s4_c, x/fx_c, y/fy_c, r_new)
r_c = np.sqrt(x_c**2+y_c**2)

fx_g = mtx_c[0][0]
fy_g = mtx_c[1][1]
k1_g = dist_c[0][0]
k2_g = dist_c[0][1]
k3_g = dist_c[0][4]
k4_g = dist_c[0][5]
k5_g = dist_c[0][6]
k6_g = dist_c[0][7]
p1_g = dist_c[0][2]
p2_g = dist_c[0][3]
s1_g = dist_c[0][8]
s2_g = dist_c[0][9]
s3_g = dist_c[0][10]
s4_g = dist_c[0][11]

r_new = np.sqrt((x/fx_g)**2+(y/fy_g)**2)  
x_g = fx_c*full_dist_model_x(k1_g, k2_g, k3_g, k4_g, k5_g, k6_g, p1_g, p2_g, s1_g, s2_g, x/fx_g, y/fy_g, r_new)
y_g = fx_c*full_dist_model_y(k1_g, k2_g, k3_g, k4_g, k5_g, k6_g, p1_g, p2_g, s3_g, s4_g, x/fx_g, y/fy_g, r_new)
r_g = np.sqrt(x_g**2+y_g**2)



fig, ax = plt.subplots(1,1)
ax.plot(r, r_m-r, label='Model')
ax.plot(r, r_s-r, label='0')
ax.plot(r, r_c-r, label='1')
ax.plot(r, r_g-r, label='Gauss')
ax.grid(True)
ax.legend()

# with open('params_s.txt', 'w') as f:
#     f.write('RMS reprojection error = {:}\n\n'.format(ret_s))
#     f.write('Camera Matrix:\n')
#     f.write('fx = {:} +/- {:}\n'.format(mtx_s[0][0], stdin_s[0][0]))
#     f.write('fy = {:} +/- {:}\n'.format(mtx_s[1][1], stdin_s[1][0]))
#     f.write('cx = {:} +/- {:}\n'.format(mtx_s[0][2], stdin_s[2][0]))
#     f.write('cy = {:} +/- {:}\n\n'.format(mtx_s[1][2], stdin_s[3][0]))
#     f.write('Radial Distortion:\n')
#     f.write('k1 = {:} +/- {:}\n'.format(dist_s[0][0], stdin_s[4][0]))
#     f.write('k2 = {:} +/- {:}\n'.format(dist_s[0][1], stdin_s[5][0]))
#     f.write('k3 = {:} +/- {:}\n'.format(dist_s[0][4], stdin_s[8][0]))
#     f.write('k4 = {:} +/- {:}\n'.format(dist_s[0][5], stdin_s[9][0]))
#     f.write('k5 = {:} +/- {:}\n'.format(dist_s[0][6], stdin_s[10][0]))
#     f.write('k6 = {:} +/- {:}\n\n'.format(dist_s[0][7], stdin_s[11][0]))
#     f.write('Tangential Distortion\n')
#     f.write('p1 = {:} /- {:}\n'.format(dist_s[0][2], stdin_s[6][0]))
#     f.write('p2 = {:} /- {:}\n\n'.format(dist_s[0][3], stdin_s[7][0]))
#     f.write('Thin Prism Distortion:\n')
#     f.write('s1 = {:} +/- {:}\n'.format(dist_s[0][8], stdin_s[12][0]))
#     f.write('s2 = {:} +/- {:}\n'.format(dist_s[0][9], stdin_s[13][0]))
#     f.write('s3 = {:} +/- {:}\n'.format(dist_s[0][10], stdin_s[14][0]))
#     f.write('s4 = {:} +/- {:}\n\n'.format(dist_s[0][11], stdin_s[15][0]))

# with open('params_c.txt', 'w') as f:
#     f.write('RMS reprojection error = {:}\n\n'.format(ret_c))
#     f.write('Camera Matrix:\n')
#     f.write('fx = {:} +/- {:}\n'.format(mtx_c[0][0], stdin_c[0][0]))
#     f.write('fy = {:} +/- {:}\n'.format(mtx_c[1][1], stdin_c[1][0]))
#     f.write('cx = {:} +/- {:}\n'.format(mtx_c[0][2], stdin_c[2][0]))
#     f.write('cy = {:} +/- {:}\n\n'.format(mtx_c[1][2], stdin_c[3][0]))
#     f.write('Radial Distortion:\n')
#     f.write('k1 = {:} +/- {:}\n'.format(dist_c[0][0], stdin_c[4][0]))
#     f.write('k2 = {:} +/- {:}\n'.format(dist_c[0][1], stdin_c[5][0]))
#     f.write('k3 = {:} +/- {:}\n'.format(dist_c[0][4], stdin_c[8][0]))
#     f.write('k4 = {:} +/- {:}\n'.format(dist_c[0][5], stdin_c[9][0]))
#     f.write('k5 = {:} +/- {:}\n'.format(dist_c[0][6], stdin_c[10][0]))
#     f.write('k6 = {:} +/- {:}\n\n'.format(dist_c[0][7], stdin_c[11][0]))
#     f.write('Tangential Distortion\n')
#     f.write('p1 = {:} /- {:}\n'.format(dist_c[0][2], stdin_c[6][0]))
#     f.write('p2 = {:} /- {:}\n\n'.format(dist_c[0][3], stdin_c[7][0]))
#     f.write('Thin Prism Distortion:\n')
#     f.write('s1 = {:} +/- {:}\n'.format(dist_c[0][8], stdin_c[12][0]))
#     f.write('s2 = {:} +/- {:}\n'.format(dist_c[0][9], stdin_c[13][0]))
#     f.write('s3 = {:} +/- {:}\n'.format(dist_c[0][10], stdin_c[14][0]))
#     f.write('s4 = {:} +/- {:}\n\n'.format(dist_c[0][11], stdin_c[15][0]))

# with open('params_g.txt', 'w') as f:
#     f.write('RMS reprojection error = {:}\n\n'.format(ret_g))
#     f.write('Camera Matrix:\n')
#     f.write('fx = {:} +/- {:}\n'.format(mtx_g[0][0], stdin_g[0][0]))
#     f.write('fy = {:} +/- {:}\n'.format(mtx_g[1][1], stdin_g[1][0]))
#     f.write('cx = {:} +/- {:}\n'.format(mtx_g[0][2], stdin_g[2][0]))
#     f.write('cy = {:} +/- {:}\n\n'.format(mtx_g[1][2], stdin_g[3][0]))
#     f.write('Radial Distortion:\n')
#     f.write('k1 = {:} +/- {:}\n'.format(dist_g[0][0], stdin_g[4][0]))
#     f.write('k2 = {:} +/- {:}\n'.format(dist_g[0][1], stdin_g[5][0]))
#     f.write('k3 = {:} +/- {:}\n'.format(dist_g[0][4], stdin_g[8][0]))
#     f.write('k4 = {:} +/- {:}\n'.format(dist_g[0][5], stdin_g[9][0]))
#     f.write('k5 = {:} +/- {:}\n'.format(dist_g[0][6], stdin_g[10][0]))
#     f.write('k6 = {:} +/- {:}\n\n'.format(dist_g[0][7], stdin_g[11][0]))
#     f.write('Tangential Distortion\n')
#     f.write('p1 = {:} /- {:}\n'.format(dist_g[0][2], stdin_g[6][0]))
#     f.write('p2 = {:} /- {:}\n\n'.format(dist_g[0][3], stdin_g[7][0]))
#     f.write('Thin Prism Distortion:\n')
#     f.write('s1 = {:} +/- {:}\n'.format(dist_g[0][8], stdin_g[12][0]))
#     f.write('s2 = {:} +/- {:}\n'.format(dist_g[0][9], stdin_g[13][0]))
#     f.write('s3 = {:} +/- {:}\n'.format(dist_g[0][10], stdin_g[14][0]))
#     f.write('s4 = {:} +/- {:}\n\n'.format(dist_g[0][11], stdin_g[15][0]))

# print elapsed time
print((time.time()-t_ref)/60) 
