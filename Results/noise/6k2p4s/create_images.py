import numpy as np
import cv2
import random
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


def checkerboard(r, t):
    # empty image
    im = np.ones((2464, 3280), dtype=np.uint8)*170

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
        
    return im

t_ref = time.time()

# define rotation and translation
r1 = np.array([0.7, 0.7, 0], dtype=np.float64)
t1 = np.array([50, -80, 200], dtype=np.float64)
r2 = np.array([-0.7, 0.7, 0], dtype=np.float64)
t2 = np.array([50, 80, 200], dtype=np.float64)
r3 = np.array([0.7, -0.7, 0], dtype=np.float64)
t3 = np.array([-50, -80, 200], dtype=np.float64)
r4 = np.array([-0.7, -0.7, 0], dtype=np.float64)
t4 = np.array([-50, 80, 200], dtype=np.float64)

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

r = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24]
t = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24]

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


# add noise to the first 20
for i in range(20):
    t[i][0] += np.random.uniform(-20, 20)
    t[i][1] += np.random.uniform(-20, 20)
    t[i][2] += np.random.uniform(-20, 20)
    r[i][0] += np.random.uniform(-0.1, 0.1)
    r[i][1] += np.random.uniform(-0.1, 0.1)
    r[i][2] += np.random.uniform(-0.1, 0.1)    

for i in range(len(t)):
    im = checkerboard(r[i], t[i])
    np.savetxt('np_images/im{:}.txt'.format(i), im)
    cv2.imwrite('png_images/im{:}.png'.format(i), im)

    print(i)
    # show image
    # cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('im', 820, 616)
    # cv2.imshow('im', im)
    # cv2.waitKey()

# print elapsed time
print((time.time()-t_ref)/60) 
