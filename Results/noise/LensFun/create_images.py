import numpy as np
import cv2
import random
import time

def ptlens(r):
    a = 0.016943
    b = -0.0048353
    c = 0
    return r*(1-a-b-c+c*r+b*r**2+a*r**3)

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
                        r_u = np.sqrt(x_2D**2+y_2D**2)
                        
                        s = 1
                        if y_2D <=0:
                            s = -1
                        
                        r_d = ptlens(r_u) 
                        x_2D_dst = s*f*r_d*(x_2D/(y_2D*np.sqrt((x_2D**2/y_2D**2)+1)))
                        y_2D_dst = s*f*r_d*(1/np.sqrt((x_2D**2/y_2D**2)+1))
                    
                        # new index with respect to image-cooridnates
                        x_new = int(round(x_2D_dst + x_im))
                        y_new = int(round(y_2D_dst + y_im))
                        
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

r = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20]
t = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20]

# distortion model
f = 2700

# add noise
for i in range(20):
    t[i][0] += np.random.normal(0, 10)
    t[i][1] += np.random.normal(0, 10)
    t[i][2] += np.random.normal(0, 10)
    r[i][0] += np.random.normal(0, 0.05)
    r[i][1] += np.random.normal(0, 0.05)
    r[i][2] += np.random.normal(0, 0.05)    

for i in range(20):
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
