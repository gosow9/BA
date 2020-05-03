import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
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

# x and y
res = 1
x = np.arange(0, 2464/2, res)
y = np.arange(0, 3280/2, res/2464*3280)
r = np.sqrt(x**2+y**2)

# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

# prepare plots
fig, ax = plt.subplots(1, 1)
ax.set_title(r'6 radial, 2 tangential, 4 prism')
ax.set_xlabel(r'Radius (pixel)')
ax.set_ylabel(r'Distortion (pixel)')

# load data
files = glob.glob('checkerboards/*.npy')
n = int(len(files)/2)

f = open('6k2p4s_params.txt', 'w')

for i in range(n):
    objp = np.load('checkerboards/objp{:}.npy'.format(i))
    imgp = np.load('checkerboards/imgp{:}.npy'.format(i))

    # select model with 6 ks, 2ps and 2 4'
    flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
    ret, mtx, dist, rvecs, tvecs, newobjp, stdin, stdex, pve, stdnewobjp = cv2.calibrateCameraROExtended(objp, imgp, (3280, 2464), 1, None, None, flags=flags)
      
    # generate plot       
    k1 = dist[0][0]
    k2 = dist[0][1]
    k3 = dist[0][4]
    k4 = dist[0][5]
    k5 = dist[0][6]
    k6 = dist[0][7]
    p1 = dist[0][2]
    p2 = dist[0][3]
    s1 = dist[0][8]
    s2 = dist[0][9]
    s3 = dist[0][10]
    s4 = dist[0][11]
   
    std_k1 = stdin[4][0]
    std_k2 = stdin[5][0]
    std_k3 = stdin[8][0]
    std_k4 = stdin[9][0]
    std_k5 = stdin[10][0]
    std_k6 = stdin[11][0]
    std_p1 = stdin[6][0]
    std_p2 = stdin[7][0]
    std_s1 = stdin[12][0]
    std_s2 = stdin[13][0]
    std_s3 = stdin[14][0]
    std_s4 = stdin[15][0]

    fx = mtx[0][0]
    fy = mtx[1][1]    
    
    new_r = np.sqrt((x/fx)**2+(y/fy)**2)
    
    dist_x = fx*full_dist_model_x(k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, x/fx, y/fx, new_r)
    dist_y = fy*full_dist_model_y(k1, k2, k3, k4, k5, k6, p1, p2, s3, s4, x/fx, y/fy, new_r)
    dist_r = np.sqrt(dist_x**2 + dist_y**2)
    
    label = '{:}: re = {:.2}'.format(i+1, ret)
    ax.plot(r, dist_r-r, label=label)
    ax.legend()
    
    # write to file f
    f.write('{:}.\n'.format(i+1))
    f.write('k1 = {:.2} +/- {:.2}\n'.format(k1, std_k1))
    f.write('k2 = {:.2} +/- {:.2}\n'.format(k2, std_k2))
    f.write('k3 = {:.2} +/- {:.2}\n'.format(k3, std_k3))
    f.write('k4 = {:.2} +/- {:.2}\n'.format(k4, std_k4))
    f.write('k5 = {:.2} +/- {:.2}\n'.format(k5, std_k5))
    f.write('k6 = {:.2} +/- {:.2}\n'.format(k6, std_k6))
    f.write('p1 = {:.2} +/- {:.2}\n'.format(p1, std_p1))
    f.write('p2 = {:.2} +/- {:.2}\n'.format(p2, std_p2))
    f.write('s1 = {:.2} +/- {:.2}\n'.format(s1, std_s1))
    f.write('s2 = {:.2} +/- {:.2}\n'.format(s2, std_s2))
    f.write('s3 = {:.2} +/- {:.2}\n'.format(s3, std_s3))
    f.write('s4 = {:.2} +/- {:.2}\n\n'.format(s4, std_s4))
   
f.close() 
fig.savefig('6k2p4s.pdf')
    
# print elapsed time
print((time.time()-t_ref)/60)
