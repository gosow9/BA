import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import time

def radialDistortion(k1, k2, k3, k4 ,k5, k6, s , r):
    return s*(1 + k1*r**2 + k2*r**4 + k3*r**6)/(1 + k4*r**2 + k5*r**4 + k6*r**6) 

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
ax.set_title(r'6 radial')
ax.set_xlabel(r'Radius (pixel)')
ax.set_ylabel(r'Distortion (pixel)')

# load data
files = glob.glob('checkerboards/*.npy')
n = int(len(files)/2)

f = open('6k_params.txt', 'w')

for i in range(n):
    objp = np.load('checkerboards/objp{:}.npy'.format(i))
    imgp = np.load('checkerboards/imgp{:}.npy'.format(i))

    # select model with 5 ks and ps set to 0
    flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_ZERO_TANGENT_DIST
    ret, mtx, dist, rvecs, tvecs, newobjp, stdin, stdex, pve, stdnewobjp = cv2.calibrateCameraROExtended(objp, imgp, (3280, 2464), 1, None, None, flags=flags)
    
    
    
    # generate plot       
    k1 = dist[0][0]
    k2 = dist[0][1]
    k3 = dist[0][4]
    k4 = dist[0][5]
    k5 = dist[0][6]
    k6 = dist[0][7]
    
    std_k1 = stdin[4][0]
    std_k2 = stdin[5][0]
    std_k3 = stdin[8][0]
    std_k4 = stdin[9][0]
    std_k5 = stdin[10][0]
    std_k6 = stdin[11][0]
    
    fx = mtx[0][0]
    fy = mtx[1][1]    
    
    new_r = np.sqrt((x/fx)**2+(y/fy)**2)
    
    dist_x = fx*radialDistortion(k1, k2, k3, k4, k5, k6, x/fx, new_r)
    dist_y = fy*radialDistortion(k1, k2, k3, k4, k5, k6, y/fy, new_r)
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
    f.write('k6 = {:.2} +/- {:.2}\n\n'.format(k6, std_k6))    

f.close() 
fig.savefig('6k.pdf')
    
# print elapsed time
print((time.time()-t_ref)/60)
