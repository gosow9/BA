import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import time

def distort(k1, k2, k3, k4, s, r):
    t = np.arctan(r)
    d = t*(1 + k1*t**2 + k2*t**4 + k3*t**6 + k4*t**8)
    
    return d/r*s


t_ref = time.time()

# x and y
res = 1
x = np.arange(0, 2464/2, res).astype(np.float32)
y = np.arange(0, 3280/2, res/2464*3280).astype(np.float32)
r = np.sqrt(x**2+y**2)


    
# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

# prepare plots
fig, ax = plt.subplots(1, 1)
ax.set_title(r'fisheye')
ax.set_xlabel(r'Radius (pixel)')
ax.set_ylabel(r'Distortion (pixel)')

# load data
files = glob.glob('checkerboards/*.npy')
n = int(len(files)/2)

f = open('fisheye_params.txt', 'w')

for i in range(n):
    objp = np.load('checkerboards/objp{:}.npy'.format(i))
    imgp = np.load('checkerboards/imgp{:}.npy'.format(i))
      
    
    flags = cv2.fisheye.CALIB_FIX_SKEW
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objp.reshape(20,1,70,3), imgp, (3280, 2464), None, None, flags=flags)      

    k1 = dist[0]
    k2 = dist[1]
    k3 = dist[2]
    k4 = dist[3]

    fx = mtx[0][0]
    fy = mtx[1][1]
    
    new_r = np.sqrt((x/fx)**2 + (y/fy)**2)
    dst_x = fx*distort(k1, k2, k3, k4, x/fx, new_r)
    dst_y = fy*distort(k1, k2, k3, k4, y/fy, new_r)
    dst_r = np.sqrt(dst_x**2 + dst_y**2)
    
    label = '{:}'.format(i+1)
    
    ax.plot(r, dst_r-r, label=label)
    ax.legend()


    # write to file f
    f.write('{:}.\n'.format(i+1))
    f.write('k1 = {:.2}\n'.format(dist[0][0]))
    f.write('k2 = {:.2}\n'.format(dist[1][0]))
    f.write('k3 = {:.2}\n'.format(dist[2][0]))
    f.write('k4 = {:.2}\n\n'.format(dist[3][0]))

f.close() 
fig.savefig('fisheye.pdf')    