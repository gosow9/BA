import numpy as np
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

# load results
std_c = np.loadtxt('params/std.txt')
rpe_c = np.loadtxt('params/rpe.txt')
sharp_c = np.loadtxt('params/sharp.txt')
fx_c = np.loadtxt('params/fx.txt')
fy_c = np.loadtxt('params/fy.txt')
cx_c = np.loadtxt('params/cx.txt')
cy_c = np.loadtxt('params/cy.txt')
k1_c = np.loadtxt('params/k1.txt')
k2_c = np.loadtxt('params/k2.txt')
k3_c = np.loadtxt('params/k3.txt')
k4_c = np.loadtxt('params/k4.txt')
k5_c = np.loadtxt('params/k5.txt')
k6_c = np.loadtxt('params/k6.txt')
p1_c = np.loadtxt('params/p1.txt')
p2_c = np.loadtxt('params/p2.txt')
s1_c = np.loadtxt('params/s1.txt')
s2_c = np.loadtxt('params/s2.txt')
s3_c = np.loadtxt('params/s3.txt')
s4_c = np.loadtxt('params/s4.txt')

# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12}) 


# x and y
res = 1
x = np.arange(0, 2464/2, res)
y = np.arange(0, 3280/2, res/2464*3280)
r = np.sqrt(x**2+y**2)

# compute model
norm_r_m = np.sqrt((x/f)**2+(y/f)**2)
dst_x_m = f*full_dist_model_x(k1_m, k2_m, k3_m, k4_m, k5_m, k6_m, p1_m, p2_m, s1_m, s2_m, x/f, y/f, norm_r_m)
dst_y_m = f*full_dist_model_y(k1_m, k2_m, k3_m, k4_m, k5_m, k6_m, p1_m, p2_m, s3_m, s4_m, x/f, y/f, norm_r_m)
dst_r_m = np.sqrt(dst_x_m**2+dst_y_m**2)

# fig1: curves
fig1, ax1 = plt.subplots(1,1)
ax1.plot(r, dst_r_m-r, label='Model')

# sigma = 0
std = [0, 10, 20, 30]
for i in std: 
    # comput curve
    norm_r_0 = np.sqrt((x/fx_c[i])**2+(y/fy_c[i])**2)
    dst_x_0 = fx_c[i]*full_dist_model_x(k1_c[i], k2_c[i], k3_c[i], k4_c[i], k5_c[i], k6_c[i], p1_c[i], p2_c[i], s1_c[i], s2_c[i], x/fx_c[i], y/fy_c[i], norm_r_0)
    dst_y_0 = fy_c[i]*full_dist_model_y(k1_c[i], k2_c[i], k3_c[i], k4_c[i], k5_c[i], k6_c[i], p1_c[i], p2_c[i], s3_c[i], s4_c[i], x/fx_c[i], y/fy_c[i], norm_r_0)
    dst_r_0 = np.sqrt(dst_x_0**2+dst_y_0**2)

    ax1.plot(r, dst_r_0-r, label=r'$\sigma = {:}$'.format(std_c[i]))

ax1.set_xlabel('Radius')
ax1.set_ylabel('Distortion')
ax1.grid(True)
ax1.legend()

# fig2: mean error
fig2, ax2 = plt.subplots(1,1)

me = []
for i in range(len(fx_c)):
    # comput curve
    norm_r_0 = np.sqrt((x/fx_c[i])**2+(y/fy_c[i])**2)
    dst_x_0 = fx_c[i]*full_dist_model_x(k1_c[i], k2_c[i], k3_c[i], k4_c[i], k5_c[i], k6_c[i], p1_c[i], p2_c[i], s1_c[i], s2_c[i], x/fx_c[i], y/fy_c[i], norm_r_0)
    dst_y_0 = fy_c[i]*full_dist_model_y(k1_c[i], k2_c[i], k3_c[i], k4_c[i], k5_c[i], k6_c[i], p1_c[i], p2_c[i], s3_c[i], s4_c[i], x/fx_c[i], y/fy_c[i], norm_r_0)
    dst_r_0 = np.sqrt(dst_x_0**2+dst_y_0**2)

    # copmute mean error
    me.append(np.mean(np.abs(dst_r_m-dst_r_0)))
 
ax2.set_xlabel('Standard Deviation')
ax2.set_ylabel('Mean Error')
ax2.plot(std_c, me, 'o')
ax2.grid(True)

# fig3: reprojection error
fig3, ax3 = plt.subplots(1,1)
ax3.plot(std_c, rpe_c)
ax3.set_xlabel('Standard Deviation')
ax3.set_ylabel('RMS Reprojection Error')
ax3.grid(True)

# fig4: sharpness
fig4, ax4 = plt.subplots(1,1)
ax4.plot(std_c, sharp_c)
ax4.set_xlabel('Standard Deviation')
ax4.set_ylabel('Sharpness')
ax4.grid(True)

# fig5: k
fig5, ax5 = plt.subplots(2,3)
ax5[0][0].plot(std_c, k1_c)
ax5[0][0].set_xlabel('Standard Deviation')
ax5[0][0].set_ylabel('k1')
ax5[0][0].grid(True)

ax5[0][1].plot(std_c, k2_c)
ax5[0][1].set_xlabel('Standard Deviation')
ax5[0][1].set_ylabel('k2')
ax5[0][1].grid(True)

ax5[0][2].plot(std_c, k3_c)
ax5[0][2].set_xlabel('Standard Deviation')
ax5[0][2].set_ylabel('k3')
ax5[0][2].grid(True)

ax5[1][0].plot(std_c, k4_c)
ax5[1][0].set_xlabel('Standard Deviation')
ax5[1][0].set_ylabel('k4')
ax5[1][0].grid(True)

ax5[1][1].plot(std_c, k5_c)
ax5[1][1].set_xlabel('Standard Deviation')
ax5[1][1].set_ylabel('k5')
ax5[1][1].grid(True)

ax5[1][2].plot(std_c, k6_c)
ax5[1][2].set_xlabel('Standard Deviation')
ax5[1][2].set_ylabel('k6')
ax5[1][2].grid(True)

# fig6: p
fig6, ax6 = plt.subplots(1, 2)
ax6[0].plot(std_c, p1_c)
ax6[0].set_xlabel('Standard Deviation')
ax6[0].set_ylabel('p1')
ax6[0].grid(True)

ax6[1].plot(std_c, p2_c)
ax6[1].set_xlabel('Standard Deviation')
ax6[1].set_ylabel('21')
ax6[1].grid(True)

# fig7: s
fig7, ax7 = plt.subplots(2,2)
ax7[0][0].plot(std_c, s1_c)
ax7[0][0].set_xlabel('Standard Deviation')
ax7[0][0].set_ylabel('s1')
ax7[0][0].grid(True)

ax7[0][1].plot(std_c, s2_c)
ax7[0][1].set_xlabel('Standard Deviation')
ax7[0][1].set_ylabel('s2')
ax7[0][1].grid(True)

ax7[1][0].plot(std_c, s3_c)
ax7[1][0].set_xlabel('Standard Deviation')
ax7[1][0].set_ylabel('s3')
ax7[1][0].grid(True)

ax7[1][1].plot(std_c, s4_c)
ax7[1][1].set_xlabel('Standard Deviation')
ax7[1][1].set_ylabel('s4')
ax7[1][1].grid(True)

# fig8: intrinsic
fig8, ax8 = plt.subplots(1,1)
ax8.plot(std_c, fx_c-f, label='fx')
ax8.plot(std_c, fy_c-f, label='fy')
ax8.plot(std_c, cx_c-cx_m, label='cx')
ax8.plot(std_c, cy_c-cy_m, label='cy')
ax8.set_xlabel('Standard Deviation')
ax8.set_ylabel('Error')
ax8.legend()
ax8.grid(True)

# fig9: sigma0-model
# comput curve
i = 0
norm_r_0 = np.sqrt((x/fx_c[i])**2+(y/fy_c[i])**2)
dst_x_0 = fx_c[i]*full_dist_model_x(k1_c[i], k2_c[i], k3_c[i], k4_c[i], k5_c[i], k6_c[i], p1_c[i], p2_c[i], s1_c[i], s2_c[i], x/fx_c[i], y/fy_c[i], norm_r_0)
dst_y_0 = fy_c[i]*full_dist_model_y(k1_c[i], k2_c[i], k3_c[i], k4_c[i], k5_c[i], k6_c[i], p1_c[i], p2_c[i], s3_c[i], s4_c[i], x/fx_c[i], y/fy_c[i], norm_r_0)
dst_r_0 = np.sqrt(dst_x_0**2+dst_y_0**2)

fig9, ax9 = plt.subplots(1,1)
ax9.plot(r, dst_r_0-dst_r_m)



