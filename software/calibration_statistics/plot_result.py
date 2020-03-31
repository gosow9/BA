import numpy as np
import matplotlib.pyplot as plt

def r_dist_model(k1, k2, k3, s, r):
    return s*(1 + k1*r**2 + k2*r**4+k3*r**6)

def t_dist_model_x(p1, p2, x, y, r):
    return x + 2*p1*x*y+p2*(r+2*x**2)

def t_dist_model_y(p1, p2, x, y, r):
    return y +  2*p2*x*y+p1*(r+2*y**2)

def full_dist_model_x(k1, k2, k3, p1, p2, x, y, r):
    r_dist = r_dist_model(k1, k2, k3, x, r)
    t_dist = t_dist_model_x(p1, p1, x, y, r)
    return r_dist + t_dist - x
    
def full_dist_model_y(k1, k2, k3, p1, p2, x, y, r):
    r_dist = r_dist_model(k1, k2, k3, y, r)
    t_dist = t_dist_model_y(p1, p1, x, y, r)
    return r_dist + t_dist - y


# load data
m_error_20 = np.loadtxt('random_result/m_error_20.txt')
r_dist_20 = np.loadtxt('random_result/r_dist_20.txt')
mtx_20 = np.loadtxt('random_result/mtx_20.txt')

m_error_22 = np.loadtxt('random_result/m_error_22.txt')
r_dist_22 = np.loadtxt('random_result/r_dist_22.txt')
t_dist_22 = np.loadtxt('random_result/t_dist_22.txt')
mtx_22 = np.loadtxt('random_result/mtx_22.txt')

m_error_30 = np.loadtxt('random_result/m_error_30.txt')
r_dist_30 = np.loadtxt('random_result/r_dist_30.txt')
mtx_30 = np.loadtxt('random_result/mtx_30.txt')

m_error_32 = np.loadtxt('random_result/m_error_32.txt')
r_dist_32 = np.loadtxt('random_result/r_dist_32.txt')
t_dist_32 = np.loadtxt('random_result/t_dist_32.txt')
mtx_32 = np.loadtxt('random_result/mtx_32.txt')

# x and y
res = 1
x = np.arange(0, 2464/2, res)
y = np.arange(0, 3280/2, res/2464*3280)
r = np.sqrt(x**2+y**2)

# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

# prepare plots
fig20, ax20 = plt.subplots(1, 1)
ax20.set_title(r'2 radial, 0 tangetial (full)')
ax20.set_xlabel(r'Radius (pixel)')
ax20.set_ylabel(r'Distortion (pixel)')

fig22_f, ax22_f = plt.subplots(1, 1)
ax22_f.set_title(r'2 radial, 2 tangential (full)')
ax22_f.set_xlabel(r'Radius (pixel)')
ax22_f.set_ylabel(r'Distortion (pixel)')

fig22_d, ax22_d = plt.subplots(2,1)
ax22_d[0].set_title(r'2 radial, 2 tangential (radial)')
ax22_d[0].set_xlabel(r'Radius (pixel)')
ax22_d[0].set_ylabel(r'Distortion (pixel)')
ax22_d[1].set_title(r'2 radial, 2 tangential (tangetial)')
ax22_d[1].set_xlabel(r'Radius (pixel)')
ax22_d[1].set_ylabel(r'Distortion (pixel)')

fig30, ax30 = plt.subplots(1, 1)
ax30.set_title(r'3 radial, 0 tangetial (full)')
ax30.set_xlabel(r'Radius (pixel)')
ax30.set_ylabel(r'Distortion (pixel)')

fig32_f, ax32_f = plt.subplots(1, 1)
ax32_f.set_title(r'3 radial, 2 tangential (full)')
ax32_f.set_xlabel(r'Radius (pixel)')
ax32_f.set_ylabel(r'Distortion (pixel)')

fig32_d, ax32_d = plt.subplots(2,1)
ax32_d[0].set_title(r'3 radial, 2 tangential (radial)')
ax32_d[0].set_xlabel(r'Radius (pixel)')
ax32_d[0].set_ylabel(r'Distortion (pixel)')
ax32_d[1].set_title(r'3 radial, 2 tangential (tangetial)')
ax32_d[1].set_xlabel(r'Radius (pixel)')
ax32_d[1].set_ylabel(r'Distortion (pixel)')

# generate plots
for i in range(len(m_error_20)):
    # figure 1 (2 radial, 0 tangential)
    k1 = r_dist_20[i][0]
    k2 = r_dist_20[i][1]
    fx = mtx_20[i*3][0]
    fy = mtx_20[i*3+1][1]
    new_r = np.sqrt((x/fx)**2+(y/fy)**2)
    l = '{:}: re = {:.2}, k1 = {:.2}, k2 = {:.2}'.format(i+1, m_error_20[i], k1, k2)
    dst_x = fx*r_dist_model(k1, k2, 0, x/fx, new_r)
    dst_y = fy*r_dist_model(k1, k2, 0, y/fy, new_r)    
    dst_r = np.sqrt(dst_x**2+dst_y**2)    
    ax20.plot(r, dst_r-r, label=l)
    ax20.legend()

    #figure 2 (2 radial, 2 tangetial (full))
    k1 = r_dist_22[i][0]
    k2 = r_dist_22[i][1]
    p1 = t_dist_22[i][0]
    p2 = t_dist_22[i][1]
    fx = mtx_22[i*3][0]
    fy = mtx_22[i*3+1][1]
    new_r = np.sqrt((x/fx)**2+(y/fy)**2)
    # full model
    l = '{:}: re = {:.2}'.format(i+1, m_error_22[i])
    dst_x = fx*full_dist_model_x(k1, k2, 0, p1, p2, x/fx, y/fy, new_r)
    dst_y = fy*full_dist_model_y(k1, k2, 0, p1, p2, x/fx, y/fy, new_r)    
    dst_r = np.sqrt(dst_x**2+dst_y**2)    
    ax22_f.plot(r, dst_r-r, label=l)
    ax22_f.legend()
    # radial distortion
    l = '{:}; k1 = {:.2}, k2 = {:.2}'.format(i+1, k1, k2)
    dst_x = fx*r_dist_model(k1, k2, 0, x/fx, new_r)
    dst_y = fy*r_dist_model(k1, k2, 0, y/fy, new_r)    
    dst_r = np.sqrt(dst_x**2+dst_y**2)    
    ax22_d[0].plot(r, dst_r-r, label=l)
    ax22_d[0].legend()
    # tangential_distortion
    l = '{:}; p1 = {:.2}, p2 = {:.2}'.format(i+1, p1, p2)
    dst_x = fx*t_dist_model_x(p1, p2, x/fx, y/fy, new_r)
    dst_y = fy*t_dist_model_y(p1, p2, x/fx, y/fy, new_r)    
    dst_r = np.sqrt(dst_x**2+dst_y**2)    
    ax22_d[1].plot(r, dst_r-r, label=l)
    ax22_d[1].legend()
    
    # figure 5 (3 radial, 0 tangential)
    k1 = r_dist_30[i][0]
    k2 = r_dist_30[i][1]
    k3 = r_dist_30[i][2]
    fx = mtx_30[i*3][0]
    fy = mtx_30[i*3+1][1]
    new_r = np.sqrt((x/fx)**2+(y/fy)**2)
    l = '{:}: re = {:.2}, k1 = {:.2}, k2 = {:.2}, k3 = {:.2}'.format(i+1, m_error_30[i], k1, k2, k3)
    dst_x = fx*r_dist_model(k1, k2, k3, x/fx, new_r)
    dst_y = fy*r_dist_model(k1, k2, k3, y/fy, new_r)    
    dst_r = np.sqrt(dst_x**2+dst_y**2)    
    ax30.plot(r, dst_r-r, label=l)
    ax30.legend()
    
    #figure 4 (3 radial, 2 tangetial)
    k1 = r_dist_32[i][0]
    k2 = r_dist_32[i][1]
    k3 = r_dist_32[i][2]
    p1 = t_dist_32[i][0]
    p2 = t_dist_32[i][1]
    fx = mtx_32[i*3][0]
    fy = mtx_32[i*3+1][1]
    new_r = np.sqrt((x/fx)**2+(y/fy)**2)
    # full model
    l = '{:}: re = {:.2}'.format(i+1, m_error_32[i])
    dst_x = fx*full_dist_model_x(k1, k2, k3, p1, p2, x/fx, y/fy, new_r)
    dst_y = fy*full_dist_model_y(k1, k2, k3, p1, p2, x/fx, y/fy, new_r)    
    dst_r = np.sqrt(dst_x**2+dst_y**2)    
    ax32_f.plot(r, dst_r-r, label=l)
    ax32_f.legend()
    # radial distortion
    l = '{:}: k1 = {:.2}, k2 = {:.2}, k3 = {:.2}'.format(i+1, k1, k2, k3)
    dst_x = fx*r_dist_model(k1, k2, k3, x/fx, new_r)
    dst_y = fy*r_dist_model(k1, k2, k3, y/fy, new_r)    
    dst_r = np.sqrt(dst_x**2+dst_y**2)    
    ax32_d[0].plot(r, dst_r-r, label=l)
    ax32_d[0].legend()
    # tangential_distortion
    l = '{:}: p1 = {:.2}, p2 = {:.2}'.format(i+1, p1, p2)
    dst_x = fx*t_dist_model_x(p1, p2, x/fx, y/fy, new_r)
    dst_y = fy*t_dist_model_y(p1, p2, x/fx, y/fy, new_r)    
    dst_r = np.sqrt(dst_x**2+dst_y**2)    
    ax32_d[1].plot(r, dst_r-r, label=l)
    ax32_d[1].legend()
    
fig20.savefig('2r0t_full.pdf')
fig22_f.savefig('2r2t_full.pdf')
fig22_d.savefig('2r2t_seperate.pdf')
fig30.savefig('3r0t_full.pdf')
fig32_f.savefig('3r3t_full.pdf')
fig32_d.savefig('3r3t_seperate.pdf')