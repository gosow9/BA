import numpy as np
from pylab import rcParams
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab


# get the data
static1 = np.loadtxt('static1.txt', skiprows=1, delimiter=';')
static2 = np.loadtxt('static2.txt', skiprows=1, delimiter=';')
static3 = np.loadtxt('static3.txt', skiprows=1, delimiter=';')
static4 = np.loadtxt('static4.txt', skiprows=1, delimiter=';')
static5 = np.loadtxt('static5.txt', skiprows=1, delimiter=';')
static6 = np.loadtxt('static6.txt', skiprows=1, delimiter=';')

dynamic1 = np.loadtxt('dynamic1.txt', skiprows=1, delimiter=';')
dynamic2 = np.loadtxt('dynamic2.txt', skiprows=1, delimiter=';')
dynamic3 = np.loadtxt('dynamic3.txt', skiprows=1, delimiter=';')
dynamic4 = np.loadtxt('dynamic4.txt', skiprows=1, delimiter=';')
dynamic5 = np.loadtxt('dynamic5.txt', skiprows=1, delimiter=';')
dynamic6 = np.loadtxt('dynamic6.txt', skiprows=1, delimiter=';')

dynamic_full = np.loadtxt('dynamic4_full.txt', skiprows=1, delimiter=';')
static_full = np.loadtxt('static4_full.txt', skiprows=1, delimiter=';')

L1_s = static1.T[0]
D1_s = static1.T[1]
L2_s = static2.T[0]
D2_s = static2.T[1]
L3_s = static3.T[0]
D3_s = static3.T[1]
L4_s = static4.T[0]
D4_s = static4.T[1]
L5_s = static5.T[0]
D5_s = static5.T[1]
L6_s = static6.T[0]
D6_s = static6.T[1]

L1_d = dynamic1.T[0]
D1_d = dynamic1.T[1]
L2_d = dynamic2.T[0]
D2_d = dynamic2.T[1]
L3_d = dynamic3.T[0]
D3_d = dynamic3.T[1]
L4_d = dynamic4.T[0]
D4_d = dynamic4.T[1]
L5_d = dynamic5.T[0]
D5_d = dynamic5.T[1]
L6_d = dynamic6.T[0]
D6_d = dynamic6.T[1]

L_df = dynamic_full.T[0]
D_df = dynamic_full.T[1]

L_sf = static_full.T[0]
D_sf = static_full.T[1]


# Gaussian
res = 200
x_L = np.arange(205, 220, (220-205)/res)

mu, std = norm.fit(L_df)
p_Ld = norm.pdf(x_L, mu, std)
mu, std = norm.fit(L_sf)
p_Ls = norm.pdf(x_L, mu, std)


res = 200
x_D = np.arange(45, 60, (60-45)/res)
mu, std = norm.fit(D_sf)
p_Ds = norm.pdf(x_D, mu, std)
mu, std = norm.fit(D_df)
p_Dd = norm.pdf(x_D, mu, std)


# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

# L
fig1, ax1 = plt.subplots(1,1)
ax1.hist(L_sf, bins=22, density=True, color='tab:orange', alpha=0.3, edgecolor='tab:orange')
ax1.plot(x_L, p_Ls, color='tab:orange', label=r'$\mu = ${:.2f}, $\sigma = ${:.2f}'.format(np.mean(L_sf), np.std(L_sf)))
ax1.hist(L_df, bins=22, density=True, color='tab:blue', alpha=0.4, edgecolor='tab:blue')
ax1.plot(x_L, p_Ld, color='tab:blue', label=r'$\mu = ${:.2f}, $\sigma = ${:.2f}'.format(np.mean(L_df), np.std(L_df)))

ax1.set_xlim([205, 220])
ax1.set_xticks([205, 210, 215, 220])

ax1.set_ylim([0, 1])
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

ax1.set_xlabel('length (mm)')
ax1.grid(True)
ax1.legend()

# D
fig2, ax2 = plt.subplots(1,1)
ax2.hist(D_sf, bins=22, density=True, color='tab:orange', alpha=0.3, edgecolor='tab:orange')
ax2.plot(x_D, p_Ds, color='tab:orange', label=r'$\mu = ${:.2f}, $\sigma = ${:.2f}'.format(np.mean(D_sf), np.std(D_sf)))
ax2.hist(D_df, bins=22, density=True, color='tab:blue', alpha=0.4, edgecolor='tab:blue')
ax2.plot(x_D, p_Dd, color='tab:blue', label=r'$\mu = ${:.2f}, $\sigma = ${:.2f}'.format(np.mean(D_df), np.std(D_df)))

ax2.set_xlim([45, 60])
ax2.set_xticks([45, 50, 55, 60])

ax2.set_ylim([0, 1.4])
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4])

ax2.set_xlabel('diameter (mm)')
ax2.grid(True)
ax2.legend()

fig1.savefig('hist_length.pdf')
fig2.savefig('hist_diameter.pdf')



error_L_s = np.std(L_sf)/np.mean(L_sf)
error_D_s = np.std(D_sf)/np.mean(D_sf)

error_L_d = np.std(L_df)/np.mean(L_df)
error_D_d = np.std(D_df)/np.mean(D_df)

print('dyn. rel. error: D = {:.2f}, L = {:.2f}'.format(error_D_d*100, error_L_d*100))
print('stat. rel. error: D = {:.2f}, L = {:.2f}'.format(error_D_s*100, error_L_s*100))


