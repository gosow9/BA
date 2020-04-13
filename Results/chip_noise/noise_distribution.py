import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm
from pylab import rcParams

img = cv2.imread('uniform1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, edg = np.histogram(gray.ravel(), bins=256, range=(0.0, 256))
h = h/np.sum(h)
x = np.arange(0, 256, 1)

# fit gauss
mu, std = norm.fit(gray)
p = norm.pdf(x, mu, std)

# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1,1)
ax.plot(x, p, color='tab:orange', label='Gaussian with\n'+r'$\mu$={:.1f}, $\sigma$={:.1f}'.format(mu, std))
ax.plot(x, h, color='tab:blue', label='Histogramm')

ax.set_xlim([0, 255])
ax.set_xticks([0, 63, 127, 191, 255])
ax.set_ylim(0,0.02)
ax.set_yticks([0, 0.005, 0.01, 0.015, 0.02])
ax.set_xlabel('Intensity')
ax.grid(True)
ax.legend()

fig.savefig('noise_distribution.pdf')
