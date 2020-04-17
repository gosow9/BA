import numpy as np
import matplotlib.pyplot as plt
import cv2
from pylab import rcParams
from matplotlib.ticker import NullFormatter

img_cv = cv2.imread('covered.png')
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# crop chessboard
gray = gray[180:2464-180, 180:3280-180]

# histogramm
h, edg = np.histogram(gray.ravel(), bins=256, range=(0.0, 256))
h = h/np.sum(h)

# workaround
fill = np.ones((1000, 3280-360))*255
gray = np.concatenate([fill, gray])


# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1,2)
fig.tight_layout()

s = ax[0].imshow(gray, cmap='gray')
ax[0].set_axis_off()
    
ax[1].set_aspect(14000)

cb = plt.colorbar(s, orientation='horizontal', ax=ax[1], pad=0.01)
cb.set_label('Intensity')
cb.set_ticks([0, 63, 127, 191,255])

ax[1].plot(np.arange(0, 256, 1), h, color='tab:blue')
ax[1].set_xlim([0,255])
ax[1].set_xticks([0, 63, 127, 191, 255])
ax[1].xaxis.set_major_formatter(NullFormatter())
ax[1].set_yticks([0,0.005, 0.01, 0.015])
ax[1].set_ylim([0, 0.015])
ax[1].grid(True)

fig.savefig('covered.pdf')

