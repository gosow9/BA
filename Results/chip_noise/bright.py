import numpy as np
import matplotlib.pyplot as plt
import cv2
from pylab import rcParams
from matplotlib.ticker import NullFormatter

img = cv2.imread('bright.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# crop chessboard
gray = gray[180:2464-180, 180:3280-180]
gray_c = cv2.cvtColor(gray.astype(np.float32), cv2.COLOR_GRAY2BGR);

h, edg = np.histogram(gray.ravel(), bins=256, range=(0.0, 256))
h = h/np.sum(h)

# workaround
fill = np.ones((1000, 3280-360))*255
gray = np.concatenate([fill, gray])
gray[0][0] = 0

# use latex-fonts in the plot
plt.rcParams['font.family'] = 'SIXTGeneral'
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1,2)
fig.tight_layout()

s = ax[0].imshow(gray, cmap='gray')
ax[0].set_axis_off()
    
ax[1].set_aspect(7000)

cb = plt.colorbar(s, orientation='horizontal', ax=ax[1], pad=0.01)
cb.set_label('Intensity')
cb.set_ticks([0, 63, 127, 191,255])

ax[1].plot(np.arange(0, 256, 1), h, color='tab:blue')
ax[1].set_xlim([0,255])
ax[1].set_xticks([0, 63, 127, 191, 255])
ax[1].xaxis.set_major_formatter(NullFormatter())
ax[1].set_yticks([0, 0.01, 0.02, 0.03])
ax[1].set_ylim([0, 0.030])
ax[1].grid(True)

fig.savefig('bright_hist.pdf')

# figure sharpness
gray_c = cv2.line(gray_c, (0, 1200), (2920, 1200), (214, 39, 40), 15)
x = np.arange(0, 2920, 1)


fig2, ax2 = plt.subplots(1,2)
fig2.tight_layout()

ax2[0].imshow(gray_c/np.max(gray_c))
ax2[0].set_axis_off()

ax2[1].plot(x, gray[1200,:], color='tab:red')
ax2[1].set_xlim([0, 2910])
ax2[1].set_xticks([])
ax2[1].set_ylim([0, 255])
ax2[1].set_yticks([0, 63, 127, 191, 255])
ax2[1].set_xlabel('Width')
ax2[1].set_ylabel('Intensity')
ax2[1].set_aspect(10)
ax2[1].grid(True)

fig2.savefig('bright_sharp.pdf')