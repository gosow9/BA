import cv2
import numpy as np


images = ['nump0.png', 'nump1.png', 'nump2.png', 'nump3.png', 'nump4.png', 'nump5.png']

for fname in images:
    im = cv2.imread(fname)
    im = cv2.GaussianBlur(im, (71,71), 0)
    
    name = fname[:5] + '_blurred.png'
    cv2.imwrite(name, im)