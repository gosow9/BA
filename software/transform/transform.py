import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('t.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

minLineLength = 100
maxLineGap = 10

edges = cv2.Canny(gray, 60, 150, apertureSize=3)
lines = cv2.HoughLines(edges,1,np.pi/180,30,minLineLength,maxLineGap)

for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

plt.imshow(edges)