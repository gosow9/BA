import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

img = cv2.imread('t.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges,1,np.pi/180,90)

for l in lines:
    rho = l[0][0]
    theta = l[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Linien
# x*cos(theta), y*sin(theta), rho
a0 = np.array([np.cos(lines[0][0][1]), np.sin(lines[0][0][1])]) 
b0 = lines[0][0][0]
a1 = np.array([np.cos(lines[1][0][1]), np.sin(lines[1][0][1])])
b1 = lines[1][0][0]
a2 = np.array([np.cos(lines[2][0][1]), np.sin(lines[2][0][1])])
b2 = lines[2][0][0]
a3 = np.array([np.cos(lines[3][0][1]), np.sin(lines[3][0][1])])
b3 = lines[3][0][0]

# Schnittpunkte
p0 = np.linalg.inv([a0, a1])@np.array([b0, b1])
p1 = np.linalg.inv([a1, a2])@np.array([b1, b2])
p2 = np.linalg.inv([a2, a3])@np.array([b2, b3])
p3 = np.linalg.inv([a3, a0])@np.array([b3, b0])

# Schnittpunkte in Array
P = np.concatenate([[p0], [p1], [p2], [p3]], axis=0)

# Schnittpunkte zeichnen
for x, y in P:
    cv2.circle(img, (x,y), 7, [255,0,0], -1)

# "richtige" Punkte
mpp = dist.euclidean(p0, p1)/48

Q = np.concatenate([[p0], [p1], np.array([[p1[0], p1[1]+29*mpp]]),
                    np.array([[p1[0]+48*mpp, p1[1]+29*mpp]])],
                    axis=0).astype(np.float32)
 
for x, y in Q:
    cv2.circle(img, (x,y), 7, [0,255,0], -1)

M = cv2.getPerspectiveTransform(P, Q)
h, w = img.shape[:2]
warped = cv2.warpPerspective(img, M, (w,h))

plt.imshow(warped)