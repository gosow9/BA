import numpy as np
import cv2

def t_dist_model_x(p1, p2, x, y, r):
    return x + 2*p1*x*y+p2*(r**2+2*x**2)

def t_dist_model_y(p1, p2, x, y, r):
    return y + 2*p2*x*y+p1*(r**2+2*y**2)

im = np.ones((2464, 3280, 3), dtype=np.uint8)*255

y_c = 2464/2
x_c = 3280/2
f = 2700

p1 = -0.15
p2 = -0.4

theta = [0, np.pi/6, 2*np.pi/6, 3*np.pi/6, 4*np.pi/6, 5*np.pi/6]

# distorted 1
for t in theta:
    s = np.sin(t)
    c = np.cos(t)

    R = np.array([[c, -s],
                  [s, c]])
    
    q = []

    for i in range(-900, 900):
        ind = np.array([i, 0])
        i_rot = R@ind
    
        x_dst = f*t_dist_model_x(p1, p2, i_rot[0]/f, i_rot[1]/f, np.sqrt((i_rot[0]/f)**2+(i_rot[1]/f)**2))
        y_dst = f*t_dist_model_y(p1, p2, i_rot[0]/f, i_rot[1]/f, np.sqrt((i_rot[0]/f)**2+(i_rot[1]/f)**2))
    
        q.append((int(round(x_dst+x_c)), int(round(y_dst+y_c))))
    
    for i in range(1, len(q)):  
        cv2.line(im, q[i-1], q[i], [40, 39, 214], 8)

# distorted circle 1
phi = np.arange(0, 2*np.pi, 0.001)
q = []
for p in phi:
    x_i = 900*np.cos(p)
    y_i = 900*np.sin(p)
       
    x_dst = f*t_dist_model_x(p1, p2, x_i/f, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2)) + x_c
    y_dst = f*t_dist_model_y(p1, p2, x_i/f, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2)) + y_c
    
    x_i = int(x_dst)
    y_i = int(y_dst)
        
    q.append((int(round(x_dst)), int(round(y_dst))))
    
    im.T[0][x_i][y_i]=0
    im.T[1][x_i][y_i]=0
    im.T[2][x_i][y_i]=0

for i in range(1, len(q)):  
    cv2.line(im, q[i-1], q[i], [40, 39, 214], 8)

p1 = 0.15
p2 = 0.4

# distorted 2
for t in theta:
    s = np.sin(t)
    c = np.cos(t)

    R = np.array([[c, -s],
                  [s, c]])
    
    q = []

    for i in range(-900, 900):
        ind = np.array([i, 0])
        i_rot = R@ind
    
        x_dst = f*t_dist_model_x(p1, p2, i_rot[0]/f, i_rot[1]/f, np.sqrt((i_rot[0]/f)**2+(i_rot[1]/f)**2))
        y_dst = f*t_dist_model_y(p1, p2, i_rot[0]/f, i_rot[1]/f, np.sqrt((i_rot[0]/f)**2+(i_rot[1]/f)**2))
    
        q.append((int(round(x_dst+x_c)), int(round(y_dst+y_c))))
    
    for i in range(1, len(q)):  
        cv2.line(im, q[i-1], q[i], [184, 128, 46], 8)

# distorted circle 2
phi = np.arange(0, 2*np.pi, 0.001)
q = []
for p in phi:
    x_i = 900*np.cos(p)
    y_i = 900*np.sin(p)
       
    x_dst = f*t_dist_model_x(p1, p2, x_i/f, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2)) + x_c
    y_dst = f*t_dist_model_y(p1, p2, x_i/f, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2)) + y_c
    
    x_i = int(x_dst)
    y_i = int(y_dst)
        
    q.append((int(round(x_dst)), int(round(y_dst))))
    
    im.T[0][x_i][y_i]=0
    im.T[1][x_i][y_i]=0
    im.T[2][x_i][y_i]=0

for i in range(1, len(q)):  
    cv2.line(im, q[i-1], q[i], [184, 128, 46], 8)

# undistorted
cv2.circle(im, (int(x_c), int(y_c)), 900, [0, 0, 0], 12)

for t in theta:
    s = np.cos(t)
    c = np.sin(t)

    R = np.array([[c, -s],
                  [s, c]])

    o1 = np.array([-900, 0])
    o2 = np.array([900, 0])
 
    o1_rot = R@o1
    o2_rot = R@o2
    
    point1 = (int(round(o1_rot[0]+x_c)), int(round(o1_rot[1]+y_c)))
    point2 = (int(round(o2_rot[0]+x_c)), int(round(o2_rot[1]+y_c)))
    
    cv2.line(im, point1, point2, [0, 0, 0], 12)
    



cv2.imwrite('tangential.png' ,im)   
 
cv2.namedWindow('im', cv2.WINDOW_NORMAL)
cv2.resizeWindow('im', 820, 616)
cv2.imshow('im', im)