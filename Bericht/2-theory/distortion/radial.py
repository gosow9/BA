import numpy as np
import cv2

def r_dist_model(k1, k2, k3, k4, k5, k6, q, r):
    return q*(1 + k1*r**2 + k2*r**4 + k3*r**6)/(1 + k4*r**2 + k5*r**4 +k6*r**6)

im = np.ones((2464, 3280, 3), dtype=np.uint8)*255

y_c = 2464/2
x_c = 3280/2
f = 2700

cv2.line(im, (int(-820+x_c), int(-616+y_c)), (int(820+x_c), int(-616+y_c)), [0, 0, 0], 12)
cv2.line(im, (int(-820+x_c), int(616+y_c)), (int(820+x_c), int(616+y_c)), [0, 0, 0], 12)
cv2.line(im, (int(-820+x_c), int(-616+y_c)), (int(-820+x_c), int(616+y_c)), [0, 0, 0], 12)
cv2.line(im, (int(820+x_c), int(-616+y_c)), (int(820+x_c), int(616+y_c)), [0, 0, 0], 12)
    
# distorted 1
k1 = 10
k2 = 11
k3 = 12
k4 = 5
k5 = 6
k6 = 7

o1 = []
o2 = []
o3 = []
o4 = []

for i in range(-820, 820):
    x_i = i
    y_i = -616
        
    x_i_new = int(np.round(x_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, x_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    y_i_new = int(np.round(y_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
      
    o1.append((x_i_new, y_i_new))
    
for i in range(-820, 820):
    x_i = i
    y_i = 616
        
    x_i_new = int(np.round(x_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, x_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    y_i_new = int(np.round(y_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    
    o2.append((x_i_new, y_i_new))
    
for i in range(-616, 616):
    y_i = i
    x_i = -820
        
    x_i_new = int(np.round(x_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, x_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    y_i_new = int(np.round(y_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
             
    o3.append((x_i_new, y_i_new))
        
for i in range(-616, 616):
    y_i = i
    x_i = 820
            
    x_i_new = int(np.round(x_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, x_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    y_i_new = int(np.round(y_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
       
    o4.append((x_i_new, y_i_new))
    
for i in range(1, len(o1)):  
    cv2.line(im, o1[i-1], o1[i], [40, 39, 214], 15)
    
for i in range(1, len(o2)):  
    cv2.line(im, o2[i-1], o2[i], [40, 39, 214], 15)

for i in range(1, len(o3)):  
    cv2.line(im, o3[i-1], o3[i], [40, 39, 214], 15)
        
for i in range(1, len(o4)):  
    cv2.line(im, o4[i-1], o4[i], [40, 39, 214], 15)

# distorted 2
k1 = -2.2
k2 = -1.2
k3 = -0.8
k4 = -0.4
k5 = -0.3
k6 = -0.2

o1 = []
o2 = []
o3 = []
o4 = []

for i in range(-820, 820):
    x_i = i
    y_i = -616
        
    x_i_new = int(np.round(x_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, x_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    y_i_new = int(np.round(y_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
      
    o1.append((x_i_new, y_i_new))
    
for i in range(-820, 820):
    x_i = i
    y_i = 616
        
    x_i_new = int(np.round(x_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, x_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    y_i_new = int(np.round(y_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    
    o2.append((x_i_new, y_i_new))
    
for i in range(-616, 616):
    y_i = i
    x_i = -820
        
    x_i_new = int(np.round(x_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, x_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    y_i_new = int(np.round(y_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
             
    o3.append((x_i_new, y_i_new))
        
for i in range(-616, 616):
    y_i = i
    x_i = 820
            
    x_i_new = int(np.round(x_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, x_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
    y_i_new = int(np.round(y_c+f*r_dist_model(k1, k2, k3, k4, k5, k6, y_i/f, np.sqrt((x_i/f)**2+(y_i/f)**2))))
       
    o4.append((x_i_new, y_i_new))
    
for i in range(1, len(o1)):  
    cv2.line(im, o1[i-1], o1[i], [184, 128, 46], 12)
    
for i in range(1, len(o2)):  
    cv2.line(im, o2[i-1], o2[i], [184, 128, 46], 12)

for i in range(1, len(o3)):  
    cv2.line(im, o3[i-1], o3[i], [184, 128, 46], 12)
        
for i in range(1, len(o4)):  
    cv2.line(im, o4[i-1], o4[i], [184, 128, 46], 12)


cv2.imwrite('radial.png' ,im)

cv2.namedWindow('im', cv2.WINDOW_NORMAL)
cv2.resizeWindow('im', 820, 616)
cv2.imshow('im', im)