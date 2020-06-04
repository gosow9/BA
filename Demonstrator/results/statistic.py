import numpy as np

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
D6_S = static6.T[1]

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

error_L_s = (np.std(L1_s)/np.mean(L1_s) + np.std(L2_s)/np.mean(L2_s) + 
             np.std(L3_s)/np.mean(L3_s) + np.std(L4_s)/np.mean(L4_s) +
             np.std(L5_s)/np.mean(L5_s) + np.std(L5_s)/np.mean(L5_s))/6

error_D_s = (np.std(D1_s)/np.mean(D1_s) + np.std(D2_s)/np.mean(D2_s) + 
             np.std(D3_s)/np.mean(D3_s) + np.std(D4_s)/np.mean(D4_s) +
             np.std(D5_s)/np.mean(D5_s) + np.std(D5_s)/np.mean(D5_s))/6

error_L_d = (np.std(L1_d)/np.mean(L1_d) + np.std(L2_d)/np.mean(L2_d) + 
             np.std(L3_d)/np.mean(L3_d) + np.std(L4_d)/np.mean(L4_d) +
             np.std(L5_d)/np.mean(L5_d) + np.std(L5_d)/np.mean(L5_d))/6

error_D_d = (np.std(D1_d)/np.mean(D1_d) + np.std(D2_d)/np.mean(D2_d) + 
             np.std(D3_d)/np.mean(D3_d) + np.std(D4_d)/np.mean(D4_d) +
             np.std(D5_d)/np.mean(D5_d) + np.std(D5_d)/np.mean(D5_d))/6

print(error_L_d*100)
print(error_D_d*100)