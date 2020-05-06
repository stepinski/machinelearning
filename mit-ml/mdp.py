import numpy as np

v = np.array([0,0,0,0])
R_up = np.array([0,1,1,10])
R_down = np.array([1,1,10,0])

# t_up = np.array([1,1,1,1])
t_down = np.array([1,1,1,1])
T_up = np.zeros((4,4))
T_down = np.zeros((4,4))
T_up[0,1]=1
T_up[1,2]= 1
T_up[2,3] =  1
T_down[1, 0] =  1
T_down[2, 1] =  1
T_down[3, 2] =  1
# right = np.array([[1,1,1,1,1]])
gamma=0.75
for i in range(0,1000):
    v_up = T_up @ (R_up + (gamma * v))
    v_down = T_down @ (R_down + (gamma * v))
    v = np.stack((v_up, v_down)).max(0)
    print(i)
    print (v)
# print(V)