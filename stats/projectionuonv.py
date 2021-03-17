import numpy as np  
v = 1/np.sqrt(5)*np.array([1, 2]) 
v_norm = np.sqrt(sum(v**2))  
# x1
u = np.array([1, 2]) 
pruv = (np.dot(u, v)/v_norm**2)*v 
print(pruv)
# x2
u = np.array([3, 4])  
pruv = (np.dot(u, v)/v_norm**2)*v 
print(pruv)
# x3
u = np.array([-1, 0])  
pruv = (np.dot(u, v)/v_norm**2)*v 
print(pruv)