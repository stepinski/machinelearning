import numpy as np
x = np.array([2,-1,-2])

A = [45,37,42,35,39]
B = [38,31,26,28,33]
C = [10,15,17,21,12]
n=3
id =np.identity(3)
ones_n = np.ones((n,1))
h=id-1/3*ones_n*ones_n.T

print(h*x.T)
#print(h*h*x.T)