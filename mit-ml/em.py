import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal 

def normalfixedsd(x,mi,sigma):
    d=x.shape[1]
    return 1/((2*np.pi)**(d/2)* (sigma)**(1/2)) * np.exp(-1/(2*sigma)*LA.norm(x-mi)**2)


# e step

mi=np.array([-3,2])
sigma=np.array([4,4])
ps=np.array([0.5,0.5])

xs= np.array([0.2,-0.9, -1,1.2, 1.8])
p0s= np.array([0,0, 0,0, 0,0])
p1s= np.array([0,0, 0,0, 0,0])
p0sum=0
p1sum=0
p0sum2=0
p1sum2=0
ind=0

for xi in np.nditer(xs):
    p0=ps[0]*normalfixedsd(xi,mi[0],sigma[0])
    p1=ps[1]*normalfixedsd(xi,mi[1],sigma[1]) 
    px = p0+p1
    p_0=p0/px
    p_1=p1/px
    p0s[ind]=p_0
    p1s[ind]=p_1
    print(ind)
    p0sum+=p_0
    p1sum+=p_1
    p0sum2+=p_0*xi
    p1sum2+=p_1*xi
    ind+=1


p0new=p0sum/2
p1new=p1sum/2

minew0=p0sum2/p0sum
minew1=p1sum2/p1sum
        
sigma0=(p0s*LA.norm(xs-mi[0])**2 )/p0s
sigma1=(p1s*LA.norm(xs-mi[1])**2 )/p1s