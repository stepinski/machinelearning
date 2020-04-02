import math
import numpy as np

def ReLU(x):
    return x * (x > 0)

def sigmoid(x): 
    return 1 / (1 + math.exp(-x)) 

t=1
x=3
w1=0.01
w2=-5
b=-1
z1=w1*x
a1=ReLU(z1)
z2=w2*a1+b
y=sigmoid(z2)
c=0.5*(float(y)-t)*(float(y)-t)  

#backward propagation
dloss_y=y-t
dsigmo=lambda xs: sigmoid(xs)*(1-sigmoid(xs))
dy_z2=dsigmo(z2)
dloss_z2=dloss_y*dy_z2

dz2_a1=w2
dloss_a1=dz2_a1*dloss_z2

dz2_w2=a1
dloss_w2=dz2_w2*dloss_z2

dz2_b=1
dloss_b=np.ones(1)*dloss_z2

drelu=lambda xs: 0 if xs<=0 else 1
da1_z1=drelu(z1)
dloss_z1=dloss_a1*da1_z1

dz1_w1=x
dloss_w1=dz1_w1*dloss_z1
print(dloss_w1)
print(dloss_w2)
print(dloss_b)

