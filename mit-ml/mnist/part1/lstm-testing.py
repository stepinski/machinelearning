import numpy as np

w=np.array([[0,0],[0,100],[0,100]])                                                                                                                                                      
b=np.array([-100,100,0])                                                                                                                                                                          
w2=np.array([-100,50,0])   
h_1=0
c_1=[0, 0, 1, 1, 1, 0]

import math 

def sigmoid(x): 
    return 1 / (1 + math.exp(-x)) 

def cnt_next(x,hprev,cprev):
    w=np.array([[0,0,-100],[0,100,100],[0,100,0]])                                                                                                                                                      
    #b=np.array([-100,100,0])                                                                                                                                                                          
    w2=np.array([-100,50,0])   
    inp=np.array([hprev,x,1])
    ft=sigmoid(np.dot(w[0,:],inp))
    it=sigmoid(np.dot(w[1,:],inp)) 	 
    ot=sigmoid(np.dot(w[2,:],inp)) 
    ct=ft*cprev+it*np.tanh(np.dot(w2,inp))	 	 
    ht=ot*np.tanh(ct)
    return (ht,ct)




input_x=np.array([1,1,0,1,1])
hprev=0
cprev=0
for xs in np.nditer(input_x):
    (hprev,cprev)=cnt_next(xs,hprev,cprev)
    print(hprev)