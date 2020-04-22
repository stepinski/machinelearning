#init
import numpy as np

z1=np.array([-5,2])
z2=np.array([0,-6])
xs=np.array([[0,-6],[4,4],[0,0],[-5,2]])
centroids=np.array([0,0,0,0])
ordn=0

def square_euclidean(x,y) : return np.sqrt(((x-y)**2).sum())

def calccost(x,xs):
    dist=0
    for xi in xs:
        # dist+=square_euclidean(xi,x)
        dist+= np.linalg.norm(xi-x)
    return dist

while(True):
    ind=0
    for xi in xs:
        # print(xi)
        dist1 = np.linalg.norm(xi-z1)
        dist2 = np.linalg.norm(xi-z2)
        # dist1 = square_euclidean(xi,z1)
        # dist2 = square_euclidean(xi,z2)
        centroids[ind]=0 if(dist1<dist2) else 1
        ind+=1 

    mindist=10000000
    z1prev=z1
    z2prev=z2
#looking for new zs
# commented for medoids only
    # for xi in xs[centroids==0]:
    #     dist = calccost(xi,xs[centroids==0]-xi)
    #     if(dist<mindist):
    #         mindist=dist
    #         z1=xi
    # mindist=10000000
    # for xi in xs[centroids==1]:
    #     dist = calccost(xi,xs[centroids==1]-xi)
    #     if(dist<mindist):
    #         mindist=dist
    #         z2=xi

    #kmeans mode
    z1= xs[centroids==0].sum(axis=0)/xs[centroids==0].shape[0]
    z2= xs[centroids==1].sum(axis=0)/xs[centroids==1].shape[0]
    print("test%s"%z1)
    if (z1prev[0]==z1[0] and z1prev[1]==z1[1] and z2prev[0]==z2[0] and z2prev[1]==z2[1]):
        break

print(z1)
print(z2)
    
    