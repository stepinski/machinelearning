"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from numpy import linalg as LA


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    def normalfixedsd(x,mi,sigma):
        d=x.shape[0]
        return 1/((2*np.pi*sigma)**(d/2)) * np.exp(-1/(2*sigma)*np.linalg.norm(x-mi)**2)

    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    ll=0.0
    llsum=0.0
    d=X.shape[0]
    
    for i in range(n):
        ll=0.0
        for j in range(K):
            norm=normalfixedsd(X[i],mixture.mu[j],mixture.var[j])
            post[i,j]=mixture.p[j]*norm
        posts=post[i].sum()
        llsum+=np.log(posts)
        post[i]=post[i]/posts  
        psum=post[i,:].sum()
        post[i,:]= post[i,:]/psum
    return post,llsum

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n,d=X.shape
    _,K=post.shape
    pnew=np.zeros(K)
    varnew=np.zeros(K)
    minew=np.zeros((K,d))

    for j in range(K):
        pnew[j] = post[:,j].sum()/n
        # print('test')
        # print(post[:,j])
        # print('test2')
        # print(X)
        # print('test3')
        # print(np.matmul(post[:,j],X))
        # print("fda")
        # print(np.matmul(post[:,j],X))
        # print('end')

        minew[j]=np.matmul(post[:,j],X)/post[:,j].sum()
        # print(np.linalg.norm(X-minew[j])**2)
        # print(post[:,j]*np.linalg.norm(X-minew[j])**2)
        # print("fda")
        accu=0.0
        for i in range(n):
            accu+=post[i,j]*np.linalg.norm(X[i]-minew[j])**2

        varnew[j]=accu/(d*post[:,j].sum())

        # varnew[j]=(post[:,j]*np.linalg.norm(X-minew[j])**2).sum()/(d*post[:,j].sum())

    mixture = GaussianMixture(minew, varnew, pnew)
    return mixture
    

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    new_ll = None
    old_ll = None
    while (old_ll is None or (new_ll - old_ll)/np.abs(new_ll) > 0.000001):
        old_ll = new_ll
        post,new_ll=estep(X,mixture)
        mixture = mstep(X, post)
        

    return mixture, post, new_ll 
