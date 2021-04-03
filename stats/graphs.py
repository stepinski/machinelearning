import numpy as np
graph=np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
[1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

