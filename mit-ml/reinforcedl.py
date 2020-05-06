import numpy as np
gama = 0.5
alfa = 0.75

data = np.array([[1, 1, 1], [1, 2, -1], [2, 1, 1]]) #(s, s', R)
Q = np.zeros((data.shape[0]+1, 2)) #(iterations, |S|)
k = 1
for d in range(data.shape[0]):
    R = data[d, 2]  #inmediate reward
    idx_s = data[d, 0] - 1  # index of state s in Q
    idx_sp = data[d, 1] - 1 #index of state s' in Q
    # Q[k, idx_s] = (1 - alfa) * Q[k - 1, idx_s] + alfa * (R + gama * np.max(Q[0:k, idx_sp]))
    Q[k, idx_s] = (1 - alfa) * Q[k - 1, idx_s] + alfa * (R + gama * Q[k-1, idx_sp])
    k += 1
print(Q)