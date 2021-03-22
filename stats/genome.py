import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

xs=np.load("../data/highdim/p1/X.npy")

rows=xs.shape[0]
cols=xs.shape[1]
max1stcol=np.max(xs[:,0])
lx=np.log2(xs+1)
max1clog=np.max(lx[:,0])

# pca = PCA(n_components=1)
# pca.fit(xs)
# print(pca.explained_variance_ratio_)

# pca2 = PCA(n_components=1)
# pca2.fit(lx)
# print(pca2.explained_variance_ratio_)

# pca = PCA().fit(xs)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# plt.show()

pca = PCA().fit(lx)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()