import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS,TSNE
from sklearn.cluster import KMeans
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

# pca = PCA().fit(lx)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# plt.show()

ys=np.load("../data/highdim/p1/y.npy")

# plt.scatter(lx[:,0], ys,c=ys)
# plt.show()



model = PCA()
# model=MDS()
# model=TSNE()


x_trans = model.fit_transform(lx)
kmeans = KMeans(n_clusters=4,n_init=100)
y=kmeans.fit_predict(lx)

# plt.scatter(x_trans[:,0],x_trans[:,1],c=y)
# plt.show()
# print(abs( pca.components_ ))

#looking for clusters with elbow method:
# allkmeans=[KMeans(n_clusters=i+1,n_init=100) for i in range(8)]
# for i in range(8):
#     allkmeans[i].fit(x_trans)

# inertias=[allkmeans[i].inertia_ for i in range(8)]
# plt.plot(np.arange(1,9),inertias)
# plt.show()