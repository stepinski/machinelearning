import numpy as np
import statistics as s
from scipy.stats import pearsonr
Xs = np.array([0.0339, 0.0423, 0.213, 0.257, 0.273, 0.273, 0.450, 0.503, 0.503, \
0.637, 0.805, 0.904, 0.904, 0.910, 0.910, 1.02, 1.11, 1.11, 1.41, \
1.72, 2.03, 2.02, 2.02, 2.02])

Ys = np.array([-19.3, 30.4, 38.7, 5.52, -33.1, -77.3, 398.0, 406.0, 436.0, 320.0, 373.0, \
93.9, 210.0, 423.0, 594.0, 829.0, 718.0, 561.0, 608.0, 1.04E3, 1.10E3, \
840.0, 801.0, 519.0])

N = 24

#sample covariance function
def cov(a, b):

    if len(a) != len(b):
        return      

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum=(np.add(a,-a_mean)*np.add(b,-b_mean)).sum()

    return sum/(len(a)-1)

print(cov(Xs,Ys))

#pearson correlation covariance(X, Y) / (stdv(X) * stdv(Y))
corr=cov(Xs,Ys)/(s.stdev(Xs) * s.stdev(Ys))
print(corr)

#pearson lib
corr2, _ = pearsonr(Xs, Ys)
print('Pearsons correlation: %.3f' % corr2)

#testing Xt = np.arange(0, 1.1, .1)
Xt = np.random.normal(-1, 1, 1000)
Yt = Xt**2

Xt_Yt_r_coeff = np.corrcoef(Xt,Yt)
corrx=pearsonr(Xt, Yt)
print(corrx)