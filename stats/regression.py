import numpy as np
import statistics as s
from scipy.stats import pearsonr
import seaborn as sns; sns.set_theme(color_codes=True)
import statsmodels.api as sm
import matplotlib.pyplot as plt
Xs = np.array([ 0.387, 0.723, 1.00, 1.52, 5.20, 9.54, 19.2, 30.1, 39.5 ])

Ys = np.array([ 0.241, 0.615, 1.00, 1.88, 11.9, 29.5, 84.0, 165.0, 248 ])

N = 9
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
print('coeff: %.3f' % corr)


# ys = sns.load_dataset([Xs,Ys])
# ax = sns.regplot(x="total_bill", y="tip", data=[Xs,Ys])
# sns.scatterplot(x=horizontal_data_1, y=vertical_data_1, ax=ax[0]);
# ax = sns.regplot(x=Xs, y=Ys)


sm.qqplot(Xs, line='s')
plt.title("X distribution")

# plt.show()
sm.qqplot(Ys, line='s')
plt.title("Y distribution")
# plt.show()
corr=cov(np.log(Xs),np.log(Ys))/(s.stdev(np.log(Xs)) * s.stdev(np.log(Ys)))
bet1=corr*(s.stdev(np.log(Ys))/s.stdev(np.log(Xs)) )
print('beta improve: %.3f' % bet1)