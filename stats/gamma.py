import pandas as pd 


gammaray=pd.read_csv("../data/gamma-ray.csv")
nullh=gammaray['count'].sum()/gammaray['seconds'].sum()
alth=np.max(gammaray['count']/gammaray['seconds'])

p_value = chi2.sf(chistat, 1)


def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta

tstm=np.concatenate((np.ones_like(morts[:,:1]), morts), axis=1)                                                                                                                                                                              
model = sm.OLS(y, morts) 
results = model.fit() 
print(results.summary())                                                                                                                                                                                                            


0.943     -1.365      0.179      -3.187       0.612
x2             3.7081      1.726      2.149      0.037       0.232       7.184
x3             3.4651      1.157      2.996      0.004       1.136       5.794
x4             1.7824      0.694      2.568      0.014       0.384       3.181
x5             8.9489      9.946      0.900      0.373     -11.083      28.981
x6             0.0105      0.005      2.029      0.048    7.65e-05       0.021
x7             2.5542      0.888      2.878      0.006       0.767       4.342
x8            -1.6619      1.522     -1.092      0.281      -4.728       1.404
x9         -3.793e-09   5.01e-06     -0.001      0.999   -1.01e-05    1.01e-05
x10          104.9013     34.848      3.010      0.004      34.715     175.088
x11           -0.0004      0.002     -0.263      0.794      -0.004       0.003
x12           -0.3906      0.559     -0.699      0.488      -1.516       0.735
x13            1.1137      1.130      0.986      0.330      -1.162       3.389
x14            0.1858 