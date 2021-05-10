import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

co2 = pd.read_csv("../data/linear/CO2.csv", header=[0, 1, 2], comment='"')

co2.columns = [
    " ".join(
        [col[0].strip(),
         col[1].strip(),
         col[2].strip()]
    ).strip()
    for col in co2.columns.values
]


co3=co2
co3['ind']=np.arange(0.0, 62.0, float(1/12))
co4=co3[['ind','CO2  [ppm]']]
co4f= co4[co4['CO2  [ppm]'] != -99.99]

# co2f= co2[co2[co2.columns[5]] != -99.99]

# X = pd.DataFrame(co2f[co2f.columns[2]])
# y = pd.DataFrame(co2f[co2f.columns[5]])

# scores = []
# kfold = KFold(n_splits=3, shuffle=True, random_state=42)
# for i, (train, test) in enumerate(kfold.split(X, y)):
#  model.fit(X.iloc[train,:], y.iloc[train,:])
#  score = model.score(X.iloc[test,:], y.iloc[test,:])
#  scores.append(score)
# print(scores)

lr = LinearRegression()
lr.fit(X,y)
print('intercept:', lr.intercept_)
print('slope:', lr.coef_)

import matplotlib.pyplot as plt
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))

#produce regression plots
fig = sm.graphics.plot_regress_exog(lr, 'points', fig=fig)


import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

co2fn = co4f.rename(columns = {'ind': 'x', 'CO2  [ppm]': 'y'}, inplace = False)
#fit simple linear regression model


train=co2fn[:587]
test = co2fn[588:]
train['x2']=train['x']**2
test['x2'] = test['x']**2
train['x3']=train['x']**3
test['x3'] = test['x']**3

# X = sm.add_constant(train[['x','x2','x3']])
X = sm.add_constant(train[['x','x2']])
# X = sm.add_constant(train[['x']])

model = sm.OLS(train['y'],X) 
# model = ols('y ~ x', data=co2fn[:587]).fit()
res=model.fit()
#view model summary
print(res.summary())

fig = plt.figure(figsize=(12,8))

#produce regression plots
fig = sm.graphics.plot_regress_exog(res, 'x', fig=fig)
plt.show()

from statsmodels.tools.eval_measures import rmse


# fit your model which you have already done

# now generate predictions
# Xt = sm.add_constant(test[['x']])
#Xt = sm.add_constant(test[['x','x2']])
Xt=sm.add_constant(test[['x','x2','x3']])
ypred = res.predict(Xt)

# calc rmse
rmse = rmse(test['y'], ypred)


from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
mean_absolute_percentage_error(test['y'], ypred)
mean_squared_error(test['y'], ypred)


X = sm.add_constant(train[['x','x2']])
model = sm.OLS(train['y'],X)
res=model.fit()
ypred = res.predict(X)
diffs=co2fn['y']-ypred