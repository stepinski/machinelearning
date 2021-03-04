import pandas as pd 


gammaray=pd.read_csv("../data/gamma-ray.csv")
nullh=gammaray['count'].sum()/gammaray['seconds'].sum()
alth=np.max(gammaray['count']/gammaray['seconds'])

p_value = chi2.sf(chistat, 1)
