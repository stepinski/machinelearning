import pandas as pd
import numpy as np
from zipfile import ZipFile 
with ZipFile("../data/release_statsreview_release.zip") as zip_file:
    zip_file.open('data/gamma-ray.csv')
    golub_data, gc = ( np.genfromtxt(zip_file.open('data/golub_data/{}'.format(fname)), delimiter=',', names=True, converters={0: lambda s: int(s.strip(b'"'))}) for fname in ['golub.csv', 'golub_cl.csv'] )

NALL=27
NAML=11

tst= tuple(tuple(map(int, tup)) for tup in gc)

dt=np.dtype([('key',int),('val',int)])
gcs=np.ndarray(len(gc),dt,np.array(tst))

ALL=AML=golub_data[gcs[gcs['val']==0]['key']]
AML=golub_data[gcs[gcs['val']==1]['key']]
# stats.ttest_ind(setosa['petal_length'], virginica['petal_length'], equal_var = False)

pvals=[]
count=0

for rind, row in golub.iterrows():
    vara=np.var(row[1:NALL])
    meana=np.mean(row[1:NALL])
    varm=np.var(row[NALL:NALL+NAML+1])
    meanm=np.mean(row[NALL:NALL+NAML+1])
     vardelta=vara+varm
     ...:     #np.sqrt(vara/NALL,varm/NAML)
     tW=abs(meana-meanm)/np.sqrt(vara/NALL+varm/NAML)
     v=((vara/NALL+varm/NAML)**2)/(1/(NALL-1)*(vara/NALL)**2 + 1/(NAML-1)*(varm/NAML)**2)
     p=1-stats.t.cdf(tW,df=v)
     pvals=np.append(pvals,p)
     if 2*p <= 0.05:
         count+=1


tst2=mt.multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
cnt2=0
for val in tst2[1]:
     if 2*val <0.05:
         cnt2+=1