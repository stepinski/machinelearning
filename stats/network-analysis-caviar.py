import pandas as pd
import networkx as nx
phases = {}
G = {}
for i in range(1,12): 
  var_name = "phase" + str(i)
  file_name = "https://raw.githubusercontent.com/ragini30/Networks-Homework/main/" + var_name + ".csv"

#   file_name = "../data/CAVIAR/" + var_name + ".csv"
  phases[i] = pd.read_csv(file_name, index_col = ["players"])
  phases[i].columns = "n" + phases[i].columns
  phases[i].index = phases[i].columns
  G[i] = nx.from_pandas_adjacency(phases[i])
  G[i].name = var_name



dc9=nx.degree_centrality(G[9]) 
skeys=['n1','n3','n12','n83'] 
[dc9[k] for k in skeys] 

bc3=nx.betweenness_centrality(G[3], normalized = True)
[bc3[k] for k in skeys] 
bc9=nx.betweenness_centrality(G[9], normalized = True)
[bc9[k] for k in skeys] 

ec3=nx.eigenvector_centrality(G[3])
[ec3[k] for k in skeys]   

ec9=nx.eigenvector_centrality(G[9])
[ec9[k] for k in skeys]  


import time

start_time = time.perf_counter()

dc9=nx.degree_centrality(G[9]) 
print (time.perf_counter() - start_time, "dc seconds") #0.000252

start_time = time.perf_counter()
bc9=nx.betweenness_centrality(G[9], normalized = True) 
print (time.perf_counter() - start_time, "bc seconds") #0.01101

start_time = time.perf_counter()
ec9=nx.eigenvector_centrality(G[9])
print (time.perf_counter() - start_time, "bc seconds") #0.0015496869

skeys=['n1','n4','n89','n83','n3','n5','n88','n85','n90','n2','n7','n54','n6','n64','n8']

import numpy as np
centralsd=[]
centralsb=[]
centralse=[]
for i in range(1,12):
  dc=nx.degree_centrality(G[i]) 
  centralsd.append(dc)

  bc=nx.betweenness_centrality(G[i], normalized = True)  
  centralsb.append(bc)
  
  ec=nx.eigenvector_centrality(G[i])
  centralse.append(ec)

import pandas as pd
ddf= pd.DataFrame(centralsd).fillna(0)
bdf= pd.DataFrame(centralsb).fillna(0)
edf= pd.DataFrame(centralse).fillna(0)

bdf.mean(axis=0).sort_values(ascending=False).head() 

edf.mean(axis=0).sort_values(ascending=False).head() 



df = pd.read_csv("../data/Cooffending/Cooffending.csv")
dfr=df.drop_duplicates(subset={'OffenderIdentifier','CrimeIdentifier'})

dfr=df.drop_duplicates(subset={'OffenderIdentifier'})

dfr=df.drop_duplicates(subset={'CrimeIdentifier'})


dfr.groupby('CrimeYear').size().reset_index(name='counts').sort_values('counts')


dfr=df.drop_duplicates(subset={'CrimeIdentifier'})
sum_column = dfr["NumberYouthOffenders"] + dfr["NumberAdultOffenders"]
dfr["TotalOffenders"] = sum_column

dfr.groupby('CrimeIdentifier').agg({'TotalOffenders':'sum'}).sort_values('TotalOffenders')
dfr.loc[dfr['CrimeIdentifier']==27849]['CrimeLocation']


dfr=df.drop_duplicates(subset={'OffenderIdentifier','CrimeIdentifier'})
dfr.groupby('OffenderIdentifier').size().reset_index(name='counts').sort_values('counts')

from scipy.sparse import csr_matrix

dfr=df.drop_duplicates(subset={'OffenderIdentifier','CrimeIdentifier'})
# coofs = nx.from_pandas_adjacency(dfr)
# adj=nx.to_numpy_array(coofs)

# csmat=csr_matrix(adj)
crimes=dfr[['OffenderIdentifier', 'CrimeIdentifier']]
crimestmp=crimes.copy()
rels=crimes.set_index('CrimeIdentifier').join(crimestmp.set_index('CrimeIdentifier'), on='CrimeIdentifier', how='left', lsuffix='_l', rsuffix='_r')

# check if removing self loops in sparse matrix is not better
# relsnos=rels.loc[rels['OffenderIdentifier_l']!=rels['OffenderIdentifier_r']]



[print(ofl) for (ofl, ofr) in zip(df['OffenderIdentifier_l'], df['OffenderIdentifier_r'])]


crimes=dfr[['OffenderIdentifier', 'CrimeIdentifier']].to_numpy()