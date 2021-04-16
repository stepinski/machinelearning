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
row = rels['OffenderIdentifier_l'].to_numpy()
col = rels['OffenderIdentifier_r'].to_numpy()
  
lng = len(row)  
# taking data
data = np.ones(lng)
  
# creating sparse matrix
cooffend_matrix = csr_matrix((data, (row, col)),shape = (row.max() + 1, col.max() + 1))
cooffend_matrix[cooffend_matrix > 0] = 1
cooffend_matrix.setdiag(0)
cooffend_matrix.eliminate_zeros() # To avoid self loops since setdiag(0) does not itself change the sparsity pattern
coofs =nx.from_scipy_sparse_matrix(cooffend_matrix)



coofs.number_of_nodes() 
coofs.number_of_edges()  
len(list(nx.isolates(coofs)))
g=coofs.copy()
g.remove_nodes_from(list(nx.isolates(g)))
sum([x>=100 for x in sorted([d for n, d in g.degree()], reverse=True)])  

len(sorted(nx.connected_components(g), key=len, reverse=True) )



cntpairs=rels.groupby(['OffenderIdentifier_r','OffenderIdentifier_l']).size().reset_index(name='counts').sort_values('counts') 
nonones=cntpairs.loc[cntpairs['counts']>1]

row2 = nonones['OffenderIdentifier_l'].to_numpy()
col2 = nonones['OffenderIdentifier_r'].to_numpy()
  
lng2 = len(row2)  
# taking data
data2 = np.ones(lng2)
  
# creating sparse matrix
cooffend_matrix2 = csr_matrix((data2, (row2, col2)),shape = (row2.max() + 1, col2.max() + 1))
cooffend_matrix2[cooffend_matrix2 > 0] = 1
cooffend_matrix2.setdiag(0)
cooffend_matrix2.eliminate_zeros() # To avoid self loops since setdiag(0) does not itself change the sparsity pattern
coofs2 =nx.from_scipy_sparse_matrix(cooffend_matrix2)
coofs2.remove_nodes_from(list(nx.isolates(coofs2)))




cntpairs2=rels.groupby(['OffenderIdentifier_r','OffenderIdentifier_l']).size().reset_index(name='counts').sort_values('counts') 
nonones2=cntpairs.loc[cntpairs['counts']==1]

row3 = nonones2['OffenderIdentifier_l'].to_numpy()
col3 = nonones2['OffenderIdentifier_r'].to_numpy()
  
lng3 = len(row3)  
# taking data
data3 = np.ones(lng3)
  
# creating sparse matrix
cooffend_matrix3 = csr_matrix((data3, (row3, col3)),shape = (row3.max() + 1, col3.max() + 1))
cooffend_matrix3[cooffend_matrix3 > 0] = 1
cooffend_matrix3.setdiag(0)
cooffend_matrix3.eliminate_zeros() # To avoid self loops since setdiag(0) does not itself change the sparsity pattern
coofs3 =nx.from_scipy_sparse_matrix(cooffend_matrix3)
coofs3.remove_nodes_from(list(nx.isolates(coofs3)))

#coofs2 is gr
#coofs3 is gnr


sum([x>=100 for x in sorted([d for n, d in g.degree()], reverse=True)])  

len(sorted(nx.connected_components(coofs2), key=len, reverse=True) )
len(sorted(nx.connected_components(coofs3), key=len, reverse=True) )

lst1=[len(c) for c in sorted(nx.connected_components(coofs2), key=len, reverse=True)] 
sum(lst1)/len(lst1) 

lst2=[len(c) for c in sorted(nx.connected_components(coofs3), key=len, reverse=True)] 
sum(lst2)/len(lst2) 

max1=max(nx.connected_components(coofs), key=len)
max2=max(nx.connected_components(coofs2), key=len)
max3=max(nx.connected_components(coofs3), key=len)

len(max2)/sum(lst1)
len(max3)/sum(lst2) 


print("{:.5f}".format(nx.density(max1)))

print("{:.5f}".format(nx.density(max2)))

print("{:.5f}".format(nx.density(max3)))


S = [(c,coofs.subgraph(c).copy())for c in nx.connected_components(coofs)]
nx.density(max(S,key=lambda item:len(item[0]))[1])  


S2 = [(c,coofs2.subgraph(c).copy())for c in nx.connected_components(coofs2)]
nx.density(max(S2,key=lambda item:len(item[0]))[1])  

S3 = [(c,coofs3.subgraph(c).copy())for c in nx.connected_components(coofs3)]
nx.density(max(S3,key=lambda item:len(item[0]))[1])  

