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