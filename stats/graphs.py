import numpy as np
graph=np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
[1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


import networkx as nx
G = nx.read_edgelist(
        "../data/rdgraph.txt",
        create_using=nx.DiGraph(),
        nodetype = int
)
G.number_of_edges() 
G.number_of_nodes()

len(list(nx.simple_cycles(G))) 
nx.number_of_selfloops(G) 
 
graph = np.array(	[[1,0,0,0],
    [1,0,0,0],
    [1,0,0,0],
    [1,0,0,0]])
G = nx.read_edgelist(
        "../data/dgraph2.txt",
        create_using=nx.DiGraph(),
        nodetype = int
)
graph = np.array(	[[0,1,1,1],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]])
graph = nx.from_numpy_matrix(np.array(graph), create_using=nx.DiGraph)
nx.katz_centrality(graph, alpha=0.1, beta=1)

G = nx.read_edgelist(
        "../data/dgraph3.txt",
        create_using=nx.DiGraph(),
        nodetype = int
)
nx.eigenvector_centrality(G) 

G = nx.read_edgelist(
        "../data/dgraph5.txt",
        create_using=nx.DiGraph(),
        nodetype = int
)
nx.networkx.katz_centrality(G,alfa=0.1,beta=1)


graph = np.array(	[[0,1,1,1],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]])
graph = nx.from_numpy_matrix(np.array(graph), create_using=nx.DiGraph)
nx.katz_centrality(graph, alpha=0.1, beta=1)

ga=np.array([[0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0]])
g= nx.from_numpy_matrix(ga, create_using=nx.DiGraph)
nx.katz_centrality(g, alpha=0.1, beta=1)
nx.networkx.pagerank(g)