import numpy as np
import networkx as nx

ga=np.array([
    [0, 3, 6, 5],
    [3, 0, 5, 6],
    [6, 5, 0, 3],
    [5, 6, 3, 0]
   ])
g= nx.from_numpy_matrix(ga, create_using=nx.cycle_graph())

#clusters
ga=np.array([
    [0, 1, 0, 1,0,0,0,0],
    [1, 0, 1, 0,0,0,0,0],
    [0, 1, 0, 1,1,0,0,0],
    [1, 0, 1, 0,0,0,0,0],
    [0, 0, 1, 0,0,1,0,0],
    [0, 0, 0, 0,1,0,1,0],
    [0, 0, 0, 0,0,1,0,1],
    [0, 0, 0, 0,0,0,1,0]
   ])
g= nx.from_numpy_matrix(ga, create_using=nx.cycle_graph(4))

vecto=nx.linalg.algebraicconnectivity.fiedler_vector(g)