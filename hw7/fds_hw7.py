
# coding: utf-8

# In[111]:


# Problem 3

"""Construct a random graph with three clusters as follows. Take three groups of vertices with n = 10 vertices each. In each group add each of the n choose 2 edges with probability 0.8 independently and uniformly at random. Then, between any two groups add each of the n**2 possible edges with probability 0.05 independently and uniformly at random. Using the networkx library, draw the graph with the positions of the vertices chosen according to the eigenvectors corresponding to the second and third smallest eigenvalues of the Laplacian of the graph.
"""
import pprint as pp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
from random import random as rnd
from itertools import combinations


# In[118]:


# Problem 3 approach:

n = 10   # points per cluster
k = 3    # clusters per graph

def gen_graph(n,k):
    g = nx.Graph()
    g.add_nodes_from([f"{cluster}{node}" for node in range(n) for cluster in range(k)])
    for i,j in combinations(g,2):
        if i[0] == j[0] and rnd() < 0.8: g.add_edge(i,j)
        if i[0] != j[0] and rnd() < 0.05: g.add_edge(i,j)
    return g

g = gen_graph(n, k)
L = nx.laplacian_matrix(g).todense()
w,v = np.linalg.eig(L)
hist(w,bins=10)
plt.show()
nx.draw_networkx(g, with_labels=True)
plt.show()
nx.draw_spectral(g, with_labels=True)

eigens = list(zip(w,v.transpose()))

ev1 = sorted(eigens)[1][1]
ev2 = sorted(eigens)[2][1]

pp.pprint(ev1)
pp.pprint(ev2)

plt.show()
plt.plot(ev1, ev2, "r.")
plt.xlabel("second smallest eigenvector")
plt.ylabel("third smallest eigenvector")
plt.show()

