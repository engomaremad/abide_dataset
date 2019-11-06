# todo: calculate stationary graph feats :degree , closeness_centrality, betweenness_centrality,
#   eigenvector_centrality

import networkx as nx
import numpy as  np
import os
import matplotlib.pyplot as plt

def construct_graph(conn_mat,th):
    num_areas = np.shape(conn_mat)[0]
    G = nx.Graph()
    for i in range(num_areas):
        for j in range(i):
            if np.abs(conn_mat[i,j])> th:
                G.add_edge(i,j,weight = np.abs(conn_mat[i,j]))
    return G
def get_graph_measures(conn_mat,th):
    num_areas = np.shape(conn_mat)[0]
    deg =np.zeros([num_areas,1])
    cs =np.zeros([num_areas,1])
    bs =np.zeros([num_areas,1])
    es =np.zeros([num_areas,1])

    G = construct_graph(conn_mat,th)
    cs_dict = nx.closeness_centrality(G)
    bs_dict = nx.betweenness_centrality(G)
   # es_dict = nx.eigenvector_centrality(G)
    d = np.asarray(nx.degree(G))
    for i in range(len(d)):
        deg[d[i, 0]] = d[i, 1]
    for i in range(num_areas):
        if i in bs_dict:
            bs[i] = bs_dict[i]
        if i in cs_dict:
            cs[i] = cs_dict[i]
        # if i in es_dict:
        #     es[i] = es_dict[i]
    return G,cs,bs,deg


