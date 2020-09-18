import networkx as nx
import numpy as np
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
from itertools import combinations
from copy import deepcopy
from collections import Counter
from itertools import permutations

def gen(n): 
    for i in range(2**(n)): 
        yield np.array([int(k) for k in "{0:b}".format(i).zfill(n)])


n = 100
L = 2
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# probs = [0.5]
e_beta_l = []
e_beta_mds = []
e_beta_cc = []
length = 10
for p in probs:
    l_sum = 0
    mds_sum = 0
    cc_sum = 0
    for e in range(0, length):
        G_small = nx.fast_gnp_random_graph(n, p, seed=None, directed=True)
        # # MinRank---------------------------------------------------------
        matrices = []
        ranks = []
        M = np.identity(n)
        print(len(G_small.edges()))
        rand_len = len(G_small.edges())
        dashes = (n * (n-1) / 2) - rand_len
        genn = gen(int(rand_len))
        for g in genn:
            # print("before", g)
            # print(rand_len)
            edges = list(G_small.edges())
            for i in range(0, n):
                for j in range(0, n):
                    if (i != j) and ((j, i) not in edges):
                        M[i][j] = -1
                    elif (i != j) and ((j, i) in edges):
                        M[i][j] = g[0]
                        g = np.delete(g, 0)
                        # print("after", g)
                    for k in range(0, n):
                        col = M.T[k]
                        unique, counts = np.unique(col, return_counts=True)
                        counts = dict(zip(unique, counts))
                        if -1 in counts.keys():
                            dash_num = counts[-1]
                            if dash_num >= L - 1:
                                for l in range(L - 1):
                                    antenna_ones = [1] * l + [0] * (dash_num - l)
                                    antenna_ones = list(permutations(antenna_ones))
                            else:
                                antenna_ones = gen(dash_num)
                            for ao in antenna_ones:
                                ao = list(ao)
                                ind = 0
                                for h in range(0, n):
                                    if M[h][k] == -1:
                                        M[h][k] = ao[ind]
                                        ind += 1
            ranks.append(matrix_rank(M))
            # print('yi=', M)
            matrices.append(M)

        # for m in matrices:
        #     r = matrix_rank(m)
        #     ranks.append(r)
        beta_l = min(ranks)
        # matrices[0]
        # l = list(gen(n, n))
        print("beta_l", beta_l)
        l_sum += beta_l

        # #MDS -------------------------------------------------------------
        indegListTuples = G_small.in_degree()
        indegList = []
        for t in indegListTuples:
            indegList.append(t[1])
        MinInDeg = min(indegList)
        # print(list (indegList))
        # print(MinInDeg)
        beta_mds = n - MinInDeg
        print("beta_mds", beta_mds)
        mds_sum += beta_mds

        # CC -------------------------------------------------------------
        G_clique_assist = deepcopy(G_small)
        c_edges = G_clique_assist.edges()
        edges_to_be_deleted = []
        for ce in c_edges:
            reverse_edge = (ce[1], ce[0])
            if reverse_edge not in c_edges:
                edges_to_be_deleted.append(ce)
        G_clique_assist.remove_edges_from(edges_to_be_deleted)
        G_clique_assist = nx.complement(G_clique_assist)
        G_clique_assist.add_edges_from(edges_to_be_deleted)
        G_only_both = G_clique_assist.to_undirected()
        i = 1
        d = nx.coloring.greedy_color(G_only_both)
        # print(len(list(Counter(list(d.values())).keys())))
        beta_cc = len(list(Counter(list(d.values())).keys()))
        cc_sum += beta_cc
        # nx.draw(G_small, with_labels = True)
        # # plt.show()
    mean_l = l_sum / length
    mean_mds = mds_sum / length
    mean_cc = cc_sum / length
    print("mean_l", mean_l)
    e_beta_l.append(mean_l)
    e_beta_mds.append(mean_mds)
    e_beta_cc.append(mean_cc)
fig = plt.figure()
l_line, = plt.plot(probs, e_beta_l) 
mds_line, = plt.plot(probs, e_beta_mds) 
cc_line, = plt.plot(probs, e_beta_cc)  
l_line.set_label('MinRank')
mds_line.set_label('MDS')
cc_line.set_label('CliqueCover')
plt.legend((mds_line, cc_line), ('MDS', 'CliqueCover'))
plt.legend(l_line, 'MinRank')
fig.suptitle('Graph Size: 5', fontsize=20)
plt.xlabel('p', fontsize=18)
plt.ylabel('number of transmissions', fontsize=16)
plt.show()