#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mercicle
"""

# Standard Libs
import networkx as nx
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# DWave libs
# DWaveSampler(solver={'qpu': True})
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

original_name = "original_network.png"
solution_name = "solution_annotated_network.png"

save_results = False
sampler = EmbeddingComposite(DWaveSampler())

n_sim_nodes = 20

# first simulate a scale-free grapha nd them coerce to undirected
G = nx.scale_free_graph(n_sim_nodes)
G = nx.DiGraph.to_undirected(G)
print(nx.info(G))
newtork_layout = nx.spring_layout(G)

# Compute MIS
dwave_mis_results = dnx.maximum_independent_set(G, 
                                                sampler = sampler, 
                                                num_reads = 10)

# Assess Results
print('MIS (n=', len(dwave_mis_results), ')')
print(dwave_mis_results)

# Compute subgraphs for network viz annotations 
mis_subgraph = G.subgraph(list(dwave_mis_results)).copy()
print(nx.info(mis_subgraph))

nodes_not_in_mis = list(set(G.nodes()) - set(dwave_mis_results))
not_mis_subgraph = G.subgraph(nodes_not_in_mis)
print(nx.info(not_mis_subgraph))

plt.figure()

# Save

nx.draw_networkx(G, 
                 pos = newtork_layout, 
                 with_labels = True)
if save_results:
    plt.savefig(original_name, bbox_inches='tight')

nx.draw_networkx(mis_subgraph, 
                 pos=newtork_layout, 
                 with_labels=True, 
                 node_color='r',
                 font_color='k')
nx.draw_networkx(not_mis_subgraph,
                 pos=newtork_layout,
                 with_labels=True, 
                 node_color='grey', 
                 font_color='w')

if save_results:
    plt.savefig(solution_name, bbox_inches='tight')
