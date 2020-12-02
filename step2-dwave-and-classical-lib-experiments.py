#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mercicle
"""

# Standard Libs
import time
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
start_time = time.time()
dwave_mis_results = dnx.maximum_independent_set(G, sampler = sampler, num_reads = 10)
end_time = time.time()
total_seconds = (end_time - start_time)
print("QA Compute time in {} seconds.".format(total_seconds))


size_dwave_mis = len(dwave_mis_results)

start_time = time.time()
classical_mis_results = nx.maximal_independent_set(G)
end_time = time.time()
total_seconds = (end_time - start_time)
print("Classical ompute time in {} seconds.".format(total_seconds))


size_classical_mis = len(classical_mis_results)

# Assess Results
print('MIS: DWave n=', size_dwave_mis, ' and Classical n= ', size_classical_mis)
print(dwave_mis_results)

########################################
##     Visualize DWave Result         ##
########################################
# Compute subgraphs for network viz annotations 
dwave_mis_subgraph = G.subgraph(list(dwave_mis_results)).copy()
print(nx.info(dwave_mis_subgraph))

nodes_not_in_dwave_mis = list(set(G.nodes()) - set(dwave_mis_results))
not_dwave_mis_subgraph = G.subgraph(nodes_not_in_dwave_mis)
print(nx.info(not_dwave_mis_subgraph))

plt.figure()

# Save

nx.draw_networkx(G, 
                 pos = newtork_layout, 
                 with_labels = True)
if save_results:
    plt.savefig(original_name, bbox_inches='tight')

nx.draw_networkx(dwave_mis_subgraph, 
                 pos=newtork_layout, 
                 with_labels=True, 
                 node_color='r',
                 font_color='k')
nx.draw_networkx(not_dwave_mis_subgraph,
                 pos=newtork_layout,
                 with_labels=True, 
                 node_color='grey', 
                 font_color='w')

if save_results:
    plt.savefig('dwave_'+solution_name, bbox_inches='tight')



########################################
##     Visualize Classical Result     ##
########################################

# Compute subgraphs for network viz annotations 
classical_mis_subgraph = G.subgraph(list(classical_mis_results)).copy()
print(nx.info(classical_mis_subgraph))

nodes_not_in_classical_mis = list(set(G.nodes()) - set(classical_mis_results))
not_classical_mis_subgraph = G.subgraph(nodes_not_in_classical_mis)
print(nx.info(not_classical_mis_subgraph))


plt.figure()

# Save
plt.title('foo')
nx.draw_networkx(G, 
                 pos = newtork_layout, 
                 with_labels = True)

nx.draw_networkx(classical_mis_subgraph, 
                 pos=newtork_layout, 
                 with_labels=True, 
                 node_color='r',
                 font_color='k')
nx.draw_networkx(not_classical_mis_subgraph,
                 pos=newtork_layout,
                 with_labels=True, 
                 node_color='grey', 
                 font_color='w')

if save_results:
    plt.savefig('classical_'+solution_name, bbox_inches='tight')

