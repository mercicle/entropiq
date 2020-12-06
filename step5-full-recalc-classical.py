#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mercicle
"""

# Standard Libs
import time
import networkx as nx
from networkx.algorithms.approximation import independent_set

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import os


# DWave libs
# DWaveSampler(solver={'qpu': True})
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

from helpers.helper_functions import jaccard_sim

solution_image_suffix = "mis_network.png"

save_results = True
sampler = EmbeddingComposite(DWaveSampler())

out_dir = os.getcwd()+'/out-data/'
out_dir_net_solution_viz = out_dir + 'mis_solutions/'
os.mkdir(out_dir_net_solution_viz)

all_bonds_df = pd.read_csv(out_dir+'all_bonds_df.csv')
all_atoms_df = pd.read_csv(out_dir+'all_atoms_df.csv')
print(all_bonds_df.head())

all_unique_molecules = all_bonds_df.molecule.unique()

save_every_n = 100  # so i only save 100 out of the 
stop_after = 2218
mol_index = 1
analysis_results_cmax_df = pd.DataFrame()
for molecule in all_unique_molecules:
    
    # molecule = all_unique_molecules[0]
    print(molecule)
    these_bonds_df = all_bonds_df[all_bonds_df.molecule == molecule]
    this_molecule_df = all_atoms_df[all_atoms_df.molecule == molecule]
    
    atom_lookup = {this_molecule_df.atom_id.values.tolist()[i]: this_molecule_df.atom_name.values.tolist()[i] for i in range(len(this_molecule_df.atom_name))} 

    these_bonds_df = these_bonds_df[['source','target']]
    these_bonds_df['source'] = [atom_lookup[id] for id in these_bonds_df.source.values.tolist()]
    these_bonds_df['target'] = [atom_lookup[id] for id in these_bonds_df.target.values.tolist()]
    this_edgelist = [ (x,y) for x , y in zip(these_bonds_df.source.values.tolist(),these_bonds_df.target.values.tolist()) ]
    G = nx.from_edgelist(this_edgelist)
    
    newtork_layout = nx.spring_layout(G)
            
    start_time = time.time()
    classical_max_results = independent_set.maximum_independent_set(G)
    end_time = time.time()
    classical_max_total_seconds = (end_time - start_time)    
    
    classical_max_subgraph = G.subgraph(list(classical_max_results)).copy()
    nodes_not_in_classical_max = list(set(G.nodes()) - set(classical_max_subgraph))
    not_classical_max_subgraph = G.subgraph(nodes_not_in_classical_max)
     
    this_comparison_df = pd.DataFrame.from_dict({'molecule':[molecule], 
                                                 'mis_sim': [this_sim],
                                                 'compute_sec_max_classical': [classical_max_total_seconds]
                                               })
    
    analysis_results_cmax_df = analysis_results_cmax_df.append(this_comparison_df)
    
    if mol_index % save_every_n == 0:

        plt.figure(figsize=(12, 8))
        plt.title('Classical Max Solution for ' + molecule)
        nx.draw_networkx(G, 
                         pos = newtork_layout, 
                         with_labels = True)
        
        nx.draw_networkx(classical_max_subgraph, 
                         pos=newtork_layout, 
                         with_labels=True, 
                         node_color='r',
                         font_color='k')
        nx.draw_networkx(not_classical_max_subgraph,
                         pos=newtork_layout,
                         with_labels=True, 
                         node_color='grey', 
                         font_color='w')
        
        if save_results:
            plt.savefig(out_dir_net_solution_viz + molecule + '_classical_max_' + solution_image_suffix, bbox_inches='tight')
    
    mol_index+=1
    
    if mol_index == stop_after:
        break
    
    #end of algo inside loop

print(analysis_results_cmax_df.head())
print(analysis_results_cmax_df.shape)
analysis_results_cmax_df.to_csv('./out-data/analysis_results_cmax_df.csv')


print(np.round(analysis_results_df.diff_sec.mean(),2))
plt.figure(figsize=(12, 8))
plt.title("Quantum vs Classical Compute Time Difference (sec)")
plt.xlabel("Difference: Quantum - Classical")
plt.ylabel("Frequency")
plt.hist(analysis_results_df.diff_sec, bins=[x/10 for x in range(0,100)], edgecolor='white', linewidth=1.2)
plt.savefig(out_dir_net_solution_viz + '__compute_time_diff', bbox_inches='tight')
plt.show()

print(np.round(analysis_results_df.mis_sim.mean(),2))
plt.figure(figsize=(12, 8))
plt.title("Quantum and Classical MIS Similarity (Jaccard)")
plt.xlabel("Jaccard Similarity")
plt.ylabel("Frequency")
plt.hist(analysis_results_df.mis_sim, bins=[x/100 for x in range(0,100) if x%2==0], edgecolor='white', linewidth=1.2)
plt.savefig(out_dir_net_solution_viz + '__jaccard_sim', bbox_inches='tight')
plt.show()

#plt.xscale('log')
 
