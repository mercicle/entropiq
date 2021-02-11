#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mercicle
"""

########################################
##          Import Libraries          ##
########################################

import time
import networkx as nx
from networkx.algorithms.approximation import independent_set
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import os
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from helpers.helper_functions import jaccard_sim

########################################
##            Setup Parms             ##
########################################

solution_image_suffix = "mis_network.png"

save_results = True
sampler = EmbeddingComposite(DWaveSampler())

out_dir = os.getcwd()+'/out-data/'
out_dir_net_solution_viz = out_dir + 'mis_solutions/'
os.mkdir(out_dir_net_solution_viz)

# read saved data sets
all_bonds_df = pd.read_csv(out_dir+'all_bonds_df.csv')
all_atoms_df = pd.read_csv(out_dir+'all_atoms_df.csv')
print(all_bonds_df.head())

all_unique_molecules = all_bonds_df.molecule.unique()

save_every_n = 100  # so i only save 100 out of the
mol_index = 1
analysis_results_df = pd.DataFrame()
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
    dwave_mis_results = dnx.maximum_independent_set(G, sampler = sampler, num_reads = 10)
    end_time = time.time()
    dwave_total_seconds = (end_time - start_time)
    size_dwave_mis = len(dwave_mis_results)

    dwave_mis_subgraph = G.subgraph(list(dwave_mis_results)).copy()
    nodes_not_in_dwave_mis = list(set(G.nodes()) - set(dwave_mis_results))
    not_dwave_mis_subgraph = G.subgraph(nodes_not_in_dwave_mis)

    print("QA Compute time in {} seconds.".format(dwave_total_seconds))

    start_time = time.time()
    classical_mis_results = nx.maximal_independent_set(G)
    end_time = time.time()
    classical_total_seconds = (end_time - start_time)
    size_classical_mis = len(classical_mis_results)

    classical_mis_subgraph = G.subgraph(list(classical_mis_results)).copy()
    nodes_not_in_classical_mis = list(set(G.nodes()) - set(classical_mis_results))
    not_classical_mis_subgraph = G.subgraph(nodes_not_in_classical_mis)

    print("Classical compute time in {} seconds.".format(classical_total_seconds))

    this_sim = jaccard_sim(dwave_mis_results, classical_mis_results)

    this_comparison_df = pd.DataFrame.from_dict({'molecule':[molecule],
                                                 'mis_sim': [this_sim],
                                                 'compute_sec_dwave': [dwave_total_seconds],
                                                 'compute_sec_classical': [classical_total_seconds],
                                                 'diff_sec': [dwave_total_seconds - classical_total_seconds]
                                               })
    analysis_results_df = analysis_results_df.append(this_comparison_df)

    if mol_index % save_every_n == 0:

        ########################################
        ##     Visualize DWave Result     ##
        ########################################

        plt.figure(figsize=(12, 8))
        plt.title('DWave QA Solution for ' + molecule)
        nx.draw_networkx(G,
                         pos = newtork_layout,
                         with_labels = True)

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
            plt.savefig(out_dir_net_solution_viz + molecule + '_dwave_'+solution_image_suffix, bbox_inches='tight')

        ########################################
        ##     Visualize Classical Result     ##
        ########################################

        plt.figure(figsize=(12, 8))
        plt.title('Classical Solution for ' + molecule)
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
            plt.savefig(out_dir_net_solution_viz + molecule + '_classical_' + solution_image_suffix, bbox_inches='tight')

    mol_index+=1

    #end of algo inside loop


print(analysis_results_df.head())
print(analysis_results_df.shape)
analysis_results_df.to_csv('./out-data/analysis_results_df.csv')

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
