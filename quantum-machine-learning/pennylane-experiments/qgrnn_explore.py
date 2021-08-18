

import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
import scipy
import networkx as nx
import copy
import os, sys

import pygraphviz as pgv # pygraphviz should be available


repo_folder_name = 'dwave_testing'
current_dir = os.getcwd()

root_repo_dir = current_dir.split(repo_folder_name)[0] + repo_folder_name + '/'
this_dir = root_repo_dir + 'quantum-machine-learning/pennylane-experiments/'

# create in/out data dirs
this_in_dir = this_dir+'in-data/'
this_out_dir = this_dir+'out-data/'

if not os.path.exists(this_in_dir):
    os.mkdir(this_in_dir)
if not os.path.exists(this_out_dir):
    os.mkdir(this_out_dir)

sys.path.append(os.path.abspath(this_dir + '/helper_functions/'))
from helpers import *

qubit_number = 4
qubits = range(qubit_number)

random_number_generator = np.random.default_rng(seed=42)

# QGRNN quantum data needed includes initial low-energy state, and a subsequent time-evolved states
low_energy_state = [(-0.054661080280306085 + 0.016713907320174026j),(0.12290003656489545 - 0.03758500591109822j),(0.3649337966440005 - 0.11158863596657455j),
                    (-0.8205175732627094 + 0.25093231967092877j),(0.010369790825776609 - 0.0031706387262686003j),(-0.02331544978544721 + 0.007129899300113728j),
                    (-0.06923183949694546 + 0.0211684344103713j),(0.15566094863283836 - 0.04760201916285508j),(0.014520590919500158 - 0.004441887836078486j),
                    (-0.032648113364535575 + 0.009988590222879195j),(-0.09694382811137187 + 0.02965579457620536j),(0.21796861485652747 - 0.06668776658411019j),
                    (-0.0027547112135013247 + 0.0008426289322652901j),(0.006193695872468649 - 0.0018948418969390599j),(0.018391279795405405 - 0.005625722994009138j),
                    (-0.041350974715649635 + 0.012650711602265649j)]

######################################################################
# Author notes: This state can be obtained by using a decoupled version of the
# :doc:`Variational Quantum Eigensolver </demos/tutorial_vqe>` algorithm (VQE).
# choose a VQE ansatz such that the circuit cannot learn the exact ground state,
# but it can get fairly close. Another way to arrive at the same result is
# to perform VQE with a reasonable ansatz, but to terminate the algorithm
# before it converges to the ground state. If we used the exact ground state
# :math:`|\psi_0\rangle`, the time-dependence would be trivial and the
# data would not provide enough information about the Hamiltonian parameters.


# cyclic graph as target interaction graph of the Ising Hamiltonian:
ising_graph = nx.cycle_graph(qubit_number)
layout_coordinates = nx.circular_layout(ising_graph)

print(f"Edges: {ising_graph.edges}")

plt.figure(figsize=(10,10))
ax = plt.gca()
ax.set_title('Target Interaction Graph of Ising Hamiltonian')
nx.draw(G = ising_graph, pos = layout_coordinates, width = 4)
nx.draw_networkx_nodes(G=complete_seed_ising_graph, pos=complete_seed_layout_coordinates, edgecolors="white")
_ = ax.axis('off')
plt.savefig(this_out_dir+'target_interaction_graph_of_ising_hamiltonian.png', format="PNG")

# initialize the “unknown” target parameters that describe the
# target Hamiltonian ( by sampling from a uniform probability distribution ranging from (-2, 2)

target_weights = [0.56, 1.24, 1.67, -0.79] # qubit interaction parameters
target_bias = [-1.44, -1.43, 1.18, -0.93]  # qubit parameters.

# we use this information to generate the matrix form of the
# Ising model Hamiltonian in the computational basis:

hamiltonian_matrix = create_hamiltonian_matrix(qubit_number, ising_graph, target_weights, target_bias)

# prints a visual representation of the Hamiltonian matrix

plt.figure(figsize=(10,10))
plt.matshow(hamiltonian_matrix, cmap="hot")
#plt.show()
plt.title('Target Hamiltonian Matrix')
plt.xlabel('2^n States')
plt.ylabel('2^n States')
plt.savefig(this_out_dir+'target_hamiltonian_matrix.png', format="PNG")

# verify that this is a low-energy state by numerically finding the lowest eigenvalue of the Hamiltonian
# and comparing it to the energy expectation of this low-energy state:

res = np.vdot(low_energy_state, (hamiltonian_matrix @ low_energy_state))
expected_energy = np.real_if_close(res)
ground_state_energy = np.real_if_close(min(np.linalg.eig(hamiltonian_matrix)[0]))

print(f"Energy Expectation: {expected_energy} Ground State Energy: {ground_state_energy}")

# We have in fact found a low-energy, non-ground state
# state. This, however, is only half of the information we need. We also require
# a collection of time-evolved, low-energy states.

############################
# Learning the Hamiltonian #
############################

# Before creating the full QGRNN and the cost function:
# complete graph is the "guessed" interaction graph because any target interaction graph will be a subgraph
# of this initial guess. Part of the idea behind the QGRNN is that
# we don’t know the interaction graph, and it has to be learned. In this case, the graph
# is learned *automatically* as the target parameters are optimized. The
# :math:`\boldsymbol\mu` parameters that correspond to edges that don't exist in
# the target graph will simply approach :math:`0`.

# Defines some fixed values
reg1 = tuple(range(qubit_number))  # First qubit register
reg2 = tuple(range(qubit_number, 2 * qubit_number))  # Second qubit register

index_of_control_qubit = 2 * qubit_number  # Index of index_of_control_qubit qubit
trotter_step = 0.01  # Trotter step size

# defines the interaction graph for the new qubit system

complete_seed_ising_graph = nx.complete_graph(reg2)
complete_seed_layout_coordinates = nx.circular_layout(complete_seed_ising_graph)
num_edges = len(complete_seed_ising_graph.edges)
print(f"Edges: {complete_seed_ising_graph.edges}")
nx.draw(complete_seed_ising_graph)

plt.figure(figsize=(10,10))
ax = plt.gca()
ax.set_title('Complete Seed Interaction Graph of Ising Hamiltonian')
nx.draw(G = complete_seed_ising_graph, pos = complete_seed_layout_coordinates, width=4)
nx.draw_networkx_nodes(G=complete_seed_ising_graph, pos=complete_seed_layout_coordinates, edgecolors="white")

_ = ax.axis('off')
plt.savefig(this_out_dir+'complete_seed_interaction_graph_of_ising_hamiltonian.png', format="PNG")

n_quantum_data = 15  # The number of pieces of quantum data that are used for each step n_quantum_data
max_time_perc = 0.1  # The maximum value of time that can be used for quantum data

####################### 
# optimization set up #
#######################

qgrnn_dev = qml.device("default.qubit", wires=2 * qubit_number + 1) # Defines the new device
qgrnn_qnode = qml.QNode(qgrnn, qgrnn_dev) # Defines the new QNode

optimization_steps = 300
optimizer = qml.AdamOptimizer(stepsize=0.5)

weights = random_number_generator.random(size = num_edges) - 0.5
bias = random_number_generator.random(size = qubit_number) - 0.5

initial_weights = copy.copy(weights)
initial_bias = copy.copy(bias)

# optimization loop
for i in range(0, optimization_steps):
    
    (weights, bias), cost = optimizer.step_and_cost(cost_function, weights, bias)

    # Prints the value of the cost function
    if i % 5 == 0:
        print(f"Cost at Step {i}: {cost}")
        print(f"Weights at Step {i}: {weights}")
        print(f"Bias at Step {i}: {bias}")
        print("---------------------------------------------")

######################################################################
# With the learned parameters, we construct a visual representation
# of the Hamiltonian to which they correspond and compare it to the
# target Hamiltonian, and the initial guessed Hamiltonian:

initial_hamiltonian_matrix = create_hamiltonian_matrix(qubit_number, nx.complete_graph(qubit_number), initial_weights, initial_bias)
inferred_hamiltonian_matrix = create_hamiltonian_matrix(qubit_number, nx.complete_graph(qubit_number), weights, bias)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

#ax.set_title('Comparison of Target, Initial, and Inferred Hamiltonian Matrix')

axes[0].matshow(hamiltonian_matrix, vmin=-7, vmax=7, cmap="hot")
axes[0].set_title("Target", y=1.13)

axes[1].matshow(initial_hamiltonian_matrix, vmin=-7, vmax=7, cmap="hot")
axes[1].set_title("Initial", y=1.13)

axes[2].matshow(inferred_hamiltonian_matrix, vmin=-7, vmax=7, cmap="hot")
axes[2].set_title("Learned", y=1.13)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
# fig.suptitle('Main title')
plt.savefig(this_out_dir+'compare_heatmaps_target_initial_inferred.png', format="PNG")


# We can also look at the exact values of the target and learned parameters.
# Recall how the target
# interaction graph has :math:`4` edges while the complete graph has :math:`6`.
# Thus, as the QGRNN converges to the optimal solution, the weights corresponding to
# edges :math:`(1, 3)` and :math:`(2, 0)` in the complete graph should go to :math:`0`, as
# this indicates that they have no effect, and effectively do not exist in the learned Hamiltonian.

# We first pick out the weights of edges (1, 3) and (2, 0)
# and then remove them from the list of target parameters

weights_noedge = []
weights_edge = []
for ii, edge in enumerate(complete_seed_ising_graph.edges):
    if (edge[0] - qubit_number, edge[1] - qubit_number) in ising_graph.edges:
        weights_edge.append(weights[ii])
    else:
        weights_noedge.append(weights[ii])

######################################################################
# Then, we print all of the weights:

print("Target parameters     Learned parameters")
print("Weights")
print("-" * 41)
for ii_target, ii_learned in zip(target_weights, weights_edge):
    print(f"{ii_target : <20}|{ii_learned : >20}")

print("\nBias")
print("-"*41)
for ii_target, ii_learned in zip(target_bias, bias):
    print(f"{ii_target : <20}|{ii_learned : >20}")

print(f"\nNon-Existing Edge Parameters: {[val.unwrap() for val in weights_noedge]}")

######################################################################
# The weights of edges :math:`(1, 3)` and :math:`(2, 0)`
# are very close to :math:`0`, indicating we have learned the cycle graph
# from the complete graph. In addition, the remaining learned weights
# are fairly close to those of the target Hamiltonian.
# Thus, the QGRNN is functioning properly, and has learned the target
# Ising Hamiltonian to a high
# degree of accuracy!

######################################################################
# References
# ----------
#
# 1. Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., &
#    Hidary, J. (2019). Quantum Graph Neural Networks. arXiv preprint
#    `arXiv:1909.12264 <https://arxiv.org/abs/1909.12264>`__.
