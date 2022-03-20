
#############################################################################################
## Using Qiskit to model 1d Brick Layer Many-body Entanglement Transition Simulation       ##
#############################################################################################
import os, timeit
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.providers.aer import QasmSimulator, extensions

import qiskit.quantum_info as qi

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import seaborn as sns
from numpy.linalg import inv

from qiskit.quantum_info import Statevector
import copy

import pickle

simulator = QasmSimulator(method='matrix_product_state')
backend = Aer.get_backend('statevector_simulator')
sim = Aer.get_backend('aer_simulator')

def isodd(num):
    return num % 2 == 1

def iseven(num):
    return num % 2 != 1

line_divider_size = 50

hilbert_space_vector_size_2qubits = 4
n_layers = 20
up_state = np.array([0,1])
random_probs_2q_test_vector = [0.25, 0.25, 0.25, 0.25]

min_qubits = 6
max_qubits = 10
n_qubit_space = list(range(min_qubits,max_qubits+1)) # 16,32
min_measurement_rate = 20
max_measurement_rate = 80
measurement_rate_step = 10
measurement_rate_space = [x/100 for x in range(min_measurement_rate, max_measurement_rate, measurement_rate_step)]

n_simulations = 10
simulation_space = list(range(0,n_simulations))
subsystem_range_divider = 2
projective_list = ['00', '01', '10', '11']

use_unitary_set = 'Random Unitaries' # 'Clifford Group' 'Random Unitaries'

# after_proj_prob_and_initstate_fix_
custom_label = 'julia_comparison_'
sim_results_label = custom_label+ str(n_simulations) + "sims_" + str(max_qubits)+'qubits_'+'_mspace'+str(min_measurement_rate)+"to"+str(max_measurement_rate)
simulation_df = pd.DataFrame()
layer_dict = dict()

for this_simulation in simulation_space:

    print("="*line_divider_size)
    print("- Simulation = " + str(this_simulation))

    for measurement_rate in measurement_rate_space:

        print("="*line_divider_size)
        print("- Measurement Rate = " + str(measurement_rate))

        measurement_rate_start_time = timeit.default_timer()

        for num_qubits in n_qubit_space:

            # num_qubits = 4
            print("-- System Size =  " + str(num_qubits))

            qubit_indices = list(range(0,num_qubits))

            subsystem_range = list(range(0,int(np.round(num_qubits/subsystem_range_divider))))
            trace_over_system = [x for x in qubit_indices if x not in subsystem_range]
            
            quantum_circuit = QuantumCircuit(num_qubits, num_qubits)

            for qubit_index in range(0, num_qubits):
                #print("--- Setting Qubit " + str(qubit_index) + " in |↑> state")
                quantum_circuit.initialize(up_state, qubit_index)

            layer_start_time = timeit.default_timer()
            keep_layer = False

            for this_layer_index in range(1, n_layers+1):

                # this_layer_index=1
                #print("--- Starting layer = " + str(this_layer_index))
                for qubit_index in range(0, num_qubits-1):

                    # qubit_index = 0
                    next_qubit_index = qubit_index + 1
                    rand_uni_0to1_draw = np.random.uniform(0,1)

                    if ((isodd(this_layer_index) and isodd(qubit_index)) or (iseven(this_layer_index) and iseven(qubit_index))):

                        if rand_uni_0to1_draw <= measurement_rate:

                            #print("---- Adding Projective Measurement " + str(qubit_index) + "-🬀-" + str(next_qubit_index))

                            if use_unitary_set == 'Clifford Group':

                                rho = qi.DensityMatrix.from_instruction(quantum_circuit)
                                unitary_pmf = rho.probabilities([qubit_index,next_qubit_index]).tolist()
                                unitary_pmf = [np.round(prob,2) for prob in unitary_pmf]

                                is_uniform = (unitary_pmf == random_probs_2q_test_vector)

                                if not is_uniform:
                                    print("|.." + str(qubit_index) + "-" + str(next_qubit_index) + str("..> is not uniform: ")+ ' - '.join([str(x) for x in unitary_pmf]))

                            elif use_unitary_set == 'Random Unitaries':

                                unitary_pmf = random_probs_2q_test_vector

                            rand_uni_proj_choice = np.random.choice(projective_list, p = unitary_pmf)

                            # projective measurement before the unitary gate

                            if rand_uni_proj_choice == '11':

                                quantum_circuit.reset(qubit_index)
                                quantum_circuit.reset(next_qubit_index)

                                quantum_circuit.x(qubit_index)
                                quantum_circuit.x(next_qubit_index)

                            elif rand_uni_proj_choice == '01':

                                quantum_circuit.reset(qubit_index)
                                quantum_circuit.reset(next_qubit_index)

                                quantum_circuit.x(next_qubit_index)

                            elif rand_uni_proj_choice == '10':

                                quantum_circuit.reset(qubit_index)
                                quantum_circuit.reset(next_qubit_index)

                                quantum_circuit.x(qubit_index)

                            elif rand_uni_proj_choice == '00':

                                quantum_circuit.reset(qubit_index)
                                quantum_circuit.reset(next_qubit_index)

                        #print("---- Starting Unitary Operation " + str(qubit_index) + "-🬀-" + str(next_qubit_index))

                        if use_unitary_set == 'Clifford Group':

                            random_clifford = qi.random_clifford(num_qubits=2)

                            quantum_circuit.append(random_clifford, [qubit_index, next_qubit_index])

                        elif use_unitary_set == 'Random Unitaries':

                            # uses randomly selected from Haar measures using Qiskit qi.random_unitary()
                            unitary_label = "rand_unit_" + str(qubit_index) + "_" + str(next_qubit_index)
                            quantum_circuit.append(qi.random_unitary(hilbert_space_vector_size_2qubits), [qubit_index, next_qubit_index])

                        else:

                            print("Now a supported set of unitaries.")

                layer_time = timeit.default_timer() - layer_start_time
                #print("--- layer took " + str(np.round(layer_time, 2)) + " seconds.")

                if this_layer_index == n_layers:
                    keep_layer = True

                #print("--- Reduced DensityMatrix Calculation " + str(this_layer_index) + "")
                rho = qi.DensityMatrix.from_instruction(quantum_circuit)

                #plt.matshow(decoherence_network)
                # why doesn't this sum to one?
                #np.sum(np.abs(decoherence_network[0,:]))

                #diagonal_vector = np.diagonal(np.asmatrix(np.real(rho))).tolist()

                #np.diagonal(np.asmatrix(np.real(reduced_rho))).tolist()

                reduced_rho = qi.partial_trace(rho, trace_over_system)
                renyi_entropy_2nd = -1.0 * np.log2( np.real( reduced_rho.purity() ) )

                if int(100*measurement_rate) == max_measurement_rate and num_qubits == max_qubits and this_simulation == 0:
                    decoherence_network = np.asmatrix(np.real(reduced_rho))
                    layer_dict[this_layer_index] = decoherence_network

                # https://qiskit.org/textbook/ch-quantum-hardware/density-matrix.html#properties
                # np.trace( np.matmul(reduced_rho, reduced_rho) ) == reduced_rho.purity()

                simulation_df = simulation_df.append(pd.DataFrame.from_dict({'simulation': [this_simulation], 'num_qubits': [num_qubits], 'measurement_rate':[measurement_rate], 'layer': [this_layer_index],'keep_layer': [keep_layer], 'renyi_entropy_2nd': [renyi_entropy_2nd] }))

        measurement_rate_value_time = timeit.default_timer() - measurement_rate_start_time
        print("--- Measurement rate " + str(measurement_rate) + " took " + str(np.round(measurement_rate_value_time/60, 2)) + " minutes.")

simulation_df.to_csv(os.getcwd() + "/out-data/"+sim_results_label+"_simulation_df.csv", sep=',')

simulation_df_final = simulation_df[simulation_df.keep_layer == True].sort_values(by=['num_qubits', 'measurement_rate','layer'])

simulation_df_summary = simulation_df.groupby(['num_qubits','measurement_rate'])[['renyi_entropy_2nd']].mean().reset_index()
simulation_df_summary.head()
simulation_df_summary.to_csv(os.getcwd() + "/out-data/"+sim_results_label+"_simulation_df_summary.csv", sep=',')


pickle.dump( favorite_color, open( "save.p", "wb" ) )


sns.set(rc = {'figure.figsize':(12,12)})
sns.set_style(style='whitegrid')

sim_plot = sns.lineplot(x='measurement_rate',
                        y='renyi_entropy_2nd',
                        data = simulation_df_summary,
                        hue='num_qubits',
                        marker='o',
                        linewidth = 3,
                        markersize = 10)
sim_plot.set_xlabel("Measurement Rate", fontsize = 20)
sim_plot.set_ylabel("2nd Renyi Entropy", fontsize = 20)
plt.savefig(os.getcwd() + '/out-data/' + sim_results_label+ '_simulation-results.pdf')


#################################
###     Layer Analysis        ###
#################################

#simulation_df['log_renyi_entropy_2nd'] = simulation_df.apply(lambda row: np.log10(row['renyi_entropy_2nd']), axis=1)
# example: https://seaborn.pydata.org/examples/faceted_lineplot.html

sns.set(rc = {'figure.figsize':(12,12)})
sns.set_style(style='whitegrid')

fig, ax = plt.subplots()

simulation_df.renyi_entropy_2nd.hist()
plt.savefig(os.getcwd() + '/out-data/' + sim_results_label+ '_simulation-results-renyi-hist.pdf')


size_order = list(np.unique(simulation_df.measurement_rate))
n_colors = len(size_order)
palette = sns.color_palette("rocket_r",10)[0:n_colors]
# sns.color_palette("rocket_r",10)
# sns.color_palette("RdPu", 10)
sns.relplot(
    data = simulation_df,
    x="layer",
    y="renyi_entropy_2nd",
    hue="measurement_rate",
    col="num_qubits",
    kind="line", palette=palette,
    facet_kws=dict(sharex=False)
)
plt.savefig(os.getcwd() + '/out-data/' + sim_results_label+ '_simulation-results-layer-evolution.pdf')



#################################
###         Simulation        ###
#################################
%matplotlib qt

initial_matrix = -1*np.log2(np.abs(layer_dict[1]))

def update(i):
    initial_matrix = -1*np.log2(np.abs(layer_dict[i+1]))
    matrix_matshow.set_array(initial_matrix)
    plt.title("Density Matrix ρ Evolution - Layer " + str(i), fontweight='bold',fontsize=15)

fig, ax = plt.subplots()
matrix_matshow = ax.matshow(initial_matrix, cmap=plt.cm.Greys_r)
plt.colorbar(matrix_matshow)
ani = animation.FuncAnimation(fig, update, frames=len(layer_dict.keys()), interval=1000)
#plt.title("Density Matrix ρ Evolution")
ax.text(75, 25, '-1*log2(ρ_ij)', bbox={'facecolor': 'white', 'pad': 10},
        fontsize=15,
        fontweight='bold')
ax.text(75, 0, 'Smaller \nDecoherence\nProbability', bbox={'facecolor': 'white', 'pad': 10},
        fontsize=15,
        fontweight='bold')
ax.text(75, 65, 'Larger \nDecoherence\nProbability', bbox={'facecolor': 'white', 'pad': 10},
        fontsize=15,
        fontweight='bold')
plt.show()

################################################################
################################################################
################################################################
