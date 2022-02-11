
#############################################################################################
## Using Qiskit to model 1d Brick Layer Many-body Entanglement Transition Simulation       ##
#############################################################################################
import os, timeit
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.providers.aer import QasmSimulator, extensions

from qiskit.circuit import Barrier
from qiskit.extensions.simulator.snapshot import snapshot

import qiskit.quantum_info as qi

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv

from qiskit.quantum_info import Statevector
import copy 

                        
                        
simulator = QasmSimulator(method='matrix_product_state')

backend = Aer.get_backend('statevector_simulator')
sim = Aer.get_backend('aer_simulator')

line_divider_size = 50

hilbert_space_vector_size_2qubits = 4
n_epochs = 100
up_state = np.array([0,1])
random_probs_2q_test_vector = [0.25, 0.25, 0.25, 0.25]

max_qubits = 8
n_qubit_space = [x for x in range(3,max_qubits+1)] # 16,32
min_measurement_rate = 5
max_measurement_rate = 50
measurement_rate_space = [x/100 for x in range(min_measurement_rate, max_measurement_rate,5)]

subsystem_range_divider = 4
projective_list = ['00', '01', '10', '11']

use_unitary_set = 'Clifford Group' # 'Clifford Group' 'Random Unitaries'
apply_snapshots = False

# after_proj_prob_and_initstate_fix_
custom_label = 'cliffords_wbarrier_fix_'
sim_results_label = custom_label+str(max_qubits)+'qubits_'+'_mspace'+str(min_measurement_rate)+"to"+str(max_measurement_rate)
simulation_df = pd.DataFrame()

for measurement_rate in measurement_rate_space:
    
    print("="*line_divider_size)
    print("- Measurement Rate = " + str(measurement_rate))
    
    measurement_rate_start_time = timeit.default_timer()

    for num_qubits in n_qubit_space:
        
        # num_qubits = 4
        print("-- System Size =  " + str(num_qubits))
        
        subsystem_range = list(range(0,int(np.round(num_qubits/subsystem_range_divider))))
        quantum_circuit = QuantumCircuit(num_qubits, num_qubits)
        
        for qubit_index in range(0, num_qubits):
            print("--- Setting Qubit " + str(qubit_index) + " in |↑> state")
            quantum_circuit.initialize(up_state, qubit_index)
        
        if apply_snapshots:
            for qubit_index in range(0, num_qubits):
                
                if qubit_index < num_qubits-1:
                    next_qubit_index = qubit_index + 1
                    snapshot_name_string = "snapshot_" + str(qubit_index) + "_" + str(next_qubit_index)
                    print("--- Setting " + snapshot_name_string)
                    quantum_circuit.snapshot(snapshot_name_string, qubits = [qubit_index,next_qubit_index])
            
        #quantum_circuit.snapshot("snapshot_all", qubits = list(range(0, num_qubits)))

        epoch_start_time = timeit.default_timer()

        for this_epoch in range(1, n_epochs):
        
            # this_epoch=1
            #print("--- Starting Epoch = " + str(this_epoch))
            for qubit_index in range(0, num_qubits-1):
        
                # qubit_index = 0
                next_qubit_index = qubit_index + 1
                rand_uni_0to1_draw = np.random.uniform(0,1)
                snapshot_name_string = "snapshot_" + str(qubit_index) + "_" + str(next_qubit_index)

                if rand_uni_0to1_draw <= measurement_rate:

                    print("---- Adding Projective Measurement " + str(qubit_index) + "-🬀-" + str(next_qubit_index))

                    if use_unitary_set == 'Clifford Group':
             
                        #result = execute(quantum_circuit, backend, shots=1).result()
                        #snapshots = result.data()['snapshots']['statevector'][snapshot_name_string][0].tolist()
    
                        #this_state_vector = qi.Statevector(snapshots)
                        #probs = this_state_vector.probabilities([qubit_index,next_qubit_index]).tolist()
                        #probs = [np.round(e, 2) for e in probs]
                        
                        #from qiskit.quantum_info import Statevector

                        #quantum_circuit_copy = copy.deepcopy(quantum_circuit)
                        #quantum_circuit_copy.save_statevector() # Save initial state

                        #zero_state = Statevector(quantum_circuit)
                        #final_state = zero_state.evolve(quantum_circuit)

                        #reduced_state = qi.partial_trace(final_state,[qubit_index,next_qubit_index])

                        rho = qi.DensityMatrix.from_instruction(quantum_circuit)
                        unitary_pmf = rho.probabilities([qubit_index,next_qubit_index]).tolist()
                        unitary_pmf = [np.round(prob,2) for prob in unitary_pmf]
                        
                        ## experiment 
                        #quantum_circuit.save_statevector()
                        #result = sim.run(quantum_circuit_copy).result()    
                        #out_state = result.get_statevector()
                        #prob_dict = out_state.probabilities_dict([qubit_index,next_qubit_index])
                        
                        #these_prob_keys = prob_dict.keys()
                        #unitary_pmf = [] 
                        #for prob_key in projective_list:
                        #    if prob_key in these_prob_keys:
                        #        unitary_pmf.append(prob_dict[prob_key])
                        #    else:
                        #        unitary_pmf.append(0)

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
                
                print("---- Starting Unitary Operation " + str(qubit_index) + "-🬀-" + str(next_qubit_index))

                if use_unitary_set == 'Clifford Group':
        
                    random_clifford = qi.random_clifford(num_qubits=2)
        
                    quantum_circuit.append(random_clifford, [qubit_index, next_qubit_index])
        
                elif use_unitary_set == 'Random Unitaries':
        
                    # uses randomly selected from Haar measures using Qiskit qi.random_unitary()
                    unitary_label = "rand_unit_" + str(qubit_index) + "_" + str(next_qubit_index)
                    quantum_circuit.append(qi.random_unitary(hilbert_space_vector_size_2qubits), [qubit_index, next_qubit_index])
        
                else:
        
                    print("Now a supported set of unitaries.")
            
            epoch_time = timeit.default_timer() - epoch_start_time
            #print("--- Epoch took " + str(np.round(epoch_time, 2)) + " seconds.")
            
            # see below: qiskit-densitymatrix-from-instruction-when-snapshots-are-present
            #quantum_circuit.data = [(Barrier(_inst[0].num_qubits), _inst[1], _inst[2]) if 'snapshot_' in _inst[0].name  else _inst for _inst in quantum_circuit.data]

            #print("--- Reduced DensityMatrix Calculation " + str(this_epoch) + "")
            rho = qi.DensityMatrix.from_instruction(quantum_circuit)
            

            #QiskitError: 'Cannot apply Instruction: snapshot'
            
            reduced_rho = qi.partial_trace(rho, subsystem_range)
            renyi_entropy_2nd = -1.0 * np.log2( np.real( np.trace( np.matmul(reduced_rho, reduced_rho) ) ) )
        
            simulation_df = simulation_df.append(pd.DataFrame.from_dict({'num_qubits': [num_qubits], 'measurement_rate':[measurement_rate], 'epoch': [this_epoch], 'renyi_entropy_2nd': [renyi_entropy_2nd] }))
        
    measurement_rate_value_time = timeit.default_timer() - measurement_rate_start_time
    print("--- Measurement rate " + str(measurement_rate) + " took " + str(np.round(measurement_rate_value_time/60, 2)) + " minutes.")

simulation_df.to_csv(os.getcwd() + "/out-data/"+sim_results_label+"_simulation_df.csv", sep=',')

simulation_df_summary = simulation_df.groupby(['num_qubits','measurement_rate'])[['renyi_entropy_2nd']].mean().reset_index()

simulation_df_summary.head()

simulation_df_summary.to_csv(os.getcwd() + "/out-data/"+sim_results_label+"_simulation_df_summary.csv", sep=',')

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


################################################################
################################################################
################################################################

# https://qiskit.org/documentation/apidoc/circuit_library.html
# https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html#multi-qubit-gates
# https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
# https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html
# https://quantumcomputing.stackexchange.com/questions/4975/how-do-i-build-a-gate-from-a-matrix-on-qiskit
# https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_unitary.html

# appending gates
#https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html

#https://qiskit.org/documentation/stubs/qiskit.quantum_info.Clifford.html
#https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_clifford.html
#https://quantumcomputing.stackexchange.com/questions/14056/what-is-the-clifford-gates-selection-probability-distribution-used-in-the-genera

#https://quantumcomputing.stackexchange.com/questions/15868/applying-a-projector-to-a-qubit-in-a-qiskit-circuit

#https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.reset.html


#  Error "too many subscripts in einsum" when system size > 10
# https://quantumcomputing.stackexchange.com/questions/16753/error-too-many-subscripts-in-einsum-unitarygate

# quantumcomputing.stackexchange.com
# https://quantumcomputing.stackexchange.com/questions/24044/qiskit-densitymatrix-from-instruction-when-snapshots-are-present/24046#24046
