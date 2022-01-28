
#############################################################################################
## Using Qiskit to model 1d Brick Layer Many-body Entanglement Transition Simulation       ##
#############################################################################################
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute
from qiskit.providers.aer import QasmSimulator

from qiskit.quantum_info.operators import Operator

from qiskit.extensions.simulator.snapshot import snapshot

import qiskit.quantum_info as qi

import pandas as pd
from numpy.linalg import inv

from qiskit.quantum_info import Clifford

import timeit

simulator = QasmSimulator(method='matrix_product_state')

line_divider_size = 50

hilbert_space_vector_size_2qubits = 4
n_epochs = 10
measurement_rate = 0.3
up_state = np.array([0,1])
n_qubit_space = [x for x in range(3,10)] # 16,32
measurement_rate_space = [x/100 for x in range(5,80,5)]

subsystem_range_divider = 4
projective_list = ['R1_P_00', 'R1_P_01', 'R1_P_10', 'R1_P_11']
use_unitary_set = 'Clifford Group' # 'Clifford Group' 'Random Unitaries'
simulation_df = pd.DataFrame()

for measurement_rate in measurement_rate_space:
    
    print("="*line_divider_size)
    print("- Measurement Rate = " + str(measurement_rate))
        
    for num_qubits in n_qubit_space:
        
        print("-- System Size =  " + str(num_qubits))
        
        subsystem_range = list(range(0,int(np.round(num_qubits/subsystem_range_divider))))
        quantum_circuit = QuantumCircuit(num_qubits, num_qubits)
        
        for qubit_index in range(0, num_qubits-1):
            print("--- Setting Qubit " + str(qubit_index) + " in |↑> state")
            quantum_circuit.initialize(up_state, 0)
        
        epoch_start_time = timeit.default_timer()

        for this_epoch in range(1, n_epochs):
        
            print("--- Starting Epoch = " + str(this_epoch))
            for qubit_index in range(0, num_qubits-1):
        
                # qubit_index = 0
                next_qubit_index = qubit_index + 1
                rand_uni_0to1_draw = np.random.uniform(0,1)
        
                if rand_uni_0to1_draw <= measurement_rate:

                    print("---- Adding Projective Measurement " + str(qubit_index) + "-🬀-" + str(next_qubit_index))

                    rand_uni_proj_choice = np.random.choice(projective_list)
                    # projective measurement before the unitary gate
        
                    if rand_uni_proj_choice == 'R1_P_11':
        
                        quantum_circuit.reset(qubit_index)
                        quantum_circuit.reset(next_qubit_index)
        
                        quantum_circuit.x(qubit_index)
                        quantum_circuit.x(next_qubit_index)
        
                    elif rand_uni_proj_choice == 'R1_P_01':
        
                        quantum_circuit.reset(qubit_index)
                        quantum_circuit.reset(next_qubit_index)
        
                        quantum_circuit.x(next_qubit_index)
        
                    elif rand_uni_proj_choice == 'R1_P_10':
        
                        quantum_circuit.reset(qubit_index)
                        quantum_circuit.reset(next_qubit_index)
        
                        quantum_circuit.x(qubit_index)
        
                    elif rand_uni_proj_choice == 'R1_P_00':
        
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
            
            epoch_start_time = timeit.default_timer()
            epoch_time = timeit.default_timer() - epoch_start_time
            print("--- Epoch took " + str(np.round(epoch_time, 2)) + " seconds.")

            print("--- Reduced DensityMatrix Calculation " + str(this_epoch) + "")
            rho = qi.DensityMatrix.from_instruction(quantum_circuit)
            reduced_rho = qi.partial_trace(rho, subsystem_range)
            renyi_entropy_2nd = -1.0 * np.log2( np.real( np.trace( np.matmul(reduced_rho, reduced_rho) ) ) )
        
            simulation_df = simulation_df.append(pd.DataFrame.from_dict({'num_qubits': [num_qubits], 'measurement_rate':[measurement_rate], 'epoch': [this_epoch], 'renyi_entropy_2nd': [renyi_entropy_2nd] }))
        

simulation_df.to_csv("./out-data/simulation_df.csv", sep=',')

simulation_df_summary = simulation_df.group_by(['num_qubits','measurement_rate']).[['renyi_entropy_2nd']].mean().reset_index()


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
