
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

simulator = QasmSimulator(method='matrix_product_state')

num_qubits = 10
quantum_circuit = QuantumCircuit(num_qubits, num_qubits)

hilbert_space_vector_size_2qubits = 4
n_epochs = 10
measurement_rate = 0.8

down_state = np.array([1,0])
up_state = np.array([0,1])

down_down_ket = np.kron(down_state, down_state)
up_up_ket = np.kron(up_state, up_state)

up_down_ket = np.kron(up_state, down_state)
down_up_ket = np.kron(down_state, up_state)

# rank 1 measurements
R1_P_00 = np.outer(up_up_ket, up_up_ket)
R1_P_01 = np.outer(up_down_ket, up_down_ket)
R1_P_10 = np.outer(down_up_ket, down_up_ket)
R1_P_11 = np.outer(down_down_ket, down_down_ket)

# test unitary
do_manual_unitary_test = False
if do_manual_unitary_test:
    this_projective_conjugate_transpose = np.asmatrix(R1_P_11).getH() # Returns the (complex) conjugate transpose of self.
    this_projective_inverse = inv(np.asmatrix(R1_P_11))
    is_unitary = np.allclose(this_projective_conjugate_transpose,this_projective_inverse)

# rank 2 measurements
R2_P_0 = R1_P_00 + R1_P_11
R2_P_1 = R1_P_01 + R1_P_10

# coerce to qiskit operators 
R1_P_00 = Operator(R1_P_00)
R1_P_01 = Operator(R1_P_01)
R1_P_10 = Operator(R1_P_10)
R1_P_11 = Operator(R1_P_11)

R2_P_0 = Operator(R2_P_0)
R2_P_1 = Operator(R2_P_1)

projective_dict = dict({'R1_P_00': R1_P_00, 'R1_P_01': R1_P_01, 'R1_P_10': R1_P_10, 'R1_P_11': R1_P_11})
projective_list = list(projective_dict.keys())

clifford_gate_dict = dict({'Hadamard': Operator(np.array([[ 0.707+0.j, 0.707-0.j],[ 0.707+0.j, -0.707+0.j]])), 
                           'sqrt_Z_phase': Operator(np.array([[1.+0.j, 0.+0.j],[0.+0.j, 0.+1.j]])), 
                           'conjugate_sqrt_Z_phase': Operator(np.array([[1.+0.j, 0.+0.j],[0.+0.j, 0.-1.j]]))
                           })
clifford_gate_list = list(clifford_gate_dict.keys())

# test using qiskit Operator.is_unitary()
Operator.is_unitary(clifford_gate_dict[clifford_gate_list[0]])
Operator.is_unitary(R1_P_00)

# Setup the 1d qubit chain
for qubit_index in range(0, num_qubits-1):
    print("Creating superposition of Qubit " + str(qubit_index))
    quantum_circuit.h(qubit_index)

# entangle neighbors i, i+1
for qubit_index in range(0, num_qubits-1):
    next_qubit_index = qubit_index + 1
    print("Entangling Qubit " + str(qubit_index) + " and " + str(next_qubit_index))
    quantum_circuit.cx(qubit_index, next_qubit_index)

use_unitary_set = 'Clifford Group'
simulation_df = pd.DataFrame()
for this_epoch in range(1, n_epochs):
    
    for qubit_index in range(0, num_qubits-1):
        
        # qubit_index = 0 
        next_qubit_index = qubit_index + 1
        rand_uni_0to1_draw = np.random.uniform(0,1)
        
        if rand_uni_0to1_draw <= measurement_rate:
            
            rand_uni_proj_choice = np.random.choice(projective_list)
            this_projective = projective_dict[rand_uni_proj_choice]
            # projective measurement before the unitary gate 
            
            quantum_circuit.append(this_projective, [qubit_index, next_qubit_index])

        if use_unitary_set == 'Clifford Group':
            
            rand_unitary_choice = np.random.choice(clifford_gate_list)
            this_clifford = clifford_gate_dict[rand_unitary_choice]
            
            quantum_circuit.append(this_clifford, [qubit_index, next_qubit_index])

        else:
            # uses randomly selected from Haar measures using Qiskit qi.random_unitary()
            unitary_label = "rand_unit_" + str(qubit_index) + "_" + str(next_qubit_index)
            quantum_circuit.append(qi.random_unitary(hilbert_space_vector_size_2qubits), [qubit_index, next_qubit_index])
            
    rho = qi.DensityMatrix.from_instruction(quantum_circuit)
    renyi_entropy_2nd = -1.0 * np.log2( np.real( np.trace( np.matmul(rho, rho) ) ) )

    simulation_df.append(pd.DataFrame.from_dict({'num_qubits': [num_qubits], 'measurement_rate':[measurement_rate], 'epoch': [this_epoch], 'renyi_entropy_2nd': [renyi_entropy_2nd]}))
    

################################################################
################################################################
################################################################

# https://qiskit.org/documentation/apidoc/circuit_library.html
# https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html#multi-qubit-gates
# https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
# https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html
# https://quantumcomputing.stackexchange.com/questions/4975/how-do-i-build-a-gate-from-a-matrix-on-qiskit
# https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_unitary.html

# appending 
#https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html
