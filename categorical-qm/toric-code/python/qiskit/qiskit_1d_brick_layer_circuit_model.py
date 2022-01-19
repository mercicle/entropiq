
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


simulator = QasmSimulator(method='matrix_product_state')

num_qubits = 16
subsystem_range = list(range(0,int(np.round(num_qubits/4))))
quantum_circuit = QuantumCircuit(num_qubits, num_qubits)

hilbert_space_vector_size_2qubits = 4
n_epochs = 10
measurement_rate = 0.3

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

# under conjugation takes pauli strings to pauli strings
clifford_gate_dict = dict({'Hadamard': Operator(np.array([[ 0.707+0.j, 0.707-0.j],[ 0.707+0.j, -0.707+0.j]])),
                           'sqrt_Z_phase': Operator(np.array([[1.+0.j, 0.+0.j],[0.+0.j, 0.+1.j]])),
                           'conjugate_sqrt_Z_phase': Operator(np.array([[1.+0.j, 0.+0.j],[0.+0.j, 0.-1.j]]))
                           })

clifford_gate_list = list(clifford_gate_dict.keys())

# test using qiskit Operator.is_unitary()
Operator.is_unitary(clifford_gate_dict[clifford_gate_list[0]])
Operator.is_unitary(R1_P_00)

for qubit_index in range(0, num_qubits-1):
    print("Setting Qubit " + str(qubit_index) + " in |↑> state")
    quantum_circuit.initialize(up_state, 0)

use_unitary_set = 'Clifford Group' # 'Clifford Group' 'Random Unitaries'
simulation_df = pd.DataFrame()
for this_epoch in range(1, n_epochs):

    print("Starting Epoch = " + str(this_epoch))
    for qubit_index in range(0, num_qubits-1):

        # qubit_index = 0
        next_qubit_index = qubit_index + 1
        rand_uni_0to1_draw = np.random.uniform(0,1)

        print("-- Starting Operation " + str(qubit_index) + "-🬀-" + str(next_qubit_index))
        if rand_uni_0to1_draw <= measurement_rate:

            rand_uni_proj_choice = np.random.choice(projective_list)
            this_projective = projective_dict[rand_uni_proj_choice]
            # projective measurement before the unitary gate

            #Operator.is_unitary(Operator(np.matmul(qi.random_unitary(hilbert_space_vector_size_2qubits), np.asmatrix(this_projective))))
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

        if use_unitary_set == 'Clifford Group':

            random_clifford = qi.random_clifford(num_qubits=2)

            quantum_circuit.append(random_clifford, [qubit_index, next_qubit_index])

        elif use_unitary_set == 'Random Unitaries':

            # uses randomly selected from Haar measures using Qiskit qi.random_unitary()
            unitary_label = "rand_unit_" + str(qubit_index) + "_" + str(next_qubit_index)
            quantum_circuit.append(qi.random_unitary(hilbert_space_vector_size_2qubits), [qubit_index, next_qubit_index])

        else:

            print("Now a supported set of unitaries.")

    print("Starting Epoch = " + str(this_epoch) + " DensityMatrix calculations")
    rho = qi.DensityMatrix.from_instruction(quantum_circuit)

    reduced_rho = qi.partial_trace(rho, subsystem_range)

    renyi_entropy_2nd = -1.0 * np.log2( np.real( np.trace( np.matmul(reduced_rho, reduced_rho) ) ) )

    simulation_df = simulation_df.append(pd.DataFrame.from_dict({'num_qubits': [num_qubits], 'measurement_rate':[measurement_rate], 'epoch': [this_epoch], 'renyi_entropy_2nd': [renyi_entropy_2nd]}))


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
