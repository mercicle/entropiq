
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute
from qiskit.providers.aer import QasmSimulator

from qiskit.quantum_info.operators import Operator

from qiskit.extensions.simulator.snapshot import snapshot


simulator = QasmSimulator(method='matrix_product_state')

num_qubits = 10
quantum_circuit = QuantumCircuit(num_qubits, num_qubits)

n_epochs = 10
measurement_rate = 0.1

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

unitary_projective_list = [R1_P_00, R1_P_01, R1_P_10, R1_P_11]
n_r1_unitary_projectives = len(unitary_projective_list)

for qubit_index in range(0, num_qubits-1):
    print("Creating superposition of Qubit " + str(qubit_index))
    quantum_circuit.h(qubit_index)
    
for qubit_index in range(0, num_qubits-1):
    next_qubit_index = qubit_index + 1
    print("Entangling Qubit " + str(qubit_index) + " and " + str(next_qubit_index))
    quantum_circuit.cx(qubit_index, next_qubit_index)


for this_epoch in range(0, n_epochs-1):
    
    for qubit_index in range(0, num_qubits-1):
        
        next_qubit_index = qubit_index + 1
        rand_uni_0to1_draw = np.random.uniform(0,1)
        rand_uni_proj_index = np.random.randint(0,n_r1_unitary_projectives)
        
        if rand_uni_0to1_draw <= measurement_rate:
            # performs a unitary projective gate which performs a projective measurement before the unitary gate 
            quantum_circuit.cx(qubit_index, next_qubit_index)
        else:
            quantum_circuit.cx(qubit_index, next_qubit_index)
            
            
    
# Measure
circ.measure(range(num_qubits), range(num_qubits))

job_sim = execute(circ, simulator)
result = job_sim.result()
print("Time taken: {} sec".format(result.time_taken))
result.get_counts()


################################################################
################################################################
################################################################

# https://qiskit.org/documentation/apidoc/circuit_library.html
# https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html#multi-qubit-gates
# https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
# https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html
