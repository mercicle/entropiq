
# Example from Qiskit
# https://qiskit.org/documentation/tutorials/simulators/7_matrix_product_state_method.html


import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator

import qiskit.quantum_info as qi

from qiskit.visualization import array_to_latex

# Select the AerSimulator from the Aer provider
simulator = AerSimulator(method='matrix_product_state')


# Construct quantum circuit
circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)
circ.measure([0,1], [0,1])

# Run and get counts, using the matrix_product_state method
tcirc = transpile(circ, simulator)
result = simulator.run(tcirc).result()
counts = result.get_counts(0)
counts

# To see the internal state vector of the circuit we can use the save_statevector instruction.
# To return the full internal MPS structure we can also use the save_matrix_product_state instruction.

#######################################
##          Max entanglement         ##
#######################################

circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)


rho = qi.DensityMatrix.from_instruction(circ)

u, s, vh = np.linalg.svd(np.real(np.matrix(rho)), full_matrices=False)

s = np.round(s,2)

reduced_rho = qi.partial_trace(rho, [1])


# Define a snapshot that shows the current state vector
circ.save_statevector(label='state_vector')
circ.save_matrix_product_state(label='MPS_representation')
circ.measure([0,1], [0,1])

# Execute and get saved data
tcirc = transpile(circ, simulator)
result = simulator.run(tcirc).result()
data = result.data(0)

#print the result data
data

{'counts': {'0x3': 503, '0x0': 521},
 'state_vector': Statevector([0.70710678+0.j, 0.        +0.j, 0.        +0.j,
              0.70710678+0.j],
             dims=(2, 2)),
 'MPS_representation': (
[(array([[1, 0]]),
    array([[0, 1]])),
   (array([[1],
           [0]]),
    array([[0],
           [1]]))],
  [array([0.70710678, 0.70710678])])}

target_qreg = []
target_qreg.append((np.array([[1, 0]], dtype=complex), np.array([[0, 1]], dtype=complex)))
target_qreg.append((np.array([[1], [0]], dtype=complex), np.array([[0], [1]], dtype=complex)))
target_qreg.append((np.array([[1]], dtype=complex), np.array([[0]], dtype=complex)))


#######################################
##          No entanglement          ##
#######################################

circ_ne = QuantumCircuit(2, 2)
circ_ne.h(0)
circ_ne.h(1)

# Define a snapshot that shows the current state vector
circ_ne.save_statevector(label='state_vector_no_entaglement')
circ_ne.save_matrix_product_state(label='MPS_representation_no_entaglement')
circ_ne.measure([0,1], [0,1])

# Execute and get saved data
tcirc_ne = transpile(circ_ne, simulator)
result_ne = simulator.run(tcirc_ne).result()
data_ne = result_ne.data(0)


rho = qi.DensityMatrix.from_instruction(circ_ne)

u, s, vh = np.linalg.svd(np.real(np.matrix(rho)), full_matrices=False)

s = np.round(s,2)
reduced_rho = qi.partial_trace(rho, [1])

unitary_pmf = rho.probabilities([0,1]).tolist()
unitary_pmf = [np.round(prob,2) for prob in unitary_pmf]

##################################
##          GHZ Example         ##
##################################

# Construct quantum circuit
num_qubits = 3
circ_ghz = QuantumCircuit(num_qubits, num_qubits)
down_state = np.array([1,0])

for qubit_index in range(0, num_qubits):
    circ_ghz.initialize(down_state, qubit_index)

circ_ghz.h(0)
circ_ghz.cx(0, 1)
circ_ghz.cx(1, 2)

# Define a snapshot that shows the current state vector
circ_ghz.save_statevector(label='state_vector_ghz')
circ_ghz.save_matrix_product_state(label='MPS_ghz')

circ_ghz.measure([0,1,2],[0,1,2])

# Run and get counts, using the matrix_product_state method
transpiled_circ_ghz = transpile(circ_ghz, simulator)
transpiled_circ_ghz_result = simulator.run(transpiled_circ_ghz).result()
ghz_measurement_counts = transpiled_circ_ghz_result.get_counts(0)
ghz_measurement_counts

transpiled_circ_ghz_result = simulator.run(transpiled_circ_ghz).result()
ghz_mps = transpiled_circ_ghz_result.data()['MPS_ghz'][0]

array_to_latex(transpiled_circ_ghz_result.data()['MPS_ghz'][0])

array_to_latex(np.array([1,0]))
([(array([[1, 0]]), array([[0, 1]])),

  (array([[1.41421356, 0],
          [0, 0]]),

   array([[0, 0],
          [0, 1.41421356]])),

  (array([[1],
          [0]]),

   array([[0],
          [1]]))],

 [array([0.70710678, 0.70710678]), array([0.70710678, 0.70710678])])
