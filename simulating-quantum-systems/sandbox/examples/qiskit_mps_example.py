
# Example from Qiskit 
# https://qiskit.org/documentation/tutorials/simulators/7_matrix_product_state_method.html


import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator

# Construct quantum circuit
circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)
circ.measure([0,1], [0,1])

# Select the AerSimulator from the Aer provider
simulator = AerSimulator(method='matrix_product_state')

# Run and get counts, using the matrix_product_state method
tcirc = transpile(circ, simulator)
result = simulator.run(tcirc).result()
counts = result.get_counts(0)
counts

# To see the internal state vector of the circuit we can use the save_statevector instruction. 
# To return the full internal MPS structure we can also use the save_matrix_product_state instruction.


circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)

# Define a snapshot that shows the current state vector
circ.save_statevector(label='my_sv')
circ.save_matrix_product_state(label='my_mps')
circ.measure([0,1], [0,1])

# Execute and get saved data
tcirc = transpile(circ, simulator)
result = simulator.run(tcirc).result()
data = result.data(0)

#print the result data
data