
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute
from qiskit.providers.aer import QasmSimulator

from qiskit.extensions.simulator.snapshot import snapshot

################################################################
## Example from https://qiskit.org/documentation/stable/0.24/tutorials/simulators/7_matrix_product_state_method.html ##
################################################################

# Construct quantum circuit
circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)
circ.measure([0,1], [0,1])

# Select the QasmSimulator from the Aer provider
simulator = QasmSimulator(method='matrix_product_state')

# Execute and get counts, using the matrix_product_state method
result = execute(circ, simulator).result()
counts = result.get_counts(circ)
counts

circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)

# Define a snapshot that shows the current state vector
circ.snapshot('my_sv', snapshot_type='statevector')
circ.measure([0,1], [0,1])

# Execute
job_sim = execute(circ, simulator)
result = job_sim.result()

#print the state vector
result.data()['snapshots']['statevector']['my_sv'][0]

result.get_counts()


num_qubits = 50
circ = QuantumCircuit(num_qubits, num_qubits)

# Create EPR state
circ.h(0)
for i in range (0, num_qubits-1):
    circ.cx(i, i+1)

# Measure
circ.measure(range(num_qubits), range(num_qubits))

job_sim = execute(circ, simulator)
result = job_sim.result()
print("Time taken: {} sec".format(result.time_taken))
result.get_counts()



################################################################
################################################################
################################################################


num_qubits = 50
circ = QuantumCircuit(num_qubits, num_qubits)

# Create EPR state
circ.h(0)
for i in range (0, num_qubits-1):
    circ.cx(i, i+1)

# Measure
circ.measure(range(num_qubits), range(num_qubits))

job_sim = execute(circ, simulator)
result = job_sim.result()
print("Time taken: {} sec".format(result.time_taken))
result.get_counts()
