
import os, sys, timeit
import numpy as np
#!pip3 install h5py
import h5py
import qiskit.quantum_info as qi

clifford_samples = int(1e5)
clifford_dict = dict()
for c in range(clifford_samples):
    clifford_dict[c] = np.array(qi.random_clifford(num_qubits=2).to_operator())

example_clifford_index = np.random.choice(list(clifford_dict.keys()))
example_clifford = clifford_dict[example_clifford_index]

this_range = range(example_clifford.shape[0])


hf = h5py.File("clifford_dict.h5", "w")
for k, v in clifford_dict.items():
    example_clifford = clifford_dict[k]
    this_list = []
    for i in this_range:
        for j in this_range:
            this_list.append((np.real(example_clifford[i,j]), np.imag(example_clifford[i,j])))

    hf.create_dataset("clifford_" + str(k), data = this_list)
hf.close()
