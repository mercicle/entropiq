#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pennylane as qml
import scipy

random_number_generator = np.random.default_rng(seed=42)

def create_hamiltonian_matrix(n_qubits, graph, weights, bias):

    # np.kron: Kronecker product of two arrays.
    # https://numpy.org/doc/stable/reference/generated/numpy.kron.html
    
    full_matrix = np.zeros((2 ** n_qubits, 2 ** n_qubits))

    # Creates the interaction component of the Hamiltonian
    for i, edge in enumerate(graph.edges):
        interaction_term = 1
        for qubit in range(0, n_qubits):
            if qubit in edge:
                interaction_term = np.kron(interaction_term, qml.PauliZ.matrix)
            else:
                interaction_term = np.kron(interaction_term, np.identity(2))
                
        full_matrix += weights[i] * interaction_term

    # Creates the bias components of the matrix
    for i in range(0, n_qubits):
        z_term = x_term = 1
        for j in range(0, n_qubits):
            if j == i:
                z_term = np.kron(z_term, qml.PauliZ.matrix)
                x_term = np.kron(x_term, qml.PauliX.matrix)
            else:
                z_term = np.kron(z_term, np.identity(2))
                x_term = np.kron(x_term, np.identity(2))
        full_matrix += bias[i] * z_term + x_term

    return full_matrix


# Evolving the low-energy state forward in time is fairly straightforward: all we
# have to do is multiply the initial state by a time-evolution unitary. This operation
# can be defined as a custom gate in PennyLane:

def state_evolve(hamiltonian, qubits, time):

    U = scipy.linalg.expm(-1j * hamiltonian * time)
    qml.QubitUnitary(U, wires=qubits)


######################################################################
# construct the QGRNN and learn the target Hamiltonian.
# Each of the exponentiated Hamiltonians in the QGRNN ansatz,
# :math:`\hat{H}^{j}_{\text{Ising}}(\boldsymbol\mu)`, are the
# :math:`ZZ`, :math:`Z`, and :math:`X` terms from the Ising
# Hamiltonian. This gives:

def qgrnn_layer(weights, bias, qubits, graph, trotter_step):

    # Applies a layer of RZZ gates (based on a graph)
    for i, edge in enumerate(graph.edges):
        qml.MultiRZ(2 * weights[i] * trotter_step, wires=(edge[0], edge[1]))

    # Applies a layer of RZ gates
    for i, qubit in enumerate(qubits):
        qml.RZ(2 * bias[i] * trotter_step, wires=qubit)

    # Applies a layer of RX gates
    for qubit in qubits:
        qml.RX(2 * trotter_step, wires=qubit)

# implement the QGRNN circuit for some given time value
def qgrnn(weights, bias, time=None):

    # Prepares the low energy state in the two registers
    qml.QubitStateVector(np.kron(low_energy_state, low_energy_state), wires = reg1 + reg2)

    # Evolves the first qubit register with the time-evolution circuit to prepare a piece of quantum data
    state_evolve(hamiltonian_matrix, reg1, time)

    # Applies the QGRNN layers to the second qubit register
    depth = time / trotter_step  # P = t/Delta
    for _ in range(0, int(depth)):
        qgrnn_layer(weights, bias, reg2, complete_seed_ising_graph, trotter_step)

    # Applies the SWAP test between the registers
    swap_test(index_of_control_qubit, reg1, reg2)

    # Returns the results of the SWAP test
    return qml.expval(qml.PauliZ(index_of_control_qubit))



######################################################################
# As was mentioned in the first section, the QGRNN has two
# registers. In one register, some piece of quantum data
# :math:`|\psi(t)\rangle` is prepared and in the other we have
# :math:`U_{H}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle`. We need a
# way to measure the similarity between these states.
# This can be done by using the fidelity, which is
# simply the modulus squared of the inner product between the states,
# :math:`| \langle \psi(t) | U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle |^2`.
# To calculate this value, we use a `SWAP
# test <https://en.wikipedia.org/wiki/Swap_test>`__ between the registers:

# After performing this procedure, the value returned from a measurement of the circuit is
# :math:`\langle Z \rangle`, with respect to the ``index_of_control_qubit`` qubit.
# The probability of measuring the :math:`|0\rangle` state
# in this index_of_control_qubit qubit is related to both the fidelity
# between registers and :math:`\langle Z \rangle`. Thus, with a bit of algebra,
# we find that :math:`\langle Z \rangle` is equal to the fidelity.
#

def swap_test(control, register1, register2):

    qml.Hadamard(wires=control)
    
    for reg1_qubit, reg2_qubit in zip(register1, register2):
        qml.CSWAP(wires=(control, reg1_qubit, reg2_qubit))
        
    qml.Hadamard(wires=control)


def cost_function(weight_params, bias_params):

    # Randomly samples times at which the QGRNN runs
    times_sampled = random_number_generator.random(size=n_quantum_data) * max_time_perc

    # Cycles through each of the sampled times and calculates the cost
    total_cost = 0
    for dt in times_sampled:
        result = qgrnn_qnode(weight_params, bias_params, time = dt)
        total_cost += -1 * result

    return total_cost / n_quantum_data

