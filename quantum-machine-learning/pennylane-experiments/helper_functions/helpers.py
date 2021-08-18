#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pennylane as qml
import scipy


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


