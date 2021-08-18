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
# :math:`\langle Z \rangle`, with respect to the ``control`` qubit.
# The probability of measuring the :math:`|0\rangle` state
# in this control qubit is related to both the fidelity
# between registers and :math:`\langle Z \rangle`. Thus, with a bit of algebra,
# we find that :math:`\langle Z \rangle` is equal to the fidelity.
#

def swap_test(control, register1, register2):

    qml.Hadamard(wires=control)
    
    for reg1_qubit, reg2_qubit in zip(register1, register2):
        qml.CSWAP(wires=(control, reg1_qubit, reg2_qubit))
        
    qml.Hadamard(wires=control)

