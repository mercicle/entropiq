

###################################
## Previous implementation       ##
###################################

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
