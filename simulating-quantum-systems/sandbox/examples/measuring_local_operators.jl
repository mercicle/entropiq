
using ITensors
using ITensors: dim as itensor_dim

num_qubits_eg = 3
ψ_eg = productstate(num_qubits)

these_inds_eg = siteinds(ψ_eg)
X_1_eg = ITensors.op("X",these_inds_eg,1)
@show X_1_eg

ket_eg = ψ_eg[1]
bra_eg = ITensors.dag(ITensors.prime(ket_eg)) # prime -> row vector then dag -> hermitian conjugation
@show ket_eg
@show bra_eg

expected_X_eg = bra_eg*X_1_eg*ket_eg
@show expected_X_eg
