
using ITensors
using ITensors: dim as itensor_dim

num_qubits = 3
ψ = productstate(num_qubits)

these_inds = siteinds(ψ)
X_1 = ITensors.op("X",these_inds,1)
@show X_1

ket = ψ[1]
bra = ITensors.dag(ITensors.prime(ket)) # prime -> row vector then dag -> hermitian conjugation
@show ket
@show bra

expected_X = bra*X_1*ket
@show expected_X
