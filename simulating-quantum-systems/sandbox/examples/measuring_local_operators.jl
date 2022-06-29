
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


using ITensors
using ITensors: dim as itensor_dim

num_qubits_eg = 3
ψ_eg = productstate(num_qubits)

gate(::GateName"Sxup") =
1/2*([1 1
       1 1])
born_probability_x_up = ITensors.expect(ψ_eg, "Sxup", sites=1)

gate(::GateName"ZOp") =
[1 0
 0 -1]

Czz = ITensors.correlation_matrix(ψ_eg,"ZOp", "ZOp", site_range = 1:2)
Czz = diag(normalize(Czz))

N = 30
m = 4

s = siteinds("S=1/2",N)
psi = randomMPS(s; linkdims=m)
Czz = correlation_matrix(psi,"Sz","Sz")
Czz = correlation_matrix(psi,[1/2 0; 0 -1/2],[1/2 0; 0 -1/2])
