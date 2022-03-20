
using ITensors
using ITensors: dim as itensor_dim

# bingo
# https://github.com/ITensor/ITensors.jl/blob/62d5ab1919bbe74168bcefc9867644b950f24e31/docs/src/examples/MPSandMPO.md

# Tensor Networks and Applications
# https://itensor.org/miles/BrazilLectures/TNAndApplications01.pdf
num_qubits = 4
state_size = 2^num_qubits
ψ = productstate(num_qubits)


subsystem_upper_bound = Int(state_size/2)
orthogonalize!(ψ, subsystem_upper_bound)
U,S,V = svd(ψ[subsystem_upper_bound], (linkind(ψ, subsystem_upper_bound-1), siteind(ψ,subsystem_upper_bound)))
SvN = 0.0
for n=1:itensor_dim(S, 1)
  p = S[n,n]^2
  SvN -= p * log(p + 1e-20)
end

T = U*S*V

subsystem_state_n = 2^subsystem_upper_bound

array(T)


# m by n complex matrix M
# U is m by m
hs_size = 2^subsystem_upper_bound
reshape(array(U), (hs_size,hs_size))

reshape(array(U)[:], (hs_size,hs_size))

A = Array(U,Index(hs_size),Index(hs_size))

@show inds(U)
@show A

U * S * V
