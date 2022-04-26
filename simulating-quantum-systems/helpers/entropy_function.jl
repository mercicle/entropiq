
using ITensors
using ITensors: dim as itensor_dim
using PastaQ
import PastaQ: gate
using DataFrames

# Von Neumann entropy at center bond von_neumann_entropy
# https://en.wikipedia.org/wiki/Von_Neumann_entropy
# http://www.scholarpedia.org/article/Quantum_entropies
# "von Neumann entropy is a limiting case of the Rényi entropy" lim α→1 Sα(ρ) = S(ρ)
# Given a family of entropies {Sα(ρ)}α, where α is some index, the entropies are monotonic in α∈ℝ

function entanglemententropy(ψ₀::MPS, subsystem_divider::Int, use_constant_size::Bool, constant_size::Int)

  # https://qiskit.org/documentation/_modules/qiskit/quantum_info/states/utils.html#partial_trace
  # https://qiskit.org/textbook/ch-quantum-hardware/density-matrix.html#reduced

  # http://itensor.org/docs.cgi?vers=cppv3&page=formulas/mps_two_rdm

  # "Among physicists, this is often called "tracing out" or "tracing over" W to leave only an operator on V in the context where W and V are Hilbert spaces associated with quantum systems (see below)."
  # https://en.wikipedia.org/wiki/Partial_trace#:~:text=In%20linear%20algebra%20and%20functional,is%20an%20operator%2Dvalued%20function.

  # http://www.fmt.if.usp.br/~gtlandi/04---reduced-dm-2.pdf

  # https://itensor.github.io/ITensors.jl/stable/examples/MPSandMPO.html
  
  # ψ₀ = ψ
  # subsystem_divider = subsystem_range_divider
  # use_constant_size
  # constant_size
  ψ_local = normalize!(copy(ψ₀))
  N = length(ψ_local)

  if use_constant_size
    bond = constant_size
  else
    bond = trunc(Int, N/subsystem_divider)
  end

  singular_values_to_keep = 2^bond

  orthogonalize!(ψ_local, bond)

  #row_inds = (linkind(ψ_local, 1), siteind(ψ_local, bond))
  row_inds = (linkind(ψ_local, bond - 1), siteind(ψ_local, bond))

  # isnan
  # SVD failed, the matrix you were trying to SVD contains NaNs.
  #http://itensor.org/docs.cgi?page=book/itensor_factorizing&vers=cppv3
  #http://itensor.org/docs.cgi?vers=cppv3&page=tutorials/SVD
  u, s, v = svd(ψ_local[bond], row_inds, mindim = singular_values_to_keep)

  S = 0.0
  sigma_rank = itensor_dim(s, 1)
  entropy_df = DataFrame()
  for n in 1:sigma_rank
    λ = s[n, n]^2
    entropy_contribution = - λ * log(λ + 1e-20)
    S = S + entropy_contribution
    this_df = DataFrame(num_qubits = N, bond_index = bond, ij= n, eigenvalue = λ, entropy_contribution = entropy_contribution)
    entropy_df = [entropy_df; this_df]
  end
  return Dict("S" => S, "entropy_df" => entropy_df)
end
