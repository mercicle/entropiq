
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

  ψ_local = normalize!(copy(ψ₀))
  N = length(ψ_local)

  if use_constant_size
    bond = constant_size
  else
    bond = trunc(Int, N/subsystem_divider)
  end

    singular_values_to_keep = 2^bond

  orthogonalize!(ψ_local, bond)

  row_inds = (linkind(ψ_local, bond - 1), siteind(ψ_local, bond))
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
