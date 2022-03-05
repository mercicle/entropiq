
# this starter code was graciously added by ITensor and PastaQ maintainers
# https://raw.githubusercontent.com/GTorlai/PastaQ.jl/master/examples/11_monitored_circuit.jl

using PastaQ
using ITensors
using Random
using Printf
using LinearAlgebra
using StatsBase: mean, sem

# define the two measurement projectors
import PastaQ: gate
gate(::GateName"Π0") =
  [1 0
   0 0]
gate(::GateName"Π1") =
  [0 0
   0 1]

let

  Random.seed!(1234)
  num_qubits = 8
  n_layers = 100
  n_simulations = 50
  measurement_rate_space = 0.0:0.02:0.2
  simulation_space = 1:n_simulations
  layer_space = 1:n_layers

  circuits_results = []
  for this_sim in simulation_space
    this_layer_results = []
    for this_layer in layer_space
      @printf("Simulation = %.3i  Layer = %.3i \n", this_sim, this_layer)
      this_entangled_layer = entangling_layer(num_qubits)
      push!(this_layer_results, this_entangled_layer)
    end
    push!(circuits_results, this_layer_results)
  end

  # loop over projective measurement probability (per site)
  for measurement_rate in measurement_rate_space

      svn = []
      num_monitored_qubits = nqubits(circuits_results[1])
      for this_circuit in circuits_results

         # initialize state ψ = |000…⟩
         ψ = productstate(num_monitored_qubits)

         # sweep over layers
         for this_layer in this_circuit

           ψ = runcircuit(ψ, this_layer; cutoff = 1e-8) # apply entangling unitary

           # perform measurements
           for monitored_qubit_index in 1:num_monitored_qubits
             if measurement_rate > rand()

               #projective_measurement!(ψ, monitored_qubit_index)
               ψ = orthogonalize!(ψ, monitored_qubit_index)
               ϕ = ψ[monitored_qubit_index]

               ρ = prime(ϕ, tags = "Site") * dag(ϕ) # 1-qubit reduced density matrix
               prob = real.(diag(array(ρ))) # Outcome probabilities
               σ = (rand() < prob[1] ? 0 : 1) # Sample
               projection_string = "Π"*"$(σ)"
               ψ = runcircuit(ψ, (projection_string, monitored_qubit_index)) # Projection
               normalize!(ψ)
             end # if measurement_rate > rand()
           end # for monitored_qubit_index in 1:num_monitored_qubits
         end # for this_layer in this_circuit
         push!(svn, entanglemententropy(ψ))

      end # for this_circuit in circuits_results
      @printf("Measurement Rate = %.2f  S(ρ) = %.5f ± %.1E \n", measurement_rate, mean(svn), sem(svn))

    end # for measurement_rate in measurement_rate_space

end # let scope

# Von Neumann entropy at center bond
# https://en.wikipedia.org/wiki/Von_Neumann_entropy
# http://www.scholarpedia.org/article/Quantum_entropies
# "von Neumann entropy is a limiting case of the Rényi entropy" lim α→1 Sα(ρ) = S(ρ)
# Given a family of entropies {Sα(ρ)}α, where α is some index, the entropies are monotonic in α∈ℝ

function entanglemententropy(ψ₀::MPS)

  ψ = normalize!(copy(ψ₀))
  N = length(ψ)
  bond = N ÷ 2
  orthogonalize!(ψ, bond)

  row_inds = (linkind(ψ, bond - 1), siteind(ψ, bond))
  u, s, v = svd(ψ[bond], row_inds)

  S = 0.0
  for n in 1:dim(s, 1)
    λ = s[n, n]^2
    S -= λ * log(λ + 1e-20)
  end
  return S
end

# build a brick-layer of random unitaries covering all
# nearest-neighbors bonds
function entangling_layer(N::Int)
  layer_odd  = randomlayer("RandomUnitary",[(j,j+1) for j in 1:2:N-1])
  layer_even = randomlayer("RandomUnitary",[(j,j+1) for j in 2:2:N-1])
  return [layer_odd..., layer_even...]
end

# perform a projective measurement at a given site
function projective_measurement!(ψ₀::MPS, site::Int)
  ψ = orthogonalize!(ψ₀, site)
  ϕ = ψ[site]
  # 1-qubit reduced density matrix
  ρ = prime(ϕ, tags="Site") * dag(ϕ)
  # Outcome probabilities
  prob = real.(diag(array(ρ)))
  # Sample
  σ = (rand() < prob[1] ? 0 : 1)
  # Projection
  ψ = runcircuit(ψ, ("Π"*"$(σ)", site))
  normalize!(ψ)
  ψ₀[:] = ψ
  return ψ₀
end

# compute average Von Neumann entropy for an ensemble of random circuits
# at a given local measurement probability rate
function monitored_circuits(circuits::Vector{<:Vector}, p::Float64)
  svn = []
  num_monitored_qubits = nqubits(circuits[1])
  for circuit in circuits
    # initialize state ψ = |000…⟩
    ψ = productstate(num_monitored_qubits)
    # sweep over layers
    for layer in circuit
      # apply entangling unitary
      ψ = runcircuit(ψ, layer; cutoff = 1e-8)
      # perform measurements
      for j in 1:num_monitored_qubits
        p > rand() && projective_measurement!(ψ, j)
      end
    end
    push!(svn, entanglemententropy(ψ))
  end
  return svn
end
