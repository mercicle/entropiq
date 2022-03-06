
# this starter code was graciously added by ITensor and PastaQ maintainers
# https://raw.githubusercontent.com/GTorlai/PastaQ.jl/master/examples/11_monitored_circuit.jl

# Julia help docs
# https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention
# https://docs.julialang.org/en/v1/manual/control-flow/
# https://docs.julialang.org/en/v1/manual/variables-and-scoping/
# https://sodocumentation.net/julia-lang

# ρ = prime(ϕ, tags = "Site")
# https://www.itensor.org/docs.cgi?vers=cppv2&page=tutorials/primes

# DataFrames
# https://github.com/bkamins/Julia-DataFrames-Tutorial/
# https://www.ahsmart.com/pub/data-wrangling-with-data-frames-jl-cheat-sheet/
#DataFrame(A=1:3, B=rand(3), C=randstring.([3,3,3]), fixed=1)

[x; x]
Pkg.add("StatsBase")
Pkg.add("Distributions")
Pkg.add("DataFrames")
Pkg.add("CSV")


using PastaQ
using ITensors
using Random
using Printf
using LinearAlgebra
using StatsBase: mean, sem

using Distributions

#
using DataFrames
using CSV


# define the two measurement projectors
import PastaQ: gate
gate(::GateName"Π0") =
  [1 0
   0 0]
gate(::GateName"Π1") =
  [0 0
   0 1]

gate(::GateName"Π00") =
[0 0 0 0
 0 0 0 0
 0 0 0 0
 0 0 0 1]
gate(::GateName"Π10") =
[0 0 0 0
 0 1 0 0
 0 0 0 0
 0 0 0 0]

gate(::GateName"Π01") =
[0 0 0 0
 0 0 0 0
 0 0 1 0
 0 0 0 0]

gate(::GateName"Π11") =
[1 0 0 0
 0 0 0 0
 0 0 0 0
 0 0 0 0]

 # Von Neumann entropy at center bond von_neumann_entropy
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

let

  Random.seed!(1234)
  num_qubits = 8
  n_layers = 100
  n_simulations = 50
  measurement_rate_space = 0.0:0.02:0.2
  simulation_space = 1:n_simulations
  layer_space = 1:n_layers

  do_single_qubit_projections = true
  qubit_index_space = nothing
  if do_single_qubit_projections
    qubit_index_space = 1:num_qubits
  else
    qubit_index_space = 1:(num_qubits-1)
  end

  projective_list = [ "00"; "01"; "10"; "11"]

  simulation_df = DataFrame()

  circuit_simulations = []
  for this_sim in simulation_space
    layers = []
    for this_layer in layer_space
      @printf("Simulation = %.3i  Layer = %.3i \n", this_sim, this_layer)

      layer_odd  = randomlayer("RandomUnitary",[(j,j+1) for j in 1:2:num_qubits-1])
      layer_even = randomlayer("RandomUnitary",[(j,j+1) for j in 2:2:num_qubits-1])
      this_entangled_layer = [layer_odd..., layer_even...]

      push!(layers, this_entangled_layer)
    end
    push!(circuit_simulations, layers)
  end

  # loop over projective measurement probability (per site)
  for measurement_rate in measurement_rate_space

      for this_circuit in circuit_simulations

         # initialize state ψ = |000…⟩
         ψ = productstate(num_qubits)

         von_neumann_entropies = []
         # sweep over layers
         for this_layer in this_circuit

           ψ = runcircuit(ψ, this_layer; cutoff = 1e-8) # apply entangling unitary

           # perform measurements
           for qubit_index in qubit_index_space

             if measurement_rate > rand()

               #projective_measurement!(ψ, qubit_index)
               if do_single_qubit_projections

                   ψ = orthogonalize!(ψ, qubit_index)
                   ϕ = ψ[qubit_index]

                   ρ = prime(ϕ, tags = "Site") * dag(ϕ) # 1-qubit reduced density matrix
                   prob = real.(diag(array(ρ))) # Outcome probabilities
                   σ = (rand() < prob[1] ? 0 : 1) # random sample

               else
                   next_qubit_index = qubit_index + 1
                   ψ = orthogonalize!(ψ, qubit_index:next_qubit_index)
                   ϕ = ψ[qubit_index:next_qubit_index]

                   ρ = prime(ϕ, tags = "Site") * dag(ϕ)
                   unitary_pmf = real.(diag(array(ρ))) # Outcome probabilities
                   σ = wsample(projective_list, unitary_pmf, 1)[1]
               end

               projection_string = "Π"*"$(σ)"
               ψ = runcircuit(ψ, (projection_string, qubit_index)) # Projection
               normalize!(ψ)

             end # if measurement_rate > rand()

           end # for qubit_index in qubit_index_space
         end # for this_layer in this_circuit
         push!(von_neumann_entropies, entanglemententropy(ψ))

      end # for this_circuit in circuit_simulations

      mean_entropy = mean(von_neumann_entropies)
      se_mean_entropy = sem(von_neumann_entropies)
      @printf("Measurement Rate = %.2f  S(ρ) = %.5f ± %.1E \n", measurement_rate, mean_entropy, se_mean_entropy)

      #append(pd.DataFrame.from_dict({'simulation': [this_simulation], 'num_qubits': [num_qubits], 'measurement_rate':[measurement_rate],
      #'layer': [this_layer],'keep_layer': [keep_layer], 'renyi_entropy_2nd': [renyi_entropy_2nd] }))
      this_simulation_df = DataFrame(measurement_rate = measurement_rate, mean_entropy = mean_entropy, se_mean_entropy=se_mean_entropy)
      simulation_df = [simulation_df; this_simulation_df]

    end # for measurement_rate in measurement_rate_space

end # let scope
