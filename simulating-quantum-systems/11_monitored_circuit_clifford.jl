
# started with code that was graciously added by ITensor and PastaQ maintainers from:
# https://raw.githubusercontent.com/GTorlai/PastaQ.jl/master/examples/11_monitored_circuit.jl

# NOTE: this version extends to 2-qubit measurements and clifford operators
# which required changes or rewrites to entangling_layer(), projective_measurement() and monitored_circuits().
# clifford_dict.h5 is being created with create_clifford_group.py using qiskit to sample from clifford group and then save to hd5 format.

using PastaQ
import PastaQ: gate

using ITensors
using ITensors: dim as itensor_dim

using Random
using Printf
using LinearAlgebra
using StatsBase: mean, sem, sample as random_sample

using Distributions
using HDF5

# JDM added new 2 qubit projectors
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

# JDM added clifford_dict prep
fid = h5open(string(@__DIR__, "/in_data/clifford_dict.h5"), "r")
clifford_samples = 99999
clifford_dict = Dict()
for c in 1:1:clifford_samples
    dataset_name = "clifford_$c"
    obj = fid[dataset_name]
    read_obj = read(obj)
    this_clifford = [
    read_obj[(1,1)...]+read_obj[(2,1)...]im read_obj[(1,2)...]+read_obj[(2,2)...]im read_obj[(1,3)...]+read_obj[(2,3)...]im read_obj[(1,4)...]+read_obj[(2,4)...]im
    read_obj[(1,5)...]+read_obj[(2,5)...]im read_obj[(1,6)...]+read_obj[(2,6)...]im read_obj[(1,7)...]+read_obj[(2,7)...]im read_obj[(1,8)...]+read_obj[(2,8)...]im
    read_obj[(1,9)...]+read_obj[(2,9)...]im read_obj[(1,10)...]+read_obj[(2,10)...]im read_obj[(1,11)...]+read_obj[(2,11)...]im read_obj[(1,12)...]+read_obj[(2,12)...]im
    read_obj[(1,13)...]+read_obj[(2,13)...]im read_obj[(1,14)...]+read_obj[(2,14)...]im read_obj[(1,15)...]+read_obj[(2,15)...]im read_obj[(1,16)...]+read_obj[(2,16)...]im
    ]
    clifford_dict["$c"] = this_clifford
end

# Von Neumann entropy at center bond
function entanglemententropy(ψ₀::MPS)
  ψ = normalize!(copy(ψ₀))
  N = length(ψ)
  bond = N ÷ 2
  orthogonalize!(ψ, bond)

  row_inds = (linkind(ψ, bond - 1), siteind(ψ, bond))
  u, s, v = svd(ψ[bond], row_inds)

  S = 0.0
  # JDM added itensor_dim vs dim()
  for n in 1:itensor_dim(s, 1)
    λ = s[n, n]^2
    S -= λ * log(λ + 1e-20)
  end
  return S
end

# build a brick-layer of random unitaries covering all
# nearest-neighbors bonds
function entangling_layer(N::Int, this_layer_index::Int)

  # JDM changed this whole functon
  # added this_layer_index for bricklayer logic
  # added sampling from clifford_dict at odd/even sites based on this_layer_index

  clifford_indices_list = [random_sample(1:clifford_samples) for j in 1:1:(N-1)]
  if isodd(this_layer_index)
      clifford_list = [clifford_dict["$j"] for j in clifford_indices_list]
      this_layer  = [(clifford_list[j], j,j+1) for j in 1:2:(N-1)]
  else
      clifford_list = [clifford_dict["$j"] for j in clifford_indices_list]
      this_layer  = [(clifford_list[j], j,j+1) for j in 2:2:(N-1)]
  end
  return this_layer
end

# modified for 2-qubit projective measurement
function projective_measurement!(ψ₀::MPS, site::Int)

  ψ = orthogonalize!(ψ₀, site)

  # JDM added * ψ[site] for 2-qubit
  ϕ = ψ[site] * ψ[site+1]

  ρ = prime(ϕ, tags="Site") * dag(ϕ)

  # JDM modified to real.(diag(reshape(array(ρ), (4,4)))) for pmf of 2-qubits
  unitary_pmf = real.(diag(reshape(array(ρ), (4,4))))

  # JDM added projective_list and wsample for σ based on state probs
  projective_list = [ "00"; "01"; "10"; "11"]
  σ = wsample(projective_list, unitary_pmf, 1)[1]

  # JDM added projection_string and (projection_string, (site, site+1)) for 2-qubit proj measurement
  projection_string = "Π"*"$(σ)"
  ψ = runcircuit(ψ, (projection_string, (site, site+1)))
  normalize!(ψ)
  ψ₀[:] = ψ

  return ψ₀
end

# compute average Von Neumann entropy for an ensemble of random circuits
# at a given local measurement probability rate
function monitored_circuits(circuits::Vector{<:Vector}, p::Float64)
  svn = []
  N = nqubits(circuits[1])
  for circuit in circuits
    # initialize state ψ = |000…⟩
    ψ = productstate(N)
    # sweep over layers
    # JDM added enumerate for layer_index
    for (layer_index, layer) in enumerate(circuit)
      # apply entangling unitary
      ψ = runcircuit(ψ, layer; cutoff = 1e-8)
      # perform measurements
      for j in 1:(N-1)
        # JDM added this logic to adhere to bricklayer logic
        if ((isodd(layer_index) && isodd(j)) || (iseven(layer_index) && iseven(j)))
          if p > rand()
            projective_measurement!(ψ, j)
          end
        end
      end
    end
    push!(svn, entanglemententropy(ψ))
  end
  return svn
end

let
  Random.seed!(1234)
  N = 8        # number of qubits
  depth = 100   # circuit's depth
  ntrials = 50  # number of random trials

  # generate random circuits
  # JDM added passing in (N, i)
  circuits = [[entangling_layer(N, i) for i in 1:depth] for _ in 1:ntrials]

  # loop over projective measurement probability (per site)
  for p in 0.0:0.02:0.2
    t = @elapsed svn = monitored_circuits(circuits, p)
    @printf("p = %.2f  S(ρ) = %.5f ± %.1E\t(elapsed = %.2fs)\n", p, mean(svn), sem(svn), t)
  end
end
