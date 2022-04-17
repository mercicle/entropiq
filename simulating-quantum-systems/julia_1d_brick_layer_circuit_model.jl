
using ITensors
using ITensors: dim as itensor_dim
using PastaQ
import PastaQ: gate

using Random
using Printf
using LinearAlgebra
using StatsBase: mean, sem, sample as random_sample
using Distributions
using DataFrames
using XLSX
using UUIDs
using Dates
using TimerOutputs
using Pickle

using HDF5
using LibPQ
using DotEnv

save_dir = string(@__DIR__, "/out_data/")
include("entropy_function.jl")

cnfg = DotEnv.config(path=string(@__DIR__, "/db_creds.env"))

db_connection_string = string(" host = ", cnfg["POSTGRES_DB_URL"],
                              " port = ", cnfg["POSTGRES_DB_PORT"],
                              " user = ", cnfg["POSTGRES_DB_USERNAME"],
                              " password = ",cnfg["POSTGRES_DB_PASSWORD"],
                              " sslmode = 'require'",
                              " dbname = ", cnfg["POSTGRES_DB_NAME"]
                              )
conn = LibPQ.Connection(db_connection_string)

# single qubit gates provided in example
gate(::GateName"Π0") =
  [1 0
   0 0]
gate(::GateName"Π1") =
  [0 0
   0 1]
# new 2 qubit projectors
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

fid = h5open(string(@__DIR__, "/clifford_dict_v2.h5"), "r")
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

rng = MersenneTwister(1234)
experiment_id = repr(uuid4(rng).value)
experiment_run_date = Dates.format(Date(Dates.today()), "mm-dd-yyyy")
experiment_label = "exp_"*experiment_id
Random.seed!(1234)
num_qubit_space = 6:1:9 #6:1:10
n_layers = 10
n_simulations = 20
measurement_rate_space = 0.0:0.1:0.9 #0.10:0.10:0.70
simulation_space = 1:n_simulations
layer_space = 1:n_layers

subsystem_range_divider = 2
use_constant_size = false
constant_size = 3

do_single_qubit_projections = false
qubit_index_space = nothing

# Options: RandomUnitaries RandomCliffords
gate_types_to_apply = "RandomUnitaries"

experiment_metadata_df = DataFrame(experiment_id = experiment_id,
                                   experiment_run_date = experiment_run_date,
                                   experiment_label = experiment_label,
                                   num_qubit_space = join(["$x" for x in num_qubit_space], ","),
                                   n_layers = n_layers,
                                   n_simulations = n_simulations,
                                   measurement_rate_space = join(["$x" for x in measurement_rate_space], ","),
                                   subsystem_range_divider = subsystem_range_divider,
                                   do_single_qubit_projections = do_single_qubit_projections,
                                   gate_types_to_apply = gate_types_to_apply
                                   )

projective_list = [ "00"; "01"; "10"; "11"]
ψ_tracker = nothing
this_layer = nothing
this_circuit = nothing
simulation_df = DataFrame()
von_neumann_entropy_df = DataFrame()
von_neumann_entropies = []

for num_qubits in num_qubit_space

  # num_qubits=6
  @printf("# Qubits = %.3i \n", num_qubits)

  if do_single_qubit_projections
    qubit_index_space = 1:num_qubits
  else
    qubit_index_space = 1:(num_qubits-1)
  end

  @printf("Preparing circuit_simulations for # Qubits = %.3i \n", num_qubits)
  circuit_simulations = []
  for this_sim in simulation_space
    layers = []
    #@printf("Preparing # Sim = %.3i \n", this_sim)
    for this_layer_index in layer_space

      this_layer = nothing
      if isodd(this_layer_index)

        if gate_types_to_apply == "RandomUnitaries"

          #randomlayer:
          #https://github.com/GTorlai/PastaQ.jl/blob/000b2524b92b5cb09295cfd09dcbb1914ddc0991/src/circuits/circuits.jl
          this_layer  = randomlayer("RandomUnitary",[(j,j+1) for j in 1:2:(num_qubits-1)])

        elseif gate_types_to_apply == "RandomCliffords"

          clifford_indices_list = [random_sample(1:clifford_samples) for j in 1:1:(num_qubits-1)]
          clifford_list = [clifford_dict["$j"] for j in clifford_indices_list]
          this_layer  = [(clifford_list[j], j,j+1) for j in 1:2:(num_qubits-1)]

        end

      else

        if gate_types_to_apply == "RandomUnitaries"

          this_layer = randomlayer("RandomUnitary",[(j,j+1) for j in 2:2:(num_qubits-1)])

        elseif gate_types_to_apply == "RandomCliffords"

          clifford_indices_list = [random_sample(1:clifford_samples) for j in 1:1:(num_qubits-1)]
          clifford_list = [clifford_dict["$j"] for j in clifford_indices_list]
          this_layer  = [(clifford_list[j], j,j+1) for j in 2:2:(num_qubits-1)]

        end

      end

      push!(layers, this_layer)
    end
    push!(circuit_simulations, layers)
  end

  to = TimerOutput()
  # loop over projective measurement probability (per site)
  for measurement_rate in measurement_rate_space

      # measurement_rate = 0.10
      this_circuit_index = 1
      von_neumann_entropies = []

      for this_circuit in circuit_simulations

         # this_circuit = circuit_simulations[8]
         N = nqubits(this_circuit)
         #@printf("# Qubits = %.3i , # Qubits = %.3i  \n", num_qubits, N)
         # initialize state ψ = |000…⟩
         ψ = productstate(num_qubits)

         this_layer_index = 1
         for this_layer in this_circuit

           @printf("# Qubits: %.3i , Measurement Rate: %.2f, Circuit Sim Index: %.3i, Layer Index: %.3i \n", num_qubits, measurement_rate, this_circuit_index, this_layer_index)

           # this_layer = this_circuit[41]
           ψ = runcircuit(ψ, this_layer; cutoff = 1e-8)

           # perform measurements
           for qubit_index in qubit_index_space

             # only apply measurements to off indices for odd layers and even indices for even layers
             if ((isodd(this_layer_index) && isodd(qubit_index)) || (iseven(this_layer_index) && iseven(qubit_index)))

               # qubit_index = 1
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
                     orthogonalize!(ψ, qubit_index)

                     #ϕ = ψ[ qubit_index:next_qubit_index ]
                     ϕ = ψ[qubit_index] * ψ[next_qubit_index]

                     ρ = prime(ϕ, tags = "Site") * dag(ϕ)

                     unitary_pmf = real.(diag(reshape(array(ρ), (4,4))))
                     σ = wsample(projective_list, unitary_pmf, 1)[1]

                 end

                 projection_string = "Π"*"$(σ)"
                 ψ = runcircuit(ψ, (projection_string, (qubit_index, next_qubit_index)))
                 normalize!(ψ)
                 # ψ[:] = ψ

               end # if measurement_rate > rand()

             end

           end # for qubit_index in qubit_index_space

           this_layer_index += 1

         end # for this_layer in this_circuit

         ψ_tracker = copy(ψ)
         this_von_neumann_entropy_dict = Dict()
         try
            this_von_neumann_entropy_dict = entanglemententropy(ψ, subsystem_range_divider, use_constant_size, constant_size)
            #@printf("Completed Entropy for Circuit: %.3i \n", this_circuit_index)
         catch e
            #println("!!SVD failed, the matrix you were trying to SVD contains NaNs.")
            @printf("!!SVD failed for Circuit: %.3i \n", this_circuit_index)
         end

         this_von_neumann_entropy = this_von_neumann_entropy_dict["S"]
         this_von_neumann_entropy_df = this_von_neumann_entropy_dict["entropy_df"]
         this_von_neumann_entropy_df = insertcols!(this_von_neumann_entropy_df, :measurement_rate => measurement_rate)
         this_von_neumann_entropy_df = insertcols!(this_von_neumann_entropy_df, :simulation_number => this_circuit_index)

         von_neumann_entropy_df = [von_neumann_entropy_df; this_von_neumann_entropy_df]

         this_circuit_index += 1

         push!(von_neumann_entropies, this_von_neumann_entropy)

      end # for this_circuit in circuit_simulations

      mean_entropy = mean(von_neumann_entropies)
      se_mean_entropy = sem(von_neumann_entropies)
      @printf("# Qubits = %.3i Measurement Rate = %.2f  S(ρ) = %.5f ± %.1E \n", num_qubits, measurement_rate, mean_entropy, se_mean_entropy)

      #append(pd.DataFrame.from_dict({'simulation': [this_simulation], 'num_qubits': [num_qubits], 'measurement_rate':[measurement_rate],
      #'layer': [this_layer],'keep_layer': [keep_layer], 'renyi_entropy_2nd': [renyi_entropy_2nd] }))
      this_simulation_df = DataFrame(num_qubits = num_qubits, measurement_rate = measurement_rate, mean_entropy = mean_entropy, se_mean_entropy=se_mean_entropy)
      simulation_df = [simulation_df; this_simulation_df]

  end # for measurement_rate in measurement_rate_space

  @timeit to "num_qubits: "*"$num_qubits" 1+1
end # for num_qubits in num_qubit_space

simulation_df = insertcols!(simulation_df, :experiment_id => experiment_id)

LibPQ.load!(
    (experiment_id = experiment_metadata_df.experiment_id,
    experiment_run_date = experiment_metadata_df.experiment_run_date,
    experiment_label = experiment_metadata_df.experiment_label,
    num_qubit_space = experiment_metadata_df.num_qubit_space,
    n_layers = experiment_metadata_df.n_layers,
    n_simulations = experiment_metadata_df.n_simulations,
    measurement_rate_space = experiment_metadata_df.measurement_rate_space,
    subsystem_range_divider = experiment_metadata_df.subsystem_range_divider,
    do_single_qubit_projections = experiment_metadata_df.do_single_qubit_projections,
    gate_types_to_apply = experiment_metadata_df.gate_types_to_apply
    ),
    conn,
    "INSERT INTO quantumlab_experiments._experiments_metadata (experiment_id, experiment_run_date, experiment_label, num_qubit_space, n_layers, n_simulations, measurement_rate_space, subsystem_range_divider, do_single_qubit_projections, gate_types_to_apply) VALUES(\$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9, \$10);"
)
execute(conn, "COMMIT;")


LibPQ.load!(
    (num_qubits = simulation_df.num_qubits,
     measurement_rate = simulation_df.measurement_rate,
     mean_entropy = simulation_df.mean_entropy,
     se_mean_entropy=simulation_df.se_mean_entropy,
     experiment_id = simulation_df.experiment_id
    ),
    conn,
    "INSERT INTO quantumlab_experiments._simulation_results (num_qubits, measurement_rate, mean_entropy, se_mean_entropy, experiment_id) VALUES(\$1, \$2, \$3, \$4, \$5);"
)
execute(conn, "COMMIT;")

XLSX.writetable(string(save_dir,experiment_label, "_metadata_df.xlsx"), experiment_metadata_df)
XLSX.writetable(string(save_dir,experiment_label, "_simulation_stats_df.xlsx"), simulation_df)
XLSX.writetable(string(save_dir,experiment_label, "_entropy_tracking_df.xlsx"), von_neumann_entropy_df)
