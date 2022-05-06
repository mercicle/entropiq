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

using TableView
using Tables

save_dir = string(@__DIR__, "/out_data/")
include("helpers/entropy_function.jl")
include("03_load_gates.jl")

cnfg = DotEnv.config(path=string(@__DIR__, "/db_creds.env"))
db_connection_string = string(" host = ", cnfg["POSTGRES_DB_URL"],
                              " port = ", cnfg["POSTGRES_DB_PORT"],
                              " user = ", cnfg["POSTGRES_DB_USERNAME"],
                              " password = ",cnfg["POSTGRES_DB_PASSWORD"],
                              " sslmode = 'require'",
                              " dbname = ", cnfg["POSTGRES_DB_NAME"]
                              )
conn = LibPQ.Connection(db_connection_string)

run_from_script = true
experiment_id, sim_status, experiment_name, experiment_description, experiment_run_date, num_qubit_space, n_layers, n_simulations, layer_space, simulation_space, measurement_rate_space, subsystem_range_divider, operation_type_to_apply, gate_types_to_apply = [nothing for _ = 1:14]
experiment_id = "0ed86a8c-bf24-11ec-80b0-328140767e06"

if run_from_script

  rng = MersenneTwister(1234)
  experiment_id = repr(uuid4(rng).value)
  sim_status = "Running"
  experiment_name = "Comare with qiskit for paper"
  experiment_description = "Comare with qiskit for paper"
  experiment_run_date = Dates.format(Date(Dates.today()), "mm-dd-yyyy")
  Random.seed!(1234)
  num_qubit_space = 6:1:9 #6:1:10
  n_layers = 20
  n_simulations = 10
  measurement_rate_space = 0.0:0.1:0.5 #0.10:0.10:0.70
  simulation_space = 1:n_simulations
  layer_space = 1:n_layers

  operation_type_to_apply = "Binary" # 'Unary', 'Binary'
  gate_types_to_apply = "Random Unitaries" # Options: Random Unitaries Random Cliffords

  subsystem_range_divider = 2
  use_constant_size = false
  constant_size = 3

  experiment_metadata_df = DataFrame(experiment_id = experiment_id,
                                     experiment_name = experiment_name,
                                     experiment_description = experiment_description,
                                     experiment_run_date = experiment_run_date,
                                     num_qubit_space = join(["$x" for x in num_qubit_space], ","),
                                     n_layers = n_layers,
                                     n_simulations = n_simulations,
                                     measurement_rate_space = join(["$x" for x in measurement_rate_space], ","),
                                     use_constant_size = use_constant_size,
                                     constant_size = constant_size,
                                     subsystem_range_divider = 1/subsystem_range_divider,
                                     operation_type_to_apply = operation_type_to_apply,
                                     gate_types_to_apply = gate_types_to_apply,
                                     status = sim_status,
                                     runtime_in_seconds = 0
                                     )


  LibPQ.load!(
     (experiment_id = experiment_metadata_df.experiment_id,
      status = experiment_metadata_df.status,
      experiment_name = experiment_metadata_df.experiment_name,
      experiment_description = experiment_metadata_df.experiment_description,
      experiment_run_date = experiment_metadata_df.experiment_run_date,
      num_qubit_space = experiment_metadata_df.num_qubit_space,
      n_layers = experiment_metadata_df.n_layers,
      n_simulations = experiment_metadata_df.n_simulations,
      measurement_rate_space = experiment_metadata_df.measurement_rate_space,
      use_constant_size = experiment_metadata_df.use_constant_size,
      constant_size = experiment_metadata_df.constant_size,
      subsystem_range_divider = experiment_metadata_df.subsystem_range_divider,
      operation_type_to_apply = experiment_metadata_df.operation_type_to_apply,
      gate_types_to_apply = experiment_metadata_df.gate_types_to_apply,
      runtime_in_seconds = experiment_metadata_df.runtime_in_seconds
     ),
     conn,
     "INSERT INTO quantumlab_experiments.experiments_metadata (experiment_id, status, experiment_name, experiment_description, experiment_run_date, num_qubit_space, n_layers, n_simulations, measurement_rate_space, use_constant_size, constant_size, subsystem_range_divider, operation_type_to_apply, gate_types_to_apply, runtime_in_seconds) VALUES(\$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9, \$10, \$11, \$12, \$13, \$14, \$15);"
  )
  execute(conn, "COMMIT;")

else

  result = execute(conn,
                   string("select * FROM quantumlab_experiments.experiments_metadata where experiment_id = '", experiment_id,"'");
                   throw_error=false
                   )

  data = columntable(result)
  experiment_name  = data.experiment_name[1]
  experiment_description  = data.experiment_description[1]
  experiment_id  = data.experiment_id[1]
  sim_status  = data.status[1]
  experiment_run_date  = data.experiment_run_date[1]

  num_qubit_space  = split(data.num_qubit_space[1],",")
  num_qubit_space = [parse(Int64,x) for x in num_qubit_space]

  n_layers = data.n_layers[1]
  layer_space = 1:n_layers

  n_simulations  = data.n_simulations[1]
  simulation_space = 1:n_simulations

  measurement_rate_space  = split(data.measurement_rate_space[1],",")
  measurement_rate_space = [parse(Float16,x)/10 for x in measurement_rate_space]

  use_constant_size = data.use_constant_size[1]
  constant_size = data.constant_size[1]

  # in app this is defined as the percentage of system
  subsystem_range_divider  = trunc(Int, (1 / data.subsystem_range_divider[1]))
  operation_type_to_apply  = data.operation_type_to_apply[1]
  gate_types_to_apply  = data.gate_types_to_apply[1]

end

projective_list = [ "00"; "01"; "10"; "11"] #[ "00"; "01"; "10"; "11"]
qubit_index_space = nothing
ψ_tracker = nothing
this_layer = nothing
this_circuit = nothing
simulation_df = DataFrame()
von_neumann_entropy_df = DataFrame()
von_neumann_entropies = []
run_times = []
start_time = time()

for (index_n, num_qubits) in enumerate(num_qubit_space)

  # num_qubits=6
  @printf("# Qubits = %.3i \n", num_qubits)

  if operation_type_to_apply == "Unary"
    qubit_index_space = 1:num_qubits
  elseif operation_type_to_apply == "Binary"
    qubit_index_space = 1:(num_qubits-1)
  end

  @printf("Preparing circuit_simulations for # Qubits = %.3i \n", num_qubits)
  circuit_simulations = []
  for (index_s, this_sim) in enumerate(simulation_space)
    layers = []
    #@printf("Preparing # Sim = %.3i \n", this_sim)
    for this_layer_index in layer_space

      this_layer = nothing
      if isodd(this_layer_index)

        if gate_types_to_apply == "Random Unitaries"

          #randomlayer:
          #https://github.com/GTorlai/PastaQ.jl/blob/000b2524b92b5cb09295cfd09dcbb1914ddc0991/src/circuits/circuits.jl
          this_layer  = randomlayer("RandomUnitary",[(j,j+1) for j in 1:2:(num_qubits-1)])

        elseif gate_types_to_apply == "Random Cliffords"

          clifford_indices_list = [random_sample(1:clifford_samples) for j in 1:1:(num_qubits-1)]
          clifford_list = [clifford_dict["$j"] for j in clifford_indices_list]
          this_layer  = [(clifford_list[j], j,j+1) for j in 1:2:(num_qubits-1)]

        end

      else

        if gate_types_to_apply == "Random Unitaries"

          this_layer = randomlayer("RandomUnitary",[(j,j+1) for j in 2:2:(num_qubits-1)])

        elseif gate_types_to_apply == "Random Cliffords"

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
  for (index_m, measurement_rate) in enumerate(measurement_rate_space)

      # measurement_rate = 0.10
      this_circuit_index = 1
      von_neumann_entropies = []
      run_times = []
      for this_circuit in circuit_simulations

         sim_start_time = time()
         # this_circuit = circuit_simulations[8]
         N = nqubits(this_circuit)
         #@printf("# Qubits = %.3i , # Qubits = %.3i  \n", num_qubits, N)
         # initialize state ψ = |000…⟩
         ψ = productstate(num_qubits)

         this_layer_index = 1
         for this_layer in this_circuit

           #@printf("# Qubits: %.3i , Measurement Rate: %.2f, Circuit Sim Index: %.3i, Layer Index: %.3i \n", num_qubits, measurement_rate, this_circuit_index, this_layer_index)

           # this_layer = this_circuit[41]
           ψ = runcircuit(ψ, this_layer; cutoff = 1e-8)

           # perform measurements
           for qubit_index in qubit_index_space

             # only apply measurements to off indices for odd layers and even indices for even layers
             if ((isodd(this_layer_index) && isodd(qubit_index)) || (iseven(this_layer_index) && iseven(qubit_index)))

               # qubit_index = 1
               if measurement_rate > rand()

                 #projective_measurement!(ψ, qubit_index)
                 if operation_type_to_apply == "Unary"

                     ψ = orthogonalize!(ψ, qubit_index)
                     ϕ = ψ[qubit_index]

                     ρ = prime(ϕ, tags = "Site") * dag(ϕ) # 1-qubit reduced density matrix
                     prob = real.(diag(array(ρ))) # Outcome probabilities
                     σ = (rand() < prob[1] ? 0 : 1) # random sample

                 elseif operation_type_to_apply == "Binary"

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
                 ψ[:] = ψ

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

         this_runtime = time() - sim_start_time
         this_runtime = round(this_runtime, digits=0)

         push!(von_neumann_entropies, this_von_neumann_entropy)
         push!(run_times, this_runtime)

      end # for this_circuit in circuit_simulations

      mean_runtime = mean(run_times)
      mean_entropy = mean(von_neumann_entropies)
      se_mean_entropy = sem(von_neumann_entropies)
      @printf("# Qubits = %.3i Measurement Rate = %.2f  S(ρ) = %.5f ± %.1E \n", num_qubits, measurement_rate, mean_entropy, se_mean_entropy)

      #append(pd.DataFrame.from_dict({'simulation': [this_simulation], 'num_qubits': [num_qubits], 'measurement_rate':[measurement_rate],
      #'layer': [this_layer],'keep_layer': [keep_layer], 'renyi_entropy_2nd': [renyi_entropy_2nd] }))
      this_simulation_df = DataFrame(num_qubits = num_qubits, measurement_rate = measurement_rate, mean_entropy = mean_entropy, se_mean_entropy=se_mean_entropy, mean_runtime = mean_runtime)
      simulation_df = [simulation_df; this_simulation_df]

  end # for measurement_rate in measurement_rate_space

  @timeit to "num_qubits: "*"$num_qubits" 1+1
end # for num_qubits in num_qubit_space

runtime_in_seconds = time() - start_time
runtime_in_seconds = round(runtime_in_seconds, digits=0)
sim_status = "Completed"
result = execute(conn,
                 string("update quantumlab_experiments.experiments_metadata set runtime_in_seconds = ", runtime_in_seconds, ", status = '", sim_status ,"' where experiment_id = '", experiment_id,"'");
                 throw_error=false
                 )

simulation_df = insertcols!(simulation_df, :experiment_id => experiment_id)

#TableView.showtable(simulation_df)

LibPQ.load!(
    (num_qubits = simulation_df.num_qubits,
     measurement_rate = simulation_df.measurement_rate,
     mean_entropy = simulation_df.mean_entropy,
     se_mean_entropy=simulation_df.se_mean_entropy,
     experiment_id = simulation_df.experiment_id,
     mean_runtime = simulation_df.mean_runtime
    ),
    conn,
    "INSERT INTO quantumlab_experiments.simulation_results (num_qubits, measurement_rate, mean_entropy, se_mean_entropy, experiment_id, mean_runtime) VALUES(\$1, \$2, \$3, \$4, \$5, \$6);"
)
execute(conn, "COMMIT;")

von_neumann_entropy_df = insertcols!(von_neumann_entropy_df, :experiment_id => experiment_id)

LibPQ.load!(
    (
    experiment_id = von_neumann_entropy_df.experiment_id,
    measurement_rate = von_neumann_entropy_df.measurement_rate,
    simulation_number = von_neumann_entropy_df.simulation_number,
    num_qubits = von_neumann_entropy_df.num_qubits,
    bond_index = von_neumann_entropy_df.bond_index,
    ij = von_neumann_entropy_df.ij,
    eigenvalue = von_neumann_entropy_df.eigenvalue,
    entropy_contribution = von_neumann_entropy_df.entropy_contribution
    ),
    conn,
    "INSERT INTO quantumlab_experiments.entropy_tracking (experiment_id, measurement_rate, simulation_number, num_qubits, bond_index, ij, eigenvalue, entropy_contribution) VALUES(\$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8);"
)
execute(conn, "COMMIT;")

#TableView.showtable(von_neumann_entropy_df)
