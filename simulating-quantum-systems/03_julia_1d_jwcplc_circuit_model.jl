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
include("helpers/quantum_sim_functions.jl")
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
experiment_id, sim_status, experiment_name, experiment_description, experiment_run_date, num_qubit_space, n_layers, n_simulations, layer_space, simulation_space, p_space, q_space, subsystem_range_divider, experimental_design_type = [nothing for _ = 1:15]
experiment_id = "0ed86a8c-bf24-11ec-80b0-328140767e06"

if run_from_script

  rng = MersenneTwister()
  experiment_id = repr(uuid4(rng).value)
  sim_status = "Running"
  experiment_name = "Testing with new run_brick_layer_sim()"
  experiment_description = "Testing with new run_brick_layer_sim()"
  experiment_run_date = Dates.format(Date(Dates.today()), "mm-dd-yyyy")

  num_qubit_space = 6:2:10
  n_layers = 20
  n_simulations = 100

  p_space = 0.10:0.10:0.9
  q_space = 0.10:0.10:0.9

  simulation_space = 1:n_simulations
  layer_space = 1:n_layers

  experimental_design_type = "Standard Bricklayer" # 'Standard Bricklayer' 'Jordan-Wigner CPLC'

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
                                     p_space = join(["$x" for x in p_space], ","),
                                     q_space = join(["$x" for x in q_space], ","),
                                     use_constant_size = use_constant_size,
                                     constant_size = constant_size,
                                     subsystem_range_divider = 1/subsystem_range_divider,
                                     status = sim_status,
                                     runtime_in_seconds = 0,
                                     experimental_design_type = experimental_design_type
                                     )

  execute(conn, "BEGIN;")
  LibPQ.load!(
     (experiment_id = experiment_metadata_df.experiment_id,
      status = experiment_metadata_df.status,
      experiment_name = experiment_metadata_df.experiment_name,
      experiment_description = experiment_metadata_df.experiment_description,
      experiment_run_date = experiment_metadata_df.experiment_run_date,
      num_qubit_space = experiment_metadata_df.num_qubit_space,
      n_layers = experiment_metadata_df.n_layers,
      n_simulations = experiment_metadata_df.n_simulations,

      p_space = experiment_metadata_df.p_space,
      q_space = experiment_metadata_df.q_space,

      use_constant_size = experiment_metadata_df.use_constant_size,
      constant_size = experiment_metadata_df.constant_size,
      subsystem_range_divider = experiment_metadata_df.subsystem_range_divider,

      runtime_in_seconds = experiment_metadata_df.runtime_in_seconds,
      experimental_design_type = experiment_metadata_df.experimental_design_type
     ),
     conn,
     "INSERT INTO quantumlab_experiments.experiments_metadata_jwcplc (experiment_id, status, experiment_name, experiment_description, experiment_run_date, num_qubit_space, n_layers, n_simulations, measurement_rate_space, use_constant_size, constant_size, subsystem_range_divider, operation_type_to_apply, gate_types_to_apply, runtime_in_seconds,experimental_design_type) VALUES(\$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9, \$10, \$11, \$12, \$13, \$14, \$15, \$16);"
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

  p_space  = split(data.p_space[1],",")
  p_space = [parse(Float16,x)/10 for x in p_space]

  q_space  = split(data.q_space[1],",")
  q_space = [parse(Float16,x)/10 for x in p_space]

  use_constant_size = data.use_constant_size[1]
  constant_size = data.constant_size[1]

  # in app this is defined as the percentage of system
  subsystem_range_divider  = trunc(Int, (1 / data.subsystem_range_divider[1]))
  experimental_design_type = data.experimental_design_type[1]

end

# start_time = time()
# brick_layer_results = run_jwcplc_sim(num_qubit_space, simulation_space, p_space, q_space, layer_space, qubit_index_space, subsystem_range_divider, use_constant_size, constant_size)
# runtime_in_seconds = time() - start_time
# runtime_in_seconds = round(runtime_in_seconds, digits=0)
#
# simulation_df = brick_layer_results["simulation_df"]
# von_neumann_entropy_df = brick_layer_results["von_neumann_entropy_df"]

projective_list = [ "00"; "01"; "10"; "11"]
qubit_index_space = nothing
ψ_tracker = nothing
this_layer = nothing
this_circuit = nothing
simulation_df = DataFrame()
von_neumann_entropy_df = DataFrame()
von_neumann_entropies = []
run_times = []

for num_qubits in num_qubit_space
  # num_qubits=6
  @printf("# Qubits = %.3i \n", num_qubits)
  qubit_index_space = 1:num_qubits

  for p in p_space
    for q in q_space
      # p = 0.90
      # q = 0.90
      von_neumann_entropies = []
      run_times = []
      for this_sim in simulation_space

         sim_start_time = time()

         # initialize state ψ = |000…⟩
         ψ = productstate(num_qubits)

         for this_layer_index in layer_space

           if isodd(this_layer_index)

             for qubit_index in qubit_index_space

               # p = 0.5
               odd_actions = ["JWCPLC_UOdd","JWCPLC_UOdd1", "JWCPLC_UOdd_Measure"]
               action_pmf = [p, (1-p)*q, (1-p)*(1-q)]
               sampled_action = wsample(odd_actions, action_pmf, 1)[1]

               if sampled_action == "JWCPLC_UOdd" || sampled_action == "JWCPLC_UOdd1"
                 action_string = "Π"*"$(sampled_action)"
               elseif sampled_action == "JWCPLC_UOdd_Measure"

                 ψ = orthogonalize!(ψ, qubit_index)
                 ket = ψ[qubit_index]
                 bra = ITensors.dag(ITensors.prime(ket)) # prime -> row vector then dag -> hermitian conjugation

                 local_X = ITensor(1/2*([1 1
                                         1 1]), Index(2,"i"), Index(2,"j"))
                 born_probability_up = bra*local_X*ket
                 born_probability_down = 1-born_probability_up
                 up_down_probabilities = [born_probability_up, born_probability_down]
                 x_measurement_list = ["X↑","X↓"]
                 action_string = wsample(x_measurement_list, up_down_probabilities, 1)[1]
               end
               ψ = runcircuit(ψ, (action_string, qubit_index))
               normalize!(ψ)

           else

             for qubit_index in qubit_index_space
               if qubit_index != num_qubits
                 even_actions = ["JWCPLC_UEven","JWCPLC_UEven1", "JWCPLC_UEven_Measure"]
                 action_pmf = [p,(1-p)*(1-q),(1-p)*q]
                 sampled_action = wsample(even_actions, action_pmf, 1)[1]

                 if sampled_action == "JWCPLC_UEven" || sampled_action == "JWCPLC_UEven1"
                   action_string = "Π"*"$(sampled_action)"
                 elseif sampled_action == "JWCPLC_UEven_Measure"

                   ψ = orthogonalize!(ψ, qubit_index)
                   ket = ψ[qubit_index] * ψ[next_qubit_index]
                   bra = ITensors.dag(ITensors.prime(ket))

                   local_ZZ_up = ITensor([1 0 0 0
                                          0 0 0 0
                                          0 0 0 0
                                          0 0 0 1]), Index(4,"i"), Index(4,"j"))

                   zz_born_probability_up = bra*local_ZZ_up*ket
                   zz_born_probability_down = 1-zz_born_probability_up
                   zz_up_down_probabilities = [zz_born_probability_up, zz_born_probability_down]
                   zz_measurement_list = ["ZZ↑","ZZ↓"]
                   action_string = wsample(zz_measurement_list, zz_up_down_probabilities, 1)[1]
                 end
                 ψ = runcircuit(ψ, (action_string, (qubit_index, next_qubit_index)))
                 normalize!(ψ)
               end #qubit_index != num_qubits
             end # for qubit_index in qubit_index_space
           end # if odd/even layer
         end # layers

         ψ_tracker = copy(ψ)
         this_von_neumann_entropy_dict = Dict()
         try
            this_von_neumann_entropy_dict = entanglemententropy(ψ, subsystem_range_divider, use_constant_size, constant_size)
         catch e
            @printf("!!SVD failed for Circuit: %.3i \n", this_sim)
         end

         this_von_neumann_entropy = this_von_neumann_entropy_dict["S"]
         this_von_neumann_entropy_df = this_von_neumann_entropy_dict["entropy_df"]
         this_von_neumann_entropy_df = insertcols!(this_von_neumann_entropy_df, :measurement_rate => measurement_rate)
         this_von_neumann_entropy_df = insertcols!(this_von_neumann_entropy_df, :simulation_number => this_sim)

         von_neumann_entropy_df = [von_neumann_entropy_df; this_von_neumann_entropy_df]

         this_runtime = time() - sim_start_time
         this_runtime = round(this_runtime, digits=3)

         push!(von_neumann_entropies, this_von_neumann_entropy)
         push!(run_times, this_runtime)

      end # simulation

      mean_runtime = mean(run_times)
      mean_entropy = mean(von_neumann_entropies)
      se_mean_entropy = sem(von_neumann_entropies)
      @printf("# Qubits = %.3i Measurement Rate = %.2f  S(ρ) = %.5f ± %.1E \n", num_qubits, measurement_rate, mean_entropy, se_mean_entropy)

      this_simulation_df = DataFrame(num_qubits = num_qubits, measurement_rate = measurement_rate, mean_entropy = mean_entropy, se_mean_entropy=se_mean_entropy, mean_runtime = mean_runtime)
      simulation_df = [simulation_df; this_simulation_df]

    end # q
  end # p
end # num_qubits

sim_status = "Completed"
result = execute(conn,
                 string("update quantumlab_experiments.experiments_metadata set runtime_in_seconds = ", runtime_in_seconds, ", status = '", sim_status ,"' where experiment_id = '", experiment_id,"'");
                 throw_error=false
                 )

simulation_df = insertcols!(simulation_df, :experiment_id => experiment_id)

#TableView.showtable(simulation_df)
execute(conn, "BEGIN;")
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

#https://invenia.github.io/LibPQ.jl/dev/#COPY-1
execute(conn, "BEGIN;")
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
