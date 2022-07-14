############################################################################
## Completely packed loop model with crossings (CPLC) Experimental Design ##
############################################################################

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

using IterTools

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
experiment_id, sim_status, experiment_name, experiment_description, experiment_run_date, num_qubit_space, n_simulations, simulation_space, p_space, q_space, subsystem_range_divider, experimental_design_type = [nothing for _ = 1:13]
experiment_id = "0ed86a8c-bf24-11ec-80b0-328140767e06"

experiments_metadata_table_name = "experiments_metadata_jwcplc"
sim_results_table_name = "simulation_results_jwcplc"
entropy_tracking_table_name  = "entropy_tracking_jwcplc"

if run_from_script

  rng = MersenneTwister()
  experiment_id = repr(uuid4(rng).value)
  sim_status = "Running"
  experiment_name = "CPLC - fixed p scaling"
  experiment_description = "CPLC - fixed p scaling"
  experiment_run_date = Dates.format(Date(Dates.today()), "mm-dd-yyyy")

  # only even system sizes and
  # depth = system size
  # percolation universality class p=0 q~1/2

  num_qubit_space = 30:10:100

  #n_layers = 20
  n_simulations = 30

  p_space = 0.0:0.05:0.0 #0.10:0.10:0.9
  q_space = 0.1:0.05:0.90

  simulation_space = 1:n_simulations

  experimental_design_type = "Jordan-Wigner CPLC" # 'Standard Bricklayer' 'Jordan-Wigner CPLC'

  subsystem_range_divider = 2
  use_constant_size = false
  constant_size = 3

  experiment_metadata_df = DataFrame(experiment_id = experiment_id,
                                     experiment_name = experiment_name,
                                     experiment_description = experiment_description,
                                     experiment_run_date = experiment_run_date,
                                     num_qubit_space = join(["$x" for x in num_qubit_space], ","),
                                     #n_layers = n_layers,
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
      #n_layers = experiment_metadata_df.n_layers,
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
     "INSERT INTO quantumlab_experiments."*"$(experiments_metadata_table_name)"*" (experiment_id, status, experiment_name, experiment_description, experiment_run_date, num_qubit_space, n_simulations, p_space, q_space,  use_constant_size, constant_size, subsystem_range_divider, runtime_in_seconds, experimental_design_type) VALUES(\$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9, \$10, \$11, \$12, \$13, \$14);"
  )
  execute(conn, "COMMIT;")

else

  result = execute(conn,
                   string("select * FROM quantumlab_experiments."*"$(experiments_metadata_table_name)"*" where experiment_id = '", experiment_id,"'");
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

  n_simulations  = data.n_simulations[1]
  simulation_space = 1:n_simulations

  p_space  = split(data.p_space[1],",")
  p_space = [parse(Float16,x)/10 for x in p_space]

  q_space  = split(data.q_space[1],",")
  q_space = [parse(Float16,x)/10 for x in q_space]

  use_constant_size = data.use_constant_size[1]
  constant_size = data.constant_size[1]

  # in app this is defined as the percentage of system
  subsystem_range_divider  = trunc(Int, (1 / data.subsystem_range_divider[1]))
  experimental_design_type = data.experimental_design_type[1]

end

start_time = time()
cplc_results = run_cplc_sim(num_qubit_space, simulation_space, measurement_rate_space, p_space, q_space, subsystem_range_divider, use_constant_size, constant_size)
runtime_in_seconds = time() - start_time
runtime_in_seconds = round(runtime_in_seconds, digits=0)

simulation_df = cplc_results["simulation_df"]
von_neumann_entropy_df = cplc_results["von_neumann_entropy_df"]

sim_status = "Completed"
result = execute(conn,
                 string("update quantumlab_experiments."*"$(experiments_metadata_table_name)"*" set runtime_in_seconds = ", runtime_in_seconds, ", status = '", sim_status ,"' where experiment_id = '", experiment_id,"'");
                 throw_error=false
                 )

simulation_df = insertcols!(simulation_df, :experiment_id => experiment_id)

#TableView.showtable(simulation_df)
execute(conn, "BEGIN;")
LibPQ.load!(
    (num_qubits = simulation_df.num_qubits,
     p = simulation_df.p,
     q = simulation_df.q,
     mean_entropy = simulation_df.mean_entropy,
     se_mean_entropy=simulation_df.se_mean_entropy,
     experiment_id = simulation_df.experiment_id,
     mean_runtime = simulation_df.mean_runtime
    ),
    conn,
    "INSERT INTO quantumlab_experiments."*"$(sim_results_table_name)"*" (num_qubits, p, q, mean_entropy, se_mean_entropy, experiment_id, mean_runtime) VALUES(\$1, \$2, \$3, \$4, \$5, \$6, \$7);"
)
execute(conn, "COMMIT;")

von_neumann_entropy_df = insertcols!(von_neumann_entropy_df, :experiment_id => experiment_id)

replace!(von_neumann_entropy_df.num_qubits, NaN => -1)
replace!(von_neumann_entropy_df.bond_index, NaN => -1)
replace!(von_neumann_entropy_df.ij, NaN => -1)
replace!(von_neumann_entropy_df.eigenvalue, NaN => -1)
replace!(von_neumann_entropy_df.entropy_contribution, NaN => -1)

von_neumann_entropy_df[!,:bond_index] = convert.(Int64,von_neumann_entropy_df[:,:bond_index])
von_neumann_entropy_df[!,:num_qubits] = convert.(Int64,von_neumann_entropy_df[:,:num_qubits])
von_neumann_entropy_df[!,:ij] = convert.(Int64,von_neumann_entropy_df[:,:ij])

row_strings = imap(eachrow(von_neumann_entropy_df)) do row
  "$(row[:experiment_id]),$(row[:p]),$(row[:q]),$(row[:simulation_number]),$(row[:num_qubits]),$(row[:bond_index]),$(row[:ij]),$(row[:eigenvalue]),$(row[:entropy_contribution])\n"
end

start_time = time()
copyin = LibPQ.CopyIn("COPY quantumlab_experiments."*"$(entropy_tracking_table_name)"*" FROM STDIN (FORMAT CSV);", row_strings)
execute(conn, copyin)
runtime_in_seconds = time() - start_time
runtime_in_seconds = round(runtime_in_seconds, digits=0)

#TableView.showtable(von_neumann_entropy_df)
