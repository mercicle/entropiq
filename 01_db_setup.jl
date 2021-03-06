
using LibPQ
using DotEnv

cnfg = DotEnv.config(path=string(@__DIR__, "/db_creds.env"))

db_connection_string = string(" host = ", cnfg["POSTGRES_DB_URL"],
                              " port = ", cnfg["POSTGRES_DB_PORT"],
                              " user = ", cnfg["POSTGRES_DB_USERNAME"],
                              " password = ",cnfg["POSTGRES_DB_PASSWORD"],
                              " sslmode = 'require'",
                              " dbname = ", cnfg["POSTGRES_DB_NAME"]
                              )
conn = LibPQ.Connection(db_connection_string)

result = LibPQ.execute(conn,"create schema quantumlab_experiments";throw_error=false)

########################
## Brick Layer Design ##
########################
create_experimental_metadata_string = "DROP TABLE IF EXISTS quantumlab_experiments.experiments_metadata; CREATE TABLE IF NOT EXISTS quantumlab_experiments.experiments_metadata (experiment_id TEXT, experiment_run_date TEXT, status TEXT, experiment_name TEXT, experiment_description TEXT, num_qubit_space TEXT, n_layers INT, n_simulations INT, measurement_rate_space TEXT, use_constant_size BOOL, constant_size INT, subsystem_range_divider FLOAT, operation_type_to_apply TEXT, gate_types_to_apply TEXT runtime_in_seconds FLOAT, experimental_design_type TEXT)";
result = execute(conn, create_experimental_metadata_string);

create_simulation_results_string = "DROP TABLE IF EXISTS quantumlab_experiments.simulation_results; CREATE TABLE IF NOT EXISTS quantumlab_experiments.simulation_results (num_qubits INT, measurement_rate FLOAT, mean_entropy FLOAT, se_mean_entropy FLOAT, experiment_id TEXT, mean_runtime FLOAT)";
result = execute(conn, create_simulation_results_string);

create_entropy_tracking_string = "DROP TABLE IF EXISTS quantumlab_experiments.entropy_tracking; CREATE TABLE IF NOT EXISTS quantumlab_experiments.entropy_tracking (experiment_id TEXT, measurement_rate FLOAT, simulation_number INT, num_qubits INT, bond_index INT, ij INT, eigenvalue FLOAT, entropy_contribution FLOAT)";
result = execute(conn, create_entropy_tracking_string);


############################################################################
## Completely packed loop model with crossings (CPLC) Experimental Design ##
############################################################################
create_experimental_metadata_string = "CREATE TABLE IF NOT EXISTS quantumlab_experiments.experiments_metadata_jwcplc (experiment_id TEXT, experiment_run_date TEXT, status TEXT, experiment_name TEXT, experiment_description TEXT, num_qubit_space TEXT, n_layers INT, n_simulations INT, p_space TEXT, q_space TEXT, use_constant_size BOOL, constant_size INT, subsystem_range_divider FLOAT, runtime_in_seconds FLOAT, experimental_design_type TEXT)";
result = execute(conn, create_experimental_metadata_string);

create_simulation_results_string = "CREATE TABLE IF NOT EXISTS quantumlab_experiments.simulation_results_jwcplc (num_qubits INT, p FLOAT, q FLOAT, mean_entropy FLOAT, se_mean_entropy FLOAT, experiment_id TEXT, mean_runtime FLOAT)";
result = execute(conn, create_simulation_results_string);

create_entropy_tracking_string = "CREATE TABLE IF NOT EXISTS quantumlab_experiments.entropy_tracking_jwcplc (experiment_id TEXT, p FLOAT, q FLOAT, simulation_number INT, num_qubits INT, bond_index INT, ij INT, eigenvalue FLOAT, entropy_contribution FLOAT)";
result = execute(conn, create_entropy_tracking_string);
