
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

create_experimenta_metadata_string = "DROP TABLE IF EXISTS quantumlab_experiments._experiments_metadata; CREATE TABLE IF NOT EXISTS quantumlab_experiments._experiments_metadata (experiment_id TEXT, experiment_run_date TEXT, experiment_label TEXT, num_qubit_space TEXT, n_layers INT, n_simulations INT, measurement_rate_space TEXT, subsystem_range_divider TEXT, do_single_qubit_projections Bool, gate_types_to_apply TEXT)";
result = execute(conn, create_experimenta_metadata_string);

create_experimenta_metadata_string = "DROP TABLE IF EXISTS quantumlab_experiments._simulation_results; CREATE TABLE IF NOT EXISTS quantumlab_experiments._simulation_results (num_qubits INT, measurement_rate FLOAT, mean_entropy FLOAT, se_mean_entropy FLOAT, experiment_id TEXT)";
result = execute(conn, create_experimenta_metadata_string);
