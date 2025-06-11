
-- CREATE EXAMPLES/TEMPLATES

CREATE TABLE IF NOT EXISTS quantumlab_experiments.experiments_metadata_jwcplc (experiment_id TEXT, experiment_run_date TEXT, status TEXT, experiment_name TEXT, experiment_description TEXT, num_qubit_space TEXT, n_simulations INT, p_space TEXT, q_space TEXT, use_constant_size BOOL, constant_size INT, subsystem_range_divider FLOAT, runtime_in_seconds FLOAT, experimental_design_type TEXT);
CREATE TABLE IF NOT EXISTS quantumlab_experiments.simulation_results_jwcplc (num_qubits INT, p FLOAT, q FLOAT, mean_entropy FLOAT, se_mean_entropy FLOAT, experiment_id TEXT, mean_runtime FLOAT);
CREATE TABLE IF NOT EXISTS quantumlab_experiments.entropy_tracking_jwcplc (experiment_id TEXT, p FLOAT, q FLOAT, simulation_number INT, num_qubits INT, bond_index INT, ij INT, eigenvalue FLOAT, entropy_contribution FLOAT);
CREATE TABLE IF NOT EXISTS quantumlab_experiments.entropy_tracking_jwcplc_copytest (experiment_id TEXT, p FLOAT, q FLOAT, simulation_number INT, num_qubits INT, bond_index INT, ij INT, eigenvalue FLOAT, entropy_contribution FLOAT);

-- SELECT EXAMPLES/TEMPLATES
select distinct experiment_id
from quantumlab_experiments.experiments_metadata

select *
from quantumlab_experiments.entropy_tracking
where experiment_id = '';

select count(*) as n
from quantumlab_experiments.entropy_tracking_jwcplc where experiment_id = '[some uuid]';

select count(*) as n
from quantumlab_experiments.simulation_results_jwcplc
where num_qubits = 26

select *
from quantumlab_experiments.entropy_tracking_jwcplc
where experiment_id = '[some uuid]'
order by p,q,num_qubits,simulation_number, ij

-- ALTER EXAMPLES/TEMPLATES

-- alter table quantumlab_experiments.experiments_metadata add column status TEXT ;
-- alter table quantumlab_experiments.experiments_metadata add column experimental_design_type TEXT;
-- alter table quantumlab_experiments.simulation_results add column mean_runtime FLOAT ;

-- DELETE EXAMPLES/TEMPLATES
--delete from quantumlab_experiments.simulation_results where experiment_id is not null;
--delete from quantumlab_experiments.simulation_results where experiment_id = '0xbcee3e8ca662403c9609615477a6fe3b';
--delete from quantumlab_experiments.entropy_tracking where experiment_id = '0xbcee3e8ca662403c9609615477a6fe3b';
--delete from quantumlab_experiments.experiments_metadata where experiment_id = '0xbcee3e8ca662403c9609615477a6fe3b';
