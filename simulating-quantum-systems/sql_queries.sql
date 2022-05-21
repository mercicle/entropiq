
CREATE TABLE IF NOT EXISTS quantumlab_experiments.experiments_metadata_jwcplc (experiment_id TEXT, experiment_run_date TEXT, status TEXT, experiment_name TEXT, experiment_description TEXT, num_qubit_space TEXT, n_simulations INT, p_space TEXT, q_space TEXT, use_constant_size BOOL, constant_size INT, subsystem_range_divider FLOAT, runtime_in_seconds FLOAT, experimental_design_type TEXT);
CREATE TABLE IF NOT EXISTS quantumlab_experiments.simulation_results_jwcplc (num_qubits INT, p FLOAT, q FLOAT, mean_entropy FLOAT, se_mean_entropy FLOAT, experiment_id TEXT, mean_runtime FLOAT);
CREATE TABLE IF NOT EXISTS quantumlab_experiments.entropy_tracking_jwcplc (experiment_id TEXT, p FLOAT, q FLOAT, simulation_number INT, num_qubits INT, bond_index INT, ij INT, eigenvalue FLOAT, entropy_contribution FLOAT);

--DROP TABLE IF EXISTS quantumlab_experiments.experiments_metadata_jwcplc;

--delete from quantumlab_experiments.experiments_metadata_jwcplc where experiment_id is not null;

--delete from quantumlab_experiments.entropy_tracking_jwcplc where experiment_id = '0x06dfa27044774ab7b9c9c484304846d1';

select *  
from quantumlab_experiments.experiments_metadata  

select distinct experiment_id
from quantumlab_experiments.experiments_metadata  

select * 
from quantumlab_experiments.simulation_results
where experiment_id = '0xafcd0de2920e4e4d92adf1f445583d2b';

select * 
from quantumlab_experiments.entropy_tracking
where experiment_id = '0xafcd0de2920e4e4d92adf1f445583d2b';

select count(*) as n
from quantumlab_experiments.entropy_tracking
where experiment_id = '0xf49163c226ae411b89f038499ea0facc';




alter table quantumlab_experiments.experiments_metadata add column status TEXT ;

alter table quantumlab_experiments.experiments_metadata add column experimental_design_type TEXT;

-- update quantumlab_experiments.experiments_metadata set experimental_design_type = 'Standard Bricklayer';


-- alter table quantumlab_experiments.simulation_results add column mean_runtime FLOAT ;

-- clear data 
-- delete from quantumlab_experiments.simulation_results where experiment_id is not null;
-- delete from quantumlab_experiments.entropy_tracking where experiment_id is not null;
-- delete from quantumlab_experiments.experiments_metadata where experiment_id is not null;

--delete from quantumlab_experiments.simulation_results where experiment_id = '0xbcee3e8ca662403c9609615477a6fe3b';
--delete from quantumlab_experiments.entropy_tracking where experiment_id = '0xbcee3e8ca662403c9609615477a6fe3b';
--delete from quantumlab_experiments.experiments_metadata where experiment_id = '0xbcee3e8ca662403c9609615477a6fe3b';


delete from quantumlab_experiments.simulation_results_jwcplc where experiment_id = '0x256d038b432d44b5aea0a12f156d08dc';

--263200
select count(*) as n
from quantumlab_experiments.entropy_tracking_jwcplc where experiment_id = '0x256d038b432d44b5aea0a12f156d08dc';
