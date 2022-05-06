select *  
from quantumlab_experiments.experiments_metadata  


alter table quantumlab_experiments.experiments_metadata add column status TEXT ;

alter table quantumlab_experiments.simulation_results add column mean_runtime FLOAT ;

-- clear data 
delete from quantumlab_experiments.simulation_results where experiment_id is not null;
delete from quantumlab_experiments.entropy_tracking where experiment_id is not null;
delete from quantumlab_experiments.experiments_metadata where experiment_id is not null;

