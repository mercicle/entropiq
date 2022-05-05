select *  
from quantumlab_experiments.experiments_metadata  


alter table quantumlab_experiments.experiments_metadata add column status TEXT ;

alter table quantumlab_experiments.simulation_results add column mean_runtime FLOAT ;
