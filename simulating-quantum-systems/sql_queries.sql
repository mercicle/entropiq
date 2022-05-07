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




alter table quantumlab_experiments.experiments_metadata add column status TEXT ;

alter table quantumlab_experiments.simulation_results add column mean_runtime FLOAT ;

-- clear data 
-- delete from quantumlab_experiments.simulation_results where experiment_id is not null;
-- delete from quantumlab_experiments.entropy_tracking where experiment_id is not null;
-- delete from quantumlab_experiments.experiments_metadata where experiment_id is not null;

-- delete from quantumlab_experiments.simulation_results where experiment_id = '0xc6a67f5ef5104321bcd95e31d1249cfa';
-- delete from quantumlab_experiments.entropy_tracking where experiment_id = '0xafcd0de2920e4e4d92adf1f445583d2b';
-- delete from quantumlab_experiments.experiments_metadata where experiment_id = '0xc6a67f5ef5104321bcd95e31d1249cfa';


--delete from quantumlab_experiments.entropy_tracking where experiment_id = '0xafcd0de2920e4e4d92adf1f445583d2b';

select count(*) as n
from quantumlab_experiments.entropy_tracking where experiment_id = '0xafcd0de2920e4e4d92adf1f445583d2b';
