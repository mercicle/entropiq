import os, sys, timeit
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

import copy
import pickle
import uuid
from datetime import date

from helpers.analysis_helpers import *
from helpers.data_helpers import *
from _app_parms import *

creds_path = os.getcwd() + "/db_creds.env"
load_environ(creds_path)

postgres_conn = get_postgres_conn()

outdata_dir = os.getcwd() + "/out_data/qiskit/"


experiment_id = "0x7a052949c1014ca39a7e43a2532b2fa8"

################################
##      Compare Runtimes      ##
################################
experiment_id = "0x1429bbf4c76d4b1b9b5f4cec3b770495"

runtime_julia_df = get_table(conn = postgres_conn, table_name = simulation_results_table_name, schema_name = core_schema, where_string = " where experiment_id = '"+experiment_id + "'")
runtime_julia_df['mean_runtime'] = runtime_julia_df['mean_runtime']/60
runtime_julia_df.rename(columns = {'mean_runtime':'mean_runtime_julia'}, inplace=True)

qiskit_experiment_id = "73fff0d4-c414-11ec-9419-328140767e06"
run_times_df_summary = pd.read_csv(outdata_dir + qiskit_experiment_id + "_runtimes.csv")
run_times_df_summary = run_times_df_summary[['num_qubits', 'measurement_rate','run_time_min']]
run_times_df_summary.rename(columns = {'run_time_min':'mean_runtime_qiskit'}, inplace=True)
merged_runtimes_df = pd.merge(run_times_df_summary, runtime_julia_df, on = ['num_qubits','measurement_rate'])

merged_runtimes_df = merged_runtimes_df[merged_runtimes_df.num_qubits == 9]
stacked_for_plot_df_julia = merged_runtimes_df[['num_qubits', 'measurement_rate', 'mean_runtime_julia']]
stacked_for_plot_df_julia.rename(columns = {'mean_runtime_julia':'mean_runtime'}, inplace=True)
stacked_for_plot_df_julia['software'] = 'ITensors'

stacked_for_plot_df_py = merged_runtimes_df[['num_qubits', 'measurement_rate', 'mean_runtime_qiskit']]
stacked_for_plot_df_py.rename(columns = {'mean_runtime_qiskit':'mean_runtime'}, inplace=True)
stacked_for_plot_df_py['software'] = 'Qiskit'

stacked_for_plot = stacked_for_plot_df_julia
stacked_for_plot = stacked_for_plot.append(stacked_for_plot_df_py).reset_index()

sns.set(rc = {'figure.figsize':(12,12)})
sns.set_style(style='whitegrid')
sim_plot = sns.lineplot(x='measurement_rate',
                        y='mean_runtime',
                        data = stacked_for_plot,
                        hue='software',
                        marker='o',
                        linewidth = 3,
                        markersize = 10)
sim_plot.set_xlabel("Measurement Rate", fontsize = 20)
sim_plot.set_ylabel("Average Runtime (Min)", fontsize = 20)
plt.savefig(outdata_dir +'___compare-qiskit-itensors-mean-runtimes.pdf')

################################
##                            ##
################################
experiment_id = "0x7a052949c1014ca39a7e43a2532b2fa8"

simulation_df = pd.read_csv(outdata_dir + experiment_id + "_simulation_df.csv")
simulation_df_final = simulation_df[simulation_df.keep_layer == True].sort_values(by=['num_qubits', 'measurement_rate','layer'])
simulation_df_summary = pd.read_csv(outdata_dir + experiment_id + "_simulation_df_summary.csv")
run_times_df_summary = pd.read_csv(outdata_dir + experiment_id + "_runtimes.csv")


decoherence_network_path = outdata_dir + experiment_id + '_decoherence_network.p'
layer_dict= pickle.load( open( decoherence_network_path, "rb" ) )


sns.set(rc = {'figure.figsize':(12,12)})
sns.set_style(style='whitegrid')
sim_plot = sns.lineplot(x='measurement_rate',
                        y='run_time_min',
                        data = run_times_df_summary,
                        hue='num_qubits',
                        marker='o',
                        linewidth = 3,
                        markersize = 10)
sim_plot.set_xlabel("Measurement Rate", fontsize = 20)
sim_plot.set_ylabel("Average Runtime (Min)", fontsize = 20)
plt.savefig(outdata_dir + experiment_id + '_simulation-results-only-converged.pdf')

################################
## Results - Only Final Layer ##
################################
simulation_df_final_summary = simulation_df_final.groupby(['num_qubits','measurement_rate'])[['renyi_entropy_2nd']].mean().reset_index()

sns.set(rc = {'figure.figsize':(12,12)})
sns.set_style(style='whitegrid')
sim_plot = sns.lineplot(x='measurement_rate',
                        y='renyi_entropy_2nd',
                        data = simulation_df_final_summary,
                        hue='num_qubits',
                        marker='o',
                        linewidth = 3,
                        markersize = 10)
sim_plot.set_xlabel("Measurement Rate", fontsize = 20)
sim_plot.set_ylabel("2nd Renyi Entropy", fontsize = 20)
plt.savefig(outdata_dir + experiment_id + '_simulation-results-only-converged.pdf')

################################
## Results - Using All Layers ##
################################
sns.set(rc = {'figure.figsize':(12,12)})
sns.set_style(style='whitegrid')
sim_plot = sns.lineplot(x='measurement_rate',
                        y='renyi_entropy_2nd',
                        data = simulation_df_summary,
                        hue='num_qubits',
                        marker='o',
                        linewidth = 3,
                        markersize = 10)
sim_plot.set_xlabel("Measurement Rate", fontsize = 20)
sim_plot.set_ylabel("2nd Renyi Entropy", fontsize = 20)
plt.savefig(outdata_dir + experiment_id+ '_simulation-results.pdf')


#################################
###     Layer Analysis        ###
#################################

#simulation_df['log_renyi_entropy_2nd'] = simulation_df.apply(lambda row: np.log10(row['renyi_entropy_2nd']), axis=1)
# example: https://seaborn.pydata.org/examples/faceted_lineplot.html

sns.set(rc = {'figure.figsize':(12,12)})
sns.set_style(style='whitegrid')

fig, ax = plt.subplots()

simulation_df.renyi_entropy_2nd.hist()
plt.savefig(outdata_dir + experiment_id+ '_simulation-results-renyi-hist.pdf')

size_order = list(np.unique(simulation_df.measurement_rate))
n_colors = len(size_order)
palette = sns.color_palette("rocket_r",10)[0:n_colors]
# sns.color_palette("rocket_r",10)
# sns.color_palette("RdPu", 10)
sns.relplot(
    data = simulation_df,
    x="layer",
    y="renyi_entropy_2nd",
    hue="measurement_rate",
    col="num_qubits",
    kind="line", palette=palette,
    facet_kws=dict(sharex=False)
)
plt.savefig(outdata_dir + experiment_id+ '_simulation-results-layer-evolution.pdf')



#################################
###         Simulation        ###
#################################
%matplotlib qt

initial_matrix = -1*np.log2(np.abs(layer_dict[1]))

def update(i):
    initial_matrix = -1*np.log2(np.abs(layer_dict[i+1]))
    matrix_matshow.set_array(initial_matrix)
    plt.title("Density Matrix ?? Evolution - Layer " + str(i), fontweight='bold',fontsize=15)

fig, ax = plt.subplots()
matrix_matshow = ax.matshow(initial_matrix, cmap=plt.cm.Greys_r)
plt.colorbar(matrix_matshow)
ani = animation.FuncAnimation(fig, update, frames=len(layer_dict.keys()), interval=1000)
#plt.title("Density Matrix ?? Evolution")
ax.text(75, 25, '-1*log2(??_ij)', bbox={'facecolor': 'white', 'pad': 10},
        fontsize=15,
        fontweight='bold')
ax.text(75, 0, 'Smaller \nDecoherence\nProbability', bbox={'facecolor': 'white', 'pad': 10},
        fontsize=15,
        fontweight='bold')
ax.text(75, 65, 'Larger \nDecoherence\nProbability', bbox={'facecolor': 'white', 'pad': 10},
        fontsize=15,
        fontweight='bold')
plt.show()
