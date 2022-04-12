
import os, sys, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import streamlit as st
from streamlit_option_menu import option_menu
import hydralit_components as hc

this_dir = os.getcwd()
repo_root_dir = this_dir.split("qc-repo")[0] + 'qc-repo/'

# works in python editor
#import julia
#from julia import Main
#Main.include("julia_test.jl")

#from julia.api import Julia
#jl = Julia(compiled_modules=False)

#julia_test_path = """include(\""""+ this_dir + """/julia_test.jl\"""" +")"""
#jl.eval(julia_test_path)


#from helpers.data_helpers import *
#from helpers.analysis_helpers import *
#from app_parms import *

#todo: implement
#db_conn = get_db_conn()
#experiment_metadata_df = get_table(conn = db_conn, table_name = experiments_metadata_table_name, schema_name = core_schema)

experiment_metadata_df = pd.DataFrame.from_dict({'experiment_id':[123], 'runtime':[1.0]})

st.set_page_config(layout = "wide")

with st.sidebar:
    selected = option_menu("Quantum Lab", ["Quantum Lab Stats", 'Launch Simulation', 'Discovery'],
                           icons=['stack', 'bricks','search'], menu_icon = "boxes", default_index=1)

if selected == "Quantum Lab Stats":

    col1, col2 = st.columns([4,4])

    n_experiments = experiment_metadata_df.shape[0]
    ave_runtime  = experiment_metadata_df['runtime'].mean()

    with col1:
        hc.info_card(title='# of Experiments', content = str(n_experiments), sentiment='good')
    with col2:
        hc.info_card(title='Average Runtime', content = str(ave_runtime), sentiment='good')

elif selected == "Launch Simulation":

    st.header('Create New Experimental Parameters')

    values = st.slider('Number of Qubit Range',6, 50, (6, 10))
    st.write('Qubit Range Selected:', values)

    qubit_step = st.number_input('Step By:',  1)

    st.write('Final Simulation: ', str(values[0]) + " to " + str(values[1]) + " by " + str(qubit_step))

    n_layers = st.number_input('Number of Layers:', 10)

    n_simulations = st.number_input('Number of Simulations:')

    mr_values = st.slider('Measurement Rate Range',0, 100, (0, 80))
    st.write('Measurement Rate Range Selected:', mr_values)

    mr_step = st.number_input('Step By Rate:',0.1)
    mr_values = [x/10 for x in mr_values]

    st.write('Final Measurement Rate: ', str(mr_values[0]) + " to " + str(mr_values[1]) + " by " + str(mr_step))


elif selected == "Discovery":

    st.header('Discovery')

    st.subheader('Experiments')
    st.table(experiment_metadata_df)

    experiment_id = st.selectbox('Select Experiment ID', experiment_metadata_df.experiment_id)

    st.subheader('Experiment Results:')

    matrix_element = st.selectbox('Select matrix element', [1,2,3])

    print("computing matrix in julia...")
    this_matrix  = np.array([[1,0],[0,1]]) #get_matrix(matrix_element)
    print("after computing matrix in julia...")

    fig, ax = plt.subplots()
    sns.heatmap(this_matrix, ax=ax)
    st.write(fig)

    #experiment_results_df = get_table(conn = db_conn, table_name = , schema_name = , where_string = " where experiment_id = '"+ experiment_id + "'")
    #st.table(experiment_results_df)
