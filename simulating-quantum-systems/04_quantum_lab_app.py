
import os, sys, json
import uuid
from datetime import date

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_option_menu import option_menu
import hydralit_components as hc

this_dir = os.getcwd()
repo_root_dir = this_dir.split("qc-repo")[0] + 'qc-repo/'

from helpers.data_helpers import *
from _app_parms import *

creds_path = os.getcwd() + "/db_creds.env"
load_environ(creds_path)

postgres_conn = get_postgres_conn()

experiment_metadata_df = get_table(conn = postgres_conn, table_name = experiments_metadata_table_name, schema_name = core_schema)

#postgres_conn.execute("drop table quantumlab_experiments._experiments_metadata;")
#postgres_conn.execute("drop table quantumlab_experiments._simulation_results;")
#postgres_conn.execute("drop table quantumlab_experiments._entropy_tracking;")

st.set_page_config(layout = "wide")

with st.sidebar:
    selected = option_menu("Quantum Lab", ["Quantum Lab Stats", 'Launch Simulation', 'Discovery'],
                           icons=['stack', 'bricks','search'], menu_icon = "boxes", default_index=1)

if selected == "Quantum Lab Stats":

    col1, col2 = st.columns([4,4])

    n_experiments = experiment_metadata_df.shape[0]
    ave_layers  = experiment_metadata_df['n_layers'].mean()
    ave_simulations  = experiment_metadata_df['n_simulations'].mean()

    last_run_date  = experiment_metadata_df['experiment_run_date'].max()

    with col1:
        hc.info_card(title='# of Experiments', content = str(int(n_experiments)), sentiment='good')
        hc.info_card(title='Last Run Date', content = str(last_run_date), sentiment='good')

    with col2:
        hc.info_card(title='Average Layers', content = str(int(ave_layers)), sentiment='good')
        hc.info_card(title='Average Simulations', content = str(ave_simulations), sentiment='good')

elif selected == "Launch Simulation":

    st.header('Experiment Configuration')

    col1, col2 = st.columns([4,4])

    experiment_id = str(uuid.uuid1())
    experiment_run_date = date.today().strftime("%m-%d-%Y")

    with col1:

        st.subheader('General Info')
        experiment_name = st.text_input('Experiment Name:')
        experiment_description = st.text_area('Experiment Description:')

        st.subheader('Qubits, Layers, and Simulatons')

        num_qubit_space = st.slider('Number of Qubit Range',6, 50, (6, 10))
        qubit_step = st.number_input('Step By:',  1)

        num_qubit_space = list(range(num_qubit_space[0], num_qubit_space[1], int(qubit_step)))

        n_layers = st.number_input('Number of Layers:', 10)
        n_simulations = st.number_input('Number of Simulations:',100)

    with col2:

        st.subheader('Gates and Measurements')
        operation_type_to_apply = st.radio("Unary or Binary Gates and Projective Measurements:", ('Unary', 'Binary'))
        gate_types_to_apply = st.radio("Gate Group to Apply:", ('Random Unitaries', 'Random Cliffords'))
        mr_values = st.slider('Measurement Rate Range (%)',0, 100, (0, 80))
        mr_step = st.number_input('Step By Rate:',0.1)

        measurement_rate_space = [x/10 for x in range(list(mr_values)[0], list(mr_values)[1] + int(10*mr_step), int(100*mr_step))]

        use_constant_size = st.checkbox('I agree')

        constant_size = 0
        if use_constant_size:
             constant_size = st.number_input('Constant Sub-system Size for Reduced Density Matrix (# of Qubits):', 3)
        else:
            subsystem_range_divider = st.selectbox('Relative Sub-system Size for Reduced Density Matrix:',[0.50, 0.25, 0.20])
            st.write('Reduced Density Matrix Based on Tracing Over:', str(np.round(100*(1-subsystem_range_divider),0)) + '% of the system.')

    st.subheader('Experiment Configuration Summary')
    this_experiment_metadata_df = pd.DataFrame.from_dict({'experiment_name': [experiment_name],
                                                          'experiment_description': [experiment_description],
                                                          'experiment_id' : [experiment_id],
                                                          'experiment_run_date' : [experiment_run_date],
                                                          'num_qubit_space' : [','.join([str(x) for x in num_qubit_space])],
                                                          'n_layers' : [n_layers],
                                                          'n_simulations' : [n_simulations],
                                                          'measurement_rate_space' : [','.join([str(x) for x in measurement_rate_space])],
                                                          'use_constant_size': [use_constant_size],
                                                          'constant_size': [constant_size],
                                                          'subsystem_range_divider' : [subsystem_range_divider],
                                                          'operation_type_to_apply' : [operation_type_to_apply],
                                                          'gate_types_to_apply' : [gate_types_to_apply]
                                                        })

    experiment_metadata_df_preview = this_experiment_metadata_df.T
    experiment_metadata_df_preview[0] = experiment_metadata_df_preview[0].astype(str)
    experiment_metadata_df_preview['Parameter'] = experiment_metadata_df_preview.index
    experiment_metadata_df_preview.reset_index(inplace=True, drop=True)
    experiment_metadata_df_preview.rename(columns={0:'Value'}, inplace=True)
    experiment_metadata_df_preview = experiment_metadata_df_preview[['Parameter', 'Value']]
    st.table(experiment_metadata_df_preview)

    action_col1, action_col2 = st.columns([0.05,1])
    with action_col1:
        if st.button('Save'):
            st.write('Saving...')
            write_table(conn = postgres_conn,
                        df = this_experiment_metadata_df,
                        table_name = experiments_metadata_table_name,
                        schema_name = core_schema,
                        append = True)
            st.write('Saved.')
    with action_col2:
        if st.button('Launch Simulation'):
            st.write('Launching Simulation...')

elif selected == "Discovery":

    st.header('Discovery')

    st.subheader('Experiments')
    #experiment_results_df = get_table(conn = db_conn, table_name = , schema_name = , where_string = " where experiment_id = '"+ experiment_id + "'")
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
