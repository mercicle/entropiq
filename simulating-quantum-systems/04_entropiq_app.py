
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

#https://streamlit-aggrid.readthedocs.io/en/docs/Usage.html#simple-use
#https://towardsdatascience.com/7-reasons-why-you-should-use-the-streamlit-aggrid-component-2d9a2b6e32f0
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

# https://pypi.org/project/streamlit-observable/

import plotly.express as px

from palettable.scientific.sequential import Devon_20
import colour

from colour import Color
colors = list(Color("#3f007d").range_to(Color("white"),20))
colors = np.flip([c.hex for c in colors])

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
    selected = option_menu("EntropiQ", ["EntropiQ Stats", 'Launch Simulation', 'Discovery'],
                           icons=['stack', 'bricks','search'], menu_icon = "boxes", default_index=1)

if selected == "EntropiQ Stats":

    col1, col2 = st.columns([4,4])

    n_experiments = experiment_metadata_df.shape[0]
    ave_layers  = experiment_metadata_df['n_layers'].mean()
    ave_simulations  = experiment_metadata_df['n_simulations'].mean()

    last_run_date  = experiment_metadata_df['experiment_run_date'].max()

    ave_runtime = experiment_metadata_df['runtime_in_seconds'].mean()

    with col1:
        hc.info_card(title='# of Experiments', content = str(int(n_experiments)), sentiment='good')
        hc.info_card(title='Last Run Date', content = str(last_run_date), sentiment='good')

    with col2:
        hc.info_card(title='Average Layers', content = str(int(ave_layers)), sentiment='good')
        hc.info_card(title='Average Simulations', content = str(int(ave_simulations)), sentiment='good')
        hc.info_card(title='Average Runtime (Min)', content = str(np.round(ave_runtime/60,2)), sentiment='good')

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
        operation_type_to_apply = st.radio("Unary or Binary Gates and Projective Measurements:", ('Binary', 'Unary'))
        gate_types_to_apply = st.radio("Gate Group to Apply:", ('Random Unitaries', 'Random Cliffords'))
        mr_values = st.slider('Measurement Rate Range (%)',0, 100, (0, 80))
        mr_step = st.number_input('Step By Rate:',0.1)

        measurement_rate_space = [x/10 for x in range(list(mr_values)[0], list(mr_values)[1] + int(10*mr_step), int(100*mr_step))]

        st.subheader('Reduced Density Matrix')
        use_constant_size = st.checkbox('Use Constant Sub-system Size for Reduced Density Matrix')

        constant_size = 0
        subsystem_range_divider = 0.50
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

    this_experiment_metadata_df['runtime_in_seconds'] = 0

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
    experiment_metadata_df = get_table(conn = postgres_conn, table_name = experiments_metadata_table_name, schema_name = core_schema)
    #st.table(experiment_metadata_df)

    gb = GridOptionsBuilder.from_dataframe(experiment_metadata_df)
    gb.configure_pagination()
    grid_options = gb.build()

    AgGrid(experiment_metadata_df, grid_options)

    experiment_id = st.selectbox('Select Experiment ID', experiment_metadata_df.experiment_id)

    st.subheader('Experiment Results:')
    experiment_results_df = get_table(conn = postgres_conn, table_name = simulation_results_table_name, schema_name = core_schema, where_string = " where experiment_id = '"+experiment_id + "'")
    experiment_results_df['num_qubits'] = experiment_results_df.num_qubits.astype(str)
    experiment_results_df['mean_runtime_min'] = experiment_results_df['mean_runtime'].apply(lambda x: np.round(x/60,3))
    AgGrid(experiment_results_df)


    n_qubits = len(experiment_results_df.num_qubits.unique())
    n_simulations = experiment_metadata_df.n_simulations.values[0]

    #these_colors = colors[ (len(colors)-n_qubits):]

    n_qubit_color_palette = list(Color("#3f007d").range_to(Color("#fcfbfd"),n_qubits))
    n_qubit_color_palette = np.flip([c.hex for c in n_qubit_color_palette])

    st.subheader('Average Entanglement Entropy by System Size and Measurement Rate')
    main_fig = px.line(experiment_results_df,
                        x='measurement_rate',
                        y='mean_entropy',
                        color='num_qubits',
                        height=800, width=800,
                        color_discrete_sequence = n_qubit_color_palette,
                        #color_continuous_scale=px.colors.sequential.Plasma_r,
                        labels={
                             "measurement_rate": "Measurement Rate (%)",
                             "mean_entropy": "Average Entropy"
                         })
    main_fig.update_traces(line = dict(width=3))
    main_fig.update_layout(font = dict(size=20))

    main_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(main_fig, use_container_width=True)

    st.subheader('Runtime Analysis')
    main_fig = px.line(experiment_results_df,
                        x='measurement_rate',
                        y='mean_runtime_min',
                        color='num_qubits',
                        height=800, width=800,
                        color_discrete_sequence=n_qubit_color_palette,
                        labels={
                             "measurement_rate": "Measurement Rate (%)",
                             "mean_runtime_min": "Average Simulation Runtime (Min)"
                         })
    main_fig.update_traces(line = dict(width=3))
    main_fig.update_layout(font = dict(size=20))
    main_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(main_fig, use_container_width=True)

    entropy_tracking_df = get_table(conn = postgres_conn, table_name = entropy_tracking_table_name, schema_name = core_schema, where_string = " where experiment_id = '"+experiment_id + "'")

    print("Dropping Experiment ID")
    entropy_tracking_df.drop(columns=['experiment_id'],inplace=True)
    entropy_tracking_df['log_state_index'] = entropy_tracking_df['ij'].apply(lambda x: np.log(x))

    st.subheader('Entropy Contribution by Measurement Rate and System Size')
    st.subheader('Preview')

    AgGrid(entropy_tracking_df.head())

    n_sim_color_palette = list(Color("#c7e9c0").range_to(Color("#006d2c"),n_simulations))
    n_sim_color_palette = np.flip([c.hex for c in n_sim_color_palette])


    et_fig = px.line(entropy_tracking_df,
                     x='log_state_index',
                     y='entropy_contribution',
                     color='simulation_number',
                     color_discrete_sequence=n_sim_color_palette,
                     facet_col='measurement_rate',
                     facet_row='num_qubits',
                     height=800, width=800,
                     labels={
                         "entropy_contribution": "ΔS",
                         "log_state_index": "Log(State Index)"
                      },
                     category_orders={
                     "simulation_number": np.sort(entropy_tracking_df.simulation_number.unique()).tolist(),
                     "num_qubits": np.sort(entropy_tracking_df.num_qubits.unique()).tolist(),
                     "measurement_rate": np.sort(entropy_tracking_df.measurement_rate.unique()).tolist()
                     })

    et_fig.update_layout(legend=dict(orientation="h"))
    et_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(et_fig, use_container_width=True)

    st.subheader('Inspection')

    select_nq = st.selectbox('# Qubits:',entropy_tracking_df.num_qubits.unique())
    select_mr = st.selectbox('Measurement rate:',entropy_tracking_df.measurement_rate.unique())

    inspect_entropy_tracking_df = entropy_tracking_df[ (entropy_tracking_df.num_qubits == select_nq) & (entropy_tracking_df.measurement_rate == select_mr)]

    eti_fig = px.line(inspect_entropy_tracking_df,
                  x='log_state_index',
                  y='entropy_contribution',
                  color='simulation_number',
                  color_discrete_sequence=n_sim_color_palette,
                  facet_col='measurement_rate',
                  facet_row='num_qubits',
                  height=800, width=800,
                  labels={
                       "entropy_contribution": "ΔS",
                       "log_state_index": "Log(State Index)"
                   })
    eti_fig.update_traces(line = dict(width=3))
    eti_fig.update_layout(font = dict(size=20), legend = dict(orientation="h"))
    eti_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(eti_fig, use_container_width=True)
    print("End plotting entropy_contribution ΔS")
