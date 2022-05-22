
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

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import plotly.express as px

import plotly.express
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from palettable.scientific.sequential import Devon_20
import colour
from colour import Color

this_dir = os.getcwd()
repo_root_dir = this_dir.split("qc-repo")[0] + 'qc-repo/'

from helpers.data_helpers import *
from _app_parms import *

creds_path = os.getcwd() + "/db_creds.env"
load_environ(creds_path)

postgres_conn = get_postgres_conn()

experiment_metadata_df = get_table(conn = postgres_conn, table_name = experiments_metadata_table_name, schema_name = core_schema)

st.set_page_config(layout = "wide")

with st.sidebar:
    selected = option_menu("EntropiQ", ["EntropiQ Stats", 'Launch Simulation', 'Discovery','Jordan-Wigner CPLC'],
                           icons=['stack', 'bricks','search','search'], menu_icon = "boxes", default_index=1)

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
        hc.info_card(title='Average Sim Runtime (Min)', content = str(np.round(ave_runtime/60,2)), sentiment='good')

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

    n_qubit_color_palette = list(Color("#3f007d").range_to(Color("#dadaeb"),n_qubits))
    n_qubit_color_palette = np.flip([c.hex for c in n_qubit_color_palette])

    st.subheader('Average Entanglement Entropy by System Size and Measurement Rate')
    main_fig = px.line(experiment_results_df,
                        x='measurement_rate',
                        y='mean_entropy',
                        color='num_qubits',
                        height=800, width=800,
                        color_discrete_sequence = n_qubit_color_palette,
                        labels={
                             "measurement_rate": "Measurement Rate (%)",
                             "mean_entropy": "Average Entanglement Entropy"
                         })
    main_fig.update_traces(line = dict(width=3))
    main_fig.update_layout(font = dict(size=20),legend_title_text='# of Qubits')

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
    main_fig.update_layout(font = dict(size=20),
                           legend_title_text='# of Qubits')
    main_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(main_fig, use_container_width=True)

    entropy_tracking_df = get_table(conn = postgres_conn, table_name = entropy_tracking_table_name, schema_name = core_schema, where_string = " where experiment_id = '"+experiment_id + "'")

    print("Dropping Experiment ID")
    entropy_tracking_df.drop(columns=['experiment_id'],inplace=True)
    entropy_tracking_df['log_state_index'] = entropy_tracking_df['ij'].apply(lambda x: np.log(x))
    entropy_tracking_df['num_qubits'] = entropy_tracking_df['num_qubits'].astype(int)

    st.subheader('Entropy Contribution by Measurement Rate and System Size')
    st.subheader('Preview')

    AgGrid(entropy_tracking_df.head())

    n_sim_color_palette = list(Color("#c7e9c0").range_to(Color("#006d2c"),n_simulations))
    n_sim_color_palette = np.flip([c.hex for c in n_sim_color_palette])

    st.subheader('Histogram - Evolution of State Probabilities')

    x_axis_ticks = [x/100 for x in range(0,105,10)]
    eti_fig_hist = px.histogram(entropy_tracking_df,
                                 x="eigenvalue",
                                 facet_col="num_qubits",
                                 facet_col_wrap = 4,
                                 color="num_qubits",
                                 color_discrete_sequence = n_qubit_color_palette,

                                 histnorm = 'probability',
                                 nbins = 20,
                                 # add the animation
                                 animation_frame="measurement_rate",
                                 category_orders={
                                 "num_qubits": np.sort(entropy_tracking_df.num_qubits.unique()).tolist(),
                                 },
                                 labels={
                                      "eigenvalue": "State Probability",
                                  },
                                 range_x = [0,1],
                                 range_y = [0,1],
                                 height=800, width=800,
    )

    eti_fig_hist.for_each_yaxis(lambda y: y.update(title = ''))
    eti_fig_hist.update_traces(marker_line_width=1,marker_line_color="white")
    eti_fig_hist.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    eti_fig_hist.update_layout(font = dict(size=20),
                               legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                               yaxis_title = "%",
                               legend_title_text='# of Qubits')

    eti_fig_hist.update_xaxes(ticktext=x_axis_ticks, tickvals=x_axis_ticks)
    eti_fig_hist.for_each_xaxis(lambda x: x.update(ticktext=x_axis_ticks, tickvals=x_axis_ticks))

    st.plotly_chart(eti_fig_hist, use_container_width=True)

    st.subheader('Inspection - State Probability Distribution')

    select_nq = st.selectbox('# Qubits:',entropy_tracking_df.num_qubits.unique())
    select_mr = st.selectbox('Measurement rate:',entropy_tracking_df.measurement_rate.unique())

    inspect_entropy_tracking_df = entropy_tracking_df[ (entropy_tracking_df.num_qubits == select_nq) & (entropy_tracking_df.measurement_rate == select_mr)]

    eti_fig = px.histogram(inspect_entropy_tracking_df,
                             x="eigenvalue",
                             histnorm = 'probability',
                             nbins = 20,
                             labels={
                                  "eigenvalue": "State Probability",
                              },
                             range_x = [0,1],
                             range_y = [0,1],
                             height=800, width=800,
    )

    eti_fig.for_each_yaxis(lambda y: y.update(title = ''))
    eti_fig.update_traces(marker_line_width=1,marker_line_color="white")
    eti_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    eti_fig.update_layout(font = dict(size=20),
                          legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                          yaxis_title = "%",
                          legend_title_text='# of Qubits')
    eti_fig.update_xaxes(ticktext=x_axis_ticks, tickvals=x_axis_ticks)
    st.plotly_chart(eti_fig, use_container_width=True)

elif selected == "Jordan-Wigner CPLC":

    st.header('Jordan-Wigner CPLC')

    st.subheader('Experiments')
    experiment_metadata_df = get_table(conn = postgres_conn, table_name = experiments_metadata_cplc_table_name, schema_name = core_schema)

    gb = GridOptionsBuilder.from_dataframe(experiment_metadata_df)
    gb.configure_pagination()
    grid_options = gb.build()

    AgGrid(experiment_metadata_df, grid_options)

    experiment_id = st.selectbox('Select Experiment ID', experiment_metadata_df.experiment_id)

    st.subheader('Experiment Results:')
    experiment_results_df = get_table(conn = postgres_conn, table_name = simulation_results_cplc_table_name, schema_name = core_schema, where_string = " where experiment_id = '"+experiment_id + "'")
    experiment_results_df['num_qubits'] = experiment_results_df.num_qubits.astype(str)
    experiment_results_df['mean_runtime_min'] = experiment_results_df['mean_runtime'].apply(lambda x: x/60)#np.round(
    AgGrid(experiment_results_df)

    n_qubits = len(experiment_results_df.num_qubits.unique())
    n_simulations = experiment_metadata_df.n_simulations.values[0]

    n_qubit_color_palette = list(Color("#3f007d").range_to(Color("#dadaeb"),n_qubits))
    n_qubit_color_palette = np.flip([c.hex for c in n_qubit_color_palette])

    st.subheader('Average Entanglement Entropy by p and q Parameters')

    facet_plot_eg = experiment_results_df[['num_qubits','p','q','mean_entropy']]
    facet_plot_eg.sort_values(["num_qubits", "p", "q"], ascending = [True, True, True], inplace = True)

    facet_plot_eg['p'] = facet_plot_eg['p'].astype(str)
    facet_plot_eg['q'] = facet_plot_eg['q'].astype(str)
    facet_plot_eg['num_qubits'] = facet_plot_eg['num_qubits'].astype(str)
    #facet_plot_eg['mean_entropy'] = facet_plot_eg['mean_entropy'].apply(lambda x: np.round(x, 5))

    max_columns = st.number_input('Maximum Columns in Heatmap Grid',  3)
    heatmap_grid_height = st.number_input('Heatmap Grid Height',  200)
    heatmap_grid_width = st.number_input('Heatmap Grid Width',  600)

    max_rows = int(np.ceil(len(facet_plot_eg['num_qubits'].unique())/max_columns))
    titles = ['Qubits='+ str(x) for x in facet_plot_eg['num_qubits'].unique().astype(str)]

    fig = make_subplots(rows=max_rows, cols=max_columns,
                        shared_yaxes=True,
                        subplot_titles = tuple(titles))

    i_index = 1
    max_index = len(facet_plot_eg['num_qubits'].unique())
    for i in facet_plot_eg['num_qubits'].unique():

        df = facet_plot_eg[facet_plot_eg['num_qubits'] == i]
        row_index = int(np.floor(i_index/(max_columns+1))+1)
        col_index = i_index % max_columns

        #print("i_index: "+ str(i_index)+" "+" row_index: "+ str(row_index)+" "+" col_index: "+ str(col_index)+" ")

        if col_index == 0:
            col_index = max_columns
        print("i_index: "+ str(i_index)+" "+" row_index: "+ str(row_index)+" "+" col_index: "+ str(col_index)+" ")

        fig.add_trace(go.Heatmap(z=df.mean_entropy, x=df.p, y=df.q, hoverinfo='text', hovertemplate='p: %{x}<br>q: %{y}<br>Mean Entropy: %{z}<extra></extra>'), row=row_index, col=col_index)

        #dtick=list(facet_plot_eg['p'].unique())
        fig.update_xaxes(title_text='p', row=row_index, col=col_index)
        fig.update_yaxes(title_text='q', row=row_index, col=col_index)

        #if i_index == max_index:
        #    fig.update_traces(xgap=1,ygap=1,showscale = True)
        #else:
        #    fig.update_traces(xgap=1,ygap=1,showscale = False)

        fig.update_layout(plot_bgcolor='black',height=heatmap_grid_height, width=heatmap_grid_width)
        fig.update_xaxes(showline=True, linewidth=0.75, linecolor='black', gridcolor='black')
        fig.update_yaxes(showline=True, linewidth=0.75, linecolor='black', gridcolor='black')
        i_index+=1

    #fig.update_layout(showlegend=False,)
    fig.update_traces(xgap=1,ygap=1,showscale = False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Average Simulation Runtime by p and q Parameters')

    facet_plot_runtime_df = experiment_results_df[['num_qubits','p','q','mean_runtime_min']]
    #facet_plot_runtime_df = facet_plot_runtime_df[~facet_plot_runtime_df.p.isin([0.10,0.90]) & ~facet_plot_runtime_df.q.isin([0.10])]

    clipped_df = pd.DataFrame()
    system_sizes = facet_plot_runtime_df.num_qubits.unique().tolist()
    for system_size in system_sizes:
        df = facet_plot_runtime_df[facet_plot_runtime_df.num_qubits == system_size]
        clip_low,clip_high = df['mean_runtime_min'].quantile([0.0,0.95])
        df['mean_runtime_min'] = df['mean_runtime_min'].clip(clip_low, clip_high)
        clipped_df = clipped_df.append(df)

    facet_plot_runtime_df = clipped_df

    fig = make_subplots(rows=max_rows, cols=max_columns,
                        shared_yaxes=True,
                        subplot_titles = tuple(titles))

    i_index = 1
    max_index = len(facet_plot_runtime_df['num_qubits'].unique())
    for i in facet_plot_runtime_df['num_qubits'].unique():

        df = facet_plot_runtime_df[facet_plot_runtime_df['num_qubits'] == i]
        row_index = int(np.floor(i_index/(max_columns+1))+1)
        col_index = i_index % max_columns

        if col_index == 0:
            col_index = max_columns

        fig.add_trace(go.Heatmap(z=df.mean_runtime_min, x=df.p, y=df.q, hoverinfo='text', hovertemplate='p: %{x}<br>q: %{y}<br>Mean Simulation Runtime: %{z}<extra></extra>'), row=row_index, col=col_index)

        fig.update_xaxes(title_text='p', row=row_index, col=col_index)
        fig.update_yaxes(title_text='q', row=row_index, col=col_index)

        fig.update_layout(plot_bgcolor='black',height=heatmap_grid_height, width=heatmap_grid_width)
        fig.update_xaxes(showline=True, linewidth=0.75, linecolor='black', gridcolor='black')
        fig.update_yaxes(showline=True, linewidth=0.75, linecolor='black', gridcolor='black')
        i_index+=1

    #fig.update_layout(showlegend=False,)
    fig.update_traces(xgap=1,ygap=1,showscale = False)
    st.plotly_chart(fig, use_container_width=True)
