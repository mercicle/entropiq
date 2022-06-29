
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
from plotly.offline import plot

this_dir = os.getcwd()
repo_root_dir = this_dir.split("qc-repo")[0] + 'qc-repo/'

from helpers.data_helpers import *
from _app_parms import *

creds_path = os.getcwd() + "/db_creds.env"
load_environ(creds_path)

postgres_conn = get_postgres_conn()


experiment_id = '0xafcd0de2920e4e4d92adf1f445583d2b'

experiment_results_df = get_table(conn = postgres_conn, table_name = simulation_results_table_name, schema_name = core_schema, where_string = " where experiment_id = '"+experiment_id + "'")
experiment_results_df['num_qubits'] = experiment_results_df.num_qubits.astype(str)


main_fig = px.line(experiment_results_df,
                    x='measurement_rate', y='mean_entropy', color='num_qubits',
                    height=800, width=800,
                    labels={
                         "measurement_rate": "Measurement Rate (%)",
                         "mean_entropy": "Average Entropy"
                     })

main_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
plot(main_fig)
#st.plotly_chart(main_fig, use_container_width=True)

entropy_tracking_df = get_table(conn = postgres_conn, table_name = entropy_tracking_table_name, schema_name = core_schema, where_string = " where experiment_id = '"+experiment_id + "'")
entropy_tracking_df.drop(columns=['experiment_id'],inplace=True)
#entropy_tracking_df['num_qubits'] = entropy_tracking_df.num_qubits.astype(int)
#entropy_tracking_df['simulation_number'] = entropy_tracking_df.simulation_number.astype(str)
#entropy_tracking_df['entropy_contribution'] = entropy_tracking_df.entropy_contribution.astype(float)
#entropy_tracking_df['state_index'] = entropy_tracking_df.entropy_contribution.astype(int)

print("Start plotting entropy_contribution ΔS")
et_fig = px.line(entropy_tracking_df,
                 x='ij', y='entropy_contribution', color='simulation_number',
                facet_col = 'measurement_rate', facet_row = 'num_qubits',
                height=800, width=800,
                labels={
                     "entropy_contribution": "ΔS",
                     "ij": "State Index"
                 })

et_fig.update_layout(legend=dict(orientation="h"))
et_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(et_fig, use_container_width=True)

st.subheader('Inspection')

select_nq = st.selectbox('# Qubits:',entropy_tracking_df.num_qubits.unique())
select_mr = st.selectbox('Measurement rate:',entropy_tracking_df.measurement_rate.unique())

inspect_entropy_tracking_df = entropy_tracking_df[ (entropy_tracking_df.num_qubits == select_nq) & (entropy_tracking_df.measurement_rate == select_mr)]
eti_fig = px.line(inspect_entropy_tracking_df,
              x='ij', y='entropy_contribution', color='simulation_number', #size='entropy_contribution',
              facet_col = 'measurement_rate', facet_row = 'num_qubits',
              height=800, width=800,
              labels={
                   "entropy_contribution": "ΔS",
                   "ij": "State Index"
               })
eti_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
eti_fig.update_layout(legend=dict(orientation="h"))
st.plotly_chart(eti_fig, use_container_width=True)
print("End plotting entropy_contribution ΔS")


