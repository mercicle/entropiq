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
et_fig.update_yaxes(matches=None)
et_fig.update_xaxes(matches=None)

st.plotly_chart(et_fig, use_container_width=True)


st.subheader('Inspection - Curves')

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
