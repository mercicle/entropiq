
function get_matrix_from_julia(v::Int)
   m=[v 0
      0 v]
 return m
end

#https://invenia.github.io/LibPQ.jl/dev/#COPY-1
# execute(conn, "BEGIN;")
# LibPQ.load!(
#     (
#     experiment_id = von_neumann_entropy_df.experiment_id,
#     p = von_neumann_entropy_df.p,
#     q = von_neumann_entropy_df.q,
#     simulation_number = von_neumann_entropy_df.simulation_number,
#     num_qubits = von_neumann_entropy_df.num_qubits,
#     bond_index = von_neumann_entropy_df.bond_index,
#     ij = von_neumann_entropy_df.ij,
#     eigenvalue = von_neumann_entropy_df.eigenvalue,
#     entropy_contribution = von_neumann_entropy_df.entropy_contribution
#     ),
#     conn,
#     "INSERT INTO quantumlab_experiments."*"$(entropy_tracking_table_name)"*" (experiment_id, p, q, simulation_number, num_qubits, bond_index, ij, eigenvalue, entropy_contribution) VALUES(\$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9);"
# )
# execute(conn, "COMMIT;")

## Test
