
using ITensors
using ITensors: dim as itensor_dim
using PastaQ
import PastaQ: gate
using DataFrames

function entanglemententropy(ψ₀::MPS, subsystem_divider::Int, use_constant_size::Bool, constant_size::Int)

  ψ_local = normalize!(copy(ψ₀))
  N = length(ψ_local)

  if use_constant_size
    bond = constant_size
  else
    bond = trunc(Int, N/subsystem_divider)
  end

    singular_values_to_keep = 2^bond

  orthogonalize!(ψ_local, bond)

  row_inds = (linkind(ψ_local, bond - 1), siteind(ψ_local, bond))
  u, s, v = svd(ψ_local[bond], row_inds, mindim = singular_values_to_keep)

  S = 0.0
  sigma_rank = itensor_dim(s, 1)
  entropy_df = DataFrame()

  #@printf("~~~~~~sigma_rank: %.3i \n", sigma_rank)

  for n in 1:sigma_rank
    λ = s[n, n]^2
    entropy_contribution = - λ * log(λ + 1e-20)
    S = S + entropy_contribution
    this_df = DataFrame(num_qubits = N, bond_index = bond, ij= n, eigenvalue = λ, entropy_contribution = entropy_contribution)
    entropy_df = [entropy_df; this_df]
  end
  return Dict("S" => S, "entropy_df" => entropy_df)
end


function run_brick_layer_sim(num_qubit_space, simulation_space, measurement_rate_space, layer_space, operation_type_to_apply, gate_types_to_apply, subsystem_range_divider, use_constant_size, constant_size)

  projective_list = [ "00"; "01"; "10"; "11"]
  qubit_index_space = nothing
  ψ_tracker = nothing
  this_layer = nothing
  this_circuit = nothing
  simulation_df = DataFrame()
  von_neumann_entropy_df = DataFrame()
  von_neumann_entropies = []
  run_times = []

  for (index_n, num_qubits) in enumerate(num_qubit_space)

    # num_qubits=6
    @printf("# Qubits = %.3i \n", num_qubits)

    if operation_type_to_apply == "Unary"
      qubit_index_space = 1:num_qubits
    elseif operation_type_to_apply == "Binary"
      qubit_index_space = 1:(num_qubits-1)
    end

    @printf("Preparing circuit_simulations for # Qubits = %.3i \n", num_qubits)
    circuit_simulations = []
    for (index_s, this_sim) in enumerate(simulation_space)
      layers = []
      for this_layer_index in layer_space

        this_layer = nothing
        if isodd(this_layer_index)

          if gate_types_to_apply == "Random Unitaries"

            this_layer  = randomlayer("RandomUnitary",[(j,j+1) for j in 1:2:(num_qubits-1)])

          elseif gate_types_to_apply == "Random Cliffords"

            clifford_indices_list = [random_sample(1:clifford_samples) for j in 1:1:(num_qubits-1)]
            clifford_list = [clifford_dict["$j"] for j in clifford_indices_list]
            this_layer  = [(clifford_list[j], j,j+1) for j in 1:2:(num_qubits-1)]

          end

        else

          if gate_types_to_apply == "Random Unitaries"

            this_layer = randomlayer("RandomUnitary",[(j,j+1) for j in 2:2:(num_qubits-1)])

          elseif gate_types_to_apply == "Random Cliffords"

            clifford_indices_list = [random_sample(1:clifford_samples) for j in 1:1:(num_qubits-1)]
            clifford_list = [clifford_dict["$j"] for j in clifford_indices_list]
            this_layer  = [(clifford_list[j], j,j+1) for j in 2:2:(num_qubits-1)]

          end
        end

        push!(layers, this_layer)
      end
      push!(circuit_simulations, layers)
    end

    for (index_m, measurement_rate) in enumerate(measurement_rate_space)

        # measurement_rate = 0.10
        this_circuit_index = 1
        von_neumann_entropies = []
        run_times = []
        for this_circuit in circuit_simulations

           sim_start_time = time()
           # this_circuit = circuit_simulations[8]
           N = nqubits(this_circuit)
           #@printf("# Qubits = %.3i , # Qubits = %.3i  \n", num_qubits, N)

           # initialize state ψ = |000…⟩
           ψ = productstate(num_qubits)

           this_layer_index = 1
           for this_layer in this_circuit

             #@printf("# Qubits: %.3i , Measurement Rate: %.2f, Circuit Sim Index: %.3i, Layer Index: %.3i \n", num_qubits, measurement_rate, this_circuit_index, this_layer_index)

             # this_layer = this_circuit[41]
             ψ = runcircuit(ψ, this_layer; cutoff = 1e-8)

             # perform measurements
             for qubit_index in qubit_index_space

               # only apply measurements to off indices for odd layers and even indices for even layers
               if ((isodd(this_layer_index) && isodd(qubit_index)) || (iseven(this_layer_index) && iseven(qubit_index)))

                 # qubit_index = 1
                 if measurement_rate > rand()

                   if operation_type_to_apply == "Unary"

                       ψ = orthogonalize!(ψ, qubit_index)
                       ϕ = ψ[qubit_index]

                       ρ = prime(ϕ, tags = "Site") * dag(ϕ)
                       prob = real.(diag(array(ρ))) # Outcome probabilities
                       σ = (rand() < prob[1] ? 0 : 1) # random sample

                   elseif operation_type_to_apply == "Binary"

                       next_qubit_index = qubit_index + 1
                       orthogonalize!(ψ, qubit_index)

                       ϕ = ψ[qubit_index] * ψ[next_qubit_index]
                       ρ = prime(ϕ, tags = "Site") * dag(ϕ)

                       unitary_pmf = real.(diag(reshape(array(ρ), (4,4))))
                       σ = wsample(projective_list, unitary_pmf, 1)[1]

                   end

                   projection_string = "Π"*"$(σ)"
                   ψ = runcircuit(ψ, (projection_string, (qubit_index, next_qubit_index)))
                   normalize!(ψ)

                 end # if measurement_rate > rand()

               end

             end # for qubit_index in qubit_index_space

             this_layer_index += 1

           end # for this_layer in this_circuit

           ψ_tracker = copy(ψ)
           this_von_neumann_entropy_dict = Dict()
           try
              this_von_neumann_entropy_dict = entanglemententropy(ψ, subsystem_range_divider, use_constant_size, constant_size)
           catch e
              @printf("!!SVD failed for Circuit: %.3i \n", this_circuit_index)
           end

           this_von_neumann_entropy = this_von_neumann_entropy_dict["S"]
           this_von_neumann_entropy_df = this_von_neumann_entropy_dict["entropy_df"]
           this_von_neumann_entropy_df = insertcols!(this_von_neumann_entropy_df, :measurement_rate => measurement_rate)
           this_von_neumann_entropy_df = insertcols!(this_von_neumann_entropy_df, :simulation_number => this_circuit_index)

           von_neumann_entropy_df = [von_neumann_entropy_df; this_von_neumann_entropy_df]

           this_circuit_index += 1

           this_runtime = time() - sim_start_time
           this_runtime = round(this_runtime, digits=3)

           push!(von_neumann_entropies, this_von_neumann_entropy)
           push!(run_times, this_runtime)

        end # for this_circuit in circuit_simulations

        mean_runtime = mean(run_times)
        mean_entropy = mean(von_neumann_entropies)
        se_mean_entropy = sem(von_neumann_entropies)
        @printf("# Qubits = %.3i Measurement Rate = %.2f  S(ρ) = %.5f ± %.1E \n", num_qubits, measurement_rate, mean_entropy, se_mean_entropy)

        this_simulation_df = DataFrame(num_qubits = num_qubits, measurement_rate = measurement_rate, mean_entropy = mean_entropy, se_mean_entropy=se_mean_entropy, mean_runtime = mean_runtime)
        simulation_df = [simulation_df; this_simulation_df]

    end # for measurement_rate in measurement_rate_space
  end # for num_qubits in num_qubit_space

  return Dict("simulation_df" => simulation_df, "von_neumann_entropy_df" => von_neumann_entropy_df)

end


function run_cplc_sim(num_qubit_space, simulation_space, measurement_rate_space, p_space, q_space, subsystem_range_divider, use_constant_size, constant_size)

  qubit_index_space = nothing
  ψ_tracker = nothing
  this_layer = nothing
  this_circuit = nothing
  simulation_df = DataFrame()
  von_neumann_entropy_df = DataFrame()
  this_von_neumann_entropy_df = DataFrame()
  this_von_neumann_entropy_dict = Dict()
  von_neumann_entropies = []
  run_times = []
  start_time = time()

  for num_qubits in num_qubit_space
    # num_qubits=6
    @printf("# Qubits = %.3i \n", num_qubits)
    qubit_index_space = 1:num_qubits
    this_layer_space = 1:num_qubits

    for p in p_space
      for q in q_space
        # p = 0.90
        # q = 0.90
        von_neumann_entropies = []
        run_times = []
        for this_sim in simulation_space

           # this_sim = 1
           sim_start_time = time()

           # initialize state ψ = |000…⟩
           ψ = productstate(num_qubits)

           for this_layer_index in this_layer_space

             # this_layer_index = 1
             if isodd(this_layer_index)

               for qubit_index in qubit_index_space

                 # qubit_index = 1
                 # p = 0.5
                 odd_actions = ["JWCPLC_UOdd","JWCPLC_UOdd1", "JWCPLC_UOdd_Measure"]
                 action_pmf = [p, (1-p)*q, (1-p)*(1-q)]
                 sampled_action = wsample(odd_actions, action_pmf, 1)[1]

                 if sampled_action == "JWCPLC_UOdd" || sampled_action == "JWCPLC_UOdd1"
                   action_string = "Π"*"$(sampled_action)"
                 elseif sampled_action == "JWCPLC_UOdd_Measure"

                   ψ = orthogonalize!(ψ, qubit_index)
                   born_probability_up = ITensors.expect(ψ, "Xup", sites=qubit_index)
                   born_probability_down = 1-born_probability_up
                   up_down_probabilities = [born_probability_up, born_probability_down]
                   X_measurement_list = ["Xup","Xdown"]
                   action_string = wsample(X_measurement_list, up_down_probabilities, 1)[1]
                 end
                 ψ = runcircuit(ψ, (action_string, qubit_index))
                 normalize!(ψ)
               end
             else

               for qubit_index in qubit_index_space
                 if qubit_index != num_qubits
                   next_qubit_index = qubit_index + 1
                   even_actions = ["JWCPLC_UEven","JWCPLC_UEven1", "JWCPLC_UEven_Measure"]
                   action_pmf = [p,(1-p)*(1-q),(1-p)*q]
                   sampled_action = wsample(even_actions, action_pmf, 1)[1]

                   if sampled_action == "JWCPLC_UEven" || sampled_action == "JWCPLC_UEven1"
                     action_string = "Π"*"$(sampled_action)"
                   elseif sampled_action == "JWCPLC_UEven_Measure"

                     ψ = orthogonalize!(ψ, qubit_index)
                     Czz = ITensors.correlation_matrix(ψ,"ZOp", "ZOp", sites = qubit_index:next_qubit_index)
                     off_diagonal = real.(Czz[1,2])

                     zz_up_down_probabilities = [1/2*(1+off_diagonal), 1/2*(1-off_diagonal)]
                     zz_measurement_list = ["ZZup","ZZdown"]
                     action_string = wsample(zz_measurement_list, zz_up_down_probabilities, 1)[1]
                   end
                   ψ = runcircuit(ψ, (action_string, (qubit_index, next_qubit_index)))
                   normalize!(ψ)
                 end #qubit_index != num_qubits
               end # for qubit_index in qubit_index_space
             end # if odd/even layer
           end # layers

           ψ_tracker = copy(ψ)
           this_von_neumann_entropy_dict = Dict()
           try
              this_von_neumann_entropy_dict = entanglemententropy(ψ, subsystem_range_divider, use_constant_size, constant_size)
           catch e
              @printf("!!SVD failed for Circuit: %.3i \n", this_sim)
           end

           this_von_neumann_entropy = this_von_neumann_entropy_dict["S"]
           if isnan(this_von_neumann_entropy)
             this_von_neumann_entropy = 0
           end

           this_von_neumann_entropy_df = this_von_neumann_entropy_dict["entropy_df"]
           if nrow(this_von_neumann_entropy_df) == 0
             this_von_neumann_entropy_df = DataFrame(num_qubits = num_qubits, bond_index = NaN, ij= NaN, eigenvalue = NaN, entropy_contribution = NaN)
           end

           this_von_neumann_entropy_df = insertcols!(this_von_neumann_entropy_df, :p => p)
           this_von_neumann_entropy_df = insertcols!(this_von_neumann_entropy_df, :q => q)
           this_von_neumann_entropy_df = insertcols!(this_von_neumann_entropy_df, :simulation_number => this_sim)

           von_neumann_entropy_df = [von_neumann_entropy_df; this_von_neumann_entropy_df]

           this_runtime = time() - sim_start_time
           this_runtime = round(this_runtime, digits=3)

           push!(von_neumann_entropies, this_von_neumann_entropy)
           push!(run_times, this_runtime)

        end # simulation

        mean_runtime = mean(run_times)
        mean_entropy = mean(von_neumann_entropies)
        se_mean_entropy = sem(von_neumann_entropies)
        if isnan(se_mean_entropy)
          se_mean_entropy = 0
        end
        @printf("# Qubits = %.3i p = %.2f q = %.2f   S(ρ) = %.5f ± %.1E \n", num_qubits, p, q, mean_entropy, se_mean_entropy)

        this_simulation_df = DataFrame(num_qubits = num_qubits, p = p, q = q, mean_entropy = mean_entropy, se_mean_entropy = se_mean_entropy, mean_runtime = mean_runtime)
        simulation_df = [simulation_df; this_simulation_df]

      end # q
    end # p
  end # num_qubits

  return Dict("simulation_df" => simulation_df, "von_neumann_entropy_df" => von_neumann_entropy_df)
end
