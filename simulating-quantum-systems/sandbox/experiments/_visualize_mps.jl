using ITensors
using ITensors: dim as itensor_dim
using PastaQ
import PastaQ: gate

using Random
using Printf
using LinearAlgebra
using StatsBase: mean, sem, sample as random_sample
using Distributions
using DataFrames
using XLSX
using UUIDs
using Dates
using TimerOutputs
using Pickle

using HDF5
using LibPQ
using DotEnv

using TableView
using Tables

save_dir = string(@__DIR__, "/out_data/")
include("helpers/entropy_function.jl")
include("03_load_gates.jl")

cnfg = DotEnv.config(path=string(@__DIR__, "/db_creds.env"))
db_connection_string = string(" host = ", cnfg["POSTGRES_DB_URL"],
                              " port = ", cnfg["POSTGRES_DB_PORT"],
                              " user = ", cnfg["POSTGRES_DB_USERNAME"],
                              " password = ",cnfg["POSTGRES_DB_PASSWORD"],
                              " sslmode = 'require'",
                              " dbname = ", cnfg["POSTGRES_DB_NAME"]
                              )
conn = LibPQ.Connection(db_connection_string)

num_qubits = 3
projective_list = [ "00"; "01"; "10"; "11"]

# initialize state ψ = |000…⟩
ψ = productstate(num_qubits)

#[1] ((dim=2|id=47|"Qubit,Site,n=1"), (dim=1|id=527|"Link,l=1"))
#[2] ((dim=1|id=527|"Link,l=1"), (dim=2|id=990|"Qubit,Site,n=2"), (dim=1|id=606|"Link,l=2"))
#[3] ((dim=1|id=606|"Link,l=2"), (dim=2|id=998|"Qubit,Site,n=3"))

this_layer = randomlayer("RandomUnitary",[(j,j+1) for j in 1:1:(num_qubits-1)])

ψ = runcircuit(ψ, this_layer; cutoff = 1e-8)

print(ψ)

#[1] ((dim=2|id=47|"Qubit,Site,n=1"), (dim=2|id=388|"Link,n=1"))
#[2] ((dim=2|id=990|"Qubit,Site,n=2"), (dim=2|id=388|"Link,n=1"), (dim=2|id=540|"Link,n=1"))
#[3] ((dim=2|id=540|"Link,n=1"), (dim=2|id=998|"Qubit,Site,n=3"))

next_qubit_index = qubit_index + 1
orthogonalize!(ψ, qubit_index)

ϕ = ψ[qubit_index] * ψ[next_qubit_index]
ρ = prime(ϕ, tags = "Site") * dag(ϕ)

unitary_pmf = real.(diag(reshape(array(ρ), (4,4))))
σ = wsample(projective_list, unitary_pmf, 1)[1]
projection_string = "Π"*"$(σ)"
ψ = runcircuit(ψ, (projection_string, (qubit_index, next_qubit_index)))
normalize!(ψ)
ψ[:] = ψ

this_von_neumann_entropy_dict = entanglemententropy(ψ, subsystem_range_divider, use_constant_size, constant_size)
