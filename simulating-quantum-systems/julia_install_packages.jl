# https://github.com/GTorlai/PastaQ.jl
# https://github.com/GTorlai/PastaQ.jl/blob/000b2524b92b5cb09295cfd09dcbb1914ddc0991/src/circuits/circuits.jl


# this starter code was graciously added by ITensor and PastaQ maintainers
# https://raw.githubusercontent.com/GTorlai/PastaQ.jl/master/examples/11_monitored_circuit.jl

# Julia help docs
# https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention
# https://docs.julialang.org/en/v1/manual/control-flow/
# https://docs.julialang.org/en/v1/manual/variables-and-scoping/
# https://sodocumentation.net/julia-lang

# ρ = prime(ϕ, tags = "Site")
# https://www.itensor.org/docs.cgi?vers=cppv2&page=tutorials/primes

# DataFrames
# https://github.com/bkamins/Julia-DataFrames-Tutorial/
# https://www.ahsmart.com/pub/data-wrangling-with-data-frames-jl-cheat-sheet/
#DataFrame(A=1:3, B=rand(3), C=randstring.([3,3,3]), fixed=1)

#https://www.reddit.com/r/Julia/comments/9p3ttr/clearing_workspace_atom/
# ctrl-j + ctrl-k will kill the current Julia process and start a new session.

# Embed Julia into Python
# https://towardsdatascience.com/how-to-embed-your-julia-code-into-python-to-speed-up-performance-e3ff0a94b6e
# https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/interfacing-julia-with-other-languages

import Pkg

Pkg.add("ITensors") # https://arxiv.org/pdf/2007.14822.pdf
Pkg.add("PastaQ") # https://github.com/GTorlai/PastaQ.jl

Pkg.add("StatsBase")
Pkg.add("Distributions")
Pkg.add("DataFrames")
Pkg.add("CSV")

Pkg.add("TimerOutputs")
Pkg.add("Pickle")

Pkg.add("HDF5")
