<img src="./readme_images/entropiq-transparent-background.png" width="800" height="200">

# Welcome to EntropiQ

We hope EntropiQ will become a community effort to build a cloud platform for entanglement analytics of many-body quantum mechanical systems using Tensor Networks.


## EntropiQ Components

- AWS Postgres Database for quantum simulation data and metadata management
- Many-body quantum system simulation pipeline templates in (Julia)
  - Julia API Templates to deploy simulations and integrate with app
- Streamlit Application (Python)

<img src="./readme_images/discovery_main.png" width="250" height="250">
<img src="./readme_images/discovery_runtimes.png" width="250" height="250">
<img src="./readme_images/discovery_state_evolution.png" width="250" height="250">
<img src="./readme_images/discovery_inspection.png" width="250" height="250">


## Relevant Papers

# Papers

## Quantum System Experimental Design
[Bricklayer Design - "Quantum Zeno Effect and the Many-body Entanglement Transition"](https://arxiv.org/pdf/1808.06134.pdf)

[Entanglement area law in superfluid 4He](https://www.nature.com/articles/nphys4075)

[Simulating Clifford's - "Hadamard-free circuits expose the structure of the Clifford group"](https://arxiv.org/abs/2003.09412)

[Measurement Protected Quantum Phases](https://arxiv.org/pdf/2004.09509.pdf)

## Tensor Networks
[This site is a resource for tensor network algorithms, theory, and software.](https://tensornetwork.org/)

[Quantum Tensor Networks in a Nutshell](https://arxiv.org/pdf/1708.00006.pdf)

[The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477)

[Hand-waving and Interpretive Dance: An Introductory Course on Tensor Networks](https://arxiv.org/pdf/1603.03039.pdf)

[Matrix Product State Based Algorithms for Ground States and Dynamics](https://qdev.nbi.ku.dk/student_theses/RGawatz_Msc.pdf)

[ITensors - very good article on measurement of local operators](http://itensor.org/docs.cgi?page=formulas/measure_mps&vers=julia)

[A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States](https://arxiv.org/pdf/1306.2164.pdf)

[Garnet Chan "Matrix product states, DMRG, and tensor networks" (Part 1 of 2)](https://www.youtube.com/watch?v=Q8bFmV6tHBs)

[Garnet Chan "Matrix product states, DMRG, and tensor networks" (Part 2 of 2)](https://www.youtube.com/watch?v=s37tvvrjlto)

[MPS Examples](https://physics.stackexchange.com/questions/266587/examples-of-matrix-product-states)

[Qiskit MPS](https://qiskit.org/documentation/stable/0.24/tutorials/simulators/7_matrix_product_state_method.html)

[MPO Matrix Product Operator](https://tensornetwork.org/mpo/#:~:text=A%20matrix%20product%20operator%20(MPO,in%20a%20chain%2Dlike%20fashion.)

## Entanglement Entropy

- https://qiskit.org/documentation/_modules/qiskit/quantum_info/states/utils.html#partial_trace
- https://qiskit.org/textbook/ch-quantum-hardware/density-matrix.html#reduced
- http://itensor.org/docs.cgi?vers=cppv3&page=formulas/mps_two_rdm

> "Among physicists, this is often called "tracing out" or "tracing over" W to leave only an operator on V in the context where W and V are Hilbert spaces associated with quantum systems (see [here](https://en.wikipedia.org/wiki/Partial_trace#:~:text=In%20linear%20algebra%20and%20functional,is%20an%20operator%2Dvalued%20function))."

[Partial Trace](http://www.fmt.if.usp.br/~gtlandi/04---reduced-dm-2.pdf)

[Partial Trace Wiki](https://en.wikipedia.org/wiki/Partial_trace)

> "von Neumann entropy is a limiting case of the Rényi entropy lim α→1 Sα(ρ) = S(ρ) Given a family of entropies {Sα(ρ)}α, where α is some index, the entropies are monotonic in α∈ℝ" (see [here](https://en.wikipedia.org/wiki/Von_Neumann_entropy)).
[Quantum Entropies](http://www.scholarpedia.org/article/Quantum_entropies)

## Misc Articles

[MSFT Azure Article - very good Pauli measurement operations](https://docs.microsoft.com/en-us/azure/quantum/concepts-pauli-measurements)

[Area Law](https://www.nist.gov/system/files/documents/itl/math/slides_fernando_brandao.pdf)

[What is the difference between general measurement and projective measurement?](https://physics.stackexchange.com/questions/184524/what-is-the-difference-between-general-measurement-and-projective-measurement)

[Validating quantum-classical programming models with tensor network simulations](https://arxiv.org/abs/1807.07914)

[Universal Quantum Simulators](https://www.science.org/doi/10.1126/science.273.5278.1073)


# Step-By-Step Onboarding

```

```

# Help Options


# Software and Library Installation Help

## Julia
To install [Julia](https://julialang.org/) on Mac:
```
brew install --cask julia
brew update && brew upgrade julia
```

Add Julia to Path
> '/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia'
> ln -fs "/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia" /usr/local/bin/julia
> or, export PATH="$PATH:/path/to/<Julia directory>/bin" or ~/.bash_profile

[Julia Docs are here.](https://docs.julialang.org/en/v1/)

[Julia Intro](https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/getting-started)

[Embed Julia in Python](https://towardsdatascience.com/how-to-embed-your-julia-code-into-python-to-speed-up-performance-e3ff0a94b6e)

[How to call Julia code from Python](https://blog.esciencecenter.nl/how-to-call-julia-code-from-python-8589a56a98f2)

[AWS Lambda Maker for Julia](https://juliahub.com/ui/Packages/LambdaMaker/oGeH6/0.1.0)

[Pluto for interactive Julia Dashboards](https://github.com/fonsp/Pluto.jl)

[LambdaMaker.jl](https://juliahub.com/ui/Packages/LambdaMaker/oGeH6/0.1.0)

[Genie is a full-stack web framework that provides a streamlined and efficient workflow for developing modern web applications. It builds on Julia's strengths (high-level, high-performance, dynamic, JIT compiled), exposing a rich API and a powerful toolset for productive web development.](https://github.com/GenieFramework/Genie.jl)

[Deploying a Julia API with Genie](https://genieframework.github.io/Genie.jl/dev/guides/Simple_API_backend.html)

[Genie Documentation](https://geniejl.readthedocs.io/en/latest/)

[Graphs.jl](https://juliagraphs.org/Graphs.jl/stable/generators/)

## ITensors and PastaQ

- [ITensor Paper](https://arxiv.org/pdf/2007.14822.pdf)
- [PastaQ GitHub](https://github.com/GTorlai/PastaQ.jl)

Starter code to understand how to run simulations using ITensor and PastaQ was graciously provided [here](https://raw.githubusercontent.com/GTorlai/PastaQ.jl/master/examples/11_monitored_circuit.jl).

Install PastaQ:

```
julia> ]
pkg> add PastaQ
julia> import Pkg; Pkg.add("ITensors"); Pkg.add("StatsBase")
julia> Pkg.add(Pkg.PackageSpec(;name="PastaQ", version="0.0.18"))
```

After installing Itensor and PastaQ, you must run this `julia> using Pkg; Pkg.update()`.
Add to `~/.zshrc`:

```
export JULIA_NUM_THREADS=4
```

## Mac M1 Support

[Julia support for Apple Silicon](https://julialang.org/blog/2021/11/julia-1.7-highlights/#support_for_apple_silicon)

Tried M1 experimental and was running into PastaQ error:

> ERROR: LoadError: UndefVarError: libscsindir not defined

So then installed Julia  macOS x86 (Intel or Rosetta)

```
softwareupdate --install-rosetta
```

## Atom Editor

[juno-makes-writing-julia-awesome](https://towardsdatascience.com/juno-makes-writing-julia-awesome-f3e1baf92ea9)

[Juno Update](https://docs.junolab.org/stable/man/update/)
```
pkg> up Atom Juno

```

## Julia help docs
- https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention
- https://docs.julialang.org/en/v1/manual/control-flow/
- https://docs.julialang.org/en/v1/manual/variables-and-scoping/
- https://sodocumentation.net/julia-lang
- https://github.com/bkamins/Julia-DataFrames-Tutorial/
- https://www.ahsmart.com/pub/data-wrangling-with-data-frames-jl-cheat-sheet/
- https://www.reddit.com/r/Julia/comments/9p3ttr/clearing_workspace_atom/
[Julia Data Format like HDF5](https://github.com/JuliaIO/JLD.jl)

## Embed Julia into Python
- https://towardsdatascience.com/how-to-embed-your-julia-code-into-python-to-speed-up-performance-e3ff0a94b6e
- https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/interfacing-julia-with-other-languages

### Online help for Streamlit and Julia Integration
[StackOverflow](https://stackoverflow.com/questions/71726946/calling-julia-from-streamlit-app-using-pyjulia)

[Streamlit Issues](https://github.com/streamlit/streamlit/issues/4585)

[PyJulia Issues](https://github.com/JuliaPy/pyjulia/issues/492)

# AWS Deployment
[Using Streamlit to build an interactive dashboard for data analysis on AWS](https://aws.amazon.com/blogs/opensource/using-streamlit-to-build-an-interactive-dashboard-for-data-analysis-on-aws/)
