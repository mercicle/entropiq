# Papers

- [Main Paper - Quantum Zeno Effect and the Many-body Entanglement Transition](https://arxiv.org/pdf/1808.06134.pdf)

- [Entanglement area law in superfluid 4He](https://www.nature.com/articles/nphys4075)

- [The ‘Quantum Zeno Effect’ Explained](https://sites.imsa.edu/hadron/2019/12/05/the-quantum-zeno-effect-explained/)
- [Q: Is the quantum zeno effect a real thing?](https://www.askamathematician.com/2012/03/q-is-the-quantum-zeno-effect-a-real-thing/)

[Simulating Clifford's - Hadamard-free circuits expose the structure of the Clifford group](https://arxiv.org/abs/2003.09412)

The Quantum Zeno Effect, also known as the Turing paradox, is a feature of quantum-mechanical systems allowing a particle’s time evolution to be arrested by measuring it frequently enough with respect to some chosen measurement setting. Simply put, it is a phenomenon in quantum physics where observing a particle prevents it from decaying as it would in the absence of observation.

- [Quantum Zeno Effect](https://addpmp.slamjam.com/index/quantum-zeno-effect)

- [Felix Pollock - The quantum Zeno effect: how curiosity can save Schrodinger's cat](https://www.youtube.com/watch?v=3WHPiH0pNeo)

- [Partial Trace](http://www.fmt.if.usp.br/~gtlandi/04---reduced-dm-2.pdf)
- [Partial Trace Wiki](https://en.wikipedia.org/wiki/Partial_trace)
- [Quantum Tensor Networks in a Nutshell](https://arxiv.org/pdf/1708.00006.pdf)
- [Measurement Protected Quantum Phases](https://arxiv.org/pdf/2004.09509.pdf)

[Effect of barren plateaus on gradient-free optimization](https://arxiv.org/abs/2011.12245)

[Quantum Algorithm Design - Classiq](https://global-uploads.webflow.com/60000db7a5f449af5e4590ac/6114676a4005b87b75c9653c_What%20is%20quantum%20algorithm%20design%20note.pdf)

[MPS Examples](https://physics.stackexchange.com/questions/266587/examples-of-matrix-product-states)

[The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477)

[Hand-waving and Interpretive Dance: An Introductory Course on Tensor Networks](https://arxiv.org/pdf/1603.03039.pdf)

[Matrix Product State Based Algorithms for Ground States and Dynamics](https://qdev.nbi.ku.dk/student_theses/RGawatz_Msc.pdf)


# Topics

[This site is a resource for tensor network algorithms, theory, and software.](https://tensornetwork.org/)
- [MPO Matrix Product Operator](https://tensornetwork.org/mpo/#:~:text=A%20matrix%20product%20operator%20(MPO,in%20a%20chain%2Dlike%20fashion.)
[DMRG - Density matrix renormalization group](https://en.wikipedia.org/wiki/Density_matrix_renormalization_group) which is a [variational method](https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics)).

[Garnet Chan "Matrix product states, DMRG, and tensor networks" (Part 1 of 2)](https://www.youtube.com/watch?v=Q8bFmV6tHBs)

[Garnet Chan "Matrix product states, DMRG, and tensor networks" (Part 2 of 2)](https://www.youtube.com/watch?v=s37tvvrjlto)

[Area Law](https://www.nist.gov/system/files/documents/itl/math/slides_fernando_brandao.pdf)

[What is the difference between general measurement and projective measurement?](https://physics.stackexchange.com/questions/184524/what-is-the-difference-between-general-measurement-and-projective-measurement)

[SPIN ONE-HALF, BRAS, KETS, AND OPERATORS](https://ocw.mit.edu/courses/physics/8-05-quantum-physics-ii-fall-2013/lecture-notes/MIT8_05F13_Chap_02.pdf)


[A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States](https://arxiv.org/pdf/1306.2164.pdf)


# Installation


## Python

[Right way to setup python on mac](https://faun.pub/the-right-way-to-set-up-python-on-your-mac-e923ffe8cf8e)

[Intro to pyenv](https://realpython.com/intro-to-pyenv/)
brew install pyenv
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9.7
otool -L ~/.pyenv/versions/3.9.7/bin/python3.9 | grep libpython
pip3 install virtualenv
brew install pyenv-virtualenv
pyenv virtualenv 3.9.7 pyforjulia

pyenv local pyforjulia
pyenv activate pyforjulia

- to exit venv
  - `pyenv local system`

To install [Julia](https://julialang.org/) on Mac:

> brew install --cask julia
> brew update && brew upgrade julia

## Add Julia to Path
- '/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia'
- ln -fs "/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia" /usr/local/bin/julia
- or, export PATH="$PATH:/path/to/<Julia directory>/bin" or ~/.bash_profile

[Julia Docs are here.](https://docs.julialang.org/en/v1/)
[Julia Intro](https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/getting-started)

[Embed Julia in Python](https://towardsdatascience.com/how-to-embed-your-julia-code-into-python-to-speed-up-performance-e3ff0a94b6e)

[How to call Julia code from Python
](https://blog.esciencecenter.nl/how-to-call-julia-code-from-python-8589a56a98f2)

[AWS Lambda Maker for Julia](https://juliahub.com/ui/Packages/LambdaMaker/oGeH6/0.1.0)

To embed Julia in Python, need to [install PyJulia in Python](https://pyjulia.readthedocs.io/en/latest/installation.html#step-2-install-pyjulia):

```
!pip3 install julia
julia.install() #dependencies
```

```
>>> import julia
>>> julia.install()
[ Info: Julia version info
Julia Version 1.7.2
Commit bf53498635 (2022-02-06 15:21 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin19.5.0)
  uname: Darwin 21.3.0 Darwin Kernel Version 21.3.0: Wed Jan  5 21:37:58 PST 2022; root:xnu-8019.80.24~20/RELEASE_ARM64_T6000 x86_64 i386
  CPU: Apple M1 Pro:
                 speed         user         nice          sys         idle          irq
       #1-10    24 MHz    8778444 s          0 s    2046948 s   11265070 s          0 s

  Memory: 32.0 GB (386.67578125 MB free)
  Uptime: 1.92383e6 sec
  Load Avg:  8.1513671875  8.60498046875  9.21142578125
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, westmere)
Environment:
  JULIA_NUM_THREADS = 4
  XPC_FLAGS = 0x0
  TERM = xterm-256color
  HOME = /Users/mercicle
  PATH = /opt/homebrew/Cellar/pyenv-virtualenv/1.1.5/shims:/Users/mercicle/opt/anaconda3/bin:/Users/mercicle/opt/anaconda3/condabin:/Library/Frameworks/Python.framework/Versions/3.10/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/mercicle/opt/anaconda3/bin/python3:/Users/mercicle/.local/bin
  HOMEBREW_PREFIX = /opt/homebrew
  HOMEBREW_CELLAR = /opt/homebrew/Cellar
  HOMEBREW_REPOSITORY = /opt/homebrew
  MANPATH = /opt/homebrew/share/man::
  INFOPATH = /opt/homebrew/share/info:
  PYTHONPATH = /Users/mercicle/opt/anaconda3/bin/python3
[ Info: Julia executable: /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia
[ Info: Trying to import PyCall...
┌ Info: PyCall is already installed and compatible with Python executable.
│
│ PyCall:
│     python: /Users/mercicle/opt/anaconda3/bin/python3
│     libpython: /Users/mercicle/opt/anaconda3/lib/libpython3.9.dylib
│ Python:
│     python: /Users/mercicle/opt/anaconda3/bin/python3
└     libpython:
>>> exit()

```
[Pluto for interactive Julia Dashboards](https://github.com/fonsp/Pluto.jl)

[LambdaMaker.jl](https://juliahub.com/ui/Packages/LambdaMaker/oGeH6/0.1.0)

[Genie is a full-stack web framework that provides a streamlined and efficient workflow for developing modern web applications. It builds on Julia's strengths (high-level, high-performance, dynamic, JIT compiled), exposing a rich API and a powerful toolset for productive web development.](https://github.com/GenieFramework/Genie.jl)

[Deploying a Julia API with Genie](https://genieframework.github.io/Genie.jl/dev/guides/Simple_API_backend.html)

[Genie Documentation](https://geniejl.readthedocs.io/en/latest/)

[Graphs.jl](https://juliagraphs.org/Graphs.jl/stable/generators/)


Install PastaQ:

```
julia> ]

pkg> add PastaQ

julia> import Pkg; Pkg.add("ITensors"); Pkg.add("StatsBase")

julia> Pkg.add(Pkg.PackageSpec(;name="PastaQ", version="0.0.18"))
```

After installing Itensor and PastaQ, you must run this
```
julia> using Pkg; Pkg.update()

julia> using Pkg; Pkg.update()
    Updating registry at `~/.julia/registries/General.toml`
    Updating `~/.julia/environments/v1.7/Project.toml`
  [30b07047] ↑ PastaQ v0.0.18 ⇒ v0.0.19
    Updating `~/.julia/environments/v1.7/Manifest.toml`
  [6e4b80f9] ↑ BenchmarkTools v0.4.3 ⇒ v1.3.1
  [523fee87] + CodecBzip2 v0.7.2
  [944b1d66] + CodecZlib v0.7.0
  [7d188eb4] + JSONSchema v0.3.4
  [b8f27783] ↑ MathOptInterface v0.9.8 ⇒ v0.9.22
  [d8a4904e] ↑ MutableArithmetics v0.1.1 ⇒ v0.2.22
  [30b07047] ↑ PastaQ v0.0.18 ⇒ v0.0.19
  [6e34b625] + Bzip2_jll v1.0.8+0
Precompiling project...
  5 dependencies successfully precompiled in 37 seconds (132 already precompiled, 1 skipped during auto due to previous errors)

```

Add to `~/.zshrc`:

```
export JULIA_NUM_THREADS=4
```

## Mac M1 Support

- https://julialang.org/blog/2021/11/julia-1.7-highlights/#support_for_apple_silicon

Tried M1 experimental and was running into PastaQ error:

> ERROR: LoadError: UndefVarError: libscsindir not defined

So then installed Julia  macOS x86 (Intel or Rosetta)

```
softwareupdate --install-rosetta
```

[Juno Update](https://docs.junolab.org/stable/man/update/)
```
pkg> up Atom Juno

```

## CSV still doesn't precompile
```
(@v1.7) pkg>  up Parsers
    Updating registry at `~/.julia/registries/General.toml`
  No Changes to `~/.julia/environments/v1.7/Project.toml`
    Updating `~/.julia/environments/v1.7/Manifest.toml`
  [6e4b80f9] ↓ BenchmarkTools v1.3.1 ⇒ v0.4.3
  [523fee87] - CodecBzip2 v0.7.2
  [7d188eb4] - JSONSchema v0.3.4
  [b8f27783] ↓ MathOptInterface v0.9.22 ⇒ v0.9.8
  [d8a4904e] ↓ MutableArithmetics v0.2.22 ⇒ v0.1.1
  [6e34b625] - Bzip2_jll v1.0.8+0
Precompiling project...
  Progress [===================================>     ]  6/7
  ✓ BenchmarkTools
  ✗ MutableArithmetics
  ◓ CSV
  ✗ MathOptInterface
  ✗ Convex
```

# Install Mongo Community Edition Local

From https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/:

```
brew tap mongodb/brew
brew install mongodb-community@5.0

#to run in background
mongod --config /opt/homebrew/etc/mongod.conf --fork

```
# Atom Editor

- https://towardsdatascience.com/juno-makes-writing-julia-awesome-f3e1baf92ea9

# Running on Hyak

[Hyak Julia Programming](https://wiki.cac.washington.edu/display/hyakusers/Hyak+Julia+programming)

# Links

UW RCC https://app.slack.com/client/T2TL8A63C/C2TL30RH7
https://wiki.cac.washington.edu/display/hyakusers/Hyak+Julia+programming
http://depts.washington.edu/uwrcc/getting-started-2/getting-started/



# Setup

```
conda create -n qiskit_dev python=3
conda activate qiskit_dev
pip3 install qiskit[visualization] qiskit-machine-learning numpy matplotlib

```

Verify with
```
conda list
```

# Googling

https://qiskit.org/documentation/apidoc/circuit_library.html
https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html#multi-qubit-gates
https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html
https://quantumcomputing.stackexchange.com/questions/4975/how-do-i-build-a-gate-from-a-matrix-on-qiskit
https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_unitary.html

appending gates
https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html

https://qiskit.org/documentation/stubs/qiskit.quantum_info.Clifford.html
https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_clifford.html
https://quantumcomputing.stackexchange.com/questions/14056/what-is-the-clifford-gates-selection-probability-distribution-used-in-the-genera

https://quantumcomputing.stackexchange.com/questions/15868/applying-a-projector-to-a-qubit-in-a-qiskit-circuit

https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.reset.html


#  Error "too many subscripts in einsum" when system size > 10
# https://quantumcomputing.stackexchange.com/questions/16753/error-too-many-subscripts-in-einsum-unitarygate

# quantumcomputing.stackexchange.com
# https://quantumcomputing.stackexchange.com/questions/24044/qiskit-densitymatrix-from-instruction-when-snapshots-are-present/24046#24046

# https://qiskit.org/documentation/stubs/qiskit.quantum_info.Statevector.probabilities.html

# TeNPy

[TeNPy Toric Code](https://tenpy.readthedocs.io/en/latest/notebooks/11_toric_code.html)

[Q&A Board TeNPy](https://tenpy.johannes-hauschild.de/viewtopic.php?t=5)

[MPO Model](https://tenpy.readthedocs.io/en/latest/reference/tenpy.models.model.MPOModel.html)

[The time-evolution block-decimation (TEBD)
](https://tensornetwork.org/mps/algorithms/timeevo/tebd.html)

# Qiskit

[Qiskit MPS](https://qiskit.org/documentation/stable/0.24/tutorials/simulators/7_matrix_product_state_method.html)

[Qiskit Density Matrix](https://qiskit.org/textbook/ch-quantum-hardware/density-matrix.html)

[Qiskit Dynamics](https://medium.com/qiskit/introducing-qiskit-dynamics-a-new-qiskit-module-for-simulating-quantum-systems-afe004f5b92b)

# Julia Packages

- [ITensor](https://arxiv.org/pdf/2007.14822.pdf)
- [PastaQ](https://github.com/GTorlai/PastaQ.jl)

## Starter code was graciously added by ITensor and PastaQ maintainers
- https://raw.githubusercontent.com/GTorlai/PastaQ.jl/master/examples/11_monitored_circuit.jl

## Julia help docs
- https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention
- https://docs.julialang.org/en/v1/manual/control-flow/
- https://docs.julialang.org/en/v1/manual/variables-and-scoping/
- https://sodocumentation.net/julia-lang

- `ρ = prime(ϕ, tags = "Site")`
  - https://www.itensor.org/docs.cgi?vers=cppv2&page=tutorials/primes

# DataFrames
- https://github.com/bkamins/Julia-DataFrames-Tutorial/
- https://www.ahsmart.com/pub/data-wrangling-with-data-frames-jl-cheat-sheet/

- https://www.reddit.com/r/Julia/comments/9p3ttr/clearing_workspace_atom/

## Embed Julia into Python
- https://towardsdatascience.com/how-to-embed-your-julia-code-into-python-to-speed-up-performance-e3ff0a94b6e
- https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/interfacing-julia-with-other-languages

## Julia Saving Data

- [Julia Data Format like HDF5](https://github.com/JuliaIO/JLD.jl)

# Online help for Streamlit and Julia Integration

[StackOverflow](https://stackoverflow.com/questions/71726946/calling-julia-from-streamlit-app-using-pyjulia)
[Streamlit Issues](https://github.com/streamlit/streamlit/issues/4585)
[PyJulia Issues](https://github.com/JuliaPy/pyjulia/issues/492)
