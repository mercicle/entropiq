
# Installation

To install [Julia](https://julialang.org/) on Mac:

> brew install --cask julia
> brew update && brew upgrade julia

[Julia Docs are here.](https://docs.julialang.org/en/v1/)

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

# Atom Editor

- https://towardsdatascience.com/juno-makes-writing-julia-awesome-f3e1baf92ea9

# Running on Hyak

[Hyak Julia Programming](https://wiki.cac.washington.edu/display/hyakusers/Hyak+Julia+programming)

# Links

UW RCC https://app.slack.com/client/T2TL8A63C/C2TL30RH7
https://wiki.cac.washington.edu/display/hyakusers/Hyak+Julia+programming
http://depts.washington.edu/uwrcc/getting-started-2/getting-started/
