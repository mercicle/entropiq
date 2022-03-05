
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

```

To update packages
```
julia> using Pkg; Pkg.update()
```

[Juno Update](https://docs.junolab.org/stable/man/update/)
```
pkg> up Atom Juno

```

# Running on Hyak

[Hyak Julia Programming](https://wiki.cac.washington.edu/display/hyakusers/Hyak+Julia+programming)

# Links

UW RCC https://app.slack.com/client/T2TL8A63C/C2TL30RH7
https://wiki.cac.washington.edu/display/hyakusers/Hyak+Julia+programming
http://depts.washington.edu/uwrcc/getting-started-2/getting-started/
