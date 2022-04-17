using ITensors
using PastaQ

using Pkg
versioninfo()
Pkg.status("ITensors")
Pkg.status("PastaQ")

N = 2
ψ = productstate(N)
U = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
runcircuit(ψ, [(U, (1, 2))])
runcircuit(ψ, [(U, 1, 2)])
runcircuit([(U, (1, 2))])
runcircuit([(U, 1, 2)])
