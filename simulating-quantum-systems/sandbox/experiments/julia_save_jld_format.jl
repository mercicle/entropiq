#import Pkg
#Pkg.add("JLD")
#Pkg.add("SparseArrays")

using JLD
using SparseArrays

a = SparseArrays.sparse(I, 2, 2)
save(string(@__DIR__, "/foo.jld"),"a",a)
