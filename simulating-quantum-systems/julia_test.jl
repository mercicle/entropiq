
#Pkg.add("Conda")
#using Conda
#Conda.rm("mkl")
#Conda.add("nomkl")

function get_matrix_from_julia(v::Int)
   m=[v 0
      0 v]
 return m
end
