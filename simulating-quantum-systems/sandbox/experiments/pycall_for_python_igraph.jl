
# https://www.juliabloggers.com/a-tutorial-on-igraph-for-julia/
#ENV["PYTHON"]="/usr/local/bin/python3"
#https://stackoverflow.com/questions/66938642/intel-mkl-fatal-error-cannot-load-libmkl-intel-thread-dylib-in-julia-pluto-not
using PyCall
run(`$(PyCall.python) -m pip install python-igraph`)
ig = pyimport("igraph");
z_ig = ig.Graph.Famous("Zachary")
