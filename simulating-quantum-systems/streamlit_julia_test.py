
import os, sys, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import streamlit as st
this_dir = os.getcwd()
repo_root_dir = this_dir.split("qc-repo")[0] + 'qc-repo/'

# works in python editor
import julia
from julia import Main
Main.include("julia_test.jl")

#from julia.api import Julia
#jl = Julia(compiled_modules=False)

#julia_test_path = """include(\""""+ this_dir + """/julia_test.jl\"""" +")"""
#jl.eval(julia_test_path)

st.header('Streamlit + Julia Example')

matrix_element = st.selectbox('Set Matrix Diagonal to:', [1,2,3])

matrix_numpy = np.array([[matrix_element,0],[0,matrix_element]])

fig, ax = plt.subplots()
sns.heatmap(matrix_numpy, ax = ax)
ax.set_title('Matrix Using Python Numpy')
st.write(fig)

matrix_julia = matrix_numpy #get_matrix(matrix_element)

fig, ax = plt.subplots()
sns.heatmap(matrix_julia, ax = ax)
ax.set_title('Matrix from External Julia Script')
st.write(fig)
