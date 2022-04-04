import os
from io import BytesIO

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# options:
# main_include
# api_compiled_false
# dont_import_julia
julia_import_method = "main_include"

if julia_import_method == "main_include":

    # works in Spyder IDE
    import julia
    from julia import Main
    Main.include("julia_test.jl")

elif julia_import_method == "api_compiled_false":

    # works in Spyder IDE
    from julia.api import Julia
    jl = Julia(compiled_modules=False)

    this_dir = os.getcwd()
    julia_test_path = """include(\""""+ this_dir + """/julia_test.jl\"""" +")"""
    print(julia_test_path)
    jl.eval(julia_test_path)
    get_matrix_from_julia = jl.eval("get_matrix_from_julia")

elif julia_import_method == "dont_import_julia":
    print("Not importing ")
else:
    ValueError("Not handling this case:" + julia_import_method)


st.header('Using Julia in Streamlit App Example')

st.text("Using Method:" + julia_import_method)

matrix_element = st.selectbox('Set Matrix Diagonal to:', [1,2,3])

matrix_numpy = np.array([[matrix_element,0],[0,matrix_element]])

col1, col2 = st.columns([4,4])

with col1:

    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(matrix_numpy, ax = ax, cmap="Blues",annot=True)
    ax.set_title('Matrix Using Python Numpy')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

with col2:

    if julia_import_method == "dont_import_julia":
        matrix_julia = matrix_numpy
    else:
        matrix_julia = get_matrix_from_julia(matrix_element)

    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(matrix_julia, ax = ax, cmap="Blues",annot=True)
    ax.set_title('Matrix from External Julia Script')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)
