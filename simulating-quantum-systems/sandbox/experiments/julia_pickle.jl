using PyCall

py"""
import pickle

def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

load_pickle = py"load_pickle"
pickle_file_path = string(@__DIR__, "/clifford_operators.p")
clifford_dict = load_pickle(pickle_file_path)
