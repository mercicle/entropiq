#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mercicle
"""

import os, sys, glob, random, subprocess

# https://docs.python.org/3/library/importlib.html
from importlib import reload

import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Phylo, AlignIO


from pybioviz import plotters

from bokeh.io import show, output_notebook, output_file
output_notebook()

import pathogenie

from ete3 import Tree, NodeStyle, TreeStyle, PhyloTree



sequence_metadata_df = pd.read_csv('./in-data/sequences-metadata.csv')
print(len(sequence_metadata_df))
print(sequence_metadata_df.columns)


sequence_metadata_df['Release_Date'] = pd.to_datetime(sequence_metadata_df.Release_Date)

print(sequence_metadata_df.Host.value_counts()[:10])

#put sequences into a set of SeqRecord objects with Biopython
seqrecs = SeqIO.to_dict(SeqIO.parse('ncbi_betacoronavirus.fasta','fasta'))