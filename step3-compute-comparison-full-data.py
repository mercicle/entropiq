#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note: for this to run you must have the current working directory set in Spyder to the main repo root
@author: mercicle
"""

########################################
##          Import Libraries          ##
########################################

import os
import pandas as pd
import numpy as np
from io import StringIO
import time
import glob

from helpers.helper_functions import bond_parser

print('This should be the root directory of the repository: \n'+os.getcwd()+'\n Correct?')

from biopandas.mol2 import PandasMol2
from biopandas.mol2 import split_multimol2

########################################
##       Example Dataset Testing      ##
########################################

mol_path = './in-data/vegfr2/databases/dud_ligands2006/vegfr2_ligands.mol2'
pdmol = PandasMol2()

# https://github.com/rasbt/biopandas/issues/54
all_atoms_df = pd.DataFrame()
for mol2 in split_multimol2(mol_path):

    pdmol.read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])

    atoms_df = pdmol.df.copy()
    atoms_df['molecule'] = mol2[0]

    print(atoms_df.head(2))

    all_atoms_df = all_atoms_df.append(atoms_df)


all_atoms_df.head()
all_atoms_df.shape

all_bonds_df.to_csv('./out-data/all_molecule_graphs_df.csv')

##########################################
##  Now Iterate Through All Datasets    ##
##########################################

out_dir = os.getcwd()+'/out-data/'
extension = ".mol2"

mol_files_in_out_data = glob.glob(out_dir + "/**/*"+extension, recursive = True)
n_files = len(mol_files_in_out_data)
print('Total: '+str(n_files))

n_downsample_molecules = 10000
all_bonds_df = pd.DataFrame()
all_atoms_df = pd.DataFrame()
file_index = 1
start_time = time.time()
for mol_file in mol_files_in_out_data:

    molecule_type = ''
    if mol_file.find('decoys')!=-1:
        molecule_type = 'decoy'
    elif mol_file.find('ligands')!=-1:
        molecule_type = 'ligand'
    elif mol_file.find('targets')!=-1:
        molecule_type = 'target'

    print("Starting (type: "+molecule_type+"): " + str(file_index) + " of " + str(n_files))

    these_bonds_df = pd.DataFrame()
    these_atoms_df = pd.DataFrame()
    file_start_time = time.time()

    for mol2 in split_multimol2(mol_file):

        pdmol.read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])

        bonds_df = bond_parser(in_mol2_lines = mol2[1])
        bonds_df['molecule'] = mol2[0]

        temp_atom_df = pdmol.df.copy()
        temp_atom_df['molecule'] = mol2[0]
        these_atoms_df = these_atoms_df.append(temp_atom_df)
        these_bonds_df = these_bonds_df.append(bonds_df)

    these_atoms_df['molecule_type'] = molecule_type
    these_bonds_df['molecule_type'] = molecule_type

    all_bonds_df = all_bonds_df.append(these_bonds_df)
    all_atoms_df = all_atoms_df.append(these_atoms_df)

    file_end_time = time.time()
    file_total_seconds = np.round((file_end_time - file_start_time)/60,2)
    print("File prep time in {} minutes.".format(file_total_seconds))

    file_index+=1

    #all_atoms_df.molecule.value_counts()

    print("Bonds DF (n="+str(all_bonds_df.shape[0])+") Atoms DF (n="+str(all_atoms_df.shape[0])+").")
    n_current_molecules = len(all_atoms_df.molecule.unique())
    if n_current_molecules > n_downsample_molecules:
        print("Completed data prep and reached n=' + str(n_current_molecules)+' molecules.")
        break

end_time = time.time()
total_seconds = np.round((end_time - start_time)/60,2)
print("Ingest time in {} minutes.".format(total_seconds))

all_atoms_df.shape
all_bonds_df.shape

# save results 
all_bonds_df.to_csv('./out-data/all_bonds_df.csv')
all_atoms_df.to_csv('./out-data/all_atoms_df.csv')
