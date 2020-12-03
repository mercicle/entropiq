#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note: for this to run you must have the current working directory set in Spyder to the main repo root 
@author: mercicle
"""

#from biopandas.mol2 import PandasMol2
#from biopandas.mol2 import split_multimol2

import os
import pandas as pd
import numpy as np
from io import StringIO

from helpers.helper_functions import bond_parser

print('This should be the root directory of the repository: \n'+os.getcwd()+'\n Correct?')


from biopandas.mol2 import PandasMol2
from biopandas.mol2 import split_multimol2

#PandasMol2 read_mol2_from_list AttributeError: module 'pandas.compat' has no attribute 'Iterable'

#import pandas as pd
mol_path = './in-data/vegfr2/databases/dud_ligands2006/vegfr2_ligands.mol2'

pdmol = PandasMol2()

# https://github.com/rasbt/biopandas/issues/54

for mol2 in split_multimol2(mol_path):
    
    
    pdmol.read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])

    bonds_df = bond_parser(in_mol2_lines = mol2[1])
    bonds_df['molecule'] = mol2[0]
    
    print(bonds_df.head())

    
    
    

f_text = ''.join(mol2[1])
f_text = f_text.split('@<TRIPOS>BOND')[1]
row_chunks = f_text.split('\n')[1:]
df = pd.DataFrame()
for i in range(len(row_chunks)):
    # i = 1
    print('i = '+str(i)+ '  row:' +row_chunks[i] )
    edge_list = [x for x in row_chunks[i].split(" ") if x!='']
    if len(edge_list)>0:
        edge_list = edge_list[1:3]
        df = df.append(pd.DataFrame.from_dict({'source':[edge_list[0]], 'target': [edge_list[1]]}))
    
# put in function
def bond_parser(in_mol2_lines):

    f_text = ''.join(in_mol2_lines)
    f_text = f_text.split('@<TRIPOS>BOND')[1]
    row_chunks = f_text.split('\n')[1:]
    df = pd.DataFrame()
    for i in range(len(row_chunks)):
        # i = 1
        #print('i = '+str(i)+ '  row:' +row_chunks[i] )
        edge_list = [x for x in row_chunks[i].split(" ") if x!='']
        if len(edge_list)>0:
            edge_list = edge_list[1:3]
            df = df.append(pd.DataFrame.from_dict({'source':[edge_list[0]], 'target': [edge_list[1]]}))
    
    return df