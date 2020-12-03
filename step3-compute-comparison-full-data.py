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
all_bonds_df = pd.DataFrame()
for mol2 in split_multimol2(mol_path):
    
    pdmol.read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])

    bonds_df = bond_parser(in_mol2_lines = mol2[1])
    bonds_df['molecule'] = mol2[0]

    print(bonds_df.head(2))
    
    all_bonds_df = all_bonds_df.append(bonds_df)
    

all_bonds_df.head()
all_bonds_df.shape

all_bonds_df.to_csv('./out-data/all_molecule_graphs_df.csv')