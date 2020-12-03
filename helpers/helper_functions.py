#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mercicle
"""

import pandas as pd
import numpy as np

def bond_parser(in_mol2_lines):

    f_text = ''.join(in_mol2_lines)
    f_text = f_text.split('@<TRIPOS>BOND')[1]
    
    # check for more @<TRIPOS> and if exists then take the first element 
    
    has_more_sections = len(f_text.split('@<TRIPOS>')) > 1
    
    if has_more_sections:
        print("Found more sections after @<TRIPOS>BOND' )
        f_text = f_text.split('@<TRIPOS>')[0]
        
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

len('foo bar baz'.split('bar'))