
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: mercicle
"""

##########################################
###### Install Python Libraries ##########
##########################################

# !pip install dwave-ocean-sdk networkx
# !pip install numpy --upgrade
# !pip install biopandas

# This is important pymol read_mol2_from_list will error without it.
# !pip install pandas==0.24
# !pip install git+git://github.com/rasbt/biopandas.git
# !pip install pandas-compat

# !pip install sh glob

##########################################
########### DWave Ocean API  #############
##########################################

# Step 1: From https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html#sapi-access, you have to run from terminal
# $ dwave config create
# and provide your API token and other info


################################################################
########## Download and Unzip DUD Folders and Files  ###########
################################################################

import os
import tarfile
import glob
from sh import gunzip

out_dir = os.getcwd()+'/out-data/'
dir_name = os.getcwd()+'/in-data/'
extension = ".gz"
os.chdir(dir_name) # change directory from working dir to dir with files

file_list = os.listdir(dir_name)
for item in file_list:
    # item = file_list[0]
    if item.endswith(extension):
        file_name = os.path.abspath(item)
        print('unzipping: '+file_name)
        tar = tarfile.open(file_name)
        new_dir = out_dir + item.replace('.tar.gz','')+'/'
        os.mkdir(new_dir)
        tar.extractall(new_dir)
        tar.close()

gz_files_in_out_data = glob.glob(out_dir + "/**/*.gz", recursive = True)

for item in gz_files_in_out_data:
    # item = gz_files_in_out_data[0]
    gunzip(item)
    print('done: '+item)
    
    
    