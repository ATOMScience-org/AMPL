# Example of curating DTC data using CYP3A4 
# Written by Ya Ju Fan (fan4@llnl.gov)
# 
# Simplified data curation steps for summer interns.
# Initial dataset downloaded from Drug Target Commons and stored here: Drug target commons 
# https://drugtargetcommons.fimm.fi/static/Excell_files/DTC_data.csv
# 
# Data location: /usr/workspace/atom/dtc
# Data file: cyp3a4.csv
# Note the "ACTION" in comments 
# Using data_curation_functions.py
# 

#import importlib as imp
#import sys
# ACTION: Specify the location of data_curation_functions.py 
#sys.path.append('/p/lustre3/fan4/data_science/code/')
#import data_curation_functions as dcf
import atomsci.ddm.utils.data_curation_functions as dcf

import atomsci.ddm.utils.curate_data as curate_data

import pandas as pd
import numpy as np
file='/usr/workspace/atom/dtc/cyp3a4.csv'

orig_df=pd.read_csv(file,sep=",",engine="python",error_bad_lines=False)
print(orig_df.shape)
print(orig_df.columns)

# ACTION: specify file location
# Save file to LC 
orig_df.to_csv('/usr/workspace/atom/public_dsets/DTC/raw/cyp3a4.csv',index=False)

# Check values
for v in orig_df['standard_units'].unique() :  
    t=orig_df[orig_df['standard_units']==v]
    print(v,t.shape)

### Specify geneNames
### Obtain unique standard_inchi_key

imp.reload(dcf)
geneNames = ['CYP3A4']
nm_df = dcf.filter_dtc_data(orig_df,geneNames)

print(nm_df.shape)
myList=nm_df['standard_inchi_key'].unique().tolist()

print(len(myList))


# Retrieve SMILES strings for compounds through PUBCHEM web interface.
# ACTION: TURN NEXT CELL TO TEXT TO AVOID RE-RUNNING (unintentionally)
'''
import imp
import atomsci.ddm.utils.pubchem_utils as pu
imp.reload(pu)
ofile='/p/lustre3/fan4/atom/pubddm/save_smiles_cyp3a4_nm_raw.csv'

## this is slow, so don't re-do if the SMILES are already downloaded
#if not -e ofile :
save_smiles_df,fail_lst,discard_lst=pu.download_smiles(myList)
save_smiles_df.to_csv(ofile)

print(len(fail_lst))
print(save_smiles_df.shape)
# 484
#(16670, 3)
'''

# ACTION: change file location
ifile='/p/lustre3/fan4/atom/pubddm/save_smiles_cyp3a4_nm_raw.csv'
save_smiles_df=pd.read_csv(ifile)
save_smiles_df.head()

## Retrieve specific CYP2D6 data
## Will include censored data in smiles 
## Combine gene data with SMILES strings and call this our starting "raw" dataset.
imp.reload(dcf)
targ_lst=['CYP3A4']
smiles_lst, shared_inchi_keys = dcf.get_smiles_dtc_data(nm_df,targ_lst,save_smiles_df)

smiles_df=pd.concat(smiles_lst)
print(smiles_df.shape)
print(smiles_df.head()

)
# Save file to LC 
smiles_df.to_csv('/usr/workspace/atom/public_dsets/DTC/raw/cyp2d6_dtc_smiles.csv',index=False)

# Apply ATOM standard 'curation' step: 
# Average replicate assays, remove duplicates and drop cases with large variance between replicates.
## Use aggregate_assay_dataB6
tolerance=10
column='PIC50'; #'standard_value'
list_bad_duplicates='No'
max_std=1

for it in range(len(targ_lst)) :
    data=smiles_lst[it]
    print("before",data.shape)
    
    temp_df=curate_data.aggregate_assay_data(data, value_col=column, output_value_col=None,
                             label_actives=True,
                             active_thresh=None,
                             id_col='standard_inchi_key', smiles_col='rdkit_smiles', relation_col='standard_relation')

    # (Yaru) Remove inf in curated_df
    temp_df = temp_df[~temp_df.isin([np.inf]).any(1)]
    #censored_curated_df = censored_curated_df[~censored_curated_df.isin([np.inf]).any(1)]

print(temp_df.head())

# Save to LC
temp_df.to_csv('/usr/workspace/atom/public_dsets/DTC/ml_ready/cyp3a4_dtc_base_smiles_all.csv')































