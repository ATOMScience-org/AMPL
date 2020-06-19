"""
data_curation_functions.py

Extract Kevin's functions for curation of public datasets
Modify them to match Jonathan's curation methods in notebook
01/30/2020
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns
import pdb

from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles
import atomsci.ddm.utils.datastore_functions as dsf
#from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.utils import curate_data as curate
import atomsci.ddm.utils.struct_utils as struct_utils
import atomsci.ddm.utils.curate_data as curate_data, imp

def set_data_root(dir):
    global data_root, data_dirs
    data_root = dir
    #data_dirs = dict(ChEMBL = '%s/ChEMBL' % data_root, DTC = '%s/DTC' % data_root, 
    #                 Excape = '%s/Excape' % data_root)
    data_dirs = dict(DTC = '%s/dtc' % data_root, 
                     Excape = '%s/excape' % data_root)


log_var_map = {
    'IC50': 'pIC50',
    'AC50': 'pIC50',
    'Solubility': 'logSolubility',
    'CL': 'logCL'
}

pub_dsets = dict(
    CYP2D6 = dict(IC50='cyp2d6'),
    CYP3A4 = dict(IC50='cyp3a4'),
    JAK1 = dict(IC50="jak1"),
    JAK2 = dict(IC50="jak2"),
    JAK3 = dict(IC50="jak3"),
)


# ----------------------------------------------------------------------------------------------------------------------
# Generic functions for all datasets
# ----------------------------------------------------------------------------------------------------------------------

# Note: Functions freq_table and labeled_freq_table have been moved to ddm.utils.curate_data module.

# ----------------------------------------------------------------------------------------------------------------------
def standardize_relations(dset_df, db='DTC'):
    """
    Standardize the censoring operators to =, < or >, and remove any rows whose operators
    don't map to a standard one.
    """
    relation_cols = dict(ChEMBL='Standard Relation', DTC='standard_relation')
    rel_col = relation_cols[db]

    dset_df[rel_col].fillna('=', inplace=True)
    ops = dset_df[rel_col].values
    if db == 'ChEMBL':
        # Remove annoying quotes around operators
        ops = [op.lstrip("'").rstrip("'") for op in ops]
    op_dict = {
        ">": ">",
        ">=": ">",
        "<": "<",
        "<=": "<",
        "=": "="
    }
    ops = np.array([op_dict.get(op, "@") for op in ops])
    dset_df[rel_col] = ops
    dset_df = dset_df[dset_df[rel_col] != "@"]
    return dset_df


# ----------------------------------------------------------------------------------------------------------------------
# DTC-specific curation functions
# ----------------------------------------------------------------------------------------------------------------------
"""
Upload a raw dataset to the datastore from the given data frame. 
Returns the datastore OID of the uploaded dataset.
"""
def upload_file_dtc_raw_data(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,file_path,
			   data_origin='journal',  species='human',  
                           force_update=False):

    bucket = 'public'
    filename = '%s.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category':assay_category, 
        'assay_endpoint' : 'multiple values',
        'curation_level': 'raw',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://doi.org/10.1016/j.chembiol.2017.11.009',
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : target, 
        'target_type' : target_type,
        'id_col' : 'compound_id'
     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        #uploaded_file = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
        #                               description=description,
        #                               tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
        #                               override_check=True, return_metadata=True)
        uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid
'''



# ----------------------------------------------------------------------------------------------------------------------
def get_dtc_jak_smiles():
    """
    Use PubChem REST API to download SMILES strings for InChi strings in DTC JAK123 data table
    """
    jak_file = "%s/jak123_dtc.csv" % data_dirs['DTC']
    dset_df = pd.read_csv(jak_file, index_col=False)
    jak_dtc_df = jak_dtc_df[~jak_dtc_df.standard_inchi_key.isna()]
    inchi_keys = sorted(set(jak_dtc_df.standard_inchi_key.values))
    smiles_df, fail_list, discard_list = pu.download_smiles(inchi_keys)
    smiles_df.to_csv('%s/jak123_inchi_smiles.csv' % data_dirs['DTC'], index=False)

# ----------------------------------------------------------------------------------------------------------------------
'''
def filter_dtc_data(orig_df,geneNames):
    """
    Extract JAK1, 2 and 3 datasets from Drug Target Commons database, filtered for data usability.
    """
    # filter criteria:
    #   gene_names == JAK1 | JAK2 | JAK3
    #   InChi key not missing
    #   standard_type IC50
    #   units NM
    #   standard_relation mappable to =, < or >
    #   wildtype_or_mutant != 'mutated'
    #   valid SMILES
    #   maps to valid RDKit base SMILES
    #   standard_value not missing
    #   pIC50 > 3
    #--------------------------------------------------
    # Filter dataset on existing columns
    dset_df = orig_df[orig_df.gene_names.isin(geneNames) &
                      ~(orig_df.standard_inchi_key.isna()) &
                      (orig_df.standard_type == 'IC50') &
                      (orig_df.standard_units == 'NM') &
                      ~orig_df.standard_value.isna() &
                      ~orig_df.compound_id.isna() &
                      (orig_df.wildtype_or_mutant != 'mutated') ]
    return dset_df

def ic50topic50(x) :
    return -np.log10((x/1000000000.0))


def down_select(df,kv_lst) :
    for k,v in kv_lst :
        df=df[df[k]==v] 
    return df


def get_smiles_dtc_data(nm_df,targ_lst,save_smiles_df):

        save_df={}

        for targ in targ_lst :
            lst1= [ ('gene_names',targ),('standard_type','IC50'),('standard_relation','=') ]
            lst1_tmp= [ ('gene_names',targ),('standard_type','IC50')]
            jak1_df=down_select(nm_df,lst1)
            jak1_df_tmp=down_select(nm_df,lst1_tmp)
            print(targ,"distinct compounds = only",jak1_df['standard_inchi_key'].nunique())
            print(targ,"distinct compounds <,>,=",jak1_df_tmp['standard_inchi_key'].nunique())
            #to ignore censored data
            #save_df[targ]=jak1_df
            #to include censored data
            save_df[targ]=jak1_df_tmp

        prev_targ=targ_lst[0]
        shared_inchi_keys=save_df[prev_targ]['standard_inchi_key']
        for it in range(1,len(targ_lst),1) :
            curr_targ=targ_lst[it]
            df=save_df[curr_targ]
            shared_inchi_keys=df[df['standard_inchi_key'].isin(shared_inchi_keys)]['standard_inchi_key']

        print("num shared compounds",shared_inchi_keys.nunique())
        lst=[]
        for targ in targ_lst :
            df=save_df[targ]
            #print(aurka_df.shape,aurkb_df.shape, shared_inchi_keys.shape)
            lst.append(df[df['standard_inchi_key'].isin(shared_inchi_keys)])
            
        shared_df=pd.concat(lst)
        # Add pIC50 values
        print('Add pIC50 values.')
        shared_df['PIC50']=shared_df['standard_value'].apply(ic50topic50)

        # Merge in SMILES strings
        print('Merge in SMILES strings.')
        smiles_lst=[]
        for targ in targ_lst :
            df=save_df[targ]
            df['PIC50']=df['standard_value'].apply(ic50topic50)
            smiles_df=df.merge(save_smiles_df,on='standard_inchi_key',suffixes=('_'+targ,'_'))
            #the file puts the SMILES string in quotes, which need to be removed
            smiles_df['smiles']=smiles_df['smiles'].str.replace('"','')
            smiles_df['rdkit_smiles']=smiles_df['smiles'].apply(struct_utils.base_smiles_from_smiles)
            smiles_df['smiles']=smiles_df['smiles'].str.replace('"','')
            print(smiles_df.shape)
            print(smiles_df['standard_inchi_key'].nunique())
            smiles_lst.append(smiles_df)


        return smiles_lst, shared_inchi_keys


def get_smiles_4dtc_data(nm_df,targ_lst,save_smiles_df):

	save_df={}
	description_str = "" 
	for targ in targ_lst :
	    lst1= [ ('gene_names',targ),('standard_type','IC50'),('standard_relation','=') ]
	    lst1_tmp= [ ('gene_names',targ),('standard_type','IC50')]
	    jak1_df=down_select(nm_df,lst1)
	    jak1_df_tmp=down_select(nm_df,lst1_tmp)
	    print(targ,"distinct compounds = only",jak1_df['standard_inchi_key'].nunique())
	    print(targ,"distinct compounds <,>,=",jak1_df_tmp['standard_inchi_key'].nunique())
	    description = '''
# '''+targ+" distinct compounds = only: "+str(jak1_df['standard_inchi_key'].nunique())+'''
# '''+targ+" distinct compounds <,>,=: "+str(jak1_df_tmp['standard_inchi_key'].nunique())
	    description_str += description
            #to ignore censored data
	    #save_df[targ]=jak1_df
	    #to include censored data
	    save_df[targ]=jak1_df_tmp

	prev_targ=targ_lst[0]
	shared_inchi_keys=save_df[prev_targ]['standard_inchi_key']
	for it in range(1,len(targ_lst),1) :
	    curr_targ=targ_lst[it]
	    df=save_df[curr_targ]
	    shared_inchi_keys=df[df['standard_inchi_key'].isin(shared_inchi_keys)]['standard_inchi_key']

	print("num shared compounds",shared_inchi_keys.nunique())
	lst=[]
	for targ in targ_lst :
	    df=save_df[targ]
	    #print(aurka_df.shape,aurkb_df.shape, shared_inchi_keys.shape)
	    lst.append(df[df['standard_inchi_key'].isin(shared_inchi_keys)])
	    
	shared_df=pd.concat(lst)
	# Add pIC50 values
	print('Add pIC50 values.')
	shared_df['PIC50']=shared_df['standard_value'].apply(ic50topic50)

	# Merge in SMILES strings
	print('Merge in SMILES strings.')
	smiles_lst=[]
	for targ in targ_lst :
	    df=save_df[targ]
	    df['PIC50']=df['standard_value'].apply(ic50topic50)
	    smiles_df=df.merge(save_smiles_df,on='standard_inchi_key',suffixes=('_'+targ,'_'))
	    #the file puts the SMILES string in quotes, which need to be removed
	    smiles_df['smiles']=smiles_df['smiles'].str.replace('"','')
	    smiles_df['rdkit_smiles']=smiles_df['smiles'].apply(struct_utils.base_smiles_from_smiles)
	    smiles_df['smiles']=smiles_df['smiles'].str.replace('"','')
	    print("Shape of dataframe:", smiles_df.shape)
	    print("Number of unique standard_inchi_key:", smiles_df['standard_inchi_key'].nunique())
	    smiles_lst.append(smiles_df)


	return smiles_lst, shared_inchi_keys, description_str



	'''
	# Standardize the relational operators
	dset_df = standardize_relations(dset_df, 'DTC')

	# Map the InChi keys to SMILES strings. Remove rows that don't map.
	smiles_file = "%s/jak123_inchi_smiles.csv" % data_dirs['DTC']
	smiles_df = pd.read_csv(smiles_file, index_col=False)[['standard_inchi_key', 'smiles']]
	smiles_df['smiles'] = [s.lstrip('"').rstrip('"') for s in smiles_df.smiles.values]
	dset_df = dset_df.merge(smiles_df, how='left', on='standard_inchi_key')
	dset_df = dset_df[~dset_df.smiles.isna()]

	# Add standardized desalted RDKit SMILES strings
	dset_df['rdkit_smiles'] = [base_smiles_from_smiles(s) for s in dset_df.smiles.values]
	dset_df = dset_df[dset_df.rdkit_smiles != '']

	# Add pIC50 values and filter on them
	dset_df['pIC50'] = 9.0 - np.log10(dset_df.standard_value.values)
	dset_df = dset_df[dset_df.pIC50 >= 3.0]

	# Add censoring relations for pIC50 values
	rels = dset_df['standard_relation'].values
	log_rels = rels.copy()
	log_rels[rels == '<'] = '>'
	log_rels[rels == '>'] = '<'
	dset_df['pIC50_relation'] = log_rels

	# Split into separate datasets by gene name
	curated_dir = "%s/curated" % data_dirs['DTC']
	os.makedirs(curated_dir, exist_ok=True)
	for gene in jak_genes:
	gene_dset_df = dset_df[dset_df.gene_names == gene]
	gene_dset_file = "%s/%s_DTC_curated.csv" % (curated_dir, gene)
	gene_dset_df.to_csv(gene_dset_file, index=False)
	print("Wrote file %s" % gene_dset_file)
	'''

def upload_df_dtc_smiles(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,smiles_df,orig_fileID,
			   data_origin='journal',  species='human',  
                           force_update=False):

    bucket = 'public'
    filename = '%s_dtc_smiles.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category': assay_category,  ## seems like this should be called 'kinase_activity'
        'assay_endpoint' : 'pic50',
        'curation_level': 'raw',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://doi.org/10.1016/j.chembiol.2017.11.009',
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : target,
        'target_type' : target_type,
        'id_col' : 'compound_id',
        'source_file_id' : orig_fileID

     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=smiles_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)
        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid


# Apply ATOM standard 'curation' step to "shared_df": Average replicate assays, remove duplicates and drop cases with large variance between replicates.
# mleqonly
def atom_curation(targ_lst, smiles_lst, shared_inchi_keys):
	
	imp.reload(curate_data)
	tolerance=10
	column='PIC50'; #'standard_value'
	list_bad_duplicates='No'
	max_std=1
	curated_lst=[]
	num_dropped_lst=[]
	#print(targ_lst)
	#print(smiles_lst)
	for it in range(len(targ_lst)) :
		data=smiles_lst[it]
		data = data[data.standard_relation.str.strip() == '=']
		print("gene_names",data.gene_names.unique())
		print("standard_type",data.standard_type.unique())
		print("standard_relation",data.standard_relation.unique())
		print("before",data.shape)
		curated_df=curate_data.average_and_remove_duplicates (column, tolerance, list_bad_duplicates, data, max_std, compound_id='standard_inchi_key',smiles_col='rdkit_smiles')

		# (Yaru) Remove inf in curated_df
		curated_df = curated_df[~curated_df.isin([np.inf]).any(1)]
		# (Yaru) Remove nan on rdkit_smiles
		curated_df = curated_df.dropna(subset=['rdkit_smiles'])

		curated_lst.append(curated_df)
		prev_cmpd_cnt=shared_inchi_keys.nunique()
		num_dropped=prev_cmpd_cnt-curated_df.shape[0]
		num_dropped_lst.append(num_dropped)
		print("After",curated_df.shape, "# of dropped compounds",num_dropped)

	return curated_lst,num_dropped_lst

# Use Kevin's "aggregate_assay_data()" to remove duplicates and generate base rdkit smiles
def aggregate_assay(targ_lst, smiles_lst):
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

	return temp_df





def upload_df_dtc_mleqonly(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,data_df,dtc_smiles_fileID,
			   data_origin='journal',  species='human',  
                           force_update=False):

    bucket = 'public'
    filename = '%s_dtc_mleqonly.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category': assay_category,  ## seems like this should be called 'kinase_activity'
        'assay_endpoint' : 'pic50',
        'curation_level': 'ml_ready',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://doi.org/10.1016/j.chembiol.2017.11.009',
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : 'CYP2D6',
        'target_type' : target_type,
        'id_col' : 'compound_id',
        'response_col' : 'VALUE_NUM_mean',
        'prediction_type' : 'regression',
        'smiles_col' : 'rdkit_smiles',
        'units' : 'unitless',
        'source_file_id' : dtc_smiles_fileID
     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid


def upload_df_dtc_mleqonly_class(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,data_df,dtc_mleqonly_fileID,
			   data_origin='journal',  species='human',  
                           force_update=False):
    bucket = 'public'
    filename = '%s_dtc_mleqonly_class.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category': assay_category,  ## seems like this should be called 'kinase_activity'
        'assay_endpoint' : 'pic50',
        'curation_level': 'ml_ready',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://doi.org/10.1016/j.chembiol.2017.11.009',
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : target,
        'target_type' : target_type,
        'id_col' : 'compound_id',
        'response_col' : 'binary_class',
        'prediction_type' : 'classification',
	'num_classes' : 2,
	'class_names' : ['inactive','active'],
        'smiles_col' : 'rdkit_smiles',
        'units' : 'unitless',
        'source_file_id' : dtc_mleqonly_fileID
     }
    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid



def upload_df_dtc_base_smiles_all(dset_name, title, description, tags,
                            functional_area,
                           target, target_type, activity, assay_category,data_df,dtc_mleqonly_fileID,
                           data_origin='journal',  species='human',
                           force_update=False):
    bucket = 'public'
    filename = '%s_dtc_base_smiles_all.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category': assay_category,  ## seems like this should be called 'kinase_activity'
        'assay_endpoint' : 'pic50',
        'curation_level': 'ml_ready',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://doi.org/10.1016/j.chembiol.2017.11.009',
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : target,
        'target_type' : target_type,
        'id_col' : 'compound_id',
        'response_col' : 'PIC50',
        'prediction_type' : 'regression',
        'smiles_col' : 'base_rdkit_smiles',
        'units' : 'unitless',
        'source_file_id' : dtc_mleqonly_fileID
     }
    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid

def upload_file_dtc_smiles_regr_all(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,file_path,dtc_smiles_fileID,
			smiles_column,  data_origin='journal',  species='human',  
                           force_update=False):

    bucket = 'public'
    filename = '%s_dtc_smiles_regr_all.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category': assay_category,  ## seems like this should be called 'kinase_activity'
        'assay_endpoint' : 'pic50',
        'curation_level': 'ml_ready',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://doi.org/10.1016/j.chembiol.2017.11.009',
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : target,
        'target_type' : target_type,
        'id_col' : 'compound_id',
        'response_col' : 'PIC50',
        'prediction_type' : 'regression',
        'smiles_col' : smiles_column,
        'units' : 'unitless',
        'source_file_id' : dtc_smiles_fileID
     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        #uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid

def upload_df_dtc_smiles_regr_all_class(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,data_df,dtc_smiles_regr_all_fileID,
			   smiles_column, data_origin='journal',  species='human',  
                           force_update=False):

    bucket = 'public'
    filename = '%s_dtc_smiles_regr_all_class.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category': assay_category,  ## seems like this should be called 'kinase_activity'
        'assay_endpoint' : 'pic50',
        'curation_level': 'ml_ready',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://doi.org/10.1016/j.chembiol.2017.11.009',
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : target,
        'target_type' : target_type,
        'id_col' : 'compound_id',
        'response_col' : 'PIC50',
        'prediction_type' : 'classification',
	'num_classes' : 2,
        'smiles_col' : smiles_column,
	'class_names' : ['inactive','active'],
        'units' : 'unitless',
        'source_file_id' : dtc_smiles_regr_all_fileID
     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid



# ----------------------------------------------------------------------------------------------------------------------
# Excape-specific curation functions
# ----------------------------------------------------------------------------------------------------------------------

"""
Upload a raw dataset to the datastore from the given data frame. 
Returns the datastore OID of the uploaded dataset.
"""
def upload_file_excape_raw_data(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,file_path,
			   data_origin='journal',  species='human',  
                           force_update=False):

    bucket = 'public'
    filename = '%s_excape.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category':assay_category, 
        'assay_endpoint' : 'multiple values',
        'curation_level': 'raw',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://dx.doi.org/10.1186%2Fs13321-017-0203-5', # ExCAPE-DB
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : target, 
        'target_type' : target_type,
        'id_col' : 'Original_Entry_ID'
     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        #uploaded_file = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
        #                               description=description,
        #                               tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
        #                               override_check=True, return_metadata=True)
        uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid


def get_smiles_excape_data(nm_df,targ_lst):
	# Delete NaN
	nm_df = nm_df.dropna(subset=['pXC50'])

	# (Yaru) Use nm_df, which has removed nan's
	# Don't need to retrieve SMILES, since already in excape file 	
	# No filtering by censored
	save_df={}
	targ = targ_lst[0]
	save_df[targ_lst[0]] = nm_df
	print(targ,"distinct compounds = only",nm_df['Ambit_InchiKey'].nunique())
	shared_inchi_keys = nm_df['Ambit_InchiKey']

	# Merge in SMILES strings
	smiles_lst=[]

	save_df[targ_lst[0]] = nm_df

	for targ in targ_lst :
	    df=save_df[targ]
	    smiles_df = df
	    #df['PIC50']=df['standard_value'].apply(ic50topic50)
	    #smiles_df=df.merge(save_smiles_df,on='standard_inchi_key',suffixes=('_'+targ,'_'))
	    #the file puts the SMILES string in quotes, which need to be removed
	    #smiles_df['smiles']=smiles_df['smiles'].str.replace('"','')
	    smiles_df['rdkit_smiles']=smiles_df['SMILES'].apply(struct_utils.base_smiles_from_smiles)
	    #smiles_df['smiles']=smiles_df['smiles'].str.replace('"','')
	    print(smiles_df.shape)
	    print(smiles_df['Ambit_InchiKey'].nunique())
	    smiles_lst.append(smiles_df)

	return smiles_lst, shared_inchi_keys


def upload_df_excape_smiles(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,smiles_df,orig_fileID,
			   data_origin='journal',  species='human',  
                           force_update=False):

    bucket = 'public'
    filename = '%s_dtc_smiles.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category': assay_category,  ## seems like this should be called 'kinase_activity'
        'assay_endpoint' : 'pic50',
        'curation_level': 'raw',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://dx.doi.org/10.1186%2Fs13321-017-0203-5', # ExCAPE-DB
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : target,
        'target_type' : target_type,
        'id_col' : 'Original_Entry_ID',
        'source_file_id' : orig_fileID

     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=smiles_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)
        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid

# Apply ATOM standard 'curation' step to "shared_df": Average replicate assays, remove duplicates and drop cases with large variance between replicates.
# mleqonly
def atom_curation_excape(targ_lst, smiles_lst, shared_inchi_keys):
	
	imp.reload(curate_data)
	tolerance=10
	column='pXC50'; #'standard_value'
	list_bad_duplicates='No'
	max_std=1
	curated_lst=[]
	#print(targ_lst)
	#print(smiles_lst)
	for it in range(len(targ_lst)) :
		data=smiles_lst[it]
		#data = data[data.standard_relation.str.strip() == '=']
		#print("gene_names",data.gene_names.unique())
		#print("standard_type",data.standard_type.unique())
		#print("standard_relation",data.standard_relation.unique())
		print("before",data.shape)
		curated_df=curate_data.average_and_remove_duplicates (column, tolerance, list_bad_duplicates, data, max_std, compound_id='standard_inchi_key',smiles_col='rdkit_smiles')

		# (Yaru) Remove inf in curated_df
		curated_df = curated_df[~curated_df.isin([np.inf]).any(1)]
		# (Yaru) Remove nan on rdkit_smiles
		curated_df = curated_df.dropna(subset=['rdkit_smiles'])
		curated_df = curated_df.dropna(subset=['VALUE_NUM_mean'])
		curated_df = curated_df.dropna(subset=['pXC50'])
		
		# (Kevin)
		# Filter criteria:
		#   pXC50 not missing
		#   rdkit_smiles not blank
		#   pXC50 > 3
		#dset_df = dset_df[dset_df.pXC50 >= 3.0]

		curated_lst.append(curated_df)
		prev_cmpd_cnt=shared_inchi_keys.nunique()
		num_dropped=prev_cmpd_cnt-curated_df.shape[0]
		print("After",curated_df.shape, "# of dropped compounds",num_dropped)

	return curated_lst


def upload_df_excape_mleqonly(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,data_df,smiles_fileID,
			   data_origin='journal',  species='human',  
                           force_update=False):

    bucket = 'public'
    filename = '%s_dtc_mleqonly.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category': assay_category,  ## seems like this should be called 'kinase_activity'
        'assay_endpoint' : 'pic50',
        'curation_level': 'ml_ready',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' : 'https://dx.doi.org/10.1186%2Fs13321-017-0203-5', # ExCAPE-DB
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : 'CYP2D6',
        'target_type' : target_type,
        'id_col' : 'Original_Entry_ID',
        'response_col' : 'VALUE_NUM_mean',
        'prediction_type' : 'regression',
        'smiles_col' : 'rdkit_smiles',
        'units' : 'unitless',
        'source_file_id' : smiles_fileID
     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid

def upload_df_excape_mleqonly_class(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,data_df,mleqonly_fileID,
			   data_origin='journal',  species='human',  
                           force_update=False):

    bucket = 'public'
    filename = '%s_dtc_mleqonly_class.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = { 'file_category': 'experimental',
        'activity': activity,
        'assay_category': assay_category,  ## seems like this should be called 'kinase_activity'
        'assay_endpoint' : 'pic50',
        'curation_level': 'ml_ready',
        'data_origin' : data_origin,
        'functional_area' : functional_area,
        'matrix' : 'multiple values',
        'journal_doi' :  'https://dx.doi.org/10.1186%2Fs13321-017-0203-5', # ExCAPE-DB
        'sample_type' : 'in_vitro',
        'species' : species,
        'target' : target,
        'target_type' : target_type,
        'id_col' : 'compound_id',
        'response_col' : 'binary_class',
        'prediction_type' : 'classification',
	'num_classes' : 2,
	'class_names' : ['inactive','active'],
        'smiles_col' : 'rdkit_smiles',
        'units' : 'unitless',
        'source_file_id' : mleqonly_fileID
     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename, 		title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid


'''
# ----------------------------------------------------------------------------------------------------------------------
def curate_excape_jak_datasets():
    """
    Extract JAK1, 2 and 3 datasets from Excape database, filtered for data usability.
    """

    # Filter criteria:
    #   pXC50 not missing
    #   rdkit_smiles not blank
    #   pXC50 > 3

    jak_file = "%s/jak123_excape_smiles.csv" % data_dirs['Excape']
    dset_df = pd.read_csv(jak_file, index_col=False)
    dset_df = dset_df[ ~dset_df.pXC50.isna() & ~dset_df.rdkit_smiles.isna() ]
    dset_df = dset_df[dset_df.pXC50 >= 3.0]
    jak_genes = ['JAK1', 'JAK2', 'JAK3']

    # Split into separate datasets by gene name
    curated_dir = "%s/curated" % data_dirs['Excape']
    os.makedirs(curated_dir, exist_ok=True)
    for gene in jak_genes:
        gene_dset_df = dset_df[dset_df.Gene_Symbol == gene]
        gene_dset_file = "%s/%s_Excape_curated.csv" % (curated_dir, gene)
        gene_dset_df.to_csv(gene_dset_file, index=False)
        print("Wrote file %s" % gene_dset_file)
'''




















'''
# ----------------------------------------------------------------------------------------------------------------------
# ChEMBL-specific curation functions
# ----------------------------------------------------------------------------------------------------------------------

# Raw ChEMBL datasets as downloaded from the interactive ChEMBL web app are labeled as compressed CSV files,
# but they are actually semicolon-separated. After decompressing them, we change the extension to .txt so we
# can open them in Excel without it getting confused. These files can be placed in data_dirs['ChEMBL']/raw
# for initial processing.


# ----------------------------------------------------------------------------------------------------------------------
def filter_chembl_dset(dset_df):
    """
    Filter rows from a raw dataset downloaded from the ChEMBL website. Standardize censoring relational operators.
    Add columns for the log-transformed value, the censoring relation for the log value, and the base RDKit SMILES
    string. Returns a filtered data frame.
    """

    # Filter out rows with no SMILES string or IC50 data
    dset_df = dset_df[~dset_df['Smiles'].isna()]
    dset_df = dset_df[~dset_df['Standard Value'].isna()]
    # Filter out rows flagged as likely duplicates
    dset_df = dset_df[dset_df['Potential Duplicate'] == False]
    # Filter out rows flagged with validity concerns (e.g., value out of typical range)
    dset_df = dset_df[dset_df['Data Validity Comment'].isna()]

    # Filter out rows with nonstandard measurement types. We assume here that the type appearing
    # most frequently is the standard one.
    type_df = curate.freq_table(dset_df, 'Standard Type')
    max_type = type_df['Standard Type'].values[0]
    if type_df.shape[0] > 1:
        print('Dataset has multiple measurement types')
        print(type_df)
    dset_df = dset_df[dset_df['Standard Type'] == max_type]

    # Filter out rows with nonstandard units. Again, we assume the unit appearing most frequently
    # is the standard one.
    unit_freq_df = curate.freq_table(dset_df, 'Standard Units')
    max_unit = unit_freq_df['Standard Units'].values[0]
    if unit_freq_df.shape[0] > 1:
        print('Dataset has multiple standard units')
        print(unit_freq_df)
    dset_df = dset_df[dset_df['Standard Units'] == max_unit]

    # Standardize the censoring operators to =, < or >, and remove any rows whose operators
    # don't map to a standard one.
    dset_df = standardize_relations(dset_df, db='ChEMBL')

    # Add a column for the pIC50 or log-transformed value, and a column for the associated censoring relation.
    # For pXC50 values, this will be the opposite of the original censoring relation.
    ops = dset_df['Standard Relation'].values
    log_ops = ops.copy()
    if (max_type in ['IC50', 'AC50']) and (max_unit == 'nM'):
        dset_df['pIC50'] = 9.0 - np.log10(dset_df['Standard Value'].values)
        log_ops[ops == '>'] = '<'
        log_ops[ops == '<'] = '>'
    elif (max_type == 'Solubility') and (max_unit == 'nM'):
        dset_df['logSolubility'] = np.log10(dset_df['Standard Value'].values) - 9.0
    elif max_type == 'CL':
        dset_df['logCL'] = np.log10(dset_df['Standard Value'].values)
    dset_df['LogVarRelation'] = log_ops

    # Add a column for the standardized base SMILES string. Remove rows with SMILES strings
    # that RDKit wasn't able to parse.
    dset_df['rdkit_smiles'] = [base_smiles_from_smiles(s) for s in dset_df.Smiles.values]
    dset_df = dset_df[dset_df.rdkit_smiles != '']

    return dset_df

# ----------------------------------------------------------------------------------------------------------------------
def filter_all_chembl_dsets(force_update=False):
    """
    Generate filtered versions of all the raw datasets present in the data_dirs['ChEMBL']/raw directory.
    Don't replace any existing filtered file unless force_update is True.
    """
    chembl_dir = data_dirs['ChEMBL']
    raw_dir = "%s/raw" % chembl_dir
    filt_dir = "%s/filtered" % chembl_dir
    os.makedirs(filt_dir, exist_ok=True)
    chembl_files = sorted(os.listdir(raw_dir))

    for fn in chembl_files:
        if fn.endswith('.txt'):
            dset_name = fn.replace('.txt', '')
            filt_path = "%s/%s_filt.csv" % (filt_dir, dset_name)
            if not os.path.exists(filt_path) or force_update:
                fpath = '%s/%s' % (raw_dir, fn)
                dset_df = pd.read_table(fpath, sep=';', index_col=False)
                print("Filtering dataset %s" % dset_name)
                dset_df = filter_chembl_dset(dset_df)
                dset_df.to_csv(filt_path, index=False)
                print("Wrote filtered data to %s" % filt_path)


# ----------------------------------------------------------------------------------------------------------------------
def summarize_chembl_dsets():
    """
    Generate a summary table describing the data in the filtered ChEMBL datasets.
    """

    chembl_dir = data_dirs['ChEMBL']
    filt_dir = "%s/filtered" % chembl_dir
    stats_dir = "%s/stats" % chembl_dir
    os.makedirs(stats_dir, exist_ok=True)
    chembl_files = sorted(os.listdir(filt_dir))
    dset_names = []
    mtype_list = []
    log_var_list = []
    units_list = []
    dset_sizes = []
    num_left = []
    num_eq = []
    num_right = []
    cmpd_counts = []
    cmpd_rep_counts = []
    max_cmpd_reps = []
    assay_counts = []
    max_assay_pts = []
    max_assay_list = []
    max_fmt_list = []

    for fn in chembl_files:
        if fn.endswith('.csv'):
            fpath = '%s/%s' % (filt_dir, fn)
            dset_df = pd.read_csv(fpath, index_col=False)
            dset_name = fn.replace('_filt.csv', '')
            dset_names.append(dset_name)
            print("Summarizing %s" % dset_name)
            dset_sizes.append(dset_df.shape[0])
            type_df = curate.freq_table(dset_df, 'Standard Type')
            max_type = type_df['Standard Type'].values[0]
            mtype_list.append(max_type)
            log_var = log_var_map[max_type]
            log_var_list.append(log_var)

            unit_freq_df = curate.freq_table(dset_df, 'Standard Units')
            max_unit = unit_freq_df['Standard Units'].values[0]
            units_list.append(max_unit)

            log_ops = dset_df.LogVarRelation.values
            uniq_ops, op_counts = np.unique(log_ops, return_counts=True)
            op_count = dict(zip(uniq_ops, op_counts))
            num_left.append(op_count.get('<', 0))
            num_eq.append(op_count.get('=', 0))
            num_right.append(op_count.get('>', 0))

            smiles_df = curate.freq_table(dset_df, 'rdkit_smiles')
            cmpd_counts.append(smiles_df.shape[0])
            smiles_df = smiles_df[smiles_df.Count > 1]
            cmpd_rep_counts.append(smiles_df.shape[0])
            if smiles_df.shape[0] > 0:
                max_cmpd_reps.append(smiles_df.Count.values[0])
            else:
                max_cmpd_reps.append(1)
            mean_values = []
            stds = []
            cmpd_assays = []
            for smiles in smiles_df.rdkit_smiles.values:
                sset_df = dset_df[dset_df.rdkit_smiles == smiles]
                vals = sset_df[log_var].values
                mean_values.append(np.mean(vals))
                stds.append(np.std(vals))
                cmpd_assays.append(len(set(sset_df['Assay ChEMBL ID'].values)))
            smiles_df['Mean_value'] = mean_values
            smiles_df['Std_dev'] = stds
            smiles_df['Num_assays'] = cmpd_assays
            smiles_file = "%s/%s_replicate_cmpd_stats.csv" % (stats_dir, dset_name)
            smiles_df.to_csv(smiles_file, index=False)

            assay_df = curate.labeled_freq_table(dset_df, ['Assay ChEMBL ID', 'Assay Description', 'BAO Label'])
            assay_counts.append(assay_df.shape[0])
            max_assay_pts.append(assay_df.Count.values[0])
            max_assay_list.append(assay_df['Assay Description'].values[0])
            max_fmt_list.append(assay_df['BAO Label'].values[0])
            assay_df = assay_df[assay_df.Count >= 20]
            assay_file = "%s/%s_top_assay_summary.csv" % (stats_dir, dset_name)
            assay_df.to_csv(assay_file, index=False)

    summary_df = pd.DataFrame(dict(
        Dataset=dset_names,
        MeasuredValue=mtype_list,
        LogValue=log_var_list,
        Units=units_list,
        NumPoints=dset_sizes,
        NumUncensored=num_eq,
        NumLeftCensored=num_left,
        NumRightCensored=num_right,
        NumCmpds=cmpd_counts,
        NumReplicatedCmpds=cmpd_rep_counts,
        MaxCmpdReps=max_cmpd_reps,
        NumAssays=assay_counts,
        MaxAssayPoints=max_assay_pts,
        MaxAssay=max_assay_list,
        MaxAssayFormat=max_fmt_list
    ))
    summary_file = "%s/chembl_public_dataset_summary.csv" % stats_dir
    summary_df.to_csv(summary_file, index=False, columns=['Dataset', 'NumPoints',
                                                          'NumUncensored', 'NumLeftCensored', 'NumRightCensored',
                                                          'NumCmpds', 'NumReplicatedCmpds', 'MaxCmpdReps',
                                                          'MeasuredValue', 'LogValue', 'Units',
                                                          'NumAssays', 'MaxAssayPoints', 'MaxAssayFormat', 'MaxAssay'])
    print("Wrote summary table to %s" % summary_file)


# ----------------------------------------------------------------------------------------------------------------------
def plot_chembl_log_distrs():
    """
    Plot distributions of the log-transformed values for each of the ChEMBL datasets
    """
    chembl_dir = data_dirs['ChEMBL']
    filt_dir = "%s/filtered" % chembl_dir
    summary_file = "%s/stats/chembl_public_dataset_summary.csv" % chembl_dir
    summary_df = pd.read_csv(summary_file, index_col=False)
    dset_names = set(summary_df.Dataset.values)

    # Plot distributions for the pairs of CYP datasets together
    cyp_dsets = dict(
        CYP2C9 = dict(AC50='CHEMBL25-CYP2C9_human_AC50_26Nov2019', IC50='CHEMBL25-CYP2C9_human_IC50_26Nov2019'),
        CYP2D6 = dict(AC50='CHEMBL25-CYP2D6_human_AC50_26Nov2019', IC50='CHEMBL25-CYP2D6_human_IC50_26Nov2019'),
        CYP3A4 = dict(AC50='CHEMBL25_CHEMBL25-CYP3A4_human_AC50_26Nov2019', IC50='CHEMBL25-CYP3A4_human_IC50_26Nov2019')
    )

    cyp_dset_names = []
    for cyp in sorted(cyp_dsets.keys()):
        ds_dict = cyp_dsets[cyp]
        cyp_dset_names.append(ds_dict['AC50'])
        cyp_dset_names.append(ds_dict['IC50'])
        ac50_path = "%s/%s_filt.csv" % (filt_dir, ds_dict['AC50'])
        ic50_path = "%s/%s_filt.csv" % (filt_dir, ds_dict['IC50'])
        ac50_df = pd.read_csv(ac50_path, index_col=False)
        ic50_df = pd.read_csv(ic50_path, index_col=False)
        ac50_smiles = set(ac50_df.Smiles.values)
        ic50_smiles = set(ic50_df.Smiles.values)
        cmn_smiles = ac50_smiles & ic50_smiles
        print("For %s: %d SMILES strings in both datasets" % (cyp, len(cmn_smiles)))

        fig, ax = plt.subplots(figsize=(10,8))
        ax = sns.distplot(ac50_df.pIC50.values, hist=False, kde_kws=dict(shade=True, bw=0.05), color='b', ax=ax, label='PubChem')
        ic50_lc_df = ic50_df[ic50_df.LogVarRelation == '<']
        ic50_rc_df = ic50_df[ic50_df.LogVarRelation == '>']
        ic50_uc_df = ic50_df[ic50_df.LogVarRelation == '=']
        ax = sns.distplot(ic50_uc_df.pIC50.values, hist=False, kde_kws=dict(shade=True, bw=0.05), color='g', ax=ax, label='Uncens')
        ax = sns.distplot(ic50_lc_df.pIC50.values, hist=False, kde_kws=dict(shade=False, bw=0.05), color='r', ax=ax, label='LeftCens')
        ax = sns.distplot(ic50_rc_df.pIC50.values, hist=False, kde_kws=dict(shade=False, bw=0.05), color='m', ax=ax, label='RightCens')
        ax.set_xlabel('pIC50')
        ax.set_title('Distributions of %s dataset values' % cyp)
        plt.show()

    other_dset_names = sorted(dset_names - set(cyp_dset_names))
    for dset_name in other_dset_names:
        log_var = summary_df.LogValue.values[summary_df.Dataset == dset_name][0]
        filt_path = "%s/%s_filt.csv" % (filt_dir, dset_name)
        dset_df = pd.read_csv(filt_path, index_col=False)
        uc_df = dset_df[dset_df.LogVarRelation == '=']
        lc_df = dset_df[dset_df.LogVarRelation == '<']
        rc_df = dset_df[dset_df.LogVarRelation == '>']
        log_uc_values = uc_df[log_var].values
        log_lc_values = lc_df[log_var].values
        log_rc_values = rc_df[log_var].values
        fig, ax = plt.subplots(figsize=(10,8))
        ax = sns.distplot(log_uc_values, hist=False, kde_kws=dict(shade=True, bw=0.05), color='b', ax=ax, label='Uncens')
        ax = sns.distplot(log_lc_values, hist=False, kde_kws=dict(shade=False, bw=0.05), color='r', ax=ax, label='LeftCens')
        ax = sns.distplot(log_rc_values, hist=False, kde_kws=dict(shade=False, bw=0.05), color='m', ax=ax, label='RightCens')
        ax.set_xlabel(log_var)
        ax.set_title('Distribution of log transformed values for %s' % dset_name)
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
def curate_chembl_xc50_assay(dset_df, target, endpoint, database='ChEMBL25'):
    """
    Examine data from individual ChEMBL assays in the given dataset to look for suspicious patterns of XC50
    values and censoring relations. Add relations where they appear to be needed, and filter out data from
    assays that seem to have only one-shot categorical data.
    """

    chembl_root = data_dirs['ChEMBL']
    assay_df = curate.freq_table(dset_df, 'Assay ChEMBL ID')
    assays = assay_df['Assay ChEMBL ID'].values
    counts = assay_df['Count'].values
    num_eq = []
    num_lt = []
    num_gt = []
    max_xc50s = []
    num_max_xc50 = []
    min_xc50s = []
    num_min_xc50 = []

    # For each assay ID, tabulate the number of occurrences of each relation, the min and max XC50
    # and the number of values reported as the max or min XC50
    for assay in assays:
        assay_dset_df = dset_df[dset_df['Assay ChEMBL ID'] == assay]

        xc50s = assay_dset_df['Standard Value'].values
        max_xc50 = max(xc50s)
        max_xc50s.append(max_xc50)
        min_xc50 = min(xc50s)
        min_xc50s.append(min_xc50)
        relations = assay_dset_df['Standard Relation'].values
        num_eq.append(sum(relations == '='))
        num_lt.append(sum(relations == '<'))
        num_gt.append(sum(relations == '>'))
        num_max_xc50.append(sum(xc50s == max_xc50))
        num_min_xc50.append(sum(xc50s == min_xc50))
    assay_df['num_eq'] = num_eq
    assay_df['num_lt'] = num_lt
    assay_df['num_gt'] = num_gt
    assay_df['max_xc50'] = max_xc50s
    assay_df['num_max_xc50'] = num_max_xc50
    assay_df['min_xc50'] = min_xc50s
    assay_df['num_min_xc50'] = num_min_xc50

    # Flag assays that appear to report one-shot screening results only (because all values are left or
    # right censored at the same threshold)
    num_eq = np.array(num_eq)
    num_lt = np.array(num_lt)
    num_gt = np.array(num_gt)
    max_xc50s = np.array(max_xc50s)
    min_xc50s = np.array(min_xc50s)
    num_max_xc50 = np.array(num_max_xc50)
    num_min_xc50 = np.array(num_min_xc50)

    one_shot =  (num_eq == 0) & (num_lt > 0) & (num_gt > 0)
    assay_df['one_shot'] = one_shot
    # Flag assays that appear not to report left-censoring correctly (because no values are censored
    # and there are multiple values at highest XC50)
    no_left_censoring = (counts == num_eq) & (num_max_xc50 >= 5)
    assay_df['no_left_censoring'] = no_left_censoring
    # Flag assays that appear not to report right-censoring correctly (because no values are censored
    # and there are multiple values at lowest XC50)
    no_right_censoring = (counts == num_eq) & (num_min_xc50 >= 5)
    assay_df['no_right_censoring'] = no_right_censoring

    assay_file = "%s/stats/%s_%s_%s_assay_stats.csv" % (chembl_root, database, target, endpoint)
    assay_df.to_csv(assay_file, index=False)
    print("Wrote %s %s assay censoring statistics to %s" % (target, endpoint, assay_file))

    # Now generate a "curated" version of the dataset
    assay_dsets = []
    for assay, is_one_shot, has_no_left_cens, has_no_right_cens in zip(assays, one_shot, no_left_censoring,
                                                                       no_right_censoring):
        # Skip over assays that appear to contain one-shot data
        if is_one_shot:
            print("Skipping apparent one-shot data from assay %s" % assay)
        else:
            assay_dset_df = dset_df[dset_df['Assay ChEMBL ID'] == assay].copy()
            xc50s = assay_dset_df['Standard Value'].values
            max_xc50 = max(xc50s)
            min_xc50 = min(xc50s)
            # Add censoring relations for rows that seem to need them
            relations = assay_dset_df['Standard Relation'].values
            log_relations = assay_dset_df['LogVarRelation'].values
            if has_no_left_cens:
                relations[xc50s == max_xc50] = '>'
                log_relations[xc50s == max_xc50] = '<'
                print("Adding missing left-censoring relations for assay %s" % assay)
            if has_no_right_cens:
                relations[xc50s == min_xc50] = '<'
                log_relations[xc50s == min_xc50] = '>'
                print("Adding missing right-censoring relations for assay %s" % assay)
            assay_dset_df['Standard Relation'] = relations
            assay_dset_df['LogVarRelation'] = log_relations
            assay_dsets.append(assay_dset_df)
    curated_df = pd.concat(assay_dsets, ignore_index=True)
    return curated_df


# ----------------------------------------------------------------------------------------------------------------------
def curate_chembl_xc50_assays(database='ChEMBL25', species='human', force_update=False):
    """
    Examine data from individual ChEMBL assays in each dataset to look for suspicious patterns of XC50
    values and censoring relations. Add relations where they appear to be needed, and filter out data from
    assays that seem to have only one-shot categorical data.
    """
    chembl_root = data_dirs['ChEMBL']

    filtered_dir = '%s/filtered' % chembl_root
    curated_dir = '%s/curated' % chembl_root
    os.makedirs(curated_dir, exist_ok=True)

    targets = sorted(chembl_dsets.keys())
    for target in targets:
        if type(chembl_dsets[target]) == dict:
            for endpoint, dset_name in chembl_dsets[target].items():
                curated_file = "%s/%s_%s_%s_%s_curated.csv" % (curated_dir, database, target, endpoint, species)
                if os.path.exists(curated_file) and not force_update:
                    print("\nCurated dataset %s already exists, skipping" % curated_file)
                    continue
                print("\n\nCurating %s data for %s" % (endpoint, target))
                dset_file = "%s/%s_filt.csv" % (filtered_dir, dset_name)
                dset_df = pd.read_csv(dset_file, index_col=False)
                curated_df = curate_chembl_xc50_assay(dset_df, target, endpoint)
                curated_df.to_csv(curated_file, index=False)
                print("Wrote %s" % curated_file)

# ----------------------------------------------------------------------------------------------------------------------
def upload_chembl_raw_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target='', database='ChEMBL25', activity='inhibition', species='human',
                           force_update=False):
    """
    Upload a raw dataset to the datastore from the given data frame. 
    Returns the datastore OID of the uploaded dataset.
    """
    raw_dir = '%s/raw' % data_dirs['ChEMBL']
    raw_path = "%s/%s.txt" % (raw_dir, dset_name)
    dset_df = pd.read_table(raw_path, sep=';', index_col=False)

    bucket = 'public'
    filename = '%s.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = {
          'activity': activity,
          'assay_category': assay_category,
          'assay_endpoint': endpoint,
          'target_type': target_type,
          'functional_area': functional_area,
          'data_origin': database,
          'species': species,
          'file_category': 'experimental',
          'curation_level': 'raw',
          'matrix': 'in vitro',
          'sample_type': 'in_vitro',
          'id_col': 'Molecule ChEMBL ID',
          'smiles_col': 'Smiles',
          'response_col': 'Standard Value'} 

    if target != '':
        kv['target'] = target

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        raw_meta = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
                                       description=description,
                                       tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
                                       override_check=True, return_metadata=True)
        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        raw_meta = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)
    raw_dset_oid = raw_meta['dataset_oid']
    return raw_dset_oid

# ----------------------------------------------------------------------------------------------------------------------
def upload_chembl_curated_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target='', database='ChEMBL25', activity='inhibition', species='human',
                           raw_dset_oid=None, force_update=False):
    """
    Upload a curated dataset to the datastore. Returns the datastore OID of the uploaded dataset.
    """
    curated_dir = '%s/curated' % data_dirs['ChEMBL']
    filtered_dir = '%s/filtered' % data_dirs['ChEMBL']

    if target == '':
        # This is a PK dataset, for which curation consists only of the initial filtering
        filename = '%s_curated.csv' % dset_name
        curated_file = "%s/%s_filt.csv" % (filtered_dir, dset_name)
    else:
        # This is a bioactivity dataset
        filename = "%s_%s_%s_%s_curated.csv" % (database, target, endpoint, species)
        curated_file = "%s/%s" % (curated_dir, filename)

    dset_df = pd.read_csv(curated_file, index_col=False)
    bucket = 'public'
    dataset_key = 'dskey_' + filename

    kv = {
          'activity': activity,
          'assay_category': assay_category,
          'assay_endpoint': endpoint,
          'target_type': target_type,
          'functional_area': functional_area,
          'data_origin': database,
          'species': species,
          'file_category': 'experimental',
          'curation_level': 'curated',
          'matrix': 'in vitro',
          'sample_type': 'in_vitro',
          'id_col': 'Molecule ChEMBL ID',
          'smiles_col': 'rdkit_smiles',
          'response_col': log_var_map[endpoint] } 
    if target != '':
        kv['target'] = target
    if raw_dset_oid is not None:
        kv['source_file_id'] = raw_dset_oid

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        curated_meta = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
                                       description=description,
                                       tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
                                       override_check=True, return_metadata=True)
        print("Uploaded curated dataset with key %s" % dataset_key)
    else:
        print("Curated dataset %s is already in datastore, skipping upload." % dataset_key)
        curated_meta = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
    curated_oid = curated_meta['dataset_oid']
    return curated_oid


# ----------------------------------------------------------------------------------------------------------------------
def create_ml_ready_chembl_dataset(dset_name, endpoint, target='', species='human', active_thresh=None,
                                   database='ChEMBL25', force_update=False):
    """
    Average replicate values from the curated version of the given dataset to give one value
    per unique compound. Select and rename columns to include only the ones we need for building
    ML models. Save the resulting dataset to disk.

    endpoint is IC50, AC50, CL or Solubility.
    """
    curated_dir = '%s/curated' % data_dirs['ChEMBL']
    filtered_dir = '%s/filtered' % data_dirs['ChEMBL']
    ml_ready_dir = '%s/ml_ready' % data_dirs['ChEMBL']
    os.makedirs(ml_ready_dir, exist_ok=True)

    if target == '':
        # This is a PK dataset, for which curation consists only of the initial filtering
        curated_file = "%s/%s_filt.csv" % (filtered_dir, dset_name)
        ml_ready_file = "%s/%s_ml_ready.csv" % (ml_ready_dir, dset_name)
    else:
        # This is a bioactivity dataset
        curated_file = "%s/%s_%s_%s_%s_curated.csv" % (curated_dir, database, target, endpoint, species)
        ml_ready_file = "%s/%s_%s_%s_%s_ml_ready.csv" % (ml_ready_dir, database, target, endpoint, species)

    if os.path.exists(ml_ready_file) and not force_update:
        return

    dset_df = pd.read_csv(curated_file, index_col=False)

    # Rename and select the columns we want from the curated dataset
    param = log_var_map[endpoint]
    agg_cols = ['compound_id', 'rdkit_smiles', 'relation', param]
    colmap = {
        'Molecule ChEMBL ID': 'compound_id',
        'LogVarRelation': 'relation'
    }
    assay_df = dset_df.rename(columns=colmap)[agg_cols]
    # Compute a single value and relational flag for each compound
    ml_ready_df = curate.aggregate_assay_data(assay_df, value_col=param, active_thresh=active_thresh,
                                              id_col='compound_id', smiles_col='rdkit_smiles',
                                              relation_col='relation')

    ml_ready_df.to_csv(ml_ready_file, index=False)
    print("Wrote ML-ready data to %s" % ml_ready_file)
    return ml_ready_df

# ----------------------------------------------------------------------------------------------------------------------
def upload_chembl_ml_ready_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target='', database='ChEMBL25', activity='inhibition', species='human',
                           curated_dset_oid=None, force_update=False):
    """
    Upload a ML-ready dataset to the datastore, previously created by create_ml_ready_chembl_dataset. 
    Returns the datastore OID of the uploaded dataset.
    """
    ml_ready_dir = '%s/ml_ready' % data_dirs['ChEMBL']

    if target == '':
        # This is a PK dataset
        filename = '%s_ml_ready.csv' % dset_name
    else:
        # This is a bioactivity dataset
        filename = "%s_%s_%s_%s_ml_ready.csv" % (database, target, endpoint, species)
    ml_ready_file = "%s/%s" % (ml_ready_dir, filename)

    dset_df = pd.read_csv(ml_ready_file, index_col=False)
    bucket = 'public'
    dataset_key = 'dskey_' + filename

    kv = {
          'activity': activity,
          'assay_category': assay_category,
          'assay_endpoint': endpoint,
          'target_type': target_type,
          'functional_area': functional_area,
          'data_origin': database,
          'species': species,
          'file_category': 'experimental',
          'curation_level': 'ml_ready',
          'matrix': 'in vitro',
          'sample_type': 'in_vitro',
          'id_col': 'compound_id',
          'smiles_col': 'base_rdkit_smiles',
          'response_col': log_var_map[endpoint]
          } 
    if target != '':
        kv['target'] = target
    if curated_dset_oid is not None:
        kv['source_file_id'] = curated_dset_oid

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        ml_ready_meta = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
                                       description=description,
                                       tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
                                       override_check=True, return_metadata=True)
        print("Uploaded ML-ready dataset with key %s" % dataset_key)
    else:
        print("ML-ready dataset %s is already in datastore, skipping upload." % dataset_key)
        ml_ready_meta = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
    ml_ready_oid = ml_ready_meta['dataset_oid']
    return ml_ready_oid

# ----------------------------------------------------------------------------------------------------------------------
def chembl_dataset_curation_pipeline(dset_table_file, force_update=False):
    """
    Run a series of ChEMBL datasets through the process of filtering, curation, and aggregation for use in
    building machine learning models. Upload the raw, curated and ML-ready datasets to the datastore.
    The datasets are described in the CSV file dset_table_file, which tabulates the attributes of each dataset
    and the metadata to be included with the uploaded datasets.
    """
    chembl_dir = data_dirs['ChEMBL']
    raw_dir = "%s/raw" % chembl_dir
    filt_dir = "%s/filtered" % chembl_dir
    curated_dir = "%s/curated" % chembl_dir
    ml_ready_dir = "%s/ml_ready" % chembl_dir

    os.makedirs(filt_dir, exist_ok=True)
    table_df = pd.read_csv(dset_table_file, index_col=False)
    table_df = table_df.fillna('')

    for i, dset_name in enumerate(table_df.Dataset.values):
        endpoint = table_df.endpoint.values[i]
        target = table_df.target.values[i]
        assay_category = table_df.assay_category.values[i]
        functional_area = table_df.functional_area.values[i]
        target_type = table_df.target_type.values[i]
        activity = table_df.activity.values[i]
        species = table_df.species.values[i]
        database = table_df.database.values[i]
        title = table_df.title.values[i]
        description = table_df.description.values[i]
        tags = ['public', 'raw']

        # Upload the raw dataset as-is
        raw_dset_oid = upload_chembl_raw_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target=target, database=database, activity=activity, species=species,
                           force_update=force_update)

        # First curation step: Filter dataset to remove rows with missing IDs, SMILES, values, etc.
        raw_path = "%s/%s.txt" % (raw_dir, dset_name)
        filt_path = "%s/%s_filt.csv" % (filt_dir, dset_name)
        if not os.path.exists(filt_path) or force_update:
            dset_df = pd.read_table(raw_path, sep=';', index_col=False)
            print("Filtering dataset %s" % dset_name)
            filt_df = filter_chembl_dset(dset_df)
            filt_df.to_csv(filt_path, index=False)
        else:
            filt_df = pd.read_csv(filt_path, index_col=False)
            print("Filtered dataset file %s already exists" % filt_path)

        # Second curation step: Fix or remove anomalous data. Currently this is only done for 
        # bioactivity data.
        if target != '':
            curated_file = "%s/%s_%s_%s_%s_curated.csv" % (curated_dir, database, target, endpoint, species)
            if not os.path.exists(curated_file) or force_update:
                print("Curating %s data for %s" % (endpoint, target))
                curated_df = curate_chembl_xc50_assay(filt_df, target, endpoint, database=database)
                curated_df.to_csv(curated_file, index=False)
            else:
                curated_df = pd.read_csv(curated_file, index_col=False)
                print("Curated %s dataset file for %s already exists" % (endpoint, target))
            description += "\nCurated using public_data_curation functions filter_chembl_dset and curate_chembl_xc50_assay."
        else:
            curated_df = filt_df
            description += "\nCurated using public_data_curation function filter_chembl_dset."
        title = title.replace('Raw', 'Curated')
        tags = ['public', 'curated']

        # Upload curated data to datastore
        curated_dset_oid = upload_chembl_curated_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target=target, database=database, activity=activity, species=species,
                           raw_dset_oid=raw_dset_oid, force_update=force_update)

        # Prepare ML-ready dataset
        if target == '':
            ml_ready_file = "%s/%s_ml_ready.csv" % (ml_ready_dir, dset_name)
        else:
            ml_ready_file = "%s/%s_%s_%s_%s_ml_ready.csv" % (ml_ready_dir, database, target, endpoint, species)
        if not os.path.exists(ml_ready_file) or force_update:
            print("Creating ML-ready dataset file %s" % ml_ready_file)
            ml_ready_df = create_ml_ready_chembl_dataset(dset_name, endpoint, target=target, species=species, active_thresh=None,
                                                         database=database, force_update=force_update)
        else:
            ml_ready_df = pd.read_csv(ml_ready_file, index_col=False)
            print("ML-ready dataset file %s already exists" % ml_ready_file)
        title = title.replace('Curated', 'ML-ready')
        description += "\nAveraged for ML model building using public_data_curation.create_ml_ready_dataset."
        tags = ['public', 'ML-ready']

        # Upload ML-ready data to the datastore
        ml_ready_dset_oid = upload_chembl_ml_ready_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target=target, database=database, activity=activity, species=species,
                           curated_dset_oid=curated_dset_oid, force_update=force_update)
        print("Done with dataset %s\n" % dset_name)       

# ----------------------------------------------------------------------------------------------------------------------
def chembl_replicate_variation(dset_df, value_col='pIC50', dset_label='', min_freq=3, num_assays=4):
    """
    Plot the variation among measurements in a ChEMBL dataset for compounds with multiple measurements
    from the same or different ChEMBL assay IDs.
    """
    rep_df = curate.freq_table(dset_df, 'rdkit_smiles', min_freq=min_freq)
    rep_df['mean_value'] = [np.mean(dset_df[dset_df.rdkit_smiles == s][value_col].values)
                            for s in rep_df.rdkit_smiles.values]
    rep_df['std_value'] = [np.std(dset_df[dset_df.rdkit_smiles == s][value_col].values)
                           for s in rep_df.rdkit_smiles.values]
    rep_df = rep_df.sort_values(by='mean_value')
    nrep = rep_df.shape[0]
    rep_df['cmpd_id'] = ['C%05d' % i for i in range(nrep)]

    rep_dset_df = dset_df[dset_df.rdkit_smiles.isin(rep_df.rdkit_smiles.values)].merge(
        rep_df, how='left', on='rdkit_smiles')

    # Label the records coming from the num_assays most common assays with the first part of
    # their assay descriptions; label the others as 'Other'.
    assay_df = curate.freq_table(rep_dset_df, 'Assay ChEMBL ID')
    other_ids = assay_df['Assay ChEMBL ID'].values[num_assays:]
    assay_labels = np.array([desc[:30]+'...' for desc in rep_dset_df['Assay Description'].values])
    assay_labels[rep_dset_df['Assay ChEMBL ID'].isin(other_ids)] = 'Other'
    rep_dset_df['Assay'] = assay_labels

    fig, ax = plt.subplots(figsize=(10,15))
    sns.stripplot(x=value_col, y='cmpd_id', hue='Assay', data=rep_dset_df,
                  order=rep_df.cmpd_id.values)
    ax.set_title(dset_label)
    return rep_df


# ----------------------------------------------------------------------------------------------------------------------
# Filename templates for curated bioactivity datasets, with a %s field to plug in the target or property name. Probably
# we should just rename the files from all data sources to follow the standard template: 
# (database)_(target)_(endpoint)_(species)_curated.csv.

curated_dset_file_templ = dict(
    ChEMBL="ChEMBL25_%s_IC50_human_curated.csv",
    DTC="%s_DTC_curated.csv",
    Excape="%s_Excape_curated.csv"
)

# ----------------------------------------------------------------------------------------------------------------------
def chembl_jak_replicate_variation(min_freq=2, num_assays=4):
    """
    Plot variation among replicate measurements for compounds in the JAK datasets
    """
    jak_genes = ['JAK1', 'JAK2', 'JAK3']
    db = 'ChEMBL'
    dsets = {}
    for gene in jak_genes:
        dset_file = "%s/curated/%s" % (data_dirs[db], curated_dset_file_templ[db] % gene)
        dset_df = pd.read_csv(dset_file, index_col=False)
        dsets[gene] = chembl_replicate_variation(dset_df, value_col='pIC50', min_freq=min_freq,
                                                 dset_label=gene,
                                                 num_assays=num_assays)
    return dsets


# ----------------------------------------------------------------------------------------------------------------------
def chembl_assay_bias(target, endpoint, database='ChEMBL25', species='human', min_cmp_assays=5, min_cmp_cmpds=10):
    """
    Investigate systematic biases among assays for target, by selecting data for compounds with data from
    multiple assays and computing deviations from mean for each compound; then reporting mean deviation for
    each assay.
    """
    curated_dir = '%s/curated' % data_dirs['ChEMBL']
    curated_file = "%s/%s_%s_%s_%s_curated.csv" % (curated_dir, database, target, endpoint, species)
    dset_df = pd.read_csv(curated_file, index_col=False)
    assay_df = curate.labeled_freq_table(dset_df, ['Assay ChEMBL ID', 'Assay Description', 'BAO Label'], min_freq=2)
    assays = assay_df['Assay ChEMBL ID'].values
    print("\nChecking bias for ChEMBL %s %s dataset:" % (target, endpoint))
    if assay_df.shape[0] == 1:
        print("Dataset %s has data for one assay only; skipping." % curated_file)
        return None
    # Restrict to data from assays with at least 2 rows of data
    dset_df = dset_df[dset_df['Assay ChEMBL ID'].isin(assay_df['Assay ChEMBL ID'].values.tolist())]

    # Tabulate overall mean and SD and compound count for each assay
    log_var = log_var_map[dset_df['Standard Type'].values[0]]
    mean_values = [np.mean(dset_df[dset_df['Assay ChEMBL ID'] == assay][log_var].values) for assay in assays]
    stds = [np.std(dset_df[dset_df['Assay ChEMBL ID'] == assay][log_var].values) for assay in assays]
    ncmpds = [len(set(dset_df[dset_df['Assay ChEMBL ID'] == assay]['rdkit_smiles'].values)) for assay in assays]
    assay_df['num_cmpds'] = ncmpds
    assay_df['mean_%s' % log_var] = mean_values
    assay_df['std_%s' % log_var] = stds
    assay_df = assay_df.rename(columns={'Count': 'num_rows'})

    # Select compounds with data from multiple assays. Compute mean values for each compound.
    # Then compute deviations from mean for each assay for each compound.
    assay_devs = {assay : [] for assay in assays}
    cmp_assays = {assay : set() for assay in assays}
    cmp_cmpds = {assay : 0 for assay in assays}
    rep_df = curate.freq_table(dset_df, 'rdkit_smiles', min_freq=2)
    for smiles in rep_df.rdkit_smiles.values:
        sset_df = dset_df[dset_df.rdkit_smiles == smiles]
        sset_assays = sset_df['Assay ChEMBL ID'].values
        sset_assay_set = set(sset_assays)
        num_assays = len(set(sset_assays))
        if num_assays > 1:
            vals = sset_df[log_var].values
            mean_val = np.mean(vals)
            deviations = vals - mean_val
            for assay, dev in zip(sset_assays, deviations):
                assay_devs[assay].append(dev)
                cmp_assays[assay] |= (sset_assay_set - set([assay]))
                cmp_cmpds[assay] += 1
    assay_df['num_cmp_assays'] = [len(cmp_assays[assay]) for assay in assays]
    assay_df['num_cmp_cmpds'] = [cmp_cmpds[assay] for assay in assays]
    mean_deviations = [np.mean(assay_devs[assay]) for assay in assays]
    assay_df['mean_deviation'] = mean_deviations
    assay_df = assay_df.sort_values(by='mean_deviation', ascending=False)
    assay_file = "%s/stats/%s_%s_assay_bias.csv" % (data_dirs['ChEMBL'], target, endpoint)
    assay_df.to_csv(assay_file, index=False)
    # Flag assays compared against at least min_cmp_assays other assays over min_cmp_cmpds compounds
    flag_df = assay_df[(assay_df.num_cmp_assays >= min_cmp_assays) & (assay_df.num_cmp_cmpds >= min_cmp_cmpds)]
    print("For %s %s data: %d assays with robust bias data:" % (target, endpoint, flag_df.shape[0]))
    if flag_df.shape[0] > 0:
        print(flag_df)
    print("Wrote assay bias statistics to %s" % assay_file)
    return assay_df

# ----------------------------------------------------------------------------------------------------------------------
def chembl_xc50_assay_bias():
    """
    Tabulate systematic biases for all the ChEMBL XC50 datasets
    """
    targets = sorted(chembl_dsets.keys())
    for target in targets:
        if type(chembl_dsets[target]) == dict:
            for endpoint in chembl_dsets[target].keys():
                bias_df = chembl_assay_bias(target, endpoint)



# ----------------------------------------------------------------------------------------------------------------------
# Functions for comparing datasets from different sources
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def compare_jak_dset_compounds():
    """
    Plot Venn diagrams for each set of public JAK datasets showing the numbers of compounds in common
    between them
    """
    jak_genes = ['JAK1', 'JAK2', 'JAK3']
    dbs = sorted(data_dirs.keys())
    for gene in jak_genes:
        dset_smiles = []
        for db in dbs:
            dset_file = "%s/curated/%s" % (data_dirs[db], curated_dset_file_templ[db] % gene)
            dset_df = pd.read_csv(dset_file, index_col=False)
            dset_smiles.append(set(dset_df.rdkit_smiles.values))
        fig, ax = plt.subplots(figsize=(8,8))
        venn3(dset_smiles, dbs)
        plt.title(gene)
        plt.show()

# ----------------------------------------------------------------------------------------------------------------------
def find_jak_dset_duplicates():
    """
    Check for potential duplication of records within and between datasets. A record is a potential
    duplicate if it has the same base SMILES string, IC50 value and standard relation.
    """
    colmap = dict(
        ChEMBL={'pIC50': 'pIC50',
                'LogVarRelation': 'Relation'
                },
        DTC={'pIC50': 'pIC50',
                'pIC50_relation': 'Relation'
                },
        Excape={'pXC50': 'pIC50'
             } )
    jak_genes = ['JAK1', 'JAK2', 'JAK3']
    dbs = sorted(data_dirs.keys())
    for gene in jak_genes:
        dedup = {}
        smiles_set = {}
        for db in dbs:
            dset_file = "%s/curated/%s" % (data_dirs[db], curated_dset_file_templ[db] % gene)
            dset_df = pd.read_csv(dset_file, index_col=False)
            dset_df = dset_df.rename(columns=colmap[db])
            if db == 'Excape':
                dset_df['Relation'] = "="
            dset_df = dset_df[['Relation', 'pIC50', 'rdkit_smiles']]
            is_dup = dset_df.duplicated().values
            print("Within %s %s dataset, %d/%d rows are potential duplicates" % (db, gene, sum(is_dup), dset_df.shape[0]))
            dedup[db] = dset_df.drop_duplicates()
            smiles_set[db] = set(dset_df.rdkit_smiles.values)
        print('\n')
        for i, db1 in enumerate(dbs[:2]):
            for db2 in dbs[i+1:]:
                combined_df = pd.concat([dedup[db1], dedup[db2]], ignore_index=True)
                is_dup = combined_df.duplicated().values
                n_cmn_smiles = len(smiles_set[db1] & smiles_set[db2])
                print("Between %s and %s %s datasets, %d common SMILES, %d identical responses" % (db1, db2, gene,
                                                                                                   n_cmn_smiles,
                                                                                                   sum(is_dup)))
        print('\n')

'''

