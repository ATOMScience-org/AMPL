"""data_curation_functions.py

Extract Kevin's functions for curation of public datasets
Modify them to match Jonathan's curation methods in notebook
01/30/2020
"""

import numpy as np
import pandas as pd


from atomsci.ddm.utils.struct_utils import mols_from_smiles
import atomsci.ddm.utils.datastore_functions as dsf
import atomsci.ddm.utils.struct_utils as struct_utils
import atomsci.ddm.utils.curate_data as curate_data
import imp

def set_data_root(dir):
    """Set global variables for data directories

    Creates paths for DTC and Excape given a root data directory.
    Global variables 'data_root' and 'data_dirs'. 'data_root' is the
    root data directory. 'data_dirs' is a dictionary that maps 'DTC' and 'Excape'
    to directores calcuated from 'data_root'

    Args:
        dir (str): root data directory containing folds 'dtc' and 'excape'

    Returns:
        None
    """
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

# The following list includes the nonmetals commonly found in organic molecules, along with alkali and alkaline earth
# metals commonly found in salts (Na, Mg, K, Ca).
organic_atomic_nums = [1, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 33, 34, 35, 53]

# ----------------------------------------------------------------------------------------------------------------------
# Generic functions for all datasets
# ----------------------------------------------------------------------------------------------------------------------

# Note: Functions freq_table and labeled_freq_table have been moved to ddm.utils.curate_data module.

def is_organometallic(mol):
    """
    Returns True if the molecule is organometallic
    """
    if mol is None:
        return True
    for at in mol.GetAtoms():
        if at.GetAtomicNum() not in organic_atomic_nums:
            return True
    return False

# ----------------------------------------------------------------------------------------------------------------------
def exclude_organometallics(df, smiles_col='rdkit_smiles'):
    """Filters data frame df based on column smiles_col to exclude organometallic compounds"""
    mols = mols_from_smiles(df[smiles_col].values.tolist(), workers=16)
    include = np.array([not is_organometallic(mol) for mol in mols])
    return df[include].copy()


# ----------------------------------------------------------------------------------------------------------------------
def standardize_relations(dset_df, db=None, rel_col=None, output_rel_col=None, invert=False):
    """Standardizes censoring operators

        Standardize the censoring operators to =, < or >, and remove any rows whose operators
        don't map to a standard one. There is a special case for db='ChEMBL' that strips
        the extra "'"s around relationship symbols. Assumes relationship columns are
        'Standard Relation', 'standard_relation' and 'activity_prefix' for ChEMBL, DTC and GoStar respectively.

        This function makes the following mappings: ">" to ">", ">=" to ">", "<" to "<",
        "<=" to "<", and "=" to "=". All other relations are removed from the DataFrame.

    Args:
        dset_df (DataFrame): Input DataFrame. Must contain either 'Standard Relation'
           or 'standard_relation'

        db (str): Source database. Must be either 'GoStar', 'DTC' or 'ChEMBL'. Required if rel_col is not specified.

        rel_col (str): Column containing relational operators. If specified, overrides the default relation column
           for db.

        output_rel_col (str): If specified, put the standardized operators in a new column with this name and leave
           the original operator column unchanged.

        invert (bool): If true, replace the inequality operators with their inverses. This is useful when a reported
           value such as IC50 is converted to its negative log such as pIC50.

    Returns:
        DataFrame: Dataframe with the standardized relationship sybmols

    """
    if rel_col is None:
        relation_cols = dict(ChEMBL='standard_relation', DTC='standard_relation', GoStar='activity_prefix')
        try:
            rel_col = relation_cols[db]
        except KeyError:
            raise ValueError(f"Unknown database {db} for standardize_relations") 

    if output_rel_col is None:
        output_rel_col = rel_col

    try:
        dset_df[rel_col] = dset_df[rel_col].fillna('=')
    except KeyError:
        raise ValueError(f"Dataset doesn't contain relation column {rel_col} expected for source database {db}")
    ops = dset_df[rel_col].values
    if db == 'ChEMBL':
        # Remove annoying quotes around operators
        ops = [op.lstrip("'").rstrip("'") for op in ops]
    op_dict = {
        ">=": ">",
        "<": "<",
        "<=": "<",
        ">": ">",
        ">R": ">",
        ">=R": ">",
        "<R": "<",
        "<=R": "<",
        "~": "=",
        "=": "="
    }
    ops = np.array([op_dict.get(op, "@") for op in ops])
    if invert:
        inv_op = {'>': '<', '<': '>'}
        ops = np.array([inv_op.get(op, op) for op in ops])
    dset_df[output_rel_col] = ops
    dset_df = dset_df[dset_df[output_rel_col] != "@"].copy()
    return dset_df

# ----------------------------------------------------------------------------------------------------------------------
# DTC-specific curation functions
# ----------------------------------------------------------------------------------------------------------------------
def upload_file_dtc_raw_data(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,file_path,
    		   data_origin='journal',  species='human',  
                           force_update=False):
    """Uploads raw DTC data to the datastore

    Upload a raw dataset to the datastore from the given DataFrame.
    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://doi.org/10.1016/j.chembiol.2017.11.009' as the doi.
    This also assumes that the id_col is 'compound_id'

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        file_path (str): The filepath of the dataset.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """

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

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        #uploaded_file = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
        #                               description=description,
        #                               tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
        #                               override_check=True, return_metadata=True)
        uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,
            title = title, description=description, tags=tags, key_values=kv, client=None,
            dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid

def filter_dtc_data(orig_df,geneNames):
    """Extracts and post processes JAK1, 2, and 3 datasets from DTC

        This is specific to the DTC database.
        Extract JAK1, 2 and 3 datasets from Drug Target Commons database, filtered for data usability.
        filter criteria:
        
            gene_names == JAK1 | JAK2 | JAK3
            InChi key not missing
            standard_type IC50
            units NM
            standard_relation mappable to =, < or >
            wildtype_or_mutant != 'mutated'
            valid SMILES
            maps to valid RDKit base SMILES
            standard_value not missing
            pIC50 > 3

    Args:
        orig_df (DataFrame): Input DataFrame. Must contain the following columns: gene_names
           standard_inchi_key, standard_type, standard_units, standard_value, compound_id,
           wildtype_or_mutant.
        
        geneNames (list): A list of gene names to filter out of orig_df e.g. ['JAK1', 'JAK2'].

    Returns:
        DataFrame: The filtered rows of the orig_df

    """
    dset_df = orig_df[orig_df.gene_names.isin(geneNames) &
                      ~(orig_df.standard_inchi_key.isna()) &
                      (orig_df.standard_type == 'IC50') &
                      (orig_df.standard_units == 'NM') &
                      ~orig_df.standard_value.isna() &
                      ~orig_df.compound_id.isna() &
                      (orig_df.wildtype_or_mutant != 'mutated') ]
    return dset_df

def ic50topic50(x) :
    """Calculates pIC50 from IC50

    Args:
        x (float): An IC50 in nanomolar (nM) units.

    Returns:
        float: The pIC50.
    """
    print(x)
    return -np.log10((x/1000000000.0))

def compute_negative_log_responses(df, unit_col='unit', value_col='value', 
        new_value_col='average_col', relation_col=None, new_relation_col=None,
        unit_conv={'uM':lambda x: x*1e-6, 'nM':lambda x: x*1e-9}, inplace=False):
    """Given the response values in `value_col` (IC50, Ki, Kd, etc.), compute their negative base 10 logarithms
    (pIC50, pKi, pKd, etc.) after converting them to molar units and store them in `new_value_col`.
    If `relation_col` is provided, replace any '<' or '>' relations with their opposites and store the result
    in `new_relation_col` (if provided), or in `relation_col` if note.
    Rows where the original value is 0 or negative will be dropped from the dataset.

    Args:
        df (DataFrame): A DataFrame that contains `value_col`, `unit_col` and `relation_col`.

        unit_conv (dict): A dictionary mapping concentration units found in `unit_col` to functions
        that convert the corresponding concentrations to molar. The default handles micromolar and
        nanomolar units, represented as 'uM' and 'nM' respectively.

        unit_col (str): Column containing units.

        value_col (str): Column containing input values.

        new_value_col (str): Column to receive converted values.

        relation_col (str): Column containing relational operators for censored data.

        new_relation_col (str): Column to receive inverted relations applicable to the negative log transformed values.

        inplace (bool): If True, the input DataFrame is modified in place when possible. The default is to return a copy

    Returns:
        DataFrame: A table containing the transformed values and relations.
    """

    missing_units = list(set(df[unit_col]) - set(unit_conv.keys()))
    assert len(missing_units) == 0, f"unit_conv lacks converter(s) for units {', '.join(missing_units)}"
    # Drop rows for which log can't be computed
    if np.any(df[value_col].values <= 0.0):
        df = df[df[value_col] > 0.0].copy()
    elif not inplace:
        df = df.copy()
    new_vals = []
    new_relations = []
    inverse_rel = str.maketrans('<>', '><')
    for i, row in df.iterrows():
        ic50 = row[value_col]
        pic50 = -np.log10(unit_conv[row[unit_col]](ic50))
        new_vals.append(pic50)
        if relation_col is not None:
            rel = row[relation_col]
            if isinstance(rel, str):
                rel = rel.translate(inverse_rel)
            new_relations.append(rel)
    df[new_value_col] = new_vals
    if relation_col is not None:
        if new_relation_col is None:
            df[relation_col] = new_relations
        else:
            df[new_relation_col] = new_relations

    return df


def convert_IC50_to_pIC50(df, unit_col='unit', value_col='value', 
        new_value_col='average_col', relation_col=None, new_relation_col=None,
        unit_conv={'uM':lambda x: x*1e-6, 'nM':lambda x: x*1e-9}, inplace=False):
    """For backward compatibiltiy only: equivalent to calling `compute_negative_log_responses` with the same arguments."""
    return compute_negative_log_responses(df, unit_col=unit_col, value_col=value_col, new_value_col=new_value_col,
                                          relation_col=relation_col, new_relation_col=new_relation_col,
                                          unit_conv=unit_conv, inplace=inplace)


def down_select(df,kv_lst) :
    """Filters rows given a set of values

    Given a DataFrame and a list of tuples columns (k) to values (v), this function
    filters out all rows where df[k] == v.

    Args:
        df (DataFrame): An input DataFrame.

        kv_list (list): A list of tuples of (column, value)

    Returns:
        DataFrame: Rows where all df[k] == v
    """
    for k,v in kv_lst :
        df=df[df[k]==v] 
    return df

def get_smiles_dtc_data(nm_df,targ_lst,save_smiles_df):
    """Returns SMILES strings from DTC data

    nm_df must be a DataFrame from DTC with the following columns: gene_names,
    standard_type, standard_value, 'standard_inchi_key', and standard_relation.

    This function selects all rows where nm_df['gene_names'] is in targ_lst,
    nm_df['standard_type']=='IC50', nm_df['standard_relation']=='=', and
    'standard_value' > 0.

    Then pIC50 values are calculated and added to the 'PIC50' column, and
    smiles strings are merged in from save_smiles_df

    Args:
        nm_df (DataFrame): Input DataFrame.

        targ_lst (list): A list of targets.

        save_smiles_df (DataFrame): A DataFrame with the column 'standard_inchi_key'

    Returns:
        list, list: A list of smiles and a list of inchi keys shared between targets.
    """
    save_df={}
    for targ in targ_lst :
        lst1= [ ('gene_names',targ),('standard_type','IC50'),('standard_relation','=') ]
        lst1_tmp= [ ('gene_names',targ),('standard_type','IC50')]
        jak1_df=down_select(nm_df,lst1)
        jak1_df_tmp=down_select(nm_df,lst1_tmp)
        print(targ,"distinct compounds = only",jak1_df['standard_inchi_key'].nunique())
        print(targ,"distinct compounds <,>,=",jak1_df_tmp['standard_inchi_key'].nunique())
        ## we convert to log values so make sure there are no 0 values
        save_df[targ]=jak1_df_tmp[jak1_df_tmp['standard_value']>0]

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
    print(shared_df['standard_value'])
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
    """Returns SMILES strings from DTC data

    nm_df must be a DataFrame from DTC with the following columns: gene_names,
    standard_type, standard_value, 'standard_inchi_key', and standard_relation.

    This function selects all rows where nm_df['gene_names'] is in targ_lst,
    nm_df['standard_type']=='IC50', nm_df['standard_relation']=='=', and
    'standard_value' > 0.

    Then pIC50 values are calculated and added to the 'PIC50' column, and
    smiles strings are merged in from save_smiles_df

    Args:
        nm_df (DataFrame): Input DataFrame.

        targ_lst (list): A list of targets.

        save_smiles_df (DataFrame): A DataFrame with the column 'standard_inchi_key'

    Returns:
        list, list, str: A list of smiles. A list of inchi keys shared between targets.
            And a description of the targets
    """
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

def upload_df_dtc_smiles(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,smiles_df,orig_fileID,
    		   data_origin='journal',  species='human',  
                           force_update=False):
    """Uploads DTC smiles data to the datastore

    Upload a raw dataset to the datastore from the given DataFrame.
    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://doi.org/10.1016/j.chembiol.2017.11.009' as the doi.
    This also assumes that the id_col is 'compound_id'

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        smiles_df (DataFrame): DataFrame containing SMILES to be uploaded.

        orig_fileID (str): Source file id used to generate smiles_df.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """
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

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=smiles_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)
        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid

def atom_curation(targ_lst, smiles_lst, shared_inchi_keys):
    """Apply ATOM standard 'curation' step to "shared_df": Average replicate assays,
    remove duplicates and drop cases with large variance between replicates.
    mleqonly

    Args:
        targ_lst (list): A list of targets.

        smiles_lst (list): A list of DataFrames.
            These DataFrames must contain the columns gene_names, standard_type,
            standard_relation, standard_inchi_key, PIC50, and rdkit_smiles

        shared_inchi_keys (list): A list of inchi keys used in this dataset.

    Returns:
        list, list:A list of curated DataFrames and a list of the number of compounds
            dropped during the curation process for each target.

    """
    imp.reload(curate_data)
    tolerance=10
    column='PIC50' #'standard_value'
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

def upload_df_dtc_mleqonly(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,data_df,dtc_smiles_fileID,
    		   data_origin='journal',  species='human',  
                           force_update=False):
    """Uploads DTC mleqonly data to the datastore

    Upload mleqonly data to the datastore from the given DataFrame. The DataFrame
    must contain the column 'rdkit_smiles' and 'VALUE_NUM_mean'. This function is
    meant to upload data that has been aggregated using
    atomsci.ddm.utils.curate_data.average_and_remove_duplicates.
    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://doi.org/10.1016/j.chembiol.2017.11.009' as the doi.
    This also assumes that the id_col is 'compound_id'.

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        data_df (DataFrame): DataFrame to be uploaded.

        dtc_smiles_fileID (str): Source file id used to generate data_df.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """

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
        'target' : target,
        'target_type' : target_type,
        'id_col' : 'compound_id',
        'response_col' : 'VALUE_NUM_mean',
        'prediction_type' : 'regression',
        'smiles_col' : 'rdkit_smiles',
        'units' : 'unitless',
        'source_file_id' : dtc_smiles_fileID
     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

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
    """Uploads DTC mleqonly classification data to the datastore

    Upload mleqonly classification data to the datastore from the given DataFrame. The DataFrame
    must contain the column 'rdkit_smiles' and 'binary_class'. This function is
    meant to upload data that has been aggregated using
    atomsci.ddm.utils.curate_data.average_and_remove_duplicates and then thresholded to
    make a binary classification dataset.
    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://doi.org/10.1016/j.chembiol.2017.11.009' as the doi.
    This also assumes that the id_col is 'compound_id'.

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        data_df (DataFrame): DataFrame to be uploaded.

        dtc_mleqonly_fileID (str): Source file id used to generate data_df.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """
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
    """Uploads DTC base smiles data to the datastore

    Uploads base SMILES string for the DTC dataset.

    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://doi.org/10.1016/j.chembiol.2017.11.009' as the doi.
    This also assumes that the id_col is 'compound_id', the response column is set to PIC50,
    and the SMILES are assumed to be in 'base_rdkit_smiles'.

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        data_df (DataFrame): DataFrame to be uploaded.

        dtc_mleqonly_fileID (str): Source file id used to generate data_df.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """
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
    """Uploads regression DTC data to the datastore

    Uploads regression dataset for DTC dataset.

    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://doi.org/10.1016/j.chembiol.2017.11.009' as the doi.
    This also assumes that the id_col is 'compound_id', the response column is set to PIC50.

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        data_df (DataFrame): DataFrame to be uploaded.

        dtc_smiles_fileID(str): Source file id used to generate data_df.

        smiles_column (str): Column containing SMILES.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """

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

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        #uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

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
    """Uploads DTC classification data to the datastore

    Uploads binary classiciation data for the DTC dataset. Classnames are assumed to
    be 'active' and 'inactive'

    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://doi.org/10.1016/j.chembiol.2017.11.009' as the doi.
    This also assumes that the id_col is 'compound_id', the response column is set to PIC50.

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        data_df (DataFrame): DataFrame to be uploaded.

        dtc_smiles_regr_all_fileID(str): Source file id used to generate data_df.

        smiles_column (str): Column containing SMILES.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """
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

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid

# ----------------------------------------------------------------------------------------------------------------------
# Excape-specific curation functions
# ----------------------------------------------------------------------------------------------------------------------

def upload_file_excape_raw_data(dset_name, title, description, tags,
                            functional_area, 
                           target, target_type, activity, assay_category,file_path,
    		   data_origin='journal',  species='human',  
                           force_update=False):
    """Uploads raw Excape data to the datastore

    Upload a raw dataset to the datastore from the given DataFrame.
    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://dx.doi.org/10.1186%2Fs13321-017-0203-5 as the doi.
    This also assumes that the id_col is 'Original_Entry_ID'

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        file_path (str): The filepath of the dataset.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """
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

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        #uploaded_file = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
        #                               description=description,
        #                               tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
        #                               override_check=True, return_metadata=True)
        uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid

def get_smiles_excape_data(nm_df,targ_lst):
    """Calculate base rdkit smiles

    Divides up nm_df based on target and makes one DataFrame for each target.

    Rows with NaN pXC50 values are dropped. Base rdkit SMILES are calculated
    from the SMILES column using
    atomsci.ddm.utils.struct_utils.base_rdkit_smiles_from_smiles. A new column,
    'rdkit_smiles, is added to each output DataFrame.

    Args:
        nm_df (DataFrame): DataFrame for Excape database. Should contain the columns,
            pXC50, SMILES, and Ambit_InchiKey

        targ_lst (list): A list of targets to filter out of nm_df

    Returns:
        list, list: A list of DataFrames, one for each target, and a list of
            all inchi keys used in the dataset.

    """
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
    """Uploads Excape SMILES data to the datastore

    Upload SMILES to the datastore from the given DataFrame.
    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://dx.doi.org/10.1186%2Fs13321-017-0203-5 as the doi.
    This also assumes that the id_col is 'Original_Entry_ID'

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        smiles_df (DataFrame): DataFrame containing SMILES to be uploaded.

        orig_fileID (str): Source file id used to generate smiles_df.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """
    bucket = 'public'
    #he6: this used to say _dtc_smiles.csv
    filename = '%s_excape_smiles.csv' % dset_name
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

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=smiles_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)
        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid

def atom_curation_excape(targ_lst, smiles_lst, shared_inchi_keys):
    """Apply ATOM standard 'curation' step: Average replicate assays,
    remove duplicates and drop cases with large variance between replicates.
    Rows with NaN values in rdkit_smiles, VALUE_NUM_mean, and pXC50 are dropped

    Args:
        targ_lst (list): A list of targets.

        smiles_lst (list): A of DataFrames.
            These DataFrames must contain the columns gene_names, standard_type,
            standard_relation, standard_inchi_key, pXC50, and rdkit_smiles

        shared_inchi_keys (list): A list of inchi keys used in this dataset.

    Returns:
        list:A list of curated DataFrames
    """
    imp.reload(curate_data)
    tolerance=10
    column='pXC50' #'standard_value'
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
    """Uploads Excape mleqonly data to the datastore

    Upload mleqonly to the datastore from the given DataFrame.
    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://dx.doi.org/10.1186%2Fs13321-017-0203-5 as the doi.
    This also assumes that the id_col is 'Original_Entry_ID', smiles_col is 'rdkit_smiles'
    and response_col is 'VALUE_NUM_mean'.

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        data_df (DataFrame): DataFrame containing SMILES to be uploaded.

        smiles_fileID (str): Source file id used to generate data_df.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """

    bucket = 'public'
    #he6: this used to say _dtc_mleqonly.csv
    filename = '%s_excape_mleqonly.csv' % dset_name
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
        'target' : target,
        'target_type' : target_type,
        'id_col' : 'Original_Entry_ID',
        'response_col' : 'VALUE_NUM_mean',
        'prediction_type' : 'regression',
        'smiles_col' : 'rdkit_smiles',
        'units' : 'unitless',
        'source_file_id' : smiles_fileID
     }

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

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
    """Uploads Excape mleqonly classification data to the datastore

    data_df contains a binary classification dataset with 'active' and 'incative' classes.

    Upload mleqonly classification to the datastore from the given DataFrame.
    Returns the datastore OID of the uploaded dataset. The dataset is uploaded to the
    public bucket and lists https://dx.doi.org/10.1186%2Fs13321-017-0203-5 as the doi.
    This also assumes that the id_col is 'Original_Entry_ID', smiles_col is 'rdkit_smiles'
    and response_col is 'binary_class'.

    Args:
        dset_name (str): Name of the dataset. Should not include a file extension.

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): Must be a list of strings.

        functional_area (str): The functional area.

        target (str): The target.

        target_type (str): The target type of the dataset.

        activity (str): The activity of the dataset.

        assay_category (str): The assay category of the dataset.

        data_df (DataFrame): DataFrame containing SMILES to be uploaded.

        mleqonly_fileID (str): Source file id used to generate data_df.

        data_origin (str): The origin of the dataset e.g. journal.

        species (str): The species of the dataset e.g. human, rat, dog.

        force_update (bool): Overwrite existing datasets in the datastore.

    Returns:
        str: datastore OID of the uploaded dataset.
    """
    bucket = 'public'
    #he6: this used to say _dtc_mleqonly.csv
    filename = '%s_excape_mleqonly_class.csv' % dset_name
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

    #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        uploaded_file = dsf.upload_df_to_DS(bucket=bucket, filename=filename,df=data_df, title = title, description=description, tags=tags, key_values=kv, client=None, dataset_key=dataset_key, override_check=False, return_metadata=True)

        #uploaded_file = dsf.upload_file_to_DS(bucket=bucket, filepath=file_path, filename=filename,     	title = title, description=description, tags=tags, key_values=kv, client=None, 			dataset_key=dataset_key, override_check=False, return_metadata=True)

        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        uploaded_file = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)

    raw_dset_oid = uploaded_file['dataset_oid']
    return raw_dset_oid
