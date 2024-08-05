import pytest
import atomsci.ddm.pipeline.featurization as feat
import deepchem as dc
from atomsci.ddm.pipeline import model_datasets as md

try:
    from mol_vae_features import MoleculeVAEFeaturizer
    mol_vae_supported = True
except ModuleNotFoundError:
    mol_vae_supported = False

import utils_testing as utils

#WARNING: assuming model_dataset.py can create a functional data object.
#WARNING: assuming that config_delaney.json and config_datastore_cav12.json are in the current directory. 

ownership = 'gskusers-ad'
datastore_is_down = utils.datastore_status()

(delaney_params_ecfp, data_obj_ecfp, df_delaney) = utils.delaney_objects()
featurizer_ecfp = data_obj_ecfp.featurization
md.check_task_columns(delaney_params_ecfp, df_delaney)

(delaney_params_graphconv, data_obj_graphconv, df_delaney) = utils.delaney_objects(featurizer="graphconv")
featurizer_graphconv = data_obj_graphconv.featurization
md.check_task_columns(delaney_params_graphconv, df_delaney)

if mol_vae_supported:
    (delaney_params_molvae, data_obj_molvae, df_delaney) = utils.delaney_objects(featurizer="molvae")
    featurizer_molvae = data_obj_molvae.featurization

if not datastore_is_down:
    (datastore_params, mdl_datastore, df_datastore) = utils.datastore_objects()

"""
#TODO: Will implement testing for MOE descriptors when there is more computational bandwidth (takes up to 5-10 minutes to load up the descriptors on a CPU)
#Note: dragon7 is not yet implemented. Only moe descriptors are supported

"""

#***********************************************************************************  
def test_remove_duplicate_smiles(caplog):
    """checking for removal of duplicates and activity with dataframes with no dupes"""
    #checking for a file with no dupes
    no_dupe_df = feat.remove_duplicate_smiles(df_delaney,smiles_col=delaney_params_ecfp.smiles_col)
    test = []
    test.append((len(no_dupe_df) <= len(df_delaney)))
    test.append(len(set(no_dupe_df[delaney_params_ecfp.smiles_col].values.tolist())) <= len(no_dupe_df))
    
    #checking for a file with dupes
    # TODO (ksm): Replace the CaV1.2 dataset I plugged in for datastore testing with a non-ML-ready dataset with dupes.
    if not datastore_is_down:
        no_dupe_df = feat.remove_duplicate_smiles(df_datastore,smiles_col=datastore_params.smiles_col)
        test.append(len(no_dupe_df) <= len(df_datastore))
        test.append(len(set(no_dupe_df[datastore_params.smiles_col].values.tolist())) <= len(no_dupe_df))
    assert all(test)
#***********************************************************************************

def test_create_featurization_dynamicfeaturization():
    """testing if classes are properly generated from the factory method. Asserting that the correct methods exist, and are callable. Asserting correct dc featurizer object is called"""
    test = []
    test.append(isinstance(featurizer_ecfp, feat.DynamicFeaturization))
    methods = ["featurize_data","get_feature_columns","extract_prefeaturized_data","get_feature_count","get_feature_specific_metadata","get_featurized_dset_name","get_featurized_data_subdir"]
    for method in methods:
        test.append(callable(getattr(featurizer_ecfp,method)))
        
    test.append(isinstance(featurizer_ecfp.featurizer_obj, dc.feat.CircularFingerprint))
    
    test.append(isinstance(featurizer_graphconv.featurizer_obj, dc.feat.graph_features.ConvMolFeaturizer))    

    if mol_vae_supported:
        test.append(isinstance(featurizer_molvae.featurizer_obj, MoleculeVAEFeaturizer))
    
    assert all(test)
        

    
#***********************************************************************************    
def test_get_feature_columns_dynamicfeaturization():
    """Testing that dynamic featurization is pulling out the correct number of features and 'column names'. Also technically testing get_feature_count"""
    test = []
    cols = featurizer_ecfp.get_feature_columns()
    test.append(len(cols) == delaney_params_ecfp.ecfp_size)
    test.append(cols[0] == 'c0')
    
    cols = featurizer_graphconv.get_feature_columns()
    test.append(len(cols) == 75)
    
    if mol_vae_supported:
        cols = featurizer_molvae.get_feature_columns()
        test.append(len(cols) == 292)
    assert all(test)


#***********************************************************************************
#def test_extract_prefeaturized_data_dynamicfeaturization():
    #loads in a previously featurized dataset. Should always return None or an exception if using DynamicFeaturization
    # TODO: NEEDS TO BE UPDATED SINCE dataset_file no longer exists
    # featurizer_ecfp.extract_prefeaturized_data(delaney_params_ecfp.dataset_file, data_obj_ecfp) == (None,None,None,None)
#***********************************************************************************

# def test_featurize_data_dynamicfeaturization_ecfp():
#     #testing featurization of ecfp, graphconv. Checking to see if the types of featurization are reasonable, if the lengths of the output from the featurizer are the same, and if the output of ids, values, and attr are (str, np.float64, panda.df) and that the index name and columh name match those specified in the parameters
#     test = []
#     (features, ids, values, attr) = featurizer_ecfp.featurize_data(df_delaney, data_obj_ecfp)
#     test.append(len(features) == len(ids) and len(values) == len(attr) and len(values) == len(features))
#     test.append(len(features[0]) == delaney_params_ecfp.ecfp_size)
#     test.append(isinstance(features[0], np.ndarray))
#     test.append(isinstance(ids[0], str))
#     test.append(isinstance(values[0], np.ndarray))
#     test.append(attr.index.name == delaney_params_ecfp.id_col)
#     test.append(attr.columns[0] == delaney_params_ecfp.smiles_col)
#     assert all(test)
    
# def test_featurize_data_dynamicfeaturization_graphconv():
#     test = []
#     (features, ids, values, attr) = featurizer_graphconv.featurize_data(df_delaney, data_obj_graphconv)
#     test.append(len(features) == len(ids) and len(values) == len(attr) and len(values) == len(features))
#     test.append(isinstance(features[0], dc.feat.mol_graphs.ConvMol))
#     test.append(isinstance(ids[0], str))
#     test.append(isinstance(values[0], np.ndarray))
#     test.append(attr.index.name == delaney_params_graphconv.id_col)
#     test.append(attr.columns[0] == delaney_params_graphconv.smiles_col)
#     assert all(test)
#     """  
#     #TODO: Fix molvae featurization
# def test_featurize_data_dynamicfeaturization_molvae():
#     
#     (features, ids, values, attr) = featurizer_molvae.featurize_data(df_delaney, data_obj_molvae)
#     test.append(len(features) == len(ids) and len(values) == len(attr) and len(values) == len(features)
#     #TODO: find DC type molvae
#     #test.append(isinstance(features[0], dc.feat.mol_graphs.ConvMol)
#     test.append(isinstance(ids[0], str)
#     test.append(isinstance(values[0],np.float64)
#     test.append(attr.index.name == delaney_params_molvae.id_col
#     test.append(attr.columns[0] == delaney_params_molvae.smiles_col
#     """
    

#***********************************************************************************
def test_get_featurized_dset_name_dynamicfeaturization():
    """Dynamic featurization does not support get_featurized_dset_name"""
    with pytest.raises(Exception):
        featurizer_ecfp.get_featurized_dset_name(data_obj_ecfp.dataset_name)
#***********************************************************************************

def test_get_featurized_data_subdir_dynamicfeaturization():
    """Dynamic featurization does not support get_featurized_data_subdir"""
    with pytest.raises(Exception):
        featurizer_ecfp.get_featurized_data_subdir()
#***********************************************************************************

def test_get_feature_specific_metadata_dynamicfeaturization():
    """Dynamic featurization returns a dictionary of parameter settings of the featurization object that are specific to the feature type. Testing all three currently implemented featurizers (ecfp, graphconv, molvae)"""
    ecfp_metadata = featurizer_ecfp.get_feature_specific_metadata(delaney_params_ecfp)
    assert ecfp_metadata =={'ecfp_specific':
                            {"ecfp_radius": delaney_params_ecfp.ecfp_radius,
                             "ecfp_size":delaney_params_ecfp.ecfp_size }}
    graphconv_metadata = featurizer_graphconv.get_feature_specific_metadata(delaney_params_graphconv)
    assert graphconv_metadata == {}
    
    #molvae_metadata = featurizer_molvae.get_feature_specific_metadata(delaney_params_molvae)
    #test.append(molvae_metadata =={'MolVAESpecific': 
    #                        {"mol_vae_model_file": delaney_params_molvae.mol_vae_model_file}}

#***********************************************************************************
    #TODO: will begin testing of DescriptorFeaturization and PersistentFeaturization after checking the base cases of 'ecfp, graphconv, molvae'
    #WARNING: Cannot test the descriptors featurization object with datastore integration. Descriptors are effectively too large to properly stream from the datastore.
#***********************************************************************************
#***********************************************************************************

#(desc_params, desc_data_obj, MAOA_df) = utils.moe_descriptors()
#
#
#featurizer_desc = desc_data_obj.featurization
#desc_data_obj.check_task_columns(MAOA_df)
#featurized_dset_name = featurizer_desc.get_featurized_dset_name(desc_data_obj.dataset_name)
#data_dir = os.path.join(desc_data_obj.params.output_dir, featurizer_desc.get_featurized_data_subdir())
#featurized_dset_path = os.path.join(data_dir, featurized_dset_name)

#********************************
# (desc_params_ds, desc_data_obj_ds, MAOA_df_ds) = utils.moe_descriptors(datastore = True)
# 
# featurizer_desc_ds = desc_data_obj_ds.featurization
# desc_data_obj_ds.check_task_columns(MAOA_df_ds)
    
# def test_create_featurization_descriptorfeaturization():
#     #testing if classes are properly generated from the factory method. Asserting that the correct methods exist, and are callable. Asserting correct dc featurizer object is called
#     test = []
#     test.append(isinstance(featurizer_desc, feat.DescriptorFeaturization))
#     methods = ["featurize_data","get_feature_columns","extract_prefeaturized_data","get_feature_count","get_feature_specific_metadata","get_featurized_dset_name","get_featurized_data_subdir"]
#     for method in methods:
#         test.append(callable(getattr(featurizer_desc,method)))
#     test.append(featurizer_desc.descriptor_type == desc_params.descriptor_type)
#     test.append(featurizer_desc.descriptor_key == desc_params.descriptor_key)
#     test.append(featurizer_desc.precomp_descr_table.empty)
#     assert all(test)
    
#***********************************************************************************
def get_featurized_dset_name_descriptorfeaturization():
    #sanity check that we are getting the correct featurized dset name. Also generating the featurized dset name for saving the featurized dataset. Used in model_datasets.save_featurized_data
    featurized_dset_name = featurizer_desc.get_featurized_dset_name(desc_data_obj.dataset_name)
    assert featurized_dset_name == "subset_" + featurizer_desc.descriptor_key + "_" + featurizer_desc.dataset_name + ".csv"
    
#***********************************************************************************
def get_featurized_data_subdir_descriptorfeaturization():
    #sanity check for gtting the name of the subdirectory. used in model_dataset.save_featurized_data
    assert featurizer_desc.get_featurized_data_subdir() == "scaled_descriptors"
#***********************************************************************************
def get_get_feature_columns_and_count_descriptorfeaturization():
    #sanity check for getting the feature columns. Also testing get_feature_count. Since dragon7 is not yet implemented, will not test for those columns.
    test = []
    moe_desc_cols = featurizer_desc.get_feature_columns() 
    test.append(moe_desc_cols[0] == 'ASA+_per_atom')
    test.append(len(moe_desc_cols) == 306 and featurizer_desc.get_feature_count() == 306)
    moe_desc_cols_all = featurizer_desc.get_feature_columns(include_all=True) 
    excluded_moe_desc_cols = ['E', 'E_ang', 'E_ele', 'E_nb', 'E_oop', 'E_sol', 'E_stb', 'E_str', 'E_strain', 'E_tor', 'E_vdw']
    test.append(set(excluded_moe_desc_cols).intersection(set(moe_desc_cols_all)) == set(excluded_moe_desc_cols))
    test.append(len(excluded_moe_desc_cols) == 317 and featurizer_desc.get_feature_count(include_all=False) == 317)
    assert all(test)
    
#***********************************************************************************
"""
def test_featurize_data_descriptorfeaturization():
    #testing for correct descriptor featurization. Checking the length of all ouputs are equivalence. 
    #Checks that the features are a pandas dataframe and that the columns of the features match those in moe_desc_cols. 
    #Checks for the correct typing of the ids, vals, and attr.
    #Technically is testing the save_featurized_data function in model_datasets.
    test = []
    (features,ids,vals,attr) = featurizer_desc.featurize_data(MAOA_df, desc_data_obj)
    num_descriptors = len(featurizer_desc.moe_desc_cols)
    (rows, cols) = features.shape
    test.append(rows == len(ids) and len(vals) == len(attr) and len(vals) == len(ids))
    test.append(isinstance(features, np.ndarray))
    test.append(cols == 306)
    test.append(isinstance(ids[0], str))
    test.append(isinstance(vals[0],np.float64))
    test.append(attr.index.name == desc_params.id_col)
    test.append(attr.columns[0] == desc_params.smiles_col)
    test.append(not featurizer_desc.precomp_descr_table.empty)
    (rows_fullfeature, cols_fullfeature) = featurizer_desc.precomp_descr_table.shape
    test.append(cols_fullfeature == 323)
    test.append(rows_fullfeature == 1857923)
    test.append(os.path.isfile(featurized_dset_path))
    from pathlib import Path
    path_metadata = Path(featurized_dset_path)
    test.append(path_metadata.group() == ownership)
    assert all(test)
    #*************************************************
    
    
def test_featurize_data_descriptorfeaturization_ds():
    #testing for correct descriptor featurization. Checking the length of all ouputs are equivalence. 
    #Checks that the features are a pandas dataframe and that the columns of the features match those in moe_desc_cols. 
    #Checks for the correct typing of the ids, vals, and attr.
    #Technically is testing the save_featurized_data function in model_datasets.
    test = []
    (features,ids,vals,attr) = featurizer_desc_ds.featurize_data(MAOA_df_ds, desc_data_obj_ds)
    num_descriptors = len(featurizer_desc_ds.moe_desc_cols)
    (rows, cols) = features.shape
    test.append(rows == len(ids) and len(vals) == len(attr) and len(vals) == len(ids))
    test.append(isinstance(features, np.ndarray))
    test.append(cols == 306)
    test.append(isinstance(ids[0], str))
    test.append(isinstance(vals[0],np.float64))
    test.append(attr.index.name == desc_params_ds.id_col)
    test.append(attr.columns[0] == desc_params_ds.smiles_col)
    test.append(not featurizer_desc.precomp_descr_table.empty)
    (rows_fullfeature, cols_fullfeature) = featurizer_desc.precomp_descr_table.shape
    test.append(cols_fullfeature == 323)
    test.append(rows_fullfeature == 1857923)
    assert all(test)
    #*****************************

def test_extract_prefeaturized_data_descriptorfeaturization_ds():
    #Now testing the proper reloading of the featurized data
    test = []
    (features,ids,vals,attr) = featurizer_desc_ds.featurize_data(MAOA_df_ds, desc_data_obj_ds)
    featurized_dset_key = featurizer_desc_ds.get_featurized_dset_name(desc_data_obj_ds.dataset_name)
    dset_df = csv2df(featurized_dset_key, desc_data_obj_ds.params.bucket, desc_data_obj_ds.ds_client)
    reloaded = desc_data_obj_ds.load_featurized_data()
    test.append(reloaded.equals(dset_df))
    (features_ext, ids_ext, vals_ext, attr_ext)=featurizer_desc_ds.extract_prefeaturized_data(reloaded,desc_data_obj_ds)
    diff_features = abs(features - features_ext) < 1E-13
    test.append(diff_features.all())
    test.append(set(ids) == set(ids_ext)) 
    test.append((abs(np.array(vals) - np.array(vals_ext)) < 1E-13).all()) 
    test.append(isinstance(vals, list))
    test.append(attr.equals(attr_ext))
    assert all(test)

                                      
#***********************************************************************************
def test_get_feature_specific_metadata_descriptorfeaturization(caplog):
    #testing feature metadata is properly generated
    
    metadata_dict = featurizer_desc.get_feature_specific_metadata(desc_params)
    assert metadata_dict == {'DescriptorSpecific' : 
                            {"descriptor_type" : desc_params.descriptor_type,
                           "descriptor_key" : desc_params.descriptor_key}}
                                         


#***********************************************************************************
"""
