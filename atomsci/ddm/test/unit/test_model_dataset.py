import os
import pandas as pd
import pytest
import json
import numpy as np
import atomsci.ddm.pipeline.featurization as feat
import atomsci.ddm.pipeline.splitting as split
import atomsci.ddm.utils.datastore_functions as ds
import deepchem as dc
import atomsci.ddm.pipeline.model_datasets as model_dataset
import atomsci.ddm.pipeline.parameter_parser as parse
from atomsci.ddm.pipeline import model_pipeline as mp
import utils_testing as utils

"""This testing script assumes that /ds/data/public/delaney/delaney-processed.csv is still on the same path on twintron. Assumes that the dataset_key: /ds/projdata/gsk_data/GSK_derived/PK_parameters/gsk_blood_plasma_partition_rat_crit_res_data.csv under the bucket gskdata and with the object_oid: 5af0e6368003ff018de33db5 still exists. 
"""

#The dataset object from file is a delaney dataset using an ecfp featurizer with a default scaffold split.

datastore_is_down = utils.datastore_status()


# (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
# (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(split_strategy="train_valid_test",splitter = "scaffold")

# ksm: params.response_cols is required now by our code, though it's not marked as such in parameter_parser.py.
# Therefore, the following line will fail.
#(params_from_file_noy, dataset_obj_from_file_noy, df_delaney) = utils.delaney_objects(y = None)

# ksm: Creating the following dataset will fail now, because it doesn't actually contain the response columns.
#(params_from_file_wrongy, dataset_obj_from_file_wrongy, df_delaney) = utils.delaney_objects(y = ["not","a","task"])

delaney_from_disk = pd.read_csv("delaney-processed.csv")
    

if not datastore_is_down:
    (params_from_ds, dataset_obj_from_datastore, df_datastore) = utils.datastore_objects()
    #(params_from_ds_noy, dataset_obj_from_datastore_noy, df_datastore) = utils.datastore_objects(y = None)
    #(params_from_ds_wrongy, dataset_obj_from_datastore_wrongy, df_datastore) = utils.datastore_objects(y = ["not","a","task"])

    df_datastore = ds.retrieve_dataset_by_datasetkey(params_from_ds.dataset_key,params_from_ds.bucket)
    

DD = dc.data.datasets.NumpyDataset



#***********************************************************************************
def test_create_model_dataset():
    """testing if classes are properly generated from the factory method. Asserting that the correct methods exist, and are callable. """

    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    test_list = []
    test_list.append(isinstance(dataset_obj_from_file, model_dataset.FileDataset))

    methods = ["load_full_dataset","load_featurized_data","get_featurized_data","get_dataset_tasks","split_dataset","get_split_metadata","save_split_dataset","load_presplit_dataset","save_featurized_data","combined_training_data"]
    #testing datastore
    if not datastore_is_down:
        test_list.append(isinstance(dataset_obj_from_datastore, model_dataset.DatastoreDataset))

        for method in methods:
            test_list.append(callable(getattr(dataset_obj_from_datastore,method)))
        test_list.append(dataset_obj_from_datastore.ds_client is not None)
        test_list.append(dataset_obj_from_datastore.dataset_name)

    #testing from file
    for method in methods:
        test_list.append(callable(getattr(dataset_obj_from_file,method)))

    test_list.append(isinstance(dataset_obj_from_file.featurization, feat.DynamicFeaturization))
    test_list.append(isinstance(dataset_obj_from_file.splitting, split.TrainValidTestSplitting))
    test_list.append(dataset_obj_from_file.dataset_name)
    assert all(test_list)



#***********************************************************************************

def test_load_full_dataset():
    """Full dataset is properly loaded. Comparing against datastore_functions and dataframe loading for the FileDataset and DatastoreDataset subclasses"""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    from_method = dataset_obj_from_file.load_full_dataset()

    assert from_method.equals(delaney_from_disk)
    if not datastore_is_down:
        from_method_datastore = dataset_obj_from_datastore.load_full_dataset()
        assert from_method_datastore.equals(df_datastore)
#***********************************************************************************

def test_get_dataset_tasks():
    """Testing task extraction with self.params.response_cols as a single value or a list.
        From Datastore, if y is not defined, should extract from the dataset itself.
        Returns True if tasks are found, False if they are not.
    """
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    test_list = []

    flag_tasks_from_file = dataset_obj_from_file.get_dataset_tasks(delaney_from_disk)
    test_list.append(flag_tasks_from_file)
    test_list.append(dataset_obj_from_file.tasks == params_from_file.response_cols)

    subset_delaney = delaney_from_disk[['Compound ID','smiles','measured log solubility in mols per litre','Polar Surface Area']]

    # ksm: These tests are commented out because we no longer support datasets where response_cols is None.
    #flag_tasks_from_file_noy = dataset_obj_from_file_noy.get_dataset_tasks(subset_delaney)
    #test_list.append(flag_tasks_from_file_noy)
    #test_list.append(sorted(dataset_obj_from_file_noy.tasks) == sorted(['measured log solubility in mols per litre','Polar Surface Area']))

    #flag_tasks_from_file_failure = dataset_obj_from_file_noy.get_dataset_tasks(subset_delaney[['Compound ID','smiles']])
    #test_list.append(not flag_tasks_from_file_failure)

    if not datastore_is_down:
        flag_tasks_from_ds = dataset_obj_from_datastore.get_dataset_tasks(df_datastore)
        test_list.append(flag_tasks_from_ds)
        test_list.append(dataset_obj_from_datastore.tasks == params_from_ds.response_cols)

        subset_datastore = df_datastore[['compound_id','rdkit_smiles','PIC50']]

        # TODO (ksm): The following test fails because the task name stored in the dataset metadata differs from the response_cols
        # parameter specified when the ModelDataset was created for this dataset. Need to find a better dataset.
        # flag_tasks_from_ds_noy = dataset_obj_from_datastore_noy.get_dataset_tasks(subset_datastore)
        # test_list.append(flag_tasks_from_ds_noy)
        # test_list.append(sorted(dataset_obj_from_datastore_noy.tasks) == sorted(['PIC50']))

        # ksm: For datastore datasets, the following test never fails because the task name is retrieved from the dataset metadata
        #dataset_obj_from_datastore_noy.tasks = None
        #flag_tasks_from_ds_failure = dataset_obj_from_datastore_noy.get_dataset_tasks(subset_datastore[['compound_id','rdkit_smiles']])
        #test_list.append(not flag_tasks_from_ds_failure)

        # TODO (ksm): Add tests for a multitask dataset
    print(test_list)
    assert all(test_list)



#***********************************************************************************


def test_check_task_columns():
    """Checks that self.tasks exist, then checks that the requested self.tasks all exist within the dataframe. Throws exception if self.get_dataset_tasks is False or if prediction tasks are missing. Testing for exception raising on bad task columns and success."""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    #with pytest.raises(Exception):
    #    dataset_obj_from_file_wrongy.check_task_columns(delaney_from_disk)

    model_dataset.check_task_columns(params_from_file, df_delaney)
    #assert dataset_obj_from_file.tasks == params_from_file.response_cols
    if not datastore_is_down:
        #with pytest.raises(Exception):
        #    dataset_obj_from_datastore_wrongy.check_task_columns(df_datastore)
        ds_tasks = dataset_obj_from_datastore.get_dataset_tasks(df_datastore)
        assert ds_tasks == params_from_ds.response_cols



#***********************************************************************************
"""
def test_load_featurized_data():
    #Loads prefeaturized data from datastore or filesystem. Returns a dataframe. Not implemented in super

    #Tested within test_featurizer.py using descriptor featurization
    pass
"""
#***********************************************************************************

"""
def test_save_featurized_data():
    #this method is tested in test_featurization.py
    #Not in super. Uploads featurized dataset to the datastore or uses set_group_permissions in the FileDataset subclass to save files

    pass
"""

"""
def test_set_group_permissions():
    #Unique to FileDataset subclass. Changes the owership of a particular file on a  path.
    #tested within test_featurization.py
    pass
"""
#***********************************************************************************
def test_get_featurized_data():
    """Tries to load a previously prefeaturized dataset, then creates featurization, instantiates n_features, and dumps a pickle file of the transformers if they exist. Implemented in super. The dataset object from file is a delaney dataset using an ecfp featurizer with a default scaffold split. Testing of featurization of the dataset is extensively done in test_featurization.py"""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    dataset_obj_from_file.params.transformers = True
    dataset_obj_from_file.get_featurized_data()
    test_list = []
    test_list.append(dataset_obj_from_file.n_features == params_from_file.ecfp_size)
    test_list.append(isinstance(dataset_obj_from_file.dataset, dc.data.datasets.NumpyDataset)) 
    test_list.append(len(dataset_obj_from_file.dataset) == len(df_delaney))
    test_list.append(dataset_obj_from_file.n_features == dataset_obj_from_file.params.ecfp_size)
    assert all(test_list)

#***********************************************************************************

def test_get_featurized_data_scaffold():
    """Tries to load a previously prefeaturized dataset, then creates featurization, instantiates n_features, and dumps a pickle file of the transformers if they exist. Implemented in super. The dataset object from file is a delaney dataset using an ecfp featurizer with a default scaffold split. Testing of featurization of the dataset is extensively done in test_featurization.py"""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    dataset_obj_from_file_scaffold.params.transformers = True
    dataset_obj_from_file_scaffold.get_featurized_data()

    test_list = []

    test_list.append(isinstance(dataset_obj_from_file_scaffold.dataset, dc.data.datasets.NumpyDataset))
    test_list.append(len(dataset_obj_from_file_scaffold.dataset) == len(df_delaney))
    test_list.append(dataset_obj_from_file.n_features == dataset_obj_from_file.params.ecfp_size)
    test_list.append(len(dataset_obj_from_file.dataset.y) == len(dataset_obj_from_file.dataset.ids))

    assert all(test_list)


#***********************************************************************************

def test_split_dataset():
    """Uses the split_datset method of splitting to split data. Implemented in super. Because the various splitting strategies are heavily tested in test_splitting.py, this test is simply ensuring that the attributes are appropriately created."""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    dataset_obj_from_file.split_dataset()
    (train, valid) = dataset_obj_from_file.train_valid_dsets[0]
    (train_attr, valid_attr) = dataset_obj_from_file.train_valid_attr[0]

    test_list = []
    test_list.append(len(dataset_obj_from_file.dataset) == len(train) + len(valid) + len(dataset_obj_from_file.test_dset))
    test_list.append(set(train.ids.tolist()) == set(train_attr.index.tolist()))
    test_list.append(set(valid.ids.tolist()) == set(valid_attr.index.tolist()))
    test_list.append(set(dataset_obj_from_file.test_dset.ids.tolist()) == set(dataset_obj_from_file.test_attr.index.tolist()))

    #testing that k_fold splits are properly generated
    dataset_obj_from_file.params.split_strategy = 'k_fold_cv'
    dataset_obj_from_file.split_dataset()
    (train, valid) = dataset_obj_from_file.train_valid_dsets[0]
    (train_attr, valid_attr) = dataset_obj_from_file.train_valid_attr[0]
    test_list.append(len(dataset_obj_from_file.dataset) == len(train) + len(valid) + len(dataset_obj_from_file.test_dset))
    test_list.append(set(train.ids.tolist()) == set(train_attr.index.tolist()))
    test_list.append(set(valid.ids.tolist()) == set(valid_attr.index.tolist()))
    test_list.append(set(dataset_obj_from_file.test_dset.ids.tolist()) == set(dataset_obj_from_file.test_attr.index.tolist()))
    assert all(test_list)

#***********************************************************************************

def test_save_split_dataset():
    """Saves the compound IDs and smiles strings for a split subset. Implemented in super"""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    dataset_obj_from_file.save_split_dataset()
    dir = os.path.dirname(dataset_obj_from_file.params.dataset_key)
    split_path = '{0}/{1}'.format(dir, dataset_obj_from_file._get_split_key())

    assert os.path.isfile(split_path)

#***********************************************************************************

def test_load_presplit_dataset():
    # Loads in the split files from disk. Uses splitting.get_split_prefix to specify the path of the split file. Uses splitting.select_dset_by_attr_ids. Returns True or False. Initializes self.train_valid_attr, self.train_valid_dsets, self.test_attr, self.test_dset. Implemented in super.
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    dataset_obj_from_file.get_featurized_data()
    dataset_obj_from_file.split_dataset()
    (orig_train, orig_valid) = dataset_obj_from_file.train_valid_dsets[0]
    (orig_train_attr, orig_valid_attr) = dataset_obj_from_file.train_valid_attr[0]
    dataset_obj_from_file.save_split_dataset()
    
    # Need to pass the split_uuid to recover the split we just saved
    (params_from_file2, dataset_obj_from_file2, df_delaney2) = utils.delaney_objects(split_uuid = dataset_obj_from_file.split_uuid)
    dataset_obj_from_file2.get_featurized_data()
    dataset_obj_from_file2.load_presplit_dataset()
    (train, valid) = dataset_obj_from_file2.train_valid_dsets[0]
    
    (train_attr, valid_attr) = dataset_obj_from_file2.train_valid_attr[0]
    
    test_list = []
    test_list.append((sorted(train.y) == sorted(orig_train.y)))
    test_list.append((sorted(valid.y) == sorted(orig_valid.y)))
    test_list.append(set(train_attr.index.values) == set(orig_train_attr.index.values))
    test_list.append(set(valid_attr.index.values) == set(orig_valid_attr.index.values))
    test_list.append((sorted(dataset_obj_from_file.test_dset.y) == sorted(dataset_obj_from_file2.test_dset.y)))
    test_list.append(set(dataset_obj_from_file.test_attr.index.values) == set(dataset_obj_from_file2.test_attr.index.values))

    assert all(test_list)
    


#***********************************************************************************

def test_combine_training_data():
    """Concatenates train and valid from self.train_valid_dsets[0] into a combined DiskDataset. Implemented in super."""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    dataset_obj_from_file.combined_training_data()
    (orig_train, orig_valid) = dataset_obj_from_file.train_valid_dsets[0]
    assert isinstance(dataset_obj_from_file.combined_train_valid_data,DD)
    
    concat_train_valid = np.concatenate((orig_train.ids, orig_valid.ids))
    assert (concat_train_valid == dataset_obj_from_file.combined_train_valid_data.ids).all()
#***********************************************************************************

def test_split_dataset_scaffold():
    """Uses the split_datset method of splitting to split data. Implemented in super. Because the various splitting strategies are heavily tested in test_splitting.py, this test is simply ensuring that the attributes are appropriately created."""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    dataset_obj_from_file_scaffold.split_dataset()
    (train, valid) = dataset_obj_from_file_scaffold.train_valid_dsets[0]
    (train_attr, valid_attr) = dataset_obj_from_file_scaffold.train_valid_attr[0]
    
    test_list = []
    test_list.append(len(dataset_obj_from_file.dataset) == len(train) + len(valid) + len(dataset_obj_from_file_scaffold.test_dset))
    test_list.append(set(train.ids.tolist()) == set(train_attr.index.tolist()))
    test_list.append(set(valid.ids.tolist()) == set(valid_attr.index.tolist()))
    test_list.append(set(dataset_obj_from_file_scaffold.test_dset.ids.tolist()) == set(dataset_obj_from_file_scaffold.test_attr.index.tolist()))
    assert all(test_list)
    
#***********************************************************************************
def test_combine_training_data_scaffold():
    """Concatenates train and valid from self.train_valid_dsets[0] into a combined DiskDataset. Implemented in super."""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    dataset_obj_from_file_scaffold.combined_training_data()
    (orig_train, orig_valid) = dataset_obj_from_file_scaffold.train_valid_dsets[0]
    test_list = []
    test_list.append(isinstance(dataset_obj_from_file_scaffold.combined_train_valid_data,DD))
    test_list.append(len(dataset_obj_from_file_scaffold.combined_train_valid_data.y)==len(dataset_obj_from_file_scaffold.combined_train_valid_data.ids))

    
    concat_train_valid = np.concatenate((orig_train.ids, orig_valid.ids))
    test_list.append((concat_train_valid == dataset_obj_from_file_scaffold.combined_train_valid_data.ids).all())
    test_list.append(len(orig_train.y) == len(orig_train.ids))
    assert all(test_list)

    
#***********************************************************************************

def test_get_split_metadata():
    """pulls a dictionary that contains the splitting strategy and splitter used to generate the model."""
    (params_from_file, dataset_obj_from_file, df_delaney) = utils.delaney_objects()
    (params_from_file_scaffold, dataset_obj_from_file_scaffold, df_delaney) = utils.delaney_objects(
        split_strategy="train_valid_test", splitter="scaffold")

    out_dict = dataset_obj_from_file.get_split_metadata()
    test_list = []
    test_list.append(out_dict["split_strategy"] == dataset_obj_from_file.params.split_strategy)
    test_list.append(out_dict["splitter"] == dataset_obj_from_file.params.splitter)
    # TODO: num_folds does not match. Need to identify the difference in num_folds.
    # test_list.append(out_dict["Splitting"]["num_folds"] == dataset_obj_from_file.splitting.num_folds)
    test_list.append(out_dict["split_valid_frac"] == dataset_obj_from_file.params.split_valid_frac)
    test_list.append(out_dict["split_test_frac"] == dataset_obj_from_file.params.split_test_frac)
    test_list.append(out_dict["split_uuid"] == dataset_obj_from_file.split_uuid)
   
    assert all(test_list)
    
    
#***********************************************************************************

def test_load_presplit_dataset():
    # open the test
    with open("../test_datasets/H1_hybrid.json", "r") as f:
        config = json.load(f)

    # change to some fake uuid
    config["split_uuid"] = "c63c6d89-8832-4434-b27a-17213bd6ef8"
    params = parse.wrapper(config)

    MP = mp.ModelPipeline(params)
    featurization=None
    if featurization is None:
        featurization = feat.create_featurization(MP.params)
    MP.featurization = featurization
    with pytest.raises(SystemExit) as e:
        # The command to test
        # should system exit
        MP.load_featurize_data()
    # test
    assert e.type == SystemExit
    assert e.value.code == 1