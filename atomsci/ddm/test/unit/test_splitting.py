import os
import pandas as pd
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
import atomsci.ddm.pipeline.featurization as feat
import atomsci.ddm.pipeline.model_datasets as model_datasets

import atomsci.ddm.pipeline.splitting as split
import deepchem as dc
import utils_testing as utils
from deepchem.data import DiskDataset

stratified_fixed = False

# ksm: In latest code, ModelDataset defers creating its splitting object until the call to ModelDataset.split_dataset().
# So we have to do that first in order to access the splitting object.
(params_random, data_obj_random, df_delaney) = utils.delaney_objects(split_strategy="train_valid_test", splitter = "random")
os.makedirs(params_random.output_dir, exist_ok=True)
data_obj_random.get_featurized_data()
data_obj_random.split_dataset()
splitter_random = data_obj_random.splitting

(params_scaffold, data_obj_scaffold, df_delaney) = utils.delaney_objects(split_strategy="train_valid_test", splitter = "scaffold")
data_obj_scaffold.get_featurized_data()
data_obj_scaffold.split_dataset()
splitter_scaffold = data_obj_scaffold.splitting

(params_stratified, data_obj_stratified, df_delaney) = utils.delaney_objects(split_strategy="train_valid_test", splitter = "stratified")
data_obj_stratified.get_featurized_data()
data_obj_stratified.split_dataset()
splitter_stratified = data_obj_stratified.splitting

(params_index, data_obj_index, df_delaney) = utils.delaney_objects(split_strategy="train_valid_test", splitter = "index")
data_obj_index.get_featurized_data()
data_obj_index.split_dataset()
splitter_index = data_obj_index.splitting


(params_k_fold_scaffold, data_obj_k_fold_scaffold, df_delaney) = utils.delaney_objects(split_strategy="k_fold_cv", splitter = "scaffold")
data_obj_k_fold_scaffold.get_featurized_data()
data_obj_k_fold_scaffold.split_dataset()
splitter_k_fold_scaffold = data_obj_k_fold_scaffold.splitting

(params_k_fold_random, data_obj_k_fold_random, df_delaney) = utils.delaney_objects(split_strategy="k_fold_cv", splitter = "random")
data_obj_k_fold_random.get_featurized_data()
data_obj_k_fold_random.split_dataset()
splitter_k_fold_random = data_obj_k_fold_random.splitting

frac_train = 0.8
DD = dc.data.datasets.NumpyDataset
PDF = pd.core.frame.DataFrame

#***********************************************************************************
def test_split_dataset_kfold_scaffold_from_pipeline(caplog):
    #Testing for correct type and length of dataset for k-fold splitting with a scaffold splitter     
    #Testing a 3-fold split first for uniqueness of all validation and training sets. 

    #mp.model_wrapper = model_wrapper.create_model_wrapper(mp.params, mp.featurization, mp.ds_client)
    #mp.model_wrapper.setup_model_dirs()
    mp = utils.delaney_pipeline(featurizer="ecfp", split_strategy="k_fold_cv", splitter = "scaffold")
    mp.featurization = feat.create_featurization(mp.params)
    mp.data = model_datasets.create_model_dataset(mp.params, mp.featurization, mp.ds_client)
    mp.data.get_featurized_data()
    mp.data.split_dataset()
    splitter_k_fold_scaffold = mp.data.splitting
    splitter_k_fold_scaffold.num_folds = 3
    nf = splitter_k_fold_scaffold.num_folds

    #mp.model_wrapper.create_transformers(self.data)
    #mp.data.dataset = mp.model_wrapper.transform_dataset(self.data.dataset)
    data_obj_k_fold_scaffold = mp.data

    data_obj_k_fold_scaffold.split_dataset()
    train_valid, test, train_valid_attr, test_attr = splitter_k_fold_scaffold.split_dataset(data_obj_k_fold_scaffold.dataset, data_obj_k_fold_scaffold.attr, data_obj_k_fold_scaffold.params.smiles_col)
    #assert no overlap of the k-fold validation sets between each other
    test_list = []
    for kfoldindex in range(0,nf):
        test_list.append((data_obj_k_fold_scaffold.train_valid_dsets[kfoldindex][0].X == train_valid[kfoldindex][0].X).all())
        test_list.append((data_obj_k_fold_scaffold.train_valid_dsets[kfoldindex][1].X == train_valid[kfoldindex][1].X).all())
        test_list.append((data_obj_k_fold_scaffold.train_valid_dsets[kfoldindex][1].ids == train_valid[kfoldindex][1].ids).all())
        test_list.append((data_obj_k_fold_scaffold.train_valid_dsets[kfoldindex][1].ids == train_valid[kfoldindex][1].ids).all())
        test_list.append(train_valid_attr[kfoldindex][0].equals(data_obj_k_fold_scaffold.train_valid_attr[kfoldindex][0]))
        test_list.append(train_valid_attr[kfoldindex][1].equals(data_obj_k_fold_scaffold.train_valid_attr[kfoldindex][1]))
    assert all(test_list)
    test_list = []
    concat_valid = [x[1].ids.tolist() for x in train_valid]
    concat_valid = sum(concat_valid,[])
    test_list.append(len(concat_valid) == len(set(concat_valid)))

    assert all(test_list)
    tv_split = []
    test_list = []
    #asserting that each k-fold split has no internal overlap.
    for kfoldindex in range(0,nf):
        current_tv_split = train_valid[kfoldindex][0].ids.tolist() + train_valid[kfoldindex][1].ids.tolist()
        test_list.append(len(train_valid[kfoldindex][0].ids) == len(train_valid[kfoldindex][0].y))
        test_list.append(len(train_valid[kfoldindex][1].ids) == len(train_valid[kfoldindex][1].y))
        current_full_dataset = sum([current_tv_split,test.ids.tolist()],[])
        test_list.append(len(current_full_dataset) == len(set(current_full_dataset)))
        test_list.append(set(train_valid[kfoldindex][0].ids.tolist()) == set(train_valid_attr[kfoldindex][0].index.tolist()))
        test_list.append(set(train_valid[kfoldindex][1].ids.tolist()) == set(train_valid_attr[kfoldindex][1].index.tolist()))
        #checking length of the validation set (should be length of the kv set/num_folds +/- 1)
        len_valid = round(len(current_tv_split)/nf)
        test_list.append(len_valid -1 <= len(train_valid[kfoldindex][1]) <= len_valid + 1)
        tv_split.append(current_tv_split)
        
    #asserting that all k-fold train valid sets are equivalent
    test_list.append(set.intersection(*[set(l) for l in tv_split]) == set(tv_split[0]))
    #aasserting that the test and test_attrs have the same index:
    test_list.append(set(test.ids.tolist()) == set(test_attr.index.tolist()))
    test_list.append(len(test.y) == len(test.ids))
    assert all(test_list)
    

#***********************************************************************************  
def test_create_splitting(caplog):
    """testing factory function to create splitting object"""
    test = []
    test.append(isinstance(splitter_random, split.TrainValidTestSplitting))
    test.append(isinstance(splitter_random.splitter, dc.splits.RandomSplitter))
    test.append(splitter_random.num_folds == 1)
    methods = ["get_split_prefix","split_dataset","needs_smiles","split_dataset"]
    for method in methods:
        test.append(callable(getattr(splitter_random,method)))
    test.append(isinstance(splitter_k_fold_scaffold, split.KFoldSplitting))
    test.append(isinstance(splitter_k_fold_scaffold.splitter, dc.splits.ScaffoldSplitter))
    test.append(splitter_k_fold_scaffold.num_folds == params_k_fold_scaffold.num_folds)
    for method in methods:
        test.append(callable(getattr(splitter_k_fold_scaffold,method)))
    assert all(test)
        
#***********************************************************************************  

def test_needs_smiles(caplog):
    """returns True if dc splitter requires compound IDs to be SMILES strings"""
    assert not splitter_random.needs_smiles()
    assert splitter_k_fold_scaffold.needs_smiles()
    assert splitter_scaffold.needs_smiles()
        
#***********************************************************************************  
        
def test_get_split_prefix(caplog):
    """returns a string that identifies the split strategy and the splitting method"""
    assert splitter_k_fold_scaffold.get_split_prefix(parent='test_fold') == "test_fold/" + str(splitter_k_fold_scaffold.num_folds) + "_fold_cv_" + str(splitter_k_fold_scaffold.split)
    assert splitter_random.get_split_prefix(parent='test_random') == "test_random/" + "train_valid_test_" + str(splitter_random.split)

#***********************************************************************************

def test_split_dataset_random(caplog):
    #Testing for correct type and length of dataset for trainvalidtest splitting with a random splitter
    ([(train,valid)], test_data, [(train_attr,valid_attr)],test_attr) = splitter_random.split_dataset(data_obj_random.dataset, data_obj_random.attr, data_obj_random.params.smiles_col)
    test = []
    #corect length of dataset
    test.append(len(data_obj_random.dataset) == len(train) + len(valid) + len(test_data))
    #correct size of train set
    test.append(len(train) == round(len(data_obj_random.dataset)*frac_train))
    #correct (type) of train, valid, test
    test.append(isinstance(train, DD) and isinstance(valid, DD) and isinstance(test_data, DD))
    test.append(len(train.ids) == len(train.y))
    test.append(len(valid.ids) == len(valid.y))
    test.append(len(test_data.ids) == len(test_data.y))

    #no overlap in train, valid and test
    full_dataset = sum([train.ids.tolist(),valid.ids.tolist(), test_data.ids.tolist()],[])
    test.append(len(full_dataset) == len(set(full_dataset)))
    #train, valid, and test are associated to the correct attr pd.DataFrame
    test.append(set(train.ids.tolist()) == set(train_attr.index.tolist()))
    test.append(set(valid.ids.tolist()) == set(valid_attr.index.tolist()))
    test.append(set(test_data.ids.tolist()) == set(test_attr.index.tolist()))
    #no overlap in train_attr, valid_attr, and test_attr in smiles
    full_dataset_smiles = sum([train_attr.smiles.tolist(),valid_attr.smiles.tolist(), test_attr.smiles.tolist()],[])
    test.append(len(full_dataset_smiles) == len(set(full_dataset_smiles)))
    assert all(test)
    
#***********************************************************************************
def test_split_dataset_scaffold(caplog):
    #Testing for correct type and length of dataset for trainvalidtest splitting with a scaffold splitter
    ([(train,valid)], test_data, [(train_attr,valid_attr)],test_attr) = splitter_scaffold.split_dataset(data_obj_scaffold.dataset, data_obj_scaffold.attr, data_obj_scaffold.params.smiles_col)
    test = []
    #correct length of dataset
    test.append(len(data_obj_scaffold.dataset) == len(train) + len(valid) + len(test_data))
    #correct length of trainiing set
    test.append(len(train) == round(len(data_obj_scaffold.dataset)*frac_train))
    #correct (type) for train, valid, test
    test.append(isinstance(train, DD) and isinstance(valid, DD) and isinstance(test_data, DD))
    test.append(len(train.ids) == len(train.y))
    test.append(len(valid.ids) == len(valid.y))
    test.append(len(test_data.ids) == len(test_data.y))
    #correct train, valid, test indexing
    test.append(set(train.ids.tolist()) == set(train_attr.index.tolist()))
    test.append(set(valid.ids.tolist()) == set(valid_attr.index.tolist()))
    test.append(set(test_data.ids.tolist()) == set(test_attr.index.tolist()))
    #asserting that there are no overlaps in the ids.
    full_dataset = sum([train.ids.tolist(),valid.ids.tolist(), test_data.ids.tolist()],[])
    test.append(len(full_dataset) == len(set(full_dataset)))
    #asserting that there are no overlaps in the smiles.
    full_dataset_smiles = sum([train_attr.smiles.tolist(),valid_attr.smiles.tolist(), test_attr.smiles.tolist()],[])
    test.append(len(full_dataset_smiles) == len(set(full_dataset_smiles)))
    assert all(test)
#***********************************************************************************

([(train,valid)], test_scaffold, [(train_attr,valid_attr)],test_scaffold_attr) = splitter_scaffold.split_dataset(data_obj_scaffold.dataset, data_obj_scaffold.attr, data_obj_scaffold.params.smiles_col)
dataset_scaffold = DiskDataset.from_numpy(data_obj_scaffold.dataset.X, data_obj_scaffold.dataset.y, ids=data_obj_scaffold.attr.index)


def test_select_dset_by_attr_ids_using_smiles():
    #testing that the method can split a dataset according to its attr ids into the correct deepchem diskdataframe. In this case, the attr_ids are converted back to smiles to match the input dataset.
    dataset = DiskDataset.from_numpy(data_obj_scaffold.dataset.X, data_obj_scaffold.dataset.y, ids=data_obj_scaffold.attr[data_obj_scaffold.params.smiles_col].values)
    newdf = pd.DataFrame({'compound_ids' : test_scaffold_attr.index.tolist()}, index = test_scaffold_attr.smiles)
    newDD = split.select_dset_by_attr_ids(dataset, newdf)
    assert (sorted(newDD.y) == sorted(test_scaffold.y))
#***********************************************************************************

def test_select_dset_by_attr_ids_using_compound_ids():
    #testing that the method can split a dataset according to its attr ids into the correct deepchem diskdataframe. This test uses compound_ids.
    newDD = split.select_dset_by_attr_ids(dataset_scaffold, test_scaffold_attr)
    assert (sorted(newDD.y) == sorted(test_scaffold.y))
#***********************************************************************************

def test_select_dset_by_id_list():
    #testing that the method can split a dataset according to a list of compound_ids into the correct deepchem diskdataframe.
    
    newDD = split.select_dset_by_id_list(dataset_scaffold, test_scaffold_attr.index.tolist())
    assert (sorted(newDD.y) == sorted(test_scaffold.y))

#***********************************************************************************


def test_select_attrs_by_dset_ids():
    #testing that the method can split a attr according to a disk dataset, using compound_ids
    newDD = split.select_attrs_by_dset_ids(test_scaffold, data_obj_scaffold.attr)
    assert set(newDD.index.values) == set(test_scaffold_attr.index.values)
    
#***********************************************************************************

def test_select_attrs_by_dset_smiles():
    #testing that the method can split a attr according to a disk dataset. In this case, the attr_ids need to be converted back to smiles to match the input dataset.
    dataset = DiskDataset.from_numpy(test_scaffold.X, test_scaffold.y, ids=test_scaffold_attr[data_obj_scaffold.params.smiles_col].values)

    newDD = split.select_attrs_by_dset_smiles(dataset, data_obj_scaffold.attr,data_obj_scaffold.params.smiles_col )
    assert set(newDD.index.values) == set(test_scaffold_attr.index.values)
#***********************************************************************************

def test_split_dataset_stratified(caplog):
    #Testing for correct type and length of dataset for trainvalidtest splitting with a random splitter    
    if stratified_fixed:
        test_list = []
        ([(train,valid)], test, [(train_attr,valid_attr)],test_attr) = splitter_stratified.split_dataset(data_obj_stratified.dataset, data_obj_stratified.attr, data_obj_stratified.params.smiles_col)
        test_list.append(len(data_obj_stratified.dataset) == len(train) + len(valid) + len(test))
        test_list.append(len(train) == len(train_attr))
        test_list.append(len(valid) == len(valid_attr))
        test_list.append(len(test) == len(test_attr))
        test_list.append(isinstance(train, DD) and isinstance(valid, DD) and isinstance(test, DD))
        test_list.append(isinstance(train_attr, PDF) and isinstance(valid_attr, PDF) and isinstance(test_attr, PDF))
        assert all(test_list)
    else:
        pass

#***********************************************************************************
def test_split_dataset_index(caplog):
    #Testing for correct type and length of dataset for trainvalidtest splitting with a random splitter     
    ([(train,valid)], test, [(train_attr,valid_attr)],test_attr) = splitter_index.split_dataset(data_obj_index.dataset, data_obj_index.attr, data_obj_index.params.smiles_col)
    test_list = []
    test_list.append(len(data_obj_index.dataset) == len(train) + len(valid) + len(test))
    test_list.append(len(train) == round(len(data_obj_index.dataset)*frac_train))    
    test_list.append(isinstance(train, DD) and isinstance(valid, DD) and isinstance(test, DD))
    test_list.append(set(train.ids.tolist()) == set(train_attr.index.tolist()))
    test_list.append(set(valid.ids.tolist()) == set(valid_attr.index.tolist()))
    test_list.append(set(test.ids.tolist()) == set(test_attr.index.tolist()))
    test_list.append(len(train.ids) == len(train.y))
    test_list.append(len(valid.ids) == len(valid.y))
    test_list.append(len(test.ids) == len(test.y))
    full_dataset = sum([train.ids.tolist(),valid.ids.tolist(), test.ids.tolist()],[])
    test_list.append(len(full_dataset) == len(set(full_dataset)))
    assert all(test_list)

#***********************************************************************************
def test_split_dataset_kfold_scaffold(caplog):
    #Testing for correct type and length of dataset for k-fold splitting with a scaffold splitter     
    #Testing a 3-fold split first for uniqueness of all validation and training sets. 
     
    splitter_k_fold_scaffold.num_folds = 3
    nf = splitter_k_fold_scaffold.num_folds
    train_valid, test, train_valid_attr, test_attr = splitter_k_fold_scaffold.split_dataset(data_obj_k_fold_scaffold.dataset, data_obj_k_fold_scaffold.attr, data_obj_k_fold_scaffold.params.smiles_col)
    #assert no overlap of the k-fold validation sets between each other
    concat_valid = [x[1].ids.tolist() for x in train_valid]
    concat_valid = sum(concat_valid,[])
    test_list = []
    test_list.append(len(concat_valid) == len(set(concat_valid)))
    test_list.append(not list(set(concat_valid) & set(test.ids.tolist())))
    tv_split = []
    #asserting that each k-fold split has no internal overlap.
    for kfoldindex in range(0,nf):
        current_tv_split = train_valid[kfoldindex][0].ids.tolist() + train_valid[kfoldindex][1].ids.tolist()
        test_list.append(len(train_valid[kfoldindex][0].ids) == len(train_valid[kfoldindex][0].y))
        test_list.append(len(train_valid[kfoldindex][1].ids) == len(train_valid[kfoldindex][1].y))
        current_full_dataset = sum([current_tv_split,test.ids.tolist()],[])
        test_list.append(len(current_full_dataset) == len(set(current_full_dataset)))
        test_list.append(set(train_valid[kfoldindex][0].ids.tolist()) == set(train_valid_attr[kfoldindex][0].index.tolist()))
        test_list.append(set(train_valid[kfoldindex][1].ids.tolist()) == set(train_valid_attr[kfoldindex][1].index.tolist()))
        #checking length of the validation set (should be length of the kv set/num_folds +/- 1)
        len_valid = round(len(current_tv_split)/nf)
        test_list.append(len_valid -1 <= len(train_valid[kfoldindex][1]) <= len_valid + 1)
        tv_split.append(current_tv_split)
        
    #asserting that all k-fold train valid sets are equivalent
    test_list.append(set.intersection(*[set(l) for l in tv_split]) == set(tv_split[0]))
    #aasserting that the test and test_attrs have the same index:
    test_list.append(set(test.ids.tolist()) == set(test_attr.index.tolist()))
    test_list.append(len(test.y) == len(test.ids))
    assert all(test_list)
    
    #***********************************************************************************
def test_split_dataset_kfold_random(caplog):
    #Testing for correct type and length of dataset for k-fold splitting with a random splitter         
    #Testing a 5-fold split first for uniqueness of all validation and training sets. 
    splitter_k_fold_random.num_folds = 5
    nf = splitter_k_fold_random.num_folds
    train_valid, test, train_valid_attr, test_attr = splitter_k_fold_random.split_dataset(data_obj_k_fold_random.dataset, data_obj_k_fold_random.attr, data_obj_k_fold_random.params.smiles_col)
    #assert no overlap of the k-fold validation sets between each other
    concat_valid = [x[1].ids.tolist() for x in train_valid]
    concat_valid = sum(concat_valid,[])
    test_list = []
    test_list.append(len(concat_valid) == len(set(concat_valid)))
    test_list.append(not list(set(concat_valid) & set(test.ids.tolist())))
    tv_split = []
    #asserting that each k-fold split has no internal overlap.
    for kfoldindex in range(0,nf):
        current_tv_split = train_valid[kfoldindex][0].ids.tolist() + train_valid[kfoldindex][1].ids.tolist()
        current_full_dataset = sum([current_tv_split,test.ids.tolist()],[])
        test_list.append(len(current_full_dataset) == len(set(current_full_dataset)))
        test_list.append(set(train_valid[kfoldindex][0].ids.tolist()) == set(train_valid_attr[kfoldindex][0].index.tolist()))
        test_list.append(set(train_valid[kfoldindex][1].ids.tolist()) == set(train_valid_attr[kfoldindex][1].index.tolist()))
        #checking length of the validation set (should be length of the kv set/num_folds +/- 1)
        len_valid = round(len(current_tv_split)/nf)
        test_list.append(len_valid -1 <= len(train_valid[kfoldindex][1]) <= len_valid + 1)
        tv_split.append(current_tv_split)
        
    #asserting that all k-fold train valid sets are equivalent
    test_list.append(set.intersection(*[set(l) for l in tv_split]) == set(tv_split[0]))
    #aasserting that the test and test_attrs have the same index:
    test_list.append(set(test.ids.tolist()) == set(test_attr.index.tolist()))
    assert all(test_list)

