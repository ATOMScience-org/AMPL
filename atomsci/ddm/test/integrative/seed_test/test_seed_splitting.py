#!/usr/bin/env python 
"""Testing the reproducibility of seeding a random seed in AMPL splitters to recreate split datasets."""

import pandas as pd 
import copy
import json 
import os

from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse 

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from integrative_utilities import extract_seed, modify_params_with_seed

#----------------------------------------------------------------------------------------------------------
def split_dataset(pparams):
    model_pipe = mp.ModelPipeline(pparams)
    split_uuid = model_pipe.split_dataset()
    pparams.split_uuid = split_uuid
    return pparams

def compare_splits(original_split_csv, retrained_split_csv):
    original_split = pd.read_csv(original_split_csv)
    retrained_split = pd.read_csv(retrained_split_csv)

    comparison_df = original_split.merge(retrained_split, on='cmpd_id', suffixes=('_original', '_retrained'))
    
    # Initialize a variable to track if all comparisons are valid
    all_match = True

    # Iterate through rows to compare the 'subset' and 'fold' columns
    for index, row in comparison_df.iterrows():
        subset_match = (row['subset_original'] == row['subset_retrained'])
        fold_match = (row['fold_original'] == row['fold_retrained'])
        
        if not (subset_match and fold_match):
            print(f"Mismatch found for cmpd_id {row['cmpd_id']}: "
                  f"original subset = {row['subset_original']}, "
                  f"retrained subset = {row['subset_retrained']}, "
                  f"original fold = {row['fold_original']}, "
                  f"retrained fold = {row['fold_retrained']}")
            all_match = False

    return all_match

def perform_splits_and_compare(pparams):
    starting_pparams=split_dataset(pparams)
    # original split
    script_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(script_path, '../../test_datasets/')
    if starting_pparams.split_strategy == 'k_fold_cv':
        original_split_csv = os.path.join(dataset_path, f"{starting_pparams.dataset_name}_{starting_pparams.num_folds}_fold_cv_{starting_pparams.splitter}_{starting_pparams.split_uuid}.csv")
    else:
        original_split_csv = os.path.join(dataset_path, f"{starting_pparams.dataset_name}_{starting_pparams.split_strategy}_{starting_pparams.splitter}_{starting_pparams.split_uuid}.csv")

    # extract the seed 
    metadata_path = os.path.join(starting_pparams.output_dir, 'split_metadata.json')
    seed = extract_seed(metadata_path)

    # Retrain split with the same seed
    retrain_pparams = copy.copy(pparams)
    retrain_pparams.split_uuid = None
    retrain_pparams.seed = seed
    
    retrain_pparams = split_dataset(retrain_pparams)
    script_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(script_path, '../../test_datasets/')
    if starting_pparams.split_strategy == 'k_fold_cv':
        retrained_split_csv = os.path.join(dataset_path, f"{retrain_pparams.dataset_name}_{retrain_pparams.num_folds}_fold_cv_{retrain_pparams.splitter}_{retrain_pparams.split_uuid}.csv")
    else: 
        retrained_split_csv = os.path.join(dataset_path, f"{retrain_pparams.dataset_name}_{retrain_pparams.split_strategy}_{retrain_pparams.splitter}_{retrain_pparams.split_uuid}.csv")

    # Compare splits
    
    splits_match = compare_splits(original_split_csv, retrained_split_csv)

    if splits_match is True:
        print("The splits match exactly!")
    else:
        print("The splits do not match.")

    return splits_match

#----------------------------------------------------------------------------------------------------------

def test_random_train_valid_test_split_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'split_json/test_random_train_valid_test_split.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    perform_splits_and_compare(pparams)

def test_scaffold_train_valid_test_split_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'split_json/test_scaffold_train_valid_test_split.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    perform_splits_and_compare(pparams)

def test_fingerprint_train_valid_test_split_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'split_json/test_fingerprint_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    perform_splits_and_compare(pparams)

def test_kfold_random_split_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'split_json/test_kfold_random_split.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    perform_splits_and_compare(pparams)

def test_kfold_scaffold_split_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'split_json/test_kfold_scaffold_split.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    perform_splits_and_compare(pparams)

def test_kfold_fingerprint_split_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'split_json/test_kfold_fingerprint_split.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    perform_splits_and_compare(pparams)

#----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("test_random_train_valid_test_split_reproducibility")
    test_random_train_valid_test_split_reproducibility()

    print("test_scaffold_train_valid_test_split_reproducibility")
    test_scaffold_train_valid_test_split_reproducibility()

    print("test_fingerprint_train_valid_test_split_reproducibility")
    test_fingerprint_train_valid_test_split_reproducibility()

    print("test_kfold_random_split_reproducibility")
    test_kfold_random_split_reproducibility()

    print("test_kfold_scaffold_split_reproducibility")
    test_kfold_scaffold_split_reproducibility()

    print("test_kfold_fingerprint_split_reproducibility")
    test_kfold_fingerprint_split_reproducibility()

    print("Passed!")