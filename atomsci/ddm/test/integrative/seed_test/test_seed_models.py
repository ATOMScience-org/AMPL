#!/usr/bin/env python 
"""Testing the reproducibility of seeding a random seed in AMPL to reproduce models."""
import pandas as pd 
import copy
import os
import json

import pytest

from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse 

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from atomsci.ddm.utils import llnl_utils
from integrative_utilities import extract_seed, find_best_test_metric

#-------------------------------------------------------------------
"""
This script does the following:
1. Generates a model 
2. Extracts the seed from the model's metadata 
3. Runs the model training again with seed
4. Compares the prediction scores to ensure they're identical

Creates and tests the following models:
- RF, NN
- regression, classification
- train_valid_test split, k-fold cv split
"""
#-------------------------------------------------------------------
def saved_model_identity(pparams):
    retrain_pparams = copy.copy(pparams)

    model_pipe = mp.ModelPipeline(pparams)

    if not pparams.previously_split:
        split_uuid = model_pipe.split_dataset()
        pparams.split_uuid = split_uuid
        pparams.previously_split = True 
        pparams.split_only = False

    model_pipe.train_model()

    # load model metrics from file 
    with open(os.path.join(pparams.output_dir, 'model_metrics.json'), 'r') as f:
        model_metrics = json.load(f)

    original_metrics = find_best_test_metric(model_metrics)
    if pparams.prediction_type == 'regression':
        original_mae = original_metrics['prediction_results']['mae_score']
        original_r2 = original_metrics['prediction_results']['r2_score']
        original_rms_score = original_metrics['prediction_results']['rms_score']
    elif pparams.prediction_type == 'classification':
        original_accuracy = original_metrics['prediction_results']['accuracy_score']
        original_precision = original_metrics['prediction_results']['precision']
        original_recall = original_metrics['prediction_results']['recall_score']
        original_prc_auc = original_metrics['prediction_results']['prc_auc_score']


    # extract the seed 
    metadata_path = os.path.join(pparams.output_dir, 'model_metadata.json')
    seed = extract_seed(metadata_path)

    # add the seed to the params
    retrain_pparams.seed = seed
    retrain_pparams.model_uuid = None

    # retrain the model
    retrain_pipe = mp.ModelPipeline(retrain_pparams)
    retrain_pipe.train_model() 
    #retrain_pipe = train(pparams)

    # extract the metrics from the retrained model 
    with open(os.path.join(retrain_pparams.output_dir, 'model_metrics.json'), 'r') as f:
        retrained_model_metrics = json.load(f)
    
    retrained_metrics = find_best_test_metric(retrained_model_metrics)
    if pparams.prediction_type == 'regression':
        retrained_mae = retrained_metrics['prediction_results']['mae_score']
        retrained_r2 = retrained_metrics['prediction_results']['r2_score']
        retrained_rms_score = retrained_metrics['prediction_results']['rms_score']
    elif pparams.prediction_type == 'classification':
        retrained_accuracy = retrained_metrics['prediction_results']['accuracy_score']
        retrained_precision = retrained_metrics['prediction_results']['precision']
        retrained_recall = retrained_metrics['prediction_results']['recall_score']
        retrained_prc_auc = retrained_metrics['prediction_results']['prc_auc_score']

    if pparams.prediction_type == 'regression':
        print("MAE difference:", abs(original_mae-retrained_mae))
        print("R2 difference:", abs(original_r2 - retrained_r2))
        print("RMS Score difference:", abs(original_rms_score - retrained_rms_score))

        assert abs(original_mae-retrained_mae) < 1e-9 \
            and abs(original_r2 - retrained_r2) < 1e-9 \
            and abs(original_rms_score - retrained_rms_score) < 1e-9
        
    elif pparams.prediction_type == 'classification':
        print("Accuracy difference:", abs(original_accuracy - retrained_accuracy))
        print("Precision difference:", abs(original_precision - retrained_precision))
        print("Recall difference:", abs(original_recall - retrained_recall))
        print("PRC AUC difference:", abs(original_prc_auc- retrained_prc_auc))
        
        assert abs(original_accuracy - retrained_accuracy) < 1e-9 \
            and abs(original_precision - retrained_precision) < 1e-9 \
            and abs(original_recall - retrained_recall) < 1e-9 \
            and abs(original_prc_auc - retrained_prc_auc) < 1e-9

#-------------------------------------------------------------------
# Random Forest
def test_RF_regression_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/rf_regression_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_RF_classification_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/rf_classification_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_RF_regression_kfold_cv_reproducibility(): 
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/rf_regression_kfold_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_RF_classification_kfold_cv_reproducibility(): 
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/rf_classification_kfold_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

# Neural Network
def test_NN_regression_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/nn_regression_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_NN_regression_kfold_cv_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/nn_regression_kfold_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_NN_classification_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/nn_classification_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_NN_classification_kfold_cv_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/nn_classification_kfold_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

# XGBoost 
def test_xgboost_regression_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/xgboost_regression_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_xgboost_classification_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/xgboost_classification_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_xgboost_regression_kfold_cv_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/xgboost_regression_kfold_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_xgboost_classification_kfold_cv_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/xgboost_classification_kfold_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

# graphconv
def test_graphconv_classification_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/graphconv_classification_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

def test_graphconv_regression_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/graphconv_regression_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

# DCmodels
@pytest.mark.dgl_required
def test_attentivefp_regression_reproducibility():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/attentivefp_regression_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)

@pytest.mark.dgl_required
def test_pytorchmpnn_regression_reproducibility():
    if not llnl_utils.is_lc_system():
        assert True
        return
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file =  os.path.join(script_path, 'model_json/pytorchmpnn_regression_train_valid_test.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path

    saved_model_identity(pparams)


if __name__ == "__main__":
    # ------ random forest 
    print("test_RF_regression_reproducibility")
    test_RF_regression_reproducibility()
    print("test_RF_regression_kfold_reproducibility")
    test_RF_regression_kfold_cv_reproducibility()
    print("test_RF_classification_reproducibility")
    test_RF_classification_reproducibility()
    print("test_RF_classification_kfold_reproducibility")
    test_RF_classification_kfold_cv_reproducibility()

    # ------ neural network
    print("test_NN_regression_reproducibility")
    test_NN_regression_reproducibility()
    print("test_NN_regression_kfold_reproducibility")
    test_NN_regression_kfold_cv_reproducibility()
    print("test_NN_classification_reproducibility")
    test_NN_classification_reproducibility()
    print("test_NN_classification_kfold_reproducibility")
    test_NN_classification_kfold_cv_reproducibility()

    # ------ xgboost
    print("test_xgboost_regression_reproducibility")
    test_xgboost_regression_reproducibility()
    print("test_xgboost_regression_kfold_reproducibility")
    test_xgboost_regression_kfold_cv_reproducibility()
    print("test_xgboost_classification_reproducibility")
    test_xgboost_classification_reproducibility()
    print("test_xgboost_classification_kfold_reproducibility")
    test_xgboost_classification_kfold_cv_reproducibility()

    # ------ graphconv
    print("test_graphconv_classification_reproducibility")
    test_graphconv_classification_reproducibility()
    print("test_graphconv_regression_reproducibility")
    test_graphconv_regression_reproducibility()

    # ------ dcmodels
    print("test_attentivefp_regression_reproducibility")
    test_attentivefp_regression_reproducibility()

    print("test_pytorchmpnn_regression_reproducibility")
    test_pytorchmpnn_regression_reproducibility()

    print("Passed!")
    