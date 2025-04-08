#!/usr/bin/env python 
"""Testing the sampling methods. Want to ensure that the model pipeline works and that the sampling methods are incorporated.
Based off of the test_kfold_split.py method. """
import sklearn.metrics as skmetrics 
import copy
import os
import json 

from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
import atomsci.ddm.pipeline.predict_from_model as pfm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from integrative_utilities import extract_seed, get_test_set, find_best_test_metric

#-------------------------------------------------------------------

def saved_model_identity(pparams):
    script_path = os.path.dirname(os.path.realpath(__file__))
    retrain_pparams = copy.copy(pparams)

    model_pipe = mp.ModelPipeline(pparams)

    if not pparams.previously_split:
        split_uuid = model_pipe.split_dataset()
        pparams.split_uuid = split_uuid
        pparams.previously_split = True
        pparams.split_only=False

    model_pipe.train_model()
    
    split_csv = os.path.join(script_path, '../../test_datasets/', model_pipe.data._get_split_key())
    test_df = get_test_set(pparams.dataset_key, split_csv, pparams.id_col)

    # load model metrics from file 
    with open(os.path.join(pparams.output_dir, 'model_metrics.json'), 'r') as f:
        model_metrics = json.load(f)
    
    metrics = find_best_test_metric(model_metrics)
    original_accuracy = metrics['prediction_results']['accuracy_score']
    original_precision = metrics['prediction_results']['precision']
    original_recall = metrics['prediction_results']['recall_score']
    original_prc_auc = metrics['prediction_results']['prc_auc_score']
    
    id_col = metrics['input_dataset']['id_col']
    response_col=metrics['input_dataset']['response_cols'][0]
    smiles_col = metrics['input_dataset']['smiles_col']
    test_length = metrics['prediction_results']['num_compounds']

    # predict from model
    model_tar = model_pipe.params.model_tarball_path
    pred_df = pfm.predict_from_model_file(model_tar, test_df, id_col=id_col,
                smiles_col=smiles_col, response_col=response_col)
    # generate another prediction from the same model file
    pred_df2 = pfm.predict_from_model_file(model_tar, test_df, id_col=id_col, smiles_col=smiles_col, response_col=response_col)

    X = pred_df[response_col+'_actual'].values
    y = pred_df[response_col+'_pred'].values

    accuracy = skmetrics.accuracy_score(X, y)
    precision = skmetrics.precision_score(X, y, average='weighted')
    recall = skmetrics.recall_score(X, y, average='weighted')
    prc_auc = skmetrics.average_precision_score(X, y)

    # return the metrics from the second prediction
    X2 = pred_df2[response_col+'_actual'].values
    y2 = pred_df2[response_col+'_pred'].values

    x2_accuracy = skmetrics.accuracy_score(X2, y2)
    x2_precision = skmetrics.precision_score(X2, y2, average='weighted')
    x2_recall = skmetrics.recall_score(X2, y2, average='weighted')
    x2_prc_auc = skmetrics.average_precision_score(X2, y2)

    #saved_accuracy = metrics['prediction_results']['accuracy_score']
    #saved_precision = metrics['prediction_results']['precision']
    #saved_recall = metrics['prediction_results']['recall_score']
    #saved_prc_auc = metrics['prediction_results']['prc_auc_score']

    # show results and compare the two predictions 
    print(metrics['subset'])
    print(pred_df.columns)
    print("Prediction results")
    print("Accuracy difference:", abs(accuracy - x2_accuracy))
    print("Precision difference:", abs(precision - x2_precision))
    print("Recall difference:", abs(recall-x2_recall))
    print("PRC AUC difference:", abs(prc_auc-x2_prc_auc))

    assert abs(accuracy - x2_accuracy) < 1e-9 \
        and abs(precision - x2_precision) < 1e-9 \
        and abs(recall - x2_recall) < 1e-9 \
        and abs(prc_auc - x2_prc_auc) < 1e-9 \
        and (test_length == len(test_df))

    # create another test to ensure that the sampling methods are reproducible with the seed 
    metadata_path = os.path.join(pparams.output_dir, 'model_metadata.json')
    seed = extract_seed(metadata_path)

    # create a duplicate parameters and add the seed
    retrain_pparams.seed = seed
    retrain_pparams.model_uuid = None 

    # retrain the model
    retrain_pipe = mp.ModelPipeline(retrain_pparams)
    retrain_pipe.train_model()

    # extract the metrics from the retrained model
    with open(os.path.join(retrain_pparams.output_dir, 'model_metrics.json'), 'r') as f:
        retrained_model_metrics = json.load(f)

    retrained_metrics = find_best_test_metric(retrained_model_metrics)    
    retrained_accuracy = retrained_metrics['prediction_results']['accuracy_score']
    retrained_precision = retrained_metrics['prediction_results']['precision']
    retrained_recall = retrained_metrics['prediction_results']['recall_score']       
    retrained_prc_auc = retrained_metrics['prediction_results']['prc_auc_score']
    
    print("Model reproducibility results")
    print("Accuracy difference:", abs(original_accuracy-retrained_accuracy))
    print("Precision difference:", abs(original_precision-retrained_precision))
    print("Recall difference:", abs(original_recall-retrained_recall))
    print("PRC AUC difference:", abs(original_prc_auc-retrained_prc_auc))
    
    assert abs(original_accuracy - retrained_accuracy) < 1e-9 \
        and abs(original_precision - retrained_precision) < 1e-9 \
        and abs(original_recall - retrained_recall) < 1e-9 \
        and abs(original_prc_auc - retrained_prc_auc) < 1e-9
#-------------------------------------------------------------------

#-------- random forest
def test_train_valid_test_RF_SMOTE(): 
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/train_valid_test_RF_SMOTE.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def test_k_fold_cv_RF_SMOTE():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/kfold_cv_RF_SMOTE.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def test_k_fold_cv_RF_undersampling():

    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/kfold_cv_RF_undersampling.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def test_train_valid_test_RF_undersampling(): 
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/train_valid_test_RF_undersampling.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

#-------- neural network

def test_train_valid_test_NN_SMOTE():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/train_valid_test_NN_SMOTE.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def test_train_valid_test_NN_undersampling():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/train_valid_test_NN_undersampling.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def test_k_fold_cv_NN_SMOTE():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/kfold_cv_NN_SMOTE.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def test_k_fold_cv_NN_undersampling():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/kfold_cv_NN_undersampling.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

#-------- xgboost

def test_train_valid_test_xgboost_SMOTE():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/train_valid_test_xgboost_SMOTE.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def test_train_valid_test_xgboost_undersampling():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/train_valid_test_xgboost_undersampling.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def test_k_fold_cv_xgboost_SMOTE():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/kfold_cv_xgboost_SMOTE.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def test_k_fold_cv_xgboost_undersampling():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'sampling_json/kfold_cv_xgboost_undersampling.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)
#-------------------------------------------------------------------

if __name__=='__main__':
    #print('train_valid_test_RF_SMOTE_test')
    #test_train_valid_test_RF_SMOTE()
    
    #print('train_valid_test_NN_SMOTE_test')
    #test_train_valid_test_NN_SMOTE()
    
    #print("train_valid_test_RF_undersampling_test")
    #test_train_valid_test_RF_undersampling()
    
    #print("train_valid_test_NN_undersampling_test")
    #test_train_valid_test_NN_undersampling()
    
    #print("kfold_cv_NN_SMOTE_test")
    #test_k_fold_cv_NN_SMOTE()

    #print("kfold_cv_NN_undersampling_test")
    #test_k_fold_cv_NN_undersampling()

    print("kfold_cv_RF_SMOTE_test")
    test_k_fold_cv_RF_SMOTE()

    #print("kfold_cv_RF_undersampling_test")
    #test_k_fold_cv_RF_undersampling()
    
    #print("train_valid_test_xgboost_SMOTE_test")
    #test_train_valid_test_xgboost_SMOTE()

    #print("train_valid_test_xgboost_undersampling_test")
    #test_train_valid_test_xgboost_undersampling()

    #print("k_fold_cv_xgboost_SMOTE_test")
    #test_k_fold_cv_xgboost_SMOTE()

    #print("k_fold_cv_xgboost_undersampling_test")
    #test_k_fold_cv_xgboost_undersampling()

    print("Passed!")