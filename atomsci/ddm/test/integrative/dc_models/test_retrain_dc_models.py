#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import os
import sys
import tarfile
import tempfile

import rdkit.Chem as rdC
import rdkit.Chem.Descriptors as rdCD

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.predict_from_model as pfm
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.utils.model_retrain as mr
import atomsci.ddm.utils.file_utils as futils
from atomsci.ddm.utils import llnl_utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import integrative_utilities


def clean(prefix='delaney-processed'):
    """
    Clean test files
    """
    for f in ['%s_curated.csv'%prefix,
              '%s_curated_fit.csv'%prefix,
              '%s_curated_external.csv'%prefix,
              '%s_curated_predict.csv'%prefix]:
        if os.path.isfile(f):
            os.remove(f)

def exact_mol_weight(x):
    '''
    Given SMILES, return exact mol weight
    '''
    return rdCD.ExactMolWt(rdC.MolFromSmiles(x))

def num_atoms(x):
    '''
    Given SMILES, return the number of atoms
    '''
    return len(rdC.MolFromSmiles(x).GetAtoms())

def H1_curate():
    """
    Curate dataset for model fitting
    """
    if (not os.path.isfile('H1_curated.csv') and
            not os.path.isfile('H1_curated_fit.csv') and
            not os.path.isfile('H1_curated_external.csv')):
        curated_df = pd.read_csv('../../test_datasets/H1_std.csv')
        split_df = pd.read_csv('../../test_datasets/H1_std_train_valid_test_scaffold_002251a2-83f8-4511-acf5-e8bbc5f86677.csv')
        id_col = "compound_id"
        column = "pKi_mean"

        # add additional columns for other forms of prediction
        # e.g. classification and multi task
        mean = np.mean(curated_df[column])
        column_class = column+'_class'
        curated_df[column_class] = curated_df[column].apply(lambda x: 1.0 if x>mean else 0.0)

        # make a copy of each column for multi task
        curated_df[column+'2'] = curated_df[column]
        curated_df[column_class+'2'] = curated_df[column_class]

        # make a really easy test. calc mass of each molecule
        curated_df['exact_mol_weight'] = curated_df['base_rdkit_smiles'].apply(lambda x: exact_mol_weight(x))

        # make a really easy test. count the number of atoms in a molecule
        curated_df['num_atoms'] = curated_df['base_rdkit_smiles'].apply(lambda x: num_atoms(x))

        curated_df.to_csv('H1_curated.csv', index=False)
        split_df.to_csv('H1_curated_fit_train_valid_test_scaffold_002251a2-83f8-4511-acf5-e8bbc5f86677.csv', index=False)

        split_external = split_df[split_df['subset']=='test']
        curated_external_df = curated_df[curated_df[id_col].isin(split_external['cmpd_id'])]
        # Create second test set by reproducible index for prediction
        curated_df.to_csv('H1_curated_fit.csv', index=False)
        curated_external_df.to_csv('H1_curated_external.csv', index=False)

    assert (os.path.isfile('H1_curated.csv'))
    assert (os.path.isfile('H1_curated_fit.csv'))
    assert (os.path.isfile('H1_curated_external.csv'))
    assert (os.path.isfile('H1_curated_fit_train_valid_test_scaffold_002251a2-83f8-4511-acf5-e8bbc5f86677.csv'))

def train_and_predict(train_json_f, prefix='delaney-processed'):
    # Train model
    # -----------
    # Read parameter JSON file
    with open(train_json_f) as f:
        config = json.loads(f.read())

    # Parse parameters
    params = parse.wrapper(config)

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    # Get uuid and reload directory
    # -----------------------------
    model_type = params.model_type
    prediction_type = params.prediction_type
    descriptor_type = params.descriptor_type
    featurizer = params.featurizer
    splitter = params.splitter
    model_dir = 'result/%s_curated_fit/%s_%s_%s_%s'%(prefix, model_type, featurizer, splitter, prediction_type)
    uuid = model.params.model_uuid
    tar_f = 'result/%s_curated_fit_model_%s.tar.gz'%(prefix, uuid)
    reload_dir = model_dir+'/'+uuid

    # Check training statistics
    # -------------------------
    if prediction_type == 'regression':
        threshold = 0.4
        if 'perf_threshold' in config:
            threshold = float(config['perf_threshold'])

        integrative_utilities.training_statistics_file(reload_dir, 'test', threshold, 'r2_score')
        score = integrative_utilities.read_training_statistics_file(reload_dir, 'test', 'r2_score')
    else:
        threshold = 0.7
        if 'perf_threshold' in config:
            threshold = float(config['perf_threshold'])
        integrative_utilities.training_statistics_file(reload_dir, 'test', threshold, 'accuracy_score')
        score = integrative_utilities.read_training_statistics_file(reload_dir, 'test', 'accuracy_score')

    print("Final test score:", score)

    # Load second test set
    # --------------------
    data = pd.read_csv('%s_curated_external.csv'%prefix)

    predict = pfm.predict_from_model_file(tar_f, data, id_col=params.id_col, 
        smiles_col=params.smiles_col, response_col=params.response_cols)
    pred_cols = [f for f in predict.columns if f.endswith('_pred')]

    pred = predict[pred_cols].to_numpy()

    # Check predictions
    # -----------------
    assert (pred.shape[0] == len(data)), 'Error: Incorrect number of predictions'
    assert (np.all(np.isfinite(pred))), 'Error: Predictions are not numbers'

    # Save predictions with experimental values
    # -----------------------------------------
    predict.reset_index(level=0, inplace=True)
    combined = pd.merge(data, predict, on=params.id_col, how='inner')
    pred_csv_name = '%s_curated_%s_%s_%s_%s_%d_%s_predict.csv'%(
            prefix, model_type, prediction_type, descriptor_type, featurizer, 
            len(model.params.response_cols), model.params.splitter)
    combined.to_csv(pred_csv_name)
    assert (os.path.isfile(pred_csv_name)
            and os.path.getsize(pred_csv_name) > 0), 'Error: Prediction file not created'

    return tar_f

def verify_saved_params(original_json_f, tar_f):
    '''
    compares saved params in a tar file with original json
    '''
    reload_dir = tempfile.mkdtemp()
    with tarfile.open(tar_f, mode='r:gz') as tar:
        futils.safe_extract(tar, path=reload_dir)

    # read config from tar file
    config_file_path = os.path.join(reload_dir, 'model_metadata.json')
    with open(config_file_path) as f:
        tar_config = json.loads(f.read())

    # read original config
    with open(original_json_f) as f:
        original_config = json.loads(f.read())

    original_pp = parse.wrapper(original_config)
    original_model_params = parse.extract_model_params(original_pp)
    original_feat_params = parse.extract_featurizer_params(original_pp)

    tar_pp = parse.wrapper(tar_config)
    tar_model_params = parse.extract_model_params(tar_pp)
    tar_feat_params = parse.extract_featurizer_params(tar_pp)

    print('-----------------------------------')
    print('model params')
    print(original_model_params)
    print(tar_model_params)
    assert original_model_params == tar_model_params
    print('-----------------------------------')
    print('feat params')
    print(original_feat_params)
    print(tar_feat_params)
    assert original_feat_params == tar_feat_params

def retrain(tar_f, prefix='H1'):
    '''
    retrain a model from tar_f
    '''
    model = mr.train_model_from_tar(tar_f, 'result')

    uuid = model.params.model_uuid
    re_tar_f = f'result/{prefix}_curated_fit_model_{uuid}.tar.gz'

    assert os.path.exists(re_tar_f)

    return re_tar_f

def H1_init():
    """
    Test full model pipeline: Curate data, fit model, and predict property for new compounds
    """

    # Clean
    # -----
    integrative_utilities.clean_fit_predict()
    clean('H1')

    # Curate
    # ------
    H1_curate()

# Train and Predict
# -----
def test_reg_config_H1_fit_AttentiveFPModel():
    if not llnl_utils.is_lc_system():
        assert True
        return
    
    H1_init()
    json_f = 'reg_config_H1_fit_AttentiveFPModel.json'
    tar_f = train_and_predict(json_f, prefix='H1') # crashes during run

    verify_saved_params(json_f, tar_f)

    re_tar_f = retrain(tar_f, 'H1')

    verify_saved_params(json_f, re_tar_f)

# -----
def test_reg_config_H1_fit_GCNModel():
    if not llnl_utils.is_lc_system():
        assert True
        return
        
    H1_init()
    json_f = 'reg_config_H1_fit_GCNModel.json'
    tar_f = train_and_predict(json_f, prefix='H1') # crashes during run

    verify_saved_params(json_f, tar_f)

    re_tar_f = retrain(tar_f, 'H1')

    verify_saved_params(json_f, re_tar_f)

# -----
def test_reg_config_H1_fit_MPNNModel():
    if not llnl_utils.is_lc_system():
        assert True
        return
    
    H1_init()
    json_f = 'reg_config_H1_fit_MPNNModel.json'
    tar_f = train_and_predict(json_f, prefix='H1') # crashes during run

    verify_saved_params(json_f, tar_f)

    re_tar_f = retrain(tar_f, 'H1')

    verify_saved_params(json_f, re_tar_f)

def test_reg_config_H1_fit_GraphConvModel():
    if not llnl_utils.is_lc_system():
        assert True
        return
    
    H1_init()
    json_f = 'reg_config_H1_fit_GraphConvModel.json'
    tar_f = train_and_predict(json_f, prefix='H1') # crashes during run

    verify_saved_params(json_f, tar_f)

    re_tar_f = retrain(tar_f, 'H1')

    verify_saved_params(json_f, re_tar_f)

def test_reg_config_H1_fit_PytorchMPNNModel():
    if not llnl_utils.is_lc_system():
        assert True
        return
    
    H1_init()
    json_f = 'reg_config_H1_fit_PytorchMPNNModel.json'
    tar_f = train_and_predict(json_f, prefix='H1') # crashes during run

    verify_saved_params(json_f, tar_f)

    re_tar_f = retrain(tar_f, 'H1')

    verify_saved_params(json_f, re_tar_f)

if __name__ == '__main__':
    test_reg_config_H1_fit_PytorchMPNNModel() # Pytorch implementation of MPNNModel
    test_reg_config_H1_fit_GraphConvModel() # the same model as graphconv
    test_reg_config_H1_fit_MPNNModel() # uses the WeaveFeaturizer
    test_reg_config_H1_fit_GCNModel()
    test_reg_config_H1_fit_AttentiveFPModel() #works fine?
