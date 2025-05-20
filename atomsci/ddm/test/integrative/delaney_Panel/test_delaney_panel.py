#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import os
import sys
import pytest

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.predict_from_model as pfm
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.utils.curate_data as curate_data
import atomsci.ddm.utils.struct_utils as struct_utils
from atomsci.ddm.utils import llnl_utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import integrative_utilities


def clean(prefix='delaney-processed'):
    """Clean test files"""
    for f in ['%s_curated.csv'%prefix,
              '%s_curated_fit.csv'%prefix,
              '%s_curated_external.csv'%prefix,
              '%s_curated_predict.csv'%prefix]:
        if os.path.isfile(f):
            os.remove(f)

def curate():
    """Curate dataset for model fitting"""
    if (not os.path.isfile('delaney-processed_curated.csv') and
            not os.path.isfile('delaney-processed_curated_fit.csv') and
            not os.path.isfile('delaney-processed_curated_external.csv')):
        raw_df = pd.read_csv('delaney-processed.csv')

        # Generate smiles, inchi
        raw_df['rdkit_smiles'] = raw_df['smiles'].apply(curate_data.base_smiles_from_smiles)
        raw_df['inchi_key'] = raw_df['smiles'].apply(struct_utils.smiles_to_inchi_key)

        # Check for duplicate compounds based on SMILES string
        # Average the response value for duplicates
        # Remove compounds where response value variation is above the threshold
        # tolerance=% of individual respsonse value is allowed to different from the average to be included in averaging.
        # max_std = maximum allowed standard deviation for computed average response value
        tolerance = 10  # percentage
        column = 'measured log solubility in mols per litre'
        list_bad_duplicates = 'Yes'
        data = raw_df
        max_std = 100000  # esentially turned off in this example
        data['compound_id'] = data['inchi_key']
        curated_df = curate_data.average_and_remove_duplicates(
            column, tolerance, list_bad_duplicates, data, max_std, compound_id='compound_id', smiles_col='rdkit_smiles')

        # add additional columns for other forms of prediction
        # e.g. classification and multi task
        mean = np.mean(curated_df[column])
        column_class = column+'_class'
        curated_df[column_class] = curated_df[column].apply(lambda x: 1.0 if x>mean else 0.0)
        
        # make a copy of each column for multi task
        curated_df[column+'2'] = curated_df[column]
        curated_df[column_class+'2'] = curated_df[column_class]

        # Check distribution of response values
        assert (curated_df.shape[0] == 1116), 'Error: Incorrect number of compounds'

        curated_df.to_csv('delaney-processed_curated.csv')

        # Create second test set by reproducible index for prediction
        curated_df.tail(999).to_csv('delaney-processed_curated_fit.csv')
        curated_df.head(117).to_csv('delaney-processed_curated_external.csv')

    assert (os.path.isfile('delaney-processed_curated.csv'))
    assert (os.path.isfile('delaney-processed_curated_fit.csv'))
    assert (os.path.isfile('delaney-processed_curated_external.csv'))

def H1_curate():
    """Curate dataset for model fitting"""
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

def duplicate_df(original, id_col):
    """Copies all rows in the original and appends _dupe to the ids
    Returns a dataframe with each row duplicated once
    """
    second_df = original.copy()
    second_df[id_col] = second_df[id_col].apply(lambda x: x+'_dupe')
    return pd.concat([original, second_df])

def H1_double_curate():
    """Curate dataset for model fitting"""
    if (not os.path.isfile('H1_double_curated.csv') and
            not os.path.isfile('H1_double_curated_fit.csv') and
            not os.path.isfile('H1_double_curated_external.csv')):
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

        # duplicate some of the SMILES
        curated_df = duplicate_df(curated_df, id_col)
        split_df = duplicate_df(split_df, 'cmpd_id')

        curated_df.to_csv('H1_double_curated.csv', index=False)
        split_df.to_csv('H1_double_curated_fit_train_valid_test_scaffold_002251a2-83f8-4511-acf5-e8bbc5f86677.csv', index=False)

        split_external = split_df[split_df['subset']=='test']
        curated_external_df = curated_df[curated_df[id_col].isin(split_external['cmpd_id'])]
        # Create second test set by reproducible index for prediction
        curated_df.to_csv('H1_double_curated_fit.csv', index=False)
        curated_external_df.to_csv('H1_double_curated_external.csv', index=False)

    assert (os.path.isfile('H1_double_curated.csv'))
    assert (os.path.isfile('H1_double_curated_fit.csv'))
    assert (os.path.isfile('H1_double_curated_external.csv'))
    assert (os.path.isfile('H1_double_curated_fit_train_valid_test_scaffold_002251a2-83f8-4511-acf5-e8bbc5f86677.csv'))

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
    assert len(pred_cols) == len(params.response_cols)

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

def init():
    """Test full model pipeline: Curate data, fit model, and predict property for new compounds"""

    # Clean
    # -----
    integrative_utilities.clean_fit_predict()
    clean()

    # Copy Data
    # --------
    integrative_utilities.copy_delaney()

    # Curate
    # ------
    curate()

def H1_init():
    """Test full model pipeline: Curate data, fit model, and predict property for new compounds"""

    # Clean
    # -----
    integrative_utilities.clean_fit_predict()
    clean('H1')

    # Curate
    # ------
    H1_curate()

def H1_double_init():
    """Test full model pipeline: Curate data, fit model, and predict property for new compounds"""

    # Clean
    # -----
    integrative_utilities.clean_fit_predict()
    clean('H1_double')

    # Curate
    # ------
    H1_double_curate()


# Train and Predict
# -----
def test_reg_config_delaney_fit_RF_3fold():
    init()
    train_and_predict('jsons/config_delaney_fit_rf_3fold_cv.json') # fine

def test_reg_config_delaney_fit_XGB_3fold():
    init()
    train_and_predict('jsons/config_delaney_fit_xgb_3fold_cv.json') # fine

def test_reg_config_delaney_fit_NN_graphconv():
    init()
    train_and_predict('jsons/reg_config_delaney_fit_NN_graphconv.json') # fine

def test_reg_config_delaney_fit_XGB_mordred_filtered():
    init()
    train_and_predict('jsons/reg_config_delaney_fit_XGB_mordred_filtered.json') # fine

def test_reg_config_delaney_fit_RF_mordred_filtered():
    init()
    train_and_predict('jsons/reg_config_delaney_fit_RF_mordred_filtered.json') # predict_full_dataset broken

def test_reg_kfold_config_delaney_fit_NN_graphconv():
    init()
    train_and_predict('jsons/reg_kfold_config_delaney_fit_NN_graphconv.json') # fine

def test_class_config_delaney_fit_XGB_mordred_filtered():
    init()
    train_and_predict('jsons/class_config_delaney_fit_XGB_mordred_filtered.json') # breaks because labels aren't numbers

def test_class_config_delaney_fit_NN_ecfp():
    init()
    train_and_predict('jsons/class_config_delaney_fit_NN_ecfp.json') # only works for class

def test_multi_class_random_config_delaney_fit_NN_mordred_filtered():
    init()
    train_and_predict('jsons/multi_class_random_config_delaney_fit_NN_mordred_filtered.json') # crashes during run

def test_multi_class_config_delaney_fit_NN_graphconv():
    init()
    train_and_predict('jsons/multi_class_config_delaney_fit_NN_graphconv.json') # fine

def test_multi_reg_config_delaney_fit_NN_graphconv():
    init()
    train_and_predict('jsons/multi_reg_config_delaney_fit_NN_graphconv.json') # fine

# MOE doesn't seem to predict delaney very well
# these are run using H1
# -------
@pytest.mark.moe_required
def test_reg_config_H1_fit_XGB_moe():
    H1_init()
    if llnl_utils.is_lc_system():
        train_and_predict('jsons/reg_config_H1_fit_XGB_moe.json', prefix='H1')

@pytest.mark.moe_required
def test_reg_config_H1_fit_NN_moe():
    H1_init()
    if llnl_utils.is_lc_system():
        train_and_predict('jsons/reg_config_H1_fit_NN_moe.json', prefix='H1')

@pytest.mark.moe_required
def test_reg_config_H1_double_fit_NN_moe():
    H1_double_init()
    if llnl_utils.is_lc_system():
        train_and_predict('jsons/reg_config_H1_double_fit_NN_moe.json', prefix='H1_double')

@pytest.mark.moe_required
def test_multi_class_random_config_H1_fit_NN_moe():
    H1_init()
    if llnl_utils.is_lc_system():
        train_and_predict('jsons/multi_class_config_H1_fit_NN_moe.json', prefix='H1')

@pytest.mark.moe_required
def test_class_config_H1_fit_NN_moe():
    H1_init()
    if llnl_utils.is_lc_system():
        train_and_predict('jsons/class_config_H1_fit_NN_moe.json', prefix='H1')

if __name__ == '__main__':
    test_reg_kfold_config_delaney_fit_NN_graphconv()
    #pass
