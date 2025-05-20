#!/usr/bin/env python

import json
import numpy as np
import os
import pandas as pd
import shutil
import sys
import tarfile

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.utils.curate_data as curate_data
import atomsci.ddm.utils.struct_utils as struct_utils
import atomsci.ddm.utils.file_utils as futils
import atomsci.ddm.pipeline.compare_models as cm
import atomsci.ddm.pipeline.predict_from_model as pfm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import integrative_utilities

def clean():
    """Clean test files"""
    for f in ['hlm_clearance_curated_predict.csv',
              'hlm_clearance_curated_external.csv',
              'hlm_clearance_curated_fit.csv',
              'hlm_clearance_curated.csv']:
        if os.path.isfile(f):
            os.remove(f)

    if os.path.exists('clearance'):
        shutil.rmtree('clearance')

    if os.path.exists('scaled_descriptors'):
        shutil.rmtree('scaled_descriptors')


def curate():
    """Curate dataset for model fitting"""
    with tarfile.open('ci8b00785_si_001.tar.gz', mode='r:gz') as tar:
        futils.safe_extract(tar, 'clearance')

    raw_df = pd.read_csv('clearance/SuppInfo/Dataset_chembl_clearcaco.txt', sep=";", dtype='str')

    #response variable
    #hlm_clearance[mL.min-1.g-1]

    # replace commas with decimal point
    raw_df['hlm_clearance[mL.min-1.g-1]']=raw_df['hlm_clearance[mL.min-1.g-1]'].str.replace(',','.')
    # record as floating point values.
    raw_df['hlm_clearance[mL.min-1.g-1]']=raw_df['hlm_clearance[mL.min-1.g-1]'].astype(float)

    hlmc_df=raw_df.rename(columns={'hlm_clearance[mL.min-1.g-1]':'value'}, inplace=False)
    hlmc_df.rename(columns={'ID':'compound_id'}, inplace=True)
    hlmc_df['rdkit_smiles']=hlmc_df['Canonical_Smiles'].apply(struct_utils.base_smiles_from_smiles)
    col = ['compound_id', 'rdkit_smiles', 'value']
    hlmc_df = hlmc_df[col]
    # drop NaN values
    hlmc_df=hlmc_df.dropna()
    print(hlmc_df.shape)

    assert(hlmc_df.shape == (5348, 3)), 'Error: Incorrect data size'

    tolerance=10
    column='value'
    list_bad_duplicates='Yes'
    data=hlmc_df
    max_std=20
    curated_df=curate_data.average_and_remove_duplicates (column, tolerance,list_bad_duplicates, data, max_std, compound_id='compound_id',smiles_col='rdkit_smiles')

    data_filename="hlm_clearance_curated.csv"

    nr=curated_df.shape[0]
    nc=curated_df.shape[1]

    curated_df.to_csv(data_filename, index=False)

    # Create second test set by reproducible index for prediction
    curated_df.tail(4989).to_csv('hlm_clearance_curated_fit.csv')
    curated_df.head(348).to_csv('hlm_clearance_curated_external.csv')


def check_for_data_zip():
    assert(os.path.isfile('ci8b00785_si_001.tar.gz'))


def test():
    """Test full model pipeline: Curate data, fit model, and predict property for new compounds"""
    # Clean
    # -----
    integrative_utilities.clean_fit_predict()
    clean()

    # Check for data
    # --------
    check_for_data_zip()

    # Curate
    # ------
    curate()

    # Train model
    # -----------
    # Read parameter JSON file
    with open('config_wenzel_fit_NN.json') as f:
        config = json.loads(f.read())

    # Parse parameters
    params = parse.wrapper(config)

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    # Get uuid and reload directory
    # -----------------------------
    uuid = integrative_utilities.get_subdirectory('result/hlm_clearance_curated_fit/NN_computed_descriptors_scaffold_regression')
    reload_dir = 'result/hlm_clearance_curated_fit/NN_computed_descriptors_scaffold_regression/'+uuid

    # Check training statistics
    # -------------------------
    integrative_utilities.training_statistics_file(reload_dir, 'valid', 0.1)

    # Make prediction using the trained model
    # -------------------------
    result_df = cm.get_filesystem_perf_results('result', pred_type='regression')

    # There should only be one model trained
    # -------------------------
    assert len(result_df) == 1
    model_path = result_df.model_path[0] # this is the path to a tar file

    # Load second test set
    # --------------------
    data = pd.read_csv('hlm_clearance_curated_external.csv')
    # Make prediction pipeline
    # ------------------------
    predict = pfm.predict_from_model_file(model_path, data,
                                id_col=params.id_col,
                                smiles_col=params.smiles_col,
                                response_col=params.response_cols,
                                is_featurized=False)

    # Check predictions
    # -----------------
    assert (predict['VALUE_NUM_mean_pred'].shape[0] == 348), 'Error: Incorrect number of predictions'
    assert (np.all(np.isfinite(predict['VALUE_NUM_mean_pred'].values))), 'Error: Predictions are not numbers'

    # Save predictions with experimental values
    # -----------------------------------------
    predict.reset_index(level=0, inplace=True)
    combined = pd.merge(data, predict, on=params.id_col, how='inner')
    combined.to_csv('hlm_clearance_curated_predict.csv')
    assert (os.path.isfile('hlm_clearance_curated_predict.csv')
            and os.path.getsize('hlm_clearance_curated_predict.csv') > 0), 'Error: Prediction file not created'

    clean()
    integrative_utilities.clean_fit_predict()

if __name__ == '__main__':
    test()
