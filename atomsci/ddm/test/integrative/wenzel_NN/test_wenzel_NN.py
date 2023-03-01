#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import os
import shutil
import sys
import zipfile

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.utils.curate_data as curate_data
import atomsci.ddm.utils.struct_utils as struct_utils
import atomsci.ddm.utils.file_utils as futils
from atomsci.ddm.utils import llnl_utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import integrative_utilities


def clean():
    """
    Clean test files
    """
    if not llnl_utils.is_lc_system():
        assert True
        return
        
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
    """
    Curate dataset for model fitting
    """
    with zipfile.ZipFile('ci8b00785_si_001.zip', 'r') as zip_ref:
        futils.safe_extract(zip_ref, 'clearance')

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


def download():
    """
    Separate download function so that download can be run separately if there is no internet.
    """
    if (not os.path.isfile('ci8b00785_si_001.zip')):
        integrative_utilities.download_save(
            'https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.8b00785/suppl_file/ci8b00785_si_001.zip', 'ci8b00785_si_001.zip', verify=False)

    assert(os.path.isfile('ci8b00785_si_001.zip'))


def test():
    """
    Test full model pipeline: Curate data, fit model, and predict property for new compounds
    """

    # Clean
    # -----
    integrative_utilities.clean_fit_predict()
    clean()

    # Download
    # --------
    download()

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
    integrative_utilities.training_statistics_file(reload_dir, 'valid', 0.3)

    # Make prediction parameters
    # --------------------------
    # Read prediction parameter JSON file
    with open('config_wenzel_predict_NN.json', 'r') as f:
        predict_parameters_dict = json.loads(f.read())

    # Set transformer key here because model uuid is not known before fit
    predict_parameters_dict['transformer_key'] = reload_dir+'transformers.pkl'

    predict_parameters = parse.wrapper(predict_parameters_dict)

    # Load second test set
    # --------------------
    data = pd.read_csv('hlm_clearance_curated_external.csv')

    # Select columns and rename response column
    data = data[[predict_parameters.id_col, predict_parameters.smiles_col, predict_parameters.response_cols[0]]]
    data = data.rename(columns={predict_parameters.response_cols[0]: 'experimental_values'})

    # Make prediction pipeline
    # ------------------------
    pp = mp.create_prediction_pipeline_from_file(predict_parameters, reload_dir)

    # Predict
    # -------
    predict = pp.predict_on_dataframe(data)

    # Check predictions
    # -----------------
    assert (predict['pred'].shape[0] == 348), 'Error: Incorrect number of predictions'
    assert (np.all(np.isfinite(predict['pred'].values))), 'Error: Predictions are not numbers'

    # Save predictions with experimental values
    # -----------------------------------------
    predict.reset_index(level=0, inplace=True)
    combined = pd.merge(data, predict, on=predict_parameters.id_col, how='inner')
    combined.to_csv('hlm_clearance_curated_predict.csv')
    assert (os.path.isfile('hlm_clearance_curated_predict.csv')
            and os.path.getsize('hlm_clearance_curated_predict.csv') > 0), 'Error: Prediction file not created'


if __name__ == '__main__':
    test()
