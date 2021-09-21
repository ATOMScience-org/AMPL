#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import os
import sys

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.utils.curate_data as curate_data
import atomsci.ddm.utils.struct_utils as struct_utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import integrative_utilities


def clean():
    """
    Clean test files
    """
    for f in ['delaney-processed_curated.csv',
              'delaney-processed_curated_fit.csv',
              'delaney-processed_curated_external.csv',
              'delaney-processed_curated_predict.csv']:
        if os.path.isfile(f):
            os.remove(f)


def curate():
    """
    Curate dataset for model fitting
    """
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
        curated_df[column_class] = curated_df[column].apply(lambda x: 1 if x>mean else 0)
        
        # make a copy of each column for multi task
        curated_df[column+'2'] = curated_df[column]
        curated_df[column_class+'2'] = curated_df[column_class]

        # Check distribution of response values
        assert (curated_df.shape[0] == 1117), 'Error: Incorrect number of compounds'

        curated_df.to_csv('delaney-processed_curated.csv')

        # Create second test set by reproducible index for prediction
        curated_df.tail(1000).to_csv('delaney-processed_curated_fit.csv')
        curated_df.head(117).to_csv('delaney-processed_curated_external.csv')

    assert (os.path.isfile('delaney-processed_curated.csv'))
    assert (os.path.isfile('delaney-processed_curated_fit.csv'))
    assert (os.path.isfile('delaney-processed_curated_external.csv'))


def download():
    """
    Separate download function so that download can be run separately if there is no internet.
    """
    if (not os.path.isfile('delaney-processed.csv')):
        integrative_utilities.download_save(
            'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
            'delaney-processed.csv')

    assert (os.path.isfile('delaney-processed.csv'))

#train_json_f,'config_delaney_fit_XGB.json'
#pred_json_f,'config_delaney_predict_XGB.json'
def train_and_predict(train_json_f, pred_json_f):
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
    model_dir = 'result/delaney-processed_curated_fit/%s_computed_descriptors_scaffold_regression'%(model_type)
    uuid = integrative_utilities.get_subdirectory(model_dir)
    reload_dir = model_dir+'/'+uuid

    # Check training statistics
    # -------------------------
    if prediction_type == 'regression':
        integrative_utilities.training_statistics_file(reload_dir, 'test', 0.6, 'r2_score')
    else:
        integrative_utilities.training_statistics_file(reload_dir, 'test', 0.6, 'roc_auc')

    # Make prediction parameters
    # --------------------------
    # Read prediction parameter JSON file
    with open(pred_json_f, 'r') as f:
        predict_parameters_dict = json.loads(f.read())

    # Set transformer key here because model uuid is not known before fit
    predict_parameters_dict['transformer_key'] = reload_dir+'transformers.pkl'
    # Set output directory for xgboost (XGB) model
    predict_parameters_dict['result_dir'] = reload_dir

    predict_parameters = parse.wrapper(predict_parameters_dict)

    # Load second test set
    # --------------------
    data = pd.read_csv('delaney-processed_curated_external.csv')

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
    assert (predict['pred'].shape[0] == 117), 'Error: Incorrect number of predictions'
    assert (np.all(np.isfinite(predict['pred'].values))), 'Error: Predictions are not numbers'

    # Save predictions with experimental values
    # -----------------------------------------
    predict.reset_index(level=0, inplace=True)
    combined = pd.merge(data, predict, on=predict_parameters.id_col, how='inner')
    combined.to_csv('delaney-processed_curated_predict.csv')
    assert (os.path.isfile('delaney-processed_curated_predict.csv')
            and os.path.getsize('delaney-processed_curated_predict.csv') > 0), 'Error: Prediction file not created'



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

    # Train and Predict
    # -----
    train_and_predict('config_delaney_fit_XGB.json', 
        'config_delaney_predict_XGB.json')


def make_json_files():
    


if __name__ == '__main__':
    test()
