"""
Functions to run predictions from a pre-trained model against user-provided data.
"""

import tempfile
import numpy as np
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles

# =====================================================================================================
def predict_from_tracker_model(model_uuid, collection, input_df, id_col='compound_id', smiles_col='rdkit_smiles',
                     response_col=None, is_featurized=False, dont_standardize=False):
    """
    Loads a pretrained model from the model tracker database and runs predictions on compounds in an input
    data frame.

    Args:
        model_uuid (str): The unique identifier of the model

        collection (str): Name of the collection in the model tracker DB containing the model.

        input_df (DataFrame): Input data to run predictions on; must at minimum contain SMILES strings.

        id_col (str): Name of the column containing compound IDs. If none is provided, sequential IDs will be
        generated.

        smiles_col (str): Name of the column containing SMILES strings; required.

        response_col (str): Name of an optional column containing actual response values; if it is provided, 
        the actual values will be included in the returned data frame to make it easier for you to assess performance.

        dont_standardize (bool): By default, SMILES strings are salt-stripped and standardized using RDKit; 
        if you have already done this, or don't want them to be standardized, set dont_standardize to True.

    Return: 
        A data frame with compound IDs, SMILES strings and predicted response values. Actual response values
        will be included if response_col is provided. Standard prediction error estimates will be included
        if the model was trained with uncertainty=True. Note that the predicted and actual response
        columns will be labeled according to the response_col setting in the original training data,
        not the response_col passed to this function; e.g. if the original model response_col was 'pIC50',
        the returned data frame will contain columns 'pIC50_actual', 'pIC50_pred' and 'pIC50_std'.
    """
    input_df, pred_params = _prepare_input_data(input_df, id_col, smiles_col, response_col, dont_standardize)
    has_responses = ('response_cols' in pred_params)
    pred_params = parse.wrapper(pred_params)
    pipe = mp.create_prediction_pipeline(pred_params, model_uuid, collection)
    pred_df = pipe.predict_full_dataset(input_df, contains_responses=has_responses, is_featurized=is_featurized,
                                        dset_params=pred_params)
    pred_df = pred_df.sort_values(by=id_col)
    return pred_df

# =====================================================================================================
def predict_from_model_file(model_path, input_df, id_col='compound_id', smiles_col='rdkit_smiles',
                     response_col=None, is_featurized=False, dont_standardize=False):
    """
    Loads a pretrained model from a model tarball file and runs predictions on compounds in an input
    data frame.

    Args:
        model_path (str): File path of the model tarball file.

        input_df (DataFrame): Input data to run predictions on; must at minimum contain SMILES strings.

        id_col (str): Name of the column containing compound IDs. If none is provided, sequential IDs will be
        generated.

        smiles_col (str): Name of the column containing SMILES strings; required.

        response_col (str): Name of an optional column containing actual response values; if it is provided, 
        the actual values will be included in the returned data frame to make it easier for you to assess performance.

        dont_standardize (bool): By default, SMILES strings are salt-stripped and standardized using RDKit; 
        if you have already done this, or don't want them to be standardized, set dont_standardize to True.

    Return: 
        A data frame with compound IDs, SMILES strings and predicted response values. Actual response values
        will be included if response_col is provided. Standard prediction error estimates will be included
        if the model was trained with uncertainty=True. Note that the predicted and actual response
        columns will be labeled according to the response_col setting in the original training data,
        not the response_col passed to this function; e.g. if the original model response_col was 'pIC50',
        the returned data frame will contain columns 'pIC50_actual', 'pIC50_pred' and 'pIC50_std'.
    """

    input_df, pred_params = _prepare_input_data(input_df, id_col, smiles_col, response_col, dont_standardize)

    has_responses = ('response_cols' in pred_params)
    pred_params = parse.wrapper(pred_params)

    pipe = mp.create_prediction_pipeline_from_file(pred_params, reload_dir=None, model_path=model_path)
    pred_df = pipe.predict_full_dataset(input_df, contains_responses=has_responses, is_featurized=is_featurized,
                                        dset_params=pred_params)
    pred_df = pred_df.sort_values(by=id_col)
    return pred_df


# =====================================================================================================
def predict_cNr_from_model_file(model_path, input_df, id_col='compound_id', smiles_col='rdkit_smiles',
                     response_col=None, is_featurized=False, dont_standardize=False):
    """
    Loads two pretrained models, a classification and a regression from two
    model tarball files and runs predictions on compounds in an input data
    frame. Each compound will first be predicted in the classfication model,
    only active compounds will then be input into the regression model to
    predict activity values.

    Args:
        model_path (list of str): File path of the two model tarball files, 1st
        one is the classification model, the 2nd one is regression.

        input_df (DataFrame): Input data to run predictions on; must at minimum contain SMILES strings.

        id_col (str): Name of the column containing compound IDs. If none is provided, sequential IDs will be
        generated.

        smiles_col (str): Name of the column containing SMILES strings; required.

        response_col (list of str): Name of two optional columns
        (classification and regression) containing actual response values; if it is provided, 
        the actual values will be included in the returned data frame to make it easier for you to assess performance.

        dont_standardize (bool): By default, SMILES strings are salt-stripped and standardized using RDKit; 
        if you have already done this, or don't want them to be standardized, set dont_standardize to True.

    Return: 
        A data frame with compound IDs, SMILES strings and predicted response values. Actual response values
        will be included if response_col is provided. Standard prediction error estimates will be included
        if the model was trained with uncertainty=True. Note that the predicted and actual response
        columns will be labeled according to the response_col setting in the original training data,
        not the response_col passed to this function; e.g. if the original model response_col was 'pIC50',
        the returned data frame will contain columns 'pIC50_actual', 'pIC50_pred' and 'pIC50_std'.
    """

    if not response_col:
        response_col = [None, None]

    if not isinstance(model_path, list) or not isinstance(response_col, list):
        raise Exception("model_path and response_col should be a list with two elements.")

    class_model = model_path[0]
    reg_model = model_path[1]
    class_response = response_col[0]
    reg_response = response_col[1]

    class_pred = predict_from_model_file(class_model, input_df, id_col=id_col, smiles_col=smiles_col, response_col=class_response, is_featurized=is_featurized, dont_standardize=dont_standardize)
    class_pred_col = [e for e in class_pred.columns if e != smiles_col]
    pred_df = input_df.merge(class_pred[class_pred_col], on=id_col)

    reg_pred = predict_from_model_file(reg_model, input_df, id_col=id_col, smiles_col=smiles_col, response_col=reg_response, is_featurized=is_featurized, dont_standardize=dont_standardize)
    reg_pred_col = [e for e in reg_pred.columns if e != smiles_col]
    pred_df = pred_df.merge(reg_pred[reg_pred_col], on=id_col)

    pred_df = pred_df.sort_values(by=id_col)
    return pred_df

# =====================================================================================================
def _prepare_input_data(input_df, id_col, smiles_col, response_col, dont_standardize):
    """
    Prepare input data frame for running predictions
    """
    colnames = set(input_df.columns.values)
    if (id_col is None) or (id_col not in colnames):
        input_df['compound_id'] = ['compound_%.6d' % i for i in range(input_df.shape[0])]
        id_col = 'compound_id'
    if smiles_col not in colnames:
        raise ValueError('smiles_col parameter not specified or column not in input file.')
    if dont_standardize:
        std_smiles_col = smiles_col
    else:
        print("Standardizing SMILES strings for %d compounds." % input_df.shape[0])
        orig_ncmpds = input_df.shape[0]
        std_smiles = base_smiles_from_smiles(input_df[smiles_col].values.tolist(), workers=16)
        input_df['standardized_smiles'] = std_smiles
        input_df = input_df[input_df.standardized_smiles != '']
        if input_df.shape[0] == 0:
            raise ValueError("No valid SMILES strings to predict on.")
        nlost = orig_ncmpds - input_df.shape[0]
        input_df = input_df.sort_values(by=id_col)
        if nlost > 0:
            print("Could not parse %d SMILES strings; will predict on the remainder." % nlost)
        std_smiles_col = 'standardized_smiles'

    pred_params = {
        'featurizer': 'computed_descriptors',
        'result_dir': tempfile.mkdtemp(),
        'id_col': id_col,
        'smiles_col': std_smiles_col
    }
    if (response_col is not None) and (response_col in input_df.columns.values.tolist()):
        pred_params['response_cols'] = response_col

    return input_df, pred_params
