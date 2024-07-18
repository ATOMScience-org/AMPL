"""Functions to run predictions from a pre-trained model against user-provided data."""

import tempfile
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles

# =====================================================================================================
def predict_from_tracker_model(model_uuid, collection, input_df, id_col='compound_id', smiles_col='rdkit_smiles',
                     response_col=None, conc_col=None, is_featurized=False, dont_standardize=False, AD_method=None, k=5, 
                     dist_metric="euclidean", max_train_records_for_AD=1000):
    """Loads a pretrained model from the model tracker database and runs predictions on compounds in an input
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

        conc_col (str): Name of an optional column containing the concentration for single concentration activity (% binding)
        prediction in hybrid models.

        is_featurized (bool): True if input_df contains precomputed feature columns. If so, input_df must contain *all*
        of the feature columns defined by the featurizer that was used when the model was trained. Default is False which
        tells AMPL to compute the necessary descriptors.

        dont_standardize (bool): By default, SMILES strings are salt-stripped and standardized using RDKit;
        if you have already done this, or don't want them to be standardized, set dont_standardize to True.

        AD_method (str or None): Method to use to compute applicability domain (AD) index; may be
        'z_score', 'local_density' or None (the default). With the default value, AD indices
        will not be calculated.

        k (int): Number of nearest neighbors of each training data point used to evaluate the AD index.

        dist_metric (str): Metric used to compute distances between feature vectors for AD index calculation.
        Valid values are 'cityblock', 'cosine', 'euclidean', 'jaccard', and 'manhattan'. If binary
        features such as fingerprints are used in model, 'jaccard' (equivalent to Tanimoto distance) may
        be a better choice than the other metrics which operate on continuous features.

        max_train_records_for_AD (int): Maximum number of training data rows to use for AD calculation.
        Note that the AD calculation time scales as the square of the number of training records used.
        If the training dataset is larger than `max_train_records_for_AD`, a random sample of rows with
        this size is used instead for the AD calculations.

    Returns:
        A data frame with compound IDs, SMILES strings, predicted response values, and (optionally) uncertainties
        and/or AD indices. In addition, actual response values will be included if `response_col` is specified.
        Standard prediction error estimates will be included if the model was trained with uncertainty=True.
        Note that the predicted and actual response columns and standard errors will be labeled according to the
        `response_col` setting in the original training data, not the `response_col` passed to this function. For example,
        if the original model response_col was 'pIC50', the returned data frame will contain columns 'pIC50_actual',
        'pIC50_pred' and 'pIC50_std'.

        For proper AD index calculation, the original data column names must be the same for the new data.
    """
    input_df, pred_params = _prepare_input_data(input_df, id_col, smiles_col, response_col, conc_col, dont_standardize)
    has_responses = ('response_cols' in pred_params)
    pred_params = parse.wrapper(pred_params)
    pipe = mp.create_prediction_pipeline(pred_params, model_uuid, collection)
    pred_df = pipe.predict_full_dataset(input_df, contains_responses=has_responses, is_featurized=is_featurized,
                                        dset_params=pred_params, AD_method=AD_method, k=k, dist_metric=dist_metric,
                                        max_train_records_for_AD=max_train_records_for_AD)
    return pred_df

# =====================================================================================================
def predict_from_model_file(model_path, input_df, id_col='compound_id', smiles_col='rdkit_smiles',
                     response_col=None, conc_col=None, is_featurized=False, dont_standardize=False, AD_method=None, k=5, dist_metric="euclidean",
                     external_training_data=None, max_train_records_for_AD=1000):
    """Loads a pretrained model from a model tarball file and runs predictions on compounds in an input
    data frame.

    Args:
        model_path (str): File path of the model tarball file.

        input_df (DataFrame): Input data to run predictions on; must at minimum contain SMILES strings.

        id_col (str): Name of the column containing compound IDs. If none is provided, sequential IDs will be
        generated.

        smiles_col (str): Name of the column containing SMILES strings; required.

        response_col (str): Name of an optional column containing actual response values; if it is provided,
        the actual values will be included in the returned data frame to make it easier for you to assess performance.

        conc_col (str): Name of an optional column containing the concentration for single concentration activity (% binding)
        prediction in hybrid models.

        is_featurized (bool): True if input_df contains precomputed feature columns. If so, input_df must contain *all*
        of the feature columns defined by the featurizer that was used when the model was trained. Default is False which
        tells AMPL to compute the necessary descriptors.

        dont_standardize (bool): By default, SMILES strings are salt-stripped and standardized using RDKit;
        if you have already done this, or don't want them to be standardized, set dont_standardize to True.

        AD_method (str or None): Method to use to compute applicability domain (AD) index; may be
        'z_score', 'local_density' or None (the default). With the default value, AD indices
        will not be calculated.

        k (int): Number of nearest neighbors of each training data point used to evaluate the AD index.

        dist_metric (str): Metric used to compute distances between feature vectors for AD index calculation.
        Valid values are 'cityblock', 'cosine', 'euclidean', 'jaccard', and 'manhattan'. If binary
        features such as fingerprints are used in model, 'jaccard' (equivalent to Tanimoto distance) may
        be a better choice than the other metrics which operate on continuous features.

        external_training_data (str): Path to a copy of the model training dataset. Used for AD index computation in
        the case where the model was trained on a different computing system, or more generally when the training
        data is not accessible at the path saved in the model metadata.

        max_train_records_for_AD (int): Maximum number of training data rows to use for AD calculation.
        Note that the AD calculation time scales as the square of the number of training records used.
        If the training dataset is larger than `max_train_records_for_AD`, a random sample of rows with
        this size is used instead for the AD calculations.

    Returns:
        A data frame with compound IDs, SMILES strings, predicted response values, and (optionally) uncertainties
        and/or AD indices. In addition, actual response values will be included if `response_col` is specified.
        Standard prediction error estimates will be included if the model was trained with uncertainty=True.
        Note that the predicted and actual response columns and standard errors will be labeled according to the
        `response_col` setting in the original training data, not the `response_col` passed to this function. For example,
        if the original model response_col was 'pIC50', the returned data frame will contain columns 'pIC50_actual',
        'pIC50_pred' and 'pIC50_std'.

        For proper AD index calculation, the original data column names must be the same for the new data.
    """

    # TODO (ksm): How to deal with response_col in the case of multitask models? User would have to provide a map
    # from the original response column names to the column names in the provided data frame.

    input_df, pred_params = _prepare_input_data(input_df, id_col, smiles_col, response_col, conc_col, dont_standardize)

    has_responses = ('response_cols' in pred_params)
    pred_params = parse.wrapper(pred_params)

    pipe = mp.create_prediction_pipeline_from_file(pred_params, reload_dir=None, model_path=model_path)
    if external_training_data is not None:
        pipe.params.dataset_key=external_training_data
    pred_df = pipe.predict_full_dataset(input_df, contains_responses=has_responses, is_featurized=is_featurized,
                                        dset_params=pred_params, AD_method=AD_method, k=k, dist_metric=dist_metric,
                                        max_train_records_for_AD=max_train_records_for_AD)
    pred_df=input_df.merge(pred_df)
    return pred_df

# =====================================================================================================
def _prepare_input_data(input_df, id_col, smiles_col, response_col, conc_col, dont_standardize):
    """Prepare input data frame for running predictions"""
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
        input_df['orig_smiles']=input_df[smiles_col]
        input_df[smiles_col] = std_smiles
        input_df = input_df[input_df[smiles_col] != '']
        if input_df.shape[0] == 0:
            raise ValueError("No valid SMILES strings to predict on.")
        nlost = orig_ncmpds - input_df.shape[0]
        if nlost > 0:
            print("Could not parse %d SMILES strings; will predict on the remainder." % nlost)

    pred_params = {
        'featurizer': 'computed_descriptors',
        'result_dir': tempfile.mkdtemp(),
        'id_col': id_col,
        'smiles_col': smiles_col
    }
    if (response_col is not None) and (response_col in input_df.columns.values):
        pred_params['response_cols'] = response_col
        if conc_col is not None and conc_col in input_df.columns.values:
            pred_params['response_cols'] += "," + conc_col
    elif conc_col is not None and conc_col in input_df.columns.values:
        pred_params['response_cols'] = "ACTIVITY," + conc_col

    return input_df, pred_params
