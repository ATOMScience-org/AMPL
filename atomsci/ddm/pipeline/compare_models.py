"""Functions for comparing and visualizing model performance. Most of these functions rely on ATOM's model tracker and
datastore services, which are not part of the standard AMPL installation, but a few functions will work on collections of
models saved as local files.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import logging
import json
import shutil
import tarfile
import tempfile
from glob import glob

from collections import OrderedDict
from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.pipeline import model_tracker as trkr
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
from atomsci.ddm.utils import file_utils as futils

logger = logging.getLogger('ATOM')
mlmt_supported = True
try:
    from atomsci.clients import MLMTClient
except (ModuleNotFoundError, ImportError):
    logger.debug("Model tracker client not supported in your environment; can look at models in filesystem only.")
    mlmt_supported = False

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('axes', labelsize=12)

logging.basicConfig(format='%(asctime)-15s %(message)s')

nan = np.float32('nan')

#------------------------------------------------------------------------------------------------------------------
def del_ignored_params(dictionary, ignored_params):
    """
    Deletes ignored parameters from the dictionary if they exist

    Args:
        dictionary (dict): A dictionary with parameters

        ignored_parameters (list(str)): A list of keys potentially in the dictionary

    Returns:
        None
    """
    for ip in ignored_params:
        if ip in dictionary:
            del dictionary[ip]

#------------------------------------------------------------------------------------------------------------------
def get_collection_datasets(collection_name):
    """Returns a list of unique training datasets used for all models in a given collection.

    Args:
        collection_name (str): Name of model tracker collection to search for models.

    Returns:
        list: List of model training (dataset_key, bucket) tuples.
    """
    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return None

    dataset_set = set()
    mlmt_client = dsf.initialize_model_tracker()
    dset_dicts = mlmt_client.model.query_datasets(collection_name=collection_name, metrics_type='training').result()
    # Convert to a list of (dataset_key, bucket) tuples
    for dset_dict in dset_dicts:
        dataset_set.add((dset_dict['dataset_key'], dset_dict['bucket']))
    return sorted(dataset_set)

#------------------------------------------------------------------------------------------------------------------
def extract_collection_perf_metrics(collection_name, output_dir, pred_type='regression'):
    """Obtain list of training datasets with models in the given collection. Get performance metrics for
    models on each dataset and save them as CSV files in the given output directory.

    Args:
        collection_name (str): Name of model tracker collection to search for models.

        output_dir (str): Directory where tables of performance metrics will be written.

        pred_type (str): Prediction type ('classification' or 'regression') of models to query.

    Returns:
        None

    """
    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return

    datasets = get_collection_datasets(collection_name)
    os.makedirs(output_dir, exist_ok=True)
    for dset_key, bucket in datasets:
        dset_perf_df = get_training_perf_table(dset_key, bucket, collection_name, pred_type=pred_type)
        dset_perf_file = '%s/%s_%s_model_perf_metrics.csv' % (output_dir, os.path.basename(dset_key).replace('.csv', ''), collection_name)
        dset_perf_df.to_csv(dset_perf_file, index=False)
        print('Wrote file %s' % dset_perf_file)

#------------------------------------------------------------------------------------------------------------------
def get_training_perf_table(dataset_key, bucket, collection_name, pred_type='regression', other_filters = {}):
    """Load performance metrics from model tracker for all models saved in the model tracker DB under
    a given collection that were trained against a particular dataset. Identify training parameters
    that vary between models, and generate plots of performance vs particular combinations of
    parameters.

    Args:
        dataset_key (str): Training dataset key.

        bucket (str): Training dataset bucket.

        collection_name (str): Name of model tracker collection to search for models.

        pred_type (str): Prediction type ('classification' or 'regression') of models to query.

        other_filters (dict): Other filter criteria to use in querying models.

    Returns:
        pd.DataFrame: Table of models and performance metrics.

    """
    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return None

    print("Finding models trained on %s dataset %s" % (bucket, dataset_key))
    mlmt_client = dsf.initialize_model_tracker()
    query_params = {
        "match_metadata": {
            "training_dataset.bucket": bucket,
            "training_dataset.dataset_key": dataset_key,
        },

        "match_metrics": {
            "metrics_type": "training",  # match only training metrics
            "label": "best",
        },
    }
    query_params['match_metadata'].update(other_filters)

    metadata_list = mlmt_client.model.query_model_metadata(
        collection_name=collection_name,
        query_params=query_params,
    ).result()
    if metadata_list == []:
        print("No matching models returned")
        return
    else:
        print("Found %d matching models" % len(metadata_list))

    model_uuid_list = []
    model_type_list = []
    max_epochs_list = []
    learning_rate_list = []
    dropouts_list = []
    layer_sizes_list = []
    featurizer_list = []
    splitter_list = []
    rf_estimators_list = []
    rf_max_features_list = []
    rf_max_depth_list = []
    xgb_learning_rate_list = []
    xgb_gamma_list = []
    xgb_max_depth_list = []
    xgb_colsample_bytree_list = []
    xgb_subsample_list = []
    xgb_n_estimators_list = []
    xgb_min_child_weight_list = []
    best_epoch_list = []
    max_epochs_list = []
    subsets = ['train', 'valid', 'test']
    score_dict = {}
    for subset in subsets:
        score_dict[subset] = []

    if pred_type == 'regression':
        metric_type = 'r2_score'
    else:
        metric_type = 'roc_auc_score'

    for metadata_dict in metadata_list:
        model_uuid = metadata_dict['model_uuid']
        #print("Got metadata for model UUID %s" % model_uuid)

        # Get model metrics for this model
        metrics_dicts = metadata_dict['training_metrics']
        #print("Got %d metrics dicts for model %s" % (len(metrics_dicts), model_uuid))
        if len(metrics_dicts) < 3:
            print("Got no or incomplete metrics for model %s, skipping..." % model_uuid)
            continue
        subset_metrics = {}
        for metrics_dict in metrics_dicts:
            subset = metrics_dict['subset']
            subset_metrics[subset] = metrics_dict['prediction_results']

        model_uuid_list.append(model_uuid)
        model_params = metadata_dict['model_parameters']
        model_type = model_params['model_type']
        model_type_list.append(model_type)
        featurizer = model_params['featurizer']
        featurizer_list.append(featurizer)
        split_params = metadata_dict['splitting_parameters']
        splitter_list.append(split_params['splitter'])
        dataset_key = metadata_dict['training_dataset']['dataset_key']
        if model_type == 'NN':
            nn_params = metadata_dict['nn_specific']
            max_epochs_list.append(nn_params['max_epochs'])
            best_epoch_list.append(nn_params['best_epoch'])
            learning_rate_list.append(nn_params['learning_rate'])
            layer_sizes_list.append(','.join(['%d' % s for s in nn_params['layer_sizes']]))
            dropouts_list.append(','.join(['%.2f' % d for d in nn_params['dropouts']]))
            rf_estimators_list.append(nan)
            rf_max_features_list.append(nan)
            rf_max_depth_list.append(nan)
            xgb_learning_rate_list.append(nan)
            xgb_gamma_list.append(nan)
            xgb_max_depth_list.append(nan)
            xgb_colsample_bytree_list.append(nan)
            xgb_subsample_list.append(nan)
            xgb_n_estimators_list.append(nan)
            xgb_min_child_weight_list.append(nan)

        if model_type == 'RF':
            rf_params = metadata_dict['rf_specific']
            rf_estimators_list.append(rf_params['rf_estimators'])
            rf_max_features_list.append(rf_params['rf_max_features'])
            rf_max_depth_list.append(rf_params['rf_max_depth'])
            max_epochs_list.append(nan)
            best_epoch_list.append(nan)
            learning_rate_list.append(nan)
            layer_sizes_list.append(nan)
            dropouts_list.append(nan)
            xgb_learning_rate_list.append(nan)
            xgb_gamma_list.append(nan)
            xgb_max_depth_list.append(nan)
            xgb_colsample_bytree_list.append(nan)
            xgb_subsample_list.append(nan)
            xgb_n_estimators_list.append(nan)
            xgb_min_child_weight_list.append(nan)
        if model_type == 'xgboost':
            xgb_params = metadata_dict['xgb_specific']
            rf_estimators_list.append(nan)
            rf_max_features_list.append(nan)
            rf_max_depth_list.append(nan)
            max_epochs_list.append(nan)
            best_epoch_list.append(nan)
            learning_rate_list.append(nan)
            layer_sizes_list.append(nan)
            dropouts_list.append(nan)
            xgb_learning_rate_list.append(xgb_params["xgb_learning_rate"])
            xgb_gamma_list.append(xgb_params["xgb_gamma"])
            xgb_max_depth_list.append(xgb_params["xgb_max_depth"])
            xgb_colsample_bytree_list.append(xgb_params["xgb_colsample_bytree"])
            xgb_subsample_list.append(xgb_params["xgb_subsample"])
            xgb_n_estimators_list.append(xgb_params["xgb_n_estimators"])
            xgb_min_child_weight_list.append(xgb_params["xgb_min_child_weight"])
        for subset in subsets:
            score_dict[subset].append(subset_metrics[subset][metric_type])

    perf_df = pd.DataFrame(dict(
                    model_uuid=model_uuid_list,
                    model_type=model_type_list,
                    dataset_key=dataset_key,
                    featurizer=featurizer_list,
                    splitter=splitter_list,
                    max_epochs=max_epochs_list,
                    best_epoch=best_epoch_list,
                    learning_rate=learning_rate_list,
                    layer_sizes=layer_sizes_list,
                    dropouts=dropouts_list,
                    rf_estimators=rf_estimators_list,
                    rf_max_features=rf_max_features_list,
                    rf_max_depth=rf_max_depth_list,
                    xgb_learning_rate = xgb_learning_rate_list,
                    xgb_gamma = xgb_gamma_list,
                    xgb_max_depth = xgb_max_depth_list,
                    xgb_colsample_bytree = xgb_colsample_bytree_list,
                    xgb_subsample = xgb_subsample_list,
                    xgb_n_estimators = xgb_n_estimators_list,
                    xgb_min_child_weight = xgb_min_child_weight_list))
    for subset in subsets:
        metric_col = '%s_%s' % (metric_type, subset)
        perf_df[metric_col] = score_dict[subset]
    sort_metric = '%s_valid' % metric_type

    perf_df = perf_df.sort_values(sort_metric, ascending=False)
    return perf_df

# -----------------------------------------------------------------------------------------------------------------
def extract_model_and_feature_parameters(metadata_dict, keep_required=True):
    """Given a config file, extract model and featurizer parameters. Looks for parameter names
    that end in *_specific. e.g. nn_specific, auto_featurizer_specific

    Args:
        model_metadict (dict): Dictionary containing NON-FLATTENED metadata for an AMPL model

    Returns:
        dictionary containing featurizer and model parameters. Most contain the following
        keys. ['max_epochs', 'best_epoch', 'learning_rate', 'layer_sizes', 'dropouts',
        'rf_estimators', 'rf_max_features', 'rf_max_depth', 'xgb_gamma', 'xgb_learning_rate',
        'xgb_max_depth', 'xgb_colsample_bytree', 'xgb_subsample', 'xgb_n_estimators', 'xgb_min_child_weight',
        'featurizer_parameters_dict', 'model_parameters_dict']
    """
    model_params = metadata_dict['model_parameters']
    model_type = model_params['model_type']
    required = ['max_epochs', 'best_epoch', 'learning_rate', 'layer_sizes', 'dropouts', 
        'rf_estimators', 'rf_max_features', 'rf_max_depth', 'xgb_gamma', 'xgb_learning_rate',
        'xgb_max_depth', 'xgb_colsample_bytree', 'xgb_subsample', 'xgb_n_estimators', 'xgb_min_child_weight'
        ]

    model_info = {}
    model_info['model_uuid'] = metadata_dict['model_uuid']

    if keep_required:
        if model_type == 'NN':
            nn_params = metadata_dict['nn_specific']
            model_info['max_epochs'] = nn_params['max_epochs']
            model_info['best_epoch'] = nn_params['best_epoch']
            model_info['learning_rate'] = nn_params['learning_rate']
            model_info['layer_sizes'] = ','.join(['%d' % s for s in nn_params['layer_sizes']])
            model_info['dropouts'] = ','.join(['%.2f' % d for d in nn_params['dropouts']])
        elif model_type == 'RF':
            rf_params = metadata_dict['rf_specific']
            model_info['rf_estimators'] = rf_params['rf_estimators']
            model_info['rf_max_features'] = rf_params['rf_max_features']
            model_info['rf_max_depth'] = rf_params['rf_max_depth']
        elif model_type == 'xgboost':
            xgb_params = metadata_dict['xgb_specific']
            model_info['xgb_gamma'] = xgb_params['xgb_gamma']
            model_info['xgb_learning_rate'] = xgb_params['xgb_learning_rate']
            model_info['xgb_max_depth'] = xgb_params['xgb_max_depth']
            model_info['xgb_colsample_bytree'] = xgb_params['xgb_colsample_bytree']
            model_info['xgb_subsample'] = xgb_params['xgb_subsample']
            model_info['xgb_n_estimators'] = xgb_params['xgb_n_estimators']
            model_info['xgb_min_child_weight'] = xgb_params['xgb_min_child_weight']
    
        for r in required:
            if r not in model_info:
                # all fields must be filled in
                model_info[r] = nan

    # the new way of extracting model parameters is to simply save them in json
    if 'nn_specific' in metadata_dict:
        model_metadata = metadata_dict['nn_specific']
        if keep_required:
            # include learning rate, max_epochs, and best_epoch for convenience 
            model_info['max_epochs'] = model_metadata.get('max_epochs', np.nan)
            model_info['best_epoch'] = model_metadata.get('best_epoch', np.nan)
            learning_rate_col = [c for c in model_metadata.keys() if c.endswith('learning_rate')]
            if len(learning_rate_col) == 1:
                model_info['learning_rate'] = model_metadata[learning_rate_col[0]]
        # delete several parameters that aren't normally saved
        ignored_params = ['batch_size','bias_init_consts','optimizer_type',
            'weight_decay_penalty','weight_decay_penalty_type','weight_init_stddevs']
        del_ignored_params(model_metadata, ignored_params)
    elif 'rf_specific' in metadata_dict:
        model_metadata = metadata_dict['rf_specific']
    elif 'xgb_specific' in metadata_dict:
        model_metadata = metadata_dict['xgb_specific']
    else:
        # no model parameters found
        model_metadata = {}
    model_info['model_parameters_dict'] = json.dumps(model_metadata)

    if 'ecfp_specific' in metadata_dict:
        feat_metadata = metadata_dict['ecfp_specific']
    elif 'auto_featurizer_specific' in metadata_dict:
        feat_metadata = metadata_dict['auto_featurizer_specific']
    elif 'autoencoder_specific' in metadata_dict:
        feat_metadata = metadata_dict['autoencoder_specific']
    else:
        # no model parameters found
        feat_metadata = {}
    model_info['feat_parameters_dict'] = json.dumps(feat_metadata)

    return model_info

# ------------------------------------------------------------------------------------------------------------------
def get_best_perf_table(metric_type, col_name=None, result_dir=None, model_uuid=None, metadata_dict=None, PK_pipe=False):
    """Extract parameters and training run performance metrics for a single model. The model may be
    specified either by a metadata dictionary, a model_uuid or a result directory; in the model_uuid case, the function
    queries the model tracker DB for the model metadata. For models saved in the filesystem, can query the performance
    data from the original result directory, but not from a saved tarball.

    Args:
        metric_type (str): Performance metric to include in result dictionary.

        col_name (str): Collection name containing model, if model is specified by model_uuid.

        result_dir (str): result directory of the model, if Model tracker is not supported and metadata_dict not provided.

        model_uuid (str): UUID of model to query, if metadata_dict is not provided.

        metadata_dict (dict): Full metadata dictionary for a model, including training metrics and
        dataset metadata.

        PK_pipe (bool): If True, include some additional parameters in the result dictionary specific to PK models.

    Returns:
        model_info (dict): Dictionary of parameter or metric name - value pairs.

    Todo:
        Add support for models saved as local tarball files.

    """
    if not mlmt_supported and not result_dir:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only, 'result_dir' needs to be provided.")
        return None
    elif mlmt_supported and col_name:
        mlmt_client = dsf.initialize_model_tracker()
        if metadata_dict is None:
            if model_uuid is None:
                print("Have to specify either metadata_dict or model_uuid")
                return
            query_params = {
                "match_metadata": {
                    "model_uuid": model_uuid,
                },

                "match_metrics": {
                    "metrics_type": "training",  # match only training metrics
                    "label": "best",
                },
            }

            metadata_list = list(mlmt_client.model.query_model_metadata(
                collection_name=col_name,
                query_params=query_params
            ).result())
            if len(metadata_list) == 0:
                print("No matching models returned")
                return None
            metadata_dict = metadata_list[0]
    elif result_dir:
        model_dir = ""
        for dirpath, dirnames, filenames in os.walk(result_dir):
            if model_uuid in dirnames:
                model_dir = os.path.join(dirpath, model_uuid)
                break
        if model_dir:
            with open(os.path.join(model_dir, 'model_metadata.json')) as f:
                metadata_dict = json.load(f)
        else:
            print(f"model_uuid ({model_uuid}) not exist in {result_dir}.")
            return None

    model_info = {}

    model_info['model_uuid'] = metadata_dict['model_uuid']
    model_info['collection_name'] = col_name

    # Get model metrics for this model
    metrics_dicts = [d for d in metadata_dict['training_metrics'] if d['label'] == 'best']
    if len(metrics_dicts) != 3:
        print("Got no or incomplete metrics for model %s, skipping..." % model_uuid)
        return None

    model_params = metadata_dict['model_parameters']
    model_info['model_type'] = model_params['model_type']
    model_info['featurizer'] = model_params['featurizer']
    split_params = metadata_dict['splitting_parameters']
    model_info['splitter'] = split_params['splitter']
    if 'split_uuid' in split_params:
        model_info['split_uuid'] = split_params['split_uuid']
    model_info['dataset_key'] = metadata_dict['training_dataset']['dataset_key']
    model_info['bucket'] = metadata_dict['training_dataset']['bucket']
    dset_meta = metadata_dict['training_dataset']['dataset_metadata']
    if PK_pipe:
        model_info['assay_name'] = dset_meta.get('assay_category', 'NA')
        model_info['response_col'] = dset_meta.get('response_cols', dset_meta.get('response_col', 'NA'))
    try:
        model_info['descriptor_type'] = metadata_dict['descriptor_specific']['descriptor_type']
    except KeyError:
        model_info['descriptor_type'] = 'NA'
    try:
        model_info['num_samples'] = dset_meta['num_row']
    except:
        # KSM: Commented out because original dataset may no longer be accessible.
        #tmp_df = dsf.retrieve_dataset_by_datasetkey(model_info['dataset_key'], model_info['bucket'])
        #model_info['num_samples'] = tmp_df.shape[0]
        model_info['num_samples'] = nan

    # add model and feature params
    # model_uuid appears in model_feature_params and will overwrite the one in model_info
    # it's the same uuid, so it should be ok
    model_feature_params = extract_model_and_feature_parameters(metadata_dict)
    model_info.update(model_feature_params)

    for metrics_dict in metrics_dicts:
        subset = metrics_dict['subset']
        metric_col = '%s_%s' % (metric_type, subset)
        model_info[metric_col] = metrics_dict['prediction_results'][metric_type]
        if (model_params['prediction_type'] == 'regression') and (metric_type != 'rms_score'):
            metric_col = 'rms_score_%s' % subset
            model_info[metric_col] = metrics_dict['prediction_results']['rms_score']

    return model_info


# ---------------------------------------------------------------------------------------------------------
def get_best_models_info(col_names=None, bucket='public', pred_type="regression", result_dir=None, PK_pipeline=False,
                         output_dir='/usr/local/data',
                         shortlist_key=None, input_dset_keys=None, save_results=False, subset='valid',
                         metric_type=None, selection_type='max', other_filters={}):
    """Tabulate parameters and performance metrics for the best models, according to a given metric, trained against
    each specified dataset.

    Args:
        col_names (list of str): List of model tracker collections to search.

        bucket (str): Datastore bucket for training datasets.

        pred_type (str): Type of models (regression or classification).

        result_dir (list of str): Result directories of the models, if model tracker is not supported.

        PK_pipeline (bool): Are we being called from PK pipeline?

        output_dir (str): Directory to write output table to.

        shortlist_key (str): Datastore key for table of datasets to query models for.

        input_dset_keys (str or list of str): List of datastore keys for datasets to query models for. Either shortlist_key
        or input_dset_keys must be specified, but not both.

        save_results (bool): If True, write the table of results to a CSV file.

        subset (str): Input dataset subset ('train', 'valid', or 'test') for which metrics are used to select best models.

        metric_type (str): Type of performance metric (r2_score, roc_auc_score, etc.) to use to select best models.

        selection_type (str): Score criterion ('max' or 'min') to use to select best models.

        other_filters (dict): Additional selection criteria to include in model query.

    Returns:
        top_models_df (DataFrame): Table of parameters and metrics for best models for each dataset.
    """

    if not mlmt_supported and not result_dir:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only, 'result_dir' needs to be provided.")
        return None

    top_models_info = []
    sort_order = {'max': -1, 'min': 1}
    sort_ascending = {'max': False, 'min': True}
    if metric_type is None:
        if pred_type == 'regression':
            metric_type = 'r2_score'
        else:
            metric_type = 'roc_auc_score'
    if other_filters is None:
        other_filters = {}
    # define dset_keys
    if input_dset_keys is not None and shortlist_key is not None:
        raise ValueError("You can specify either shortlist_key or input_dset_keys but not both.")
    elif input_dset_keys is not None and shortlist_key is None:
        if type(input_dset_keys) == str:
            dset_keys = [input_dset_keys]
        else:
            dset_keys = input_dset_keys
    elif input_dset_keys is None and shortlist_key is None:
        raise ValueError('Must specify either input_dset_keys or shortlist_key')
    else:
        dset_keys = dsf.retrieve_dataset_by_datasetkey(shortlist_key, bucket)
        if dset_keys is None:
            # define dset_keys, col_names and buckets from shortlist file
            shortlist = pd.read_csv(shortlist_key)
            if 'dataset_key' in shortlist.columns:
                dset_keys = shortlist['dataset_key'].unique()
            elif 'task_name' in shortlist.columns:
                dset_keys = shortlist['task_name'].unique()
            else:
                dset_keys = shortlist.values
            if 'collection' in shortlist.columns:
                col_names = shortlist['collection'].unique()
            if 'bucket' in shortlist.columns:
                bucket = shortlist['bucket'].unique()
    
    if mlmt_supported and col_names is not None:
        mlmt_client = dsf.initialize_model_tracker()
        if type(col_names) == str:
            col_names = [col_names]
        if type(bucket) == str:
            bucket=[bucket]
        # Get the best model over all collections for each dataset
        for dset_key in dset_keys:
            dset_key = dset_key.strip()
            dset_model_info = []
            for col_name in col_names:
                for buck in bucket:
                    try:
                        query_params = {
                            "match_metadata": {
                                "training_dataset.dataset_key": dset_key,
                                "training_dataset.bucket": buck,
                            },

                            "match_metrics": {
                                "metrics_type": "training",  # match only training metrics
                                "label": "best",
                                "subset": subset,
                                "$sort": [{"prediction_results.%s" % metric_type : sort_order[selection_type]}]
                            },
                        }
                        query_params['match_metadata'].update(other_filters)

                        try:
                            print('Querying collection %s for models trained on dataset %s, %s' % (col_name, buck, dset_key))
                            metadata_list = list(mlmt_client.model.query_model_metadata(
                                collection_name=col_name,
                                query_params=query_params,
                                limit=1
                            ).result())
                        except Exception as e:
                            print("Error returned when querying the best model for dataset %s in collection %s" % (dset_key, col_name))
                            print(e)
                            continue
                        if len(metadata_list) == 0:
                            print("No models returned for dataset %s in collection %s" % (dset_key, col_name))
                            continue
                        print('Query returned %d models' % len(metadata_list))
                        model = metadata_list[0]
                        model_info = get_best_perf_table(metric_type, col_name, metadata_dict=model, PK_pipe=PK_pipeline)
                        if model_info is not None:
                            res_df = pd.DataFrame.from_records([model_info])
                            dset_model_info.append(res_df)
                    except Exception as e:
                        print(e)
                        continue
            metric_col = '%s_%s' % (metric_type, subset)
            if len(dset_model_info) > 0:
                dset_model_df = pd.concat(dset_model_info, ignore_index=True).sort_values(
                                by=metric_col, ascending=sort_ascending[selection_type])
                top_models_info.append(dset_model_df.head(1))
                print('Adding data for bucket %s, dset_key %s' % (dset_model_df.bucket.values[0], dset_model_df.dataset_key.values[0]))
    elif result_dir:
        metric_col = '%s_%s' % (subset, metric_type)
        for rd in result_dir:
            temp_perf_df = get_filesystem_perf_results(result_dir = rd, pred_type = pred_type).sort_values(
                                by=metric_col, ascending=sort_ascending[selection_type])
            top_models_info.append(temp_perf_df.head(1))
            print(f"Adding data from '{rd}' ")

    if len(top_models_info) == 0:
        print("No metadata found")
        return None
    top_models_df = pd.concat(top_models_info, ignore_index=True)
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        if shortlist_key is not None:
            # Not including shortlist key right now because some are weirdly formed and have .csv in the middle
            top_models_df.to_csv(os.path.join(output_dir, 'best_models_metadata.csv'), index=False)
        else:
            for dset_key in input_dset_keys:
                # TODO: This doesn't make sense; why output multiple copies of the same table?
                shortened_key = dset_key.rstrip('.csv')
                top_models_df.to_csv(os.path.join(output_dir, 'best_models_metadata_%s.csv' % shortened_key), index=False)
    return top_models_df


# TODO: This function looks like work in progress, should we delete it?
'''
#---------------------------------------------------------------------------------------------------------
def _get_best_grouped_models_info(collection='pilot_fixed', pred_type='regression', top_n=1, subset='test'):
    """Get results for models in the given collection."""

    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return

    res_dir = '/usr/local/data/%s_perf' % collection
    plt_dir = '%s/Plots' % res_dir
    os.makedirs(plt_dir, exist_ok=True)
    res_files = os.listdir(res_dir)
    suffix = '_%s_model_perf_metrics.csv' % collection

    if pred_type == 'regression':
        metric_type = 'r2_score'
    else:
        metric_type = 'roc_auc_score'
    for res_file in res_files:
        try:
            if not res_file.endswith(suffix):
                continue
            res_path = os.path.join(res_dir, res_file)

            res_df = pd.read_csv(res_path, index_col=False)
            res_df['combo'] = ['%s/%s' % (m,f) for m, f in zip(res_df.model_type.values, res_df.featurizer.values)]
            dset_name = res_file.replace(suffix, '')
            datasets.append(dset_name)
            res_df['dataset'] = dset_name
            print(dset_name)
            res_df = res_df.sort_values('{0}_{1}'.format(metric_type, subset), ascending=False)
            res_df['model_type/feat'] = ['%s/%s' % (m,f) for m, f in zip(res_df.model_type.values, res_df.featurizer.values)]
            res_df = res_df.sort_values('{0}_{1}'.format(metric_type, subset), ascending=False)
            grouped_df = res_df.groupby('model_type/feat').apply(
                lambda t: t.head(top_n)
            ).reset_index(drop=True)
            top_grouped_models.append(grouped_df)
            top_combo = res_df['model_type/feat'].values[0]
            top_combo_dsets.append(top_combo + dset_name.lstrip('ATOM_GSK_dskey'))
            top_score = res_df['{0}_{1}'.format(metric_type, subset)].values[0]
            top_model_feat.append(top_combo)
            top_scores.append(top_score)
            num_samples.append(res_df['Dataset Size'][0])

#------------------------------------------------------------------------------------------------------------------
def get_umap_nn_model_perf_table(dataset_key, bucket, collection_name, pred_type='regression'):
    """Load performance metrics from model tracker for all NN models with the given prediction_type saved in
    the model tracker DB under a given collection that were trained against a particular dataset. Show
    parameter settings for UMAP transformer for models where they are available.

    Args:
        dataset_key (str): Dataset key for training dataset.

        bucket (str): Dataset bucket for training dataset.

        collection_name (str): Name of model tracker collection to search for models.

        pred_type (str): Prediction type ('classification' or 'regression') of models to query.

    Returns:
        pd.DataFrame: Table of model performance metrics.

    """
    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return None

    query_params = {
        "match_metadata": {
            "training_dataset.bucket": bucket,
            "training_dataset.dataset_key": dataset_key,
            "model_parameters.model_type" : "NN",
            "model_parameters.prediction_type" : pred_type
        },

        "match_metrics": {
            "metrics_type": "training",  # match only training metrics
            "label": "best",
        },
    }
    query_params['match_metadata'].update(other_filters)

    print("Finding models trained on %s dataset %s" % (bucket, dataset_key))
    mlmt_client = dsf.initialize_model_tracker()
    metadata_list = mlmt_client.model.query_model_metadata(
        collection_name=collection_name,
        query_params=query_params,
    ).result()
    if metadata_list == []:
        print("No matching models returned")
        return
    else:
        print("Found %d matching models" % len(metadata_list))

    model_uuid_list = []
    learning_rate_list = []
    dropouts_list = []
    layer_sizes_list = []
    featurizer_list = []
    best_epoch_list = []
    max_epochs_list = []

    feature_transform_type_list = []
    umap_dim_list = []
    umap_targ_wt_list = []
    umap_neighbors_list = []
    umap_min_dist_list = []

    subsets = ['train', 'valid', 'test']

    if pred_type == 'regression':
        sort_metric = 'r2_score'
        metrics = ['r2_score', 'rms_score', 'mae_score']
    else:
        sort_metric = 'roc_auc_score'
        metrics = ['roc_auc_score', 'prc_auc_score', 'matthews_cc', 'kappa', 'confusion_matrix']
    score_dict = {}
    for subset in subsets:
        score_dict[subset] = {}
        for metric in metrics:
            score_dict[subset][metric] = []

    for metadata_dict in metadata_list:
        model_uuid = metadata_dict['model_uuid']
        #print("Got metadata for model UUID %s" % model_uuid)

        # Get model metrics for this model
        metrics_dicts = metadata_dict['training_metrics']
        #print("Got %d metrics dicts for model %s" % (len(metrics_dicts), model_uuid))
        if len(metrics_dicts) < 3:
            print("Got no or incomplete metrics for model %s, skipping..." % model_uuid)
            continue
        if len(metrics_dicts) > 3:
            raise Exception('Got more than one set of best epoch metrics for model %s' % model_uuid)
        subset_metrics = {}
        for metrics_dict in metrics_dicts:
            subset = metrics_dict['subset']
            subset_metrics[subset] = metrics_dict['prediction_results']

        model_uuid_list.append(model_uuid)
        model_params = metadata_dict['model_parameters']
        model_type = model_params['model_type']
        if model_type != 'NN':
            continue
        featurizer = model_params['featurizer']
        featurizer_list.append(featurizer)
        feature_transform_type = metadata_dict['training_dataset']['feature_transform_type']
        feature_transform_type_list.append(feature_transform_type)
        nn_params = metadata_dict['nn_specific']
        max_epochs_list.append(nn_params['max_epochs'])
        best_epoch_list.append(nn_params['best_epoch'])
        learning_rate_list.append(nn_params['learning_rate'])
        layer_sizes_list.append(','.join(['%d' % s for s in nn_params['layer_sizes']]))
        dropouts_list.append(','.join(['%.2f' % d for d in nn_params['dropouts']]))
        for subset in subsets:
            for metric in metrics:
                score_dict[subset][metric].append(subset_metrics[subset][metric])
        if 'umap_specific' in metadata_dict:
            umap_params = metadata_dict['umap_specific']
            umap_dim_list.append(umap_params['umap_dim'])
            umap_targ_wt_list.append(umap_params['umap_targ_wt'])
            umap_neighbors_list.append(umap_params['umap_neighbors'])
            umap_min_dist_list.append(umap_params['umap_min_dist'])
        else:
            umap_dim_list.append(nan)
            umap_targ_wt_list.append(nan)
            umap_neighbors_list.append(nan)
            umap_min_dist_list.append(nan)


    perf_df = pd.DataFrame(dict(
                    model_uuid=model_uuid_list,
                    learning_rate=learning_rate_list,
                    dropouts=dropouts_list,
                    layer_sizes=layer_sizes_list,
                    featurizer=featurizer_list,
                    best_epoch=best_epoch_list,
                    max_epochs=max_epochs_list,
                    feature_transform_type=feature_transform_type_list,
                    umap_dim=umap_dim_list,
                    umap_targ_wt=umap_targ_wt_list,
                    umap_neighbors=umap_neighbors_list,
                    umap_min_dist=umap_min_dist_list ))
    for subset in subsets:
        for metric in metrics:
            metric_col = '%s_%s' % (metric, subset)
            perf_df[metric_col] = score_dict[subset][metric]
    sort_by = '%s_valid' % sort_metric

    perf_df = perf_df.sort_values(sort_by, ascending=False)
    return perf_df
'''

#------------------------------------------------------------------------------------------------------------------
def get_tarball_perf_table(model_tarball, pred_type='classification'):
    """Retrieve model metadata and performance metrics for a model saved as a tarball (.tar.gz) file.

    Args:
        model_tarball (str): Path of model tarball file, named as model.tar.gz.

        pred_type (str): Prediction type ('classification' or 'regression') of model.

    Returns:
        tuple (pd.DataFrame, dict): Table of performance metrics and a dictionary of model metadata.

    """
    tarf_content = tarfile.open(model_tarball, "r")
    metadata_file = tarf_content.getmember("./model_metadata.json")
    ext_metadata = tarf_content.extractfile(metadata_file)

    meta_json = json.load(ext_metadata)
    ext_metadata.close()

    subsets = ['train', 'valid', 'test']

    if pred_type == 'regression':
        metrics = ['r2_score', 'rms_score', 'mae_score']
    else:
        metrics = ['roc_auc_score', 'prc_auc_score', 'precision', 'recall_score',
                   'accuracy_score', 'npv', 'matthews_cc', 'kappa', 'cross_entropy', 'confusion_matrix']
    score_dict = {}
    for subset in subsets:
        score_dict[subset] = {}
        for metric in metrics:
            score_dict[subset][metric] = [0,0]

    for emet in meta_json["training_metrics"]:
        label = emet["label"]
        score_ix = 0 if label == "best" else 1
        subset = emet["subset"]
        for metric in metrics:
            score_dict[subset][metric][score_ix] = emet["prediction_results"][metric]

    perf_df = pd.DataFrame()
    for subset in subsets:
        for metric in metrics:
            metric_col = '%s_%s' % (subset, metric)
            perf_df[metric_col] = score_dict[subset][metric]

    return perf_df, meta_json

#------------------------------------------------------------------------------------------------------------------
def get_filesystem_perf_results(result_dir, pred_type='classification', expand=True):
    """Retrieve metadata and performance metrics for models stored in the filesystem from a hyperparameter search run.

    Args:
        result_dir (str): Root directory for results from a hyperparameter search training run.

        pred_type (str): Prediction type ('classification' or 'regression') of models to query.

    Returns:
        pd.DataFrame: Table of metadata fields and performance metrics.

    """
    ampl_version_list = []
    model_uuid_list = []
    model_type_list = []
    featurizer_list = []
    dataset_key_list = []
    splitter_list = []
    split_strategy_list = []
    split_uuid_list = []
    model_score_type_list = []
    feature_transform_type_list = []
    weight_transform_type_list = []

    # model type specific lists
    param_list = []

    subsets = ['train', 'valid', 'test']

    if pred_type == 'regression':
        metrics = ['r2_score', 'rms_score', 'mae_score', 'num_compounds']
    else:
        metrics = ['roc_auc_score', 'prc_auc_score', 'precision', 'recall_score', 'num_compounds',
                   'accuracy_score', 'bal_accuracy', 'npv', 'matthews_cc', 'kappa', 'cross_entropy', 'confusion_matrix']
    score_dict = {}
    for subset in subsets:
        score_dict[subset] = {}
        for metric in metrics:
            score_dict[subset][metric] = []
    score_dict['valid']['model_choice_score'] = []
       
    # Navigate the results directory tree
    model_list = []
    metrics_list = []
    tar_list = []
    for dirpath, dirnames, filenames in os.walk(result_dir):
        # collect all tars for later
        tar_list = tar_list + [os.path.join(dirpath, f) for f in filenames if f.endswith('.tar.gz')]
        
        if ('model_metadata.json' in filenames) and ('model_metrics.json' in filenames):
#             print(dirpath)
            meta_path = os.path.join(dirpath, 'model_metadata.json')
            try:
                with open(meta_path, 'r') as meta_fp:
                    meta_dict = json.load(meta_fp)
                if meta_dict['model_parameters']['prediction_type']==pred_type:
                    model_list.append(meta_dict)
                    metrics_path = os.path.join(dirpath, 'model_metrics.json')
                    with open(metrics_path, 'r') as metrics_fp:
                        metrics_dicts = json.load(metrics_fp)
                    metrics_list.append(metrics_dicts)
            except:
                print(f"Can't access model {dirpath}")

    print("Found data for %d models under %s" % (len(model_list), result_dir))

    # build dictonary of tarball names
    tar_dict = {os.path.basename(tf):tf for tf in tar_list}
    path_list = []
    for metadata_dict, metrics_dicts in zip(model_list, metrics_list):
        model_uuid = metadata_dict['model_uuid']
        dataset_key = metadata_dict['training_dataset']['dataset_key']
        dataset_name = mp.build_tarball_name(mp.build_dataset_name(dataset_key), model_uuid)
        if dataset_name in tar_dict:
            path_list.append(tar_dict[dataset_name])
        else:
            # unable to find saved tar file
            path_list.append('')

        # Get list of training run metrics for this model
        if len(metrics_dicts) < 3:
            print("Got no or incomplete metrics for model %s, skipping..." % model_uuid)
            continue
        subset_metrics = {}
        for metrics_dict in metrics_dicts:
            if metrics_dict['label'] == 'best':
                subset = metrics_dict['subset']
                subset_metrics[subset] = metrics_dict['prediction_results']
        
        model_uuid_list.append(model_uuid)
        model_params = metadata_dict['model_parameters']
        ampl_version = model_params['ampl_version']
        ampl_version_list.append(ampl_version)
        model_type = model_params['model_type']
        model_type_list.append(model_type)
        model_score_type = model_params['model_choice_score_type']
        model_score_type_list.append(model_score_type)
        featurizer = model_params['featurizer']
        #mix ecfp, graphconv, moe, mordred, rdkit for concise representation
        if featurizer in ["computed_descriptors", "descriptors"]:
            featurizer = metadata_dict["descriptor_specific"]["descriptor_type"]
        featurizer_list.append(featurizer)
        split_params = metadata_dict['splitting_parameters']
        splitter_list.append(split_params['splitter'])
        split_strategy_list.append(split_params['split_strategy'])
        split_uuid_list.append(split_params['split_uuid'])
        dataset_key_list.append(metadata_dict['training_dataset']['dataset_key'])
        feature_transform_type = metadata_dict['training_dataset']['feature_transform_type']
        feature_transform_type_list.append(feature_transform_type)
        try:
            weight_transform_type_list.append(metadata_dict['training_dataset']['weight_transform_type'])
        except:
            weight_transform_type_list.append(None)

        param_list.append(extract_model_and_feature_parameters(metadata_dict, keep_required=expand))

        for subset in subsets:
            for metric in metrics:
                try:
                    score_dict[subset][metric].append(subset_metrics[subset][metric])
                except:
                    score_dict[subset][metric].append(np.nan)
        score_dict['valid']['model_choice_score'].append(subset_metrics['valid']['model_choice_score'])

    param_df = pd.DataFrame(param_list)
    perf_df = pd.DataFrame(dict(
                    model_uuid=model_uuid_list,
                    model_path = path_list,
                    ampl_version=ampl_version_list,
                    model_type=model_type_list,
                    dataset_key=dataset_key_list,
                    features=featurizer_list,
                    splitter=splitter_list,
                    split_strategy=split_strategy_list,
                    split_uuid=split_uuid_list,
                    model_score_type=model_score_type_list,
                    feature_transform_type=feature_transform_type_list,
                    weight_transform_type=weight_transform_type_list))

    perf_df['model_choice_score'] = score_dict['valid']['model_choice_score']
    for subset in subsets:
        for metric in metrics:
            metric_col = 'best_%s_%s' % (subset, metric)
            perf_df[metric_col] = score_dict[subset][metric]
    perf_df = perf_df.merge(param_df, on='model_uuid', how='inner')
    sort_by = 'model_choice_score'
    perf_df = perf_df.sort_values(sort_by, ascending=False)
    
    return perf_df


def get_filesystem_models(result_dir, pred_type):

    """Identify all models in result_dir and create perf_result table with 'tarball_path' column containing a path
    to each tarball.
    """
    perf_df = get_filesystem_perf_results(result_dir, pred_type)
    if pred_type == 'regression':
        metric = 'valid_r2_score'
    else:
        metric = 'valid_roc_auc_score'
    #best_df = perf_df.sort_values(by=metric, ascending=False).drop_duplicates(subset='dataset_key').copy()
    perf_df['dataset_names'] = perf_df['dataset_key'].apply(lambda f: os.path.splitext(os.path.basename(f))[0])
    perf_df['tarball_names'] = perf_df.apply(lambda x: '%s_model_%s.tar.gz' % (x['dataset_names'], x['model_uuid']), axis=1)
    tarball_names = set(perf_df['tarball_names'].values)

    all_filenames = []
    for dirpath, dirnames, filenames in os.walk(result_dir):
        for fn in filenames:
            if fn in tarball_names:
                all_filenames.append((fn, os.path.join(dirpath, fn)))

    found_files_df = pd.DataFrame({'tarball_names':[f[0] for f in all_filenames],
                                    'tarball_paths':[f[1] for f in all_filenames]})
    perf_df = perf_df.merge(found_files_df, on='tarball_names', how='outer')

    return perf_df

#------------------------------------------------------------------------------------------------------------------
def copy_best_filesystem_models(result_dir, dest_dir, pred_type, force_update=False):

    """Identify the best models for each dataset within a result directory tree (e.g. from a hyperparameter search).
    Copy the associated model tarballs to a destination directory.

    Args:
        result_dir (str): Path to model training result directory.

        dest_dir (str): Path of directory wherre model tarballs will be copied to.

        pred_type (str): Prediction type ('classification' or 'regression') of models to copy

        force_update (bool): If true, overwrite tarball files that already exist in dest_dir.

    Returns:
        pd.DataFrame: Table of performance metrics for best models.

    """
    perf_df = get_filesystem_perf_results(result_dir, pred_type)
    if pred_type == 'regression':
        metric = 'valid_r2_score'
    else:
        metric = 'valid_roc_auc_score'
    best_df = perf_df.sort_values(by=metric, ascending=False).drop_duplicates(subset='dataset_key').copy()
    dataset_names = [os.path.splitext(os.path.basename(f))[0] for f in best_df.dataset_key.values]
    model_uuids = best_df.model_uuid.values
    tarball_names = ['%s_model_%s.tar.gz' % (dset_name, model_uuid) for dset_name, model_uuid in zip(dataset_names, model_uuids)]
    for dirpath, dirnames, filenames in os.walk(result_dir):
        for fn in filenames:
            if (fn in tarball_names) and (force_update or not os.path.exists(os.path.join(dest_dir, fn))):
                shutil.copy2(os.path.join(dirpath, fn), dest_dir)
                print('Copied %s' % fn)
    return best_df

#------------------------------------------------------------------------------------------------------------------
def get_summary_perf_tables(collection_names=None, filter_dict={}, result_dir=None, prediction_type='regression', verbose=False):
    """Load model parameters and performance metrics from model tracker for all models saved in the model tracker DB under
    the given collection names (or result directory if Model tracker is not available) with the given prediction type.
    Tabulate the parameters and metrics including:
        
        dataset (assay name, target, parameter, key, bucket)
        dataset size (train/valid/test/total)
        number of training folds
        model type (NN or RF)
        featurizer
        transformation type
        metrics: r2_score, mae_score and rms_score for regression, or ROC AUC for classification

    Args:
        collection_names (list): Names of model tracker collections to search for models.

        filter_dict (dict): Additional filter criteria to use in model query.

        result_dir (str or list): Directories to search for models; must be provided if the model tracker DB is not available.

        prediction_type (str): Type of models (classification or regression) to query.

        verbose (bool): If true, print status messages as collections are processed.

    Returns:
        pd.DataFrame: Table of model metadata fields and performance metrics.

    """

    if not mlmt_supported and not result_dir:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only, 'result_dir' is needed.")
        return None

    collection_list = []
    ampl_version_list=[]
    model_uuid_list = []
    time_built_list = []
    model_type_list = []
    dataset_key_list = []
    bucket_list = []
    param_list = []
    featurizer_list = []
    desc_type_list = []
    transform_list = []
    dset_size_list = []
    splitter_list = []
    split_strategy_list = []
    split_uuid_list = []
    umap_dim_list = []
    umap_targ_wt_list = []
    umap_neighbors_list = []
    umap_min_dist_list = []
    split_uuid_list=[]

    model_feat_param_list = []


    if prediction_type == 'regression':
        score_types = ['r2_score', 'mae_score', 'rms_score']
    else:
        # TODO: add more classification metrics later
        score_types = ['roc_auc_score', 'prc_auc_score', 'accuracy_score', 'bal_accuracy', 'precision', 'recall_score', 'npv', 'matthews_cc', 'kappa']

    subsets = ['train', 'valid', 'test']
    score_dict = {}
    ncmpd_dict = {}
    for subset in subsets:
        score_dict[subset] = {}
        for score_type in score_types:
            score_dict[subset][score_type] = []
        ncmpd_dict[subset] = []

    metadata_list_dict = {}
    if mlmt_supported and collection_names:
        mlmt_client = dsf.initialize_model_tracker()
        filter_dict['model_parameters.prediction_type'] = prediction_type
        for collection_name in collection_names:
            print("Finding models in collection %s" % collection_name)
            query_params = {
                "match_metadata": filter_dict,

                "match_metrics": {
                    "metrics_type": "training",  # match only training metrics
                    "label": "best",
                },
            }

            metadata_list = mlmt_client.model.query_model_metadata(
                collection_name=collection_name,
                query_params=query_params,
            ).result()
            metadata_list_dict[collection_name] = metadata_list
    elif result_dir:
        if isinstance(result_dir, str):
            result_dir = [result_dir]
        for rd in result_dir:
            if rd not in metadata_list_dict:
                metadata_list_dict[rd] = []
            for dirpath, dirnames, filenames in os.walk(rd):
                if "model_metadata.json" in filenames:
                    with open(os.path.join(dirpath, 'model_metadata.json')) as f:
                        metadata_dict = json.load(f)
                    metadata_list_dict[rd].append(metadata_dict)

    for ss in metadata_list_dict:
        for i, metadata_dict in enumerate(metadata_list_dict[ss]):
            if (i % 10 == 0) and verbose:
                print('Processing collection %s model %d' % (ss, i))
            # Check that model has metrics before we go on
            if 'training_metrics' not in metadata_dict:
                continue
            collection_list.append(ss)
            model_uuid = metadata_dict['model_uuid']
            model_uuid_list.append(model_uuid)
            time_built = metadata_dict['time_built']
            time_built_list.append(time_built)

            model_params = metadata_dict['model_parameters']
            ampl_version = model_params.get('ampl_version', 'probably 1.0.0')
            ampl_version_list.append(ampl_version)
            model_type = model_params['model_type']
            model_type_list.append(model_type)
            featurizer = model_params['featurizer']
            featurizer_list.append(featurizer)
            if 'descriptor_specific' in metadata_dict:
                desc_type = metadata_dict['descriptor_specific']['descriptor_type']
            elif featurizer in ['graphconv', 'ecfp']:
                desc_type = featurizer
            else:
                desc_type = ''
            desc_type_list.append(desc_type)
            dataset_key = metadata_dict['training_dataset']['dataset_key']
            bucket = metadata_dict['training_dataset']['bucket']
            dataset_key_list.append(dataset_key)
            bucket_list.append(bucket)
            dset_metadata = metadata_dict['training_dataset']['dataset_metadata']
            param = metadata_dict['training_dataset']['response_cols'][0]
            param_list.append(param)
            transform_type = metadata_dict['training_dataset']['feature_transform_type']
            transform_list.append(transform_type)
            split_params = metadata_dict['splitting_parameters']
            splitter_list.append(split_params['splitter'])
            split_uuid_list.append(split_params.get('split_uuid', ''))
            split_strategy = split_params['split_strategy']
            split_strategy_list.append(split_strategy)

            if 'umap_specific' in metadata_dict:
                umap_params = metadata_dict['umap_specific']
                umap_dim_list.append(umap_params['umap_dim'])
                umap_targ_wt_list.append(umap_params['umap_targ_wt'])
                umap_neighbors_list.append(umap_params['umap_neighbors'])
                umap_min_dist_list.append(umap_params['umap_min_dist'])
            else:
                umap_dim_list.append(nan)
                umap_targ_wt_list.append(nan)
                umap_neighbors_list.append(nan)
                umap_min_dist_list.append(nan)

            model_feat_param_list.append(extract_model_and_feature_parameters(metadata_dict))

            # Get model metrics for this model
            metrics_dicts = metadata_dict['training_metrics']
            #print("Got %d metrics dicts for model %s" % (len(metrics_dicts), model_uuid))
            subset_metrics = {}
            for metrics_dict in metrics_dicts:
                if metrics_dict['label'] == 'best':
                    subset = metrics_dict['subset']
                    subset_metrics[subset] = metrics_dict['prediction_results']
            if split_strategy == 'k_fold_cv':
                dset_size = subset_metrics['train']['num_compounds'] + subset_metrics['test']['num_compounds']
            else:
                dset_size = subset_metrics['train']['num_compounds'] + subset_metrics['valid']['num_compounds'] + subset_metrics['test']['num_compounds']
            for subset in subsets:
                subset_size = subset_metrics[subset]['num_compounds']
                for score_type in score_types:
                    try:
                        score = subset_metrics[subset][score_type]
                    except KeyError:
                        score = float('nan')
                    score_dict[subset][score_type].append(score)
                ncmpd_dict[subset].append(subset_size)
            dset_size_list.append(dset_size)

    col_dict = dict(
                    collection=collection_list,
                    ampl_version=ampl_version_list,
                    model_uuid=model_uuid_list,
                    time_built=time_built_list,
                    model_type=model_type_list,
                    featurizer=featurizer_list,
                    features=desc_type_list,
                    transformer=transform_list,
                    splitter=splitter_list,
                    split_strategy=split_strategy_list,
                    split_uuid=split_uuid_list,
                    umap_dim=umap_dim_list,
                    umap_targ_wt=umap_targ_wt_list,
                    umap_neighbors=umap_neighbors_list,
                    umap_min_dist=umap_min_dist_list,
                    dataset_bucket=bucket_list,
                    dataset_key=dataset_key_list,
                    dataset_size=dset_size_list,
                    parameter=param_list
                    )


    perf_df = pd.DataFrame(col_dict)

    param_df = pd.DataFrame(model_feat_param_list)
    perf_df = perf_df.merge(param_df, on='model_uuid', how='inner')

    for subset in subsets:
        ncmpds_col = '%s_size' % subset
        perf_df[ncmpds_col] = ncmpd_dict[subset]
        for score_type in score_types:
            metric_col = '%s_%s' % (subset, score_type)
            perf_df[metric_col] = score_dict[subset][score_type]

    return perf_df

#------------------------------------------------------------------------------------------------------------------
def get_summary_metadata_table(uuids, collections=None):
    """Tabulate metadata fields and performance metrics for a set of models identified by specific model_uuids.

    Args:
        uuids (list): List of model UUIDs to query.

        collections (list or str): Names of collections in model tracker DB to get models from. If collections is
            a string, it must identify one collection to search for all models. If a list, it must be of the same
            length as `uuids`. If not provided, all collections will be searched.

    Returns:
        pd.DataFrame: Table of metadata fields and performance metrics for models.

    """

    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return None

    if isinstance(uuids,str):
        uuids = [uuids]

    if isinstance(collections,str):
        collections = [collections] * len(uuids)

    mlist = []
    mlmt_client = dsf.initialize_model_tracker()
    for idx,uuid in enumerate(uuids):
        if collections is not None:
            collection_name = collections[idx]
        else:
            collection_name = trkr.get_model_collection_by_uuid(uuid)

        model_meta = trkr.get_full_metadata_by_uuid(uuid, collection_name=collection_name)

        mdl_params  = model_meta['model_parameters']
        data_params = model_meta['training_dataset']
        # Get model metrics for this model
        metrics = pd.DataFrame(model_meta['training_metrics'])
        metrics = metrics[metrics['label']=='best']
        train_metrics = metrics[metrics['subset']=='train']['prediction_results'].values[0]
        valid_metrics = metrics[metrics['subset']=='valid']['prediction_results'].values[0]
        test_metrics  = metrics[metrics['subset']=='test']['prediction_results'].values[0]

        # Try to name the model something intelligible in the table
        name  = 'NA'
        if 'target' in data_params['dataset_metadata']:
            name = data_params['dataset_metadata']['target']

        if (name == 'NA') & ('assay_endpoint' in data_params['dataset_metadata']):
            name = data_params['dataset_metadata']['assay_endpoint']

        if (name == 'NA') & ('response_col' in data_params['dataset_metadata']):
            name = data_params['dataset_metadata']['response_col']

        if name  != 'NA':
            if 'param' in data_params['dataset_metadata'].keys():
                name = name + ' ' + data_params['dataset_metadata']['param']
        else:
            name = 'unknown'


        transform = 'None'
        if 'transformation' in data_params['dataset_metadata'].keys():
            transform = data_params['dataset_metadata']['transformation']

        if mdl_params['featurizer'] == 'computed_descriptors':
            featurizer = model_meta['descriptor_specific']['descriptor_type']
        else:
            featurizer = mdl_params['featurizer']

        try:
            split_uuid = model_meta['splitting_parameters']['split_uuid']
        except:
            split_uuid = 'Not Available'

        if mdl_params['prediction_type'] == 'regression':
            if mdl_params['model_type'] == 'NN':
                nn_params = model_meta['nn_specific']
                minfo = {'Name': name,
                         'Transformation': transform,
                         'AMPL version used:': mdl_params.get('ampl_version', 'probably 1.0.0'),
                         'Model Type (Featurizer)':    '%s (%s)' % (mdl_params['model_type'],featurizer),
                         'r^2 (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['r2_score'], valid_metrics['r2_score'], test_metrics['r2_score']),
                         'MAE (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['mae_score'], valid_metrics['mae_score'], test_metrics['mae_score']),
                         'RMSE(Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['rms_score'], valid_metrics['rms_score'], test_metrics['rms_score']),
                         'Data Size (Train/Valid/Test)': '%i/%i/%i' % (train_metrics["num_compounds"],valid_metrics["num_compounds"],test_metrics["num_compounds"]),
                         'Splitter':      model_meta['splitting_parameters']['splitter'],
                         'Layer Sizes':   nn_params['layer_sizes'],
                         'Optimizer':     nn_params['optimizer_type'],
                         'Learning Rate': nn_params['learning_rate'],
                         'Dropouts':      nn_params['dropouts'],
                         'Best Epoch (Max)': '%i (%i)' % (nn_params['best_epoch'],nn_params['max_epochs']),
                         'Collection':    collection_name,
                         'UUID':          model_meta['model_uuid'],
                         'Split UUID':    split_uuid,
                         'Dataset Key':   data_params['dataset_key']}
            elif mdl_params['model_type'] == 'RF':
                rf_params = model_meta['rf_specific']
                minfo = {'Name': name,
                         'Transformation': transform,
                         'AMPL version used:': mdl_params.get('ampl_version', 'probably 1.0.0'),
                         'Model Type (Featurizer)':    '%s (%s)' % (mdl_params['model_type'],featurizer),
                         'Max Depth':    rf_params['rf_max_depth'],
                         'Max Features': rf_params['rf_max_depth'],
                         'RF Estimators': rf_params['rf_estimators'],
                         'r^2 (Train/Valid/Test)':       '%0.2f/%0.2f/%0.2f' % (train_metrics['r2_score'], valid_metrics['r2_score'], test_metrics['r2_score']),
                         'MAE (Train/Valid/Test)':       '%0.2f/%0.2f/%0.2f' % (train_metrics['mae_score'], valid_metrics['mae_score'], test_metrics['mae_score']),
                         'RMSE(Train/Valid/Test)':       '%0.2f/%0.2f/%0.2f' % (train_metrics['rms_score'], valid_metrics['rms_score'], test_metrics['rms_score']),
                         'Data Size (Train/Valid/Test)': '%i/%i/%i' % (train_metrics["num_compounds"],valid_metrics["num_compounds"],test_metrics["num_compounds"]),
                         'Splitter':      model_meta['splitting_parameters']['splitter'],
                         'Collection':    collection_name,
                         'UUID':          model_meta['model_uuid'],
                         'Split UUID':    split_uuid,
                         'Dataset Key':   data_params['dataset_key']}
            elif mdl_params['model_type'] == 'xgboost':
                xgb_params = model_meta['xgb_specific']
                minfo = {'Name': name,
                         'Transformation': transform,
                         'AMPL version used:': mdl_params.get('ampl_version', 'probably 1.0.0'),
                         'Model Type (Featurizer)':    '%s (%s)' % (mdl_params['model_type'],featurizer),
                         'XGB learning rate': xgb_params['xgb_learning_rate'],
                         'Gamma':    xgb_params['xgb_gamma'],
                         'XGB max depth': xgb_params['xgb_max_depth'],
                         'Column sample fraction':    xgb_params['xgb_colsample_bytree'],
                         'Row subsample fraction':    xgb_params['xgb_subsample'],
                         'Number of estimators':    xgb_params['xgb_n_estimators'],
                         'Minimum child weight':    xgb_params['xgb_min_child_weight'],
                         'r^2 (Train/Valid/Test)':       '%0.2f/%0.2f/%0.2f' % (train_metrics['r2_score'], valid_metrics['r2_score'], test_metrics['r2_score']),
                         'MAE (Train/Valid/Test)':       '%0.2f/%0.2f/%0.2f' % (train_metrics['mae_score'], valid_metrics['mae_score'], test_metrics['mae_score']),
                         'RMSE(Train/Valid/Test)':       '%0.2f/%0.2f/%0.2f' % (train_metrics['rms_score'], valid_metrics['rms_score'], test_metrics['rms_score']),
                         'Data Size (Train/Valid/Test)': '%i/%i/%i' % (train_metrics["num_compounds"],valid_metrics["num_compounds"],test_metrics["num_compounds"]),
                         'Splitter':      model_meta['splitting_parameters']['splitter'],
                         'Collection':    collection_name,
                         'UUID':          model_meta['model_uuid'],
                         'Split UUID':    split_uuid,
                         'Dataset Key':   data_params['dataset_key']}
            else:
                architecture = 'unknown'
        elif mdl_params['prediction_type'] == 'classification':
            if mdl_params['model_type'] == 'NN':
                nn_params = model_meta['nn_specific']
                minfo = {'Name': name,
                         'Transformation': transform,
                         'AMPL version used:': mdl_params.get('ampl_version', 'probably 1.0.0'),
                         'Model Type (Featurizer)':    '%s (%s)' % (mdl_params['model_type'],featurizer),
                         'ROC AUC (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['roc_auc_score'], valid_metrics['roc_auc_score'], test_metrics['roc_auc_score']),
                         'PRC AUC (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['prc_auc_score'], valid_metrics['prc_auc_score'], test_metrics['prc_auc_score']),
                         'Balanced accuracy (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics.get('bal_accuracy', np.nan), valid_metrics.get('bal_accuracy',np.nan), test_metrics.get('bal_accuracy', np.nan)),
                         'Accuracy (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['accuracy_score'], valid_metrics['accuracy_score'], test_metrics['accuracy_score']),
                         'Precision (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['precision'], valid_metrics['precision'], test_metrics['precision']),
                         'Recall (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['recall_score'], valid_metrics['recall_score'], test_metrics['recall_score']),
                         'NPV (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['npv'], valid_metrics['npv'], test_metrics['npv']),
                         'Kappa (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['kappa'], valid_metrics['kappa'], test_metrics['kappa']),
                         'Matthews CC (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['matthews_cc'], valid_metrics['matthews_cc'], test_metrics['matthews_cc']),
                         'Cross entropy (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['cross_entropy'], valid_metrics['cross_entropy'], test_metrics['cross_entropy']),
                         'Confusion matrices (Train/Valid/Test)':     f"{str(train_metrics['confusion_matrix'])}/{str(valid_metrics['confusion_matrix'])}/{str(test_metrics['confusion_matrix'])}",
                         'Data Size (Train/Valid/Test)': '%i/%i/%i' % (train_metrics["num_compounds"],valid_metrics["num_compounds"],test_metrics["num_compounds"]),
                         'Splitter':      model_meta['splitting_parameters']['splitter'],
                         'Layer Sizes':   nn_params['layer_sizes'],
                         'Optimizer':     nn_params['optimizer_type'],
                         'Learning Rate': nn_params['learning_rate'],
                         'Dropouts':      nn_params['dropouts'],
                         'Best Epoch (Max)': '%i (%i)' % (nn_params['best_epoch'],nn_params['max_epochs']),
                         'Collection':    collection_name,
                         'UUID':          model_meta['model_uuid'],
                         'Split UUID':    split_uuid,
                         'Dataset Key':   data_params['dataset_key']}
            elif mdl_params['model_type'] == 'RF':
                rf_params = model_meta['rf_specific']
                minfo = {'Name': name,
                         'Transformation': transform,
                         'AMPL version used:': mdl_params.get('ampl_version', 'probably 1.0.0'),
                         'Model Type (Featurizer)':    '%s (%s)' % (mdl_params['model_type'],featurizer),
                         'Max Depth':    rf_params['rf_max_depth'],
                         'Max Features': rf_params['rf_max_depth'],
                         'RF Estimators': rf_params['rf_estimators'],
                         'ROC AUC (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['roc_auc_score'], valid_metrics['roc_auc_score'], test_metrics['roc_auc_score']),
                         'PRC AUC (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['prc_auc_score'], valid_metrics['prc_auc_score'], test_metrics['prc_auc_score']),
                         'Balanced accuracy (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics.get('bal_accuracy', np.nan), valid_metrics.get('bal_accuracy',np.nan), test_metrics.get('bal_accuracy', np.nan)),
                         'Accuracy (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['accuracy_score'], valid_metrics['accuracy_score'], test_metrics['accuracy_score']),
                         'Precision (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['precision'], valid_metrics['precision'], test_metrics['precision']),
                         'Recall (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['recall_score'], valid_metrics['recall_score'], test_metrics['recall_score']),
                         'NPV (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['npv'], valid_metrics['npv'], test_metrics['npv']),
                         'Kappa (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['kappa'], valid_metrics['kappa'], test_metrics['kappa']),
                         'Matthews CC (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['matthews_cc'], valid_metrics['matthews_cc'], test_metrics['matthews_cc']),
                         'Cross entropy (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['cross_entropy'], valid_metrics['cross_entropy'], test_metrics['cross_entropy']),
                         'Confusion matrices (Train/Valid/Test)':     f"{train_metrics['confusion_matrix']}/{valid_metrics['confusion_matrix']}/{test_metrics['confusion_matrix']}",
                         'Data Size (Train/Valid/Test)': '%i/%i/%i' % (train_metrics["num_compounds"],valid_metrics["num_compounds"],test_metrics["num_compounds"]),
                         'Splitter':      model_meta['splitting_parameters']['splitter'],
                         'Collection':    collection_name,
                         'UUID':          model_meta['model_uuid'],
                         'Split UUID':    split_uuid,
                         'Dataset Key':   data_params['dataset_key']}
            elif mdl_params['model_type'] == 'xgboost':
                xgb_params = model_meta['xgb_specific']
                minfo = {'Name': name,
                         'Transformation': transform,
                         'AMPL version used:': mdl_params.get('ampl_version', 'probably 1.0.0'),
                         'Model Type (Featurizer)':    '%s (%s)' % (mdl_params['model_type'],featurizer),
                         'XGB learning rate': xgb_params['xgb_learning_rate'],
                         'Gamma':    xgb_params['xgb_gamma'],
                         'XGB max depth': xgb_params['xgb_max_depth'],
                         'Column sample fraction':    xgb_params['xgb_colsample_bytree'],
                         'Row subsample fraction':    xgb_params['xgb_subsample'],
                         'Number of estimators':    xgb_params['xgb_n_estimators'],
                         'Minimum child weight':    xgb_params['xgb_min_child_weight'],
                         'ROC AUC (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['roc_auc_score'], valid_metrics['roc_auc_score'], test_metrics['roc_auc_score']),
                         'PRC AUC (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['prc_auc_score'], valid_metrics['prc_auc_score'], test_metrics['prc_auc_score']),
                         'Balanced accuracy (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics.get('bal_accuracy', np.nan), valid_metrics.get('bal_accuracy',np.nan), test_metrics.get('bal_accuracy', np.nan)),
                         'Accuracy (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['accuracy_score'], valid_metrics['accuracy_score'], test_metrics['accuracy_score']),
                         'Precision (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['precision'], valid_metrics['precision'], test_metrics['precision']),
                         'Recall (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['recall_score'], valid_metrics['recall_score'], test_metrics['recall_score']),
                         'NPV (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['npv'], valid_metrics['npv'], test_metrics['npv']),
                         'Kappa (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['kappa'], valid_metrics['kappa'], test_metrics['kappa']),
                         'Matthews CC (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['matthews_cc'], valid_metrics['matthews_cc'], test_metrics['matthews_cc']),
                         'Cross entropy (Train/Valid/Test)':     '%0.2f/%0.2f/%0.2f' % (train_metrics['cross_entropy'], valid_metrics['cross_entropy'], test_metrics['cross_entropy']),
                         'Confusion matrices (Train/Valid/Test)':     f"{train_metrics['confusion_matrix']}/{valid_metrics['confusion_matrix']}/{test_metrics['confusion_matrix']}",
                         'Data Size (Train/Valid/Test)': '%i/%i/%i' % (train_metrics["num_compounds"],valid_metrics["num_compounds"],test_metrics["num_compounds"]),
                         'Splitter':      model_meta['splitting_parameters']['splitter'],
                         'Collection':    collection_name,
                         'UUID':          model_meta['model_uuid'],
                         'Split UUID':    split_uuid,
                         'Dataset Key':   data_params['dataset_key']}
            else:
                architecture = 'unknown'

        mlist.append(OrderedDict(minfo))
    return pd.DataFrame(mlist).set_index('Name').transpose()

#------------------------------------------------------------------------------------------------------------------
def get_training_datasets(collection_names):
    """Query the model tracker DB for all the unique dataset keys and buckets used to train models in the given
    collections.

    Args:
        collection_names (list): List of names of model tracker collections to search for models.

    Returns:
        dict: Dictionary mapping collection names to lists of (dataset_key, bucket) tuples for training sets.

    """
    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return None

    result_dict = {}

    mlmt_client = dsf.initialize_model_tracker()
    for collection_name in collection_names:
        dset_list = mlmt_client.model.get_training_datasets(collection_name=collection_name).result()
        result_dict[collection_name] = dset_list

    return result_dict

#------------------------------------------------------------------------------------------------------------------
def get_dataset_models(collection_names, filter_dict={}):
    """Query the model tracker for all models saved in the model tracker DB under the given collection names. Returns a dictionary
    mapping (dataset_key,bucket) pairs to the list of (collection,model_uuid) pairs trained on the corresponding datasets.

    Args:
        collection_names (list): List of names of model tracker collections to search for models.

        filter_dict (dict): Additional filter criteria to use in model query.

    Returns:
        dict: Dictionary mapping training set (dataset_key, bucket) tuples to (collection, model_uuid) pairs.

    """
    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return None


    result_dict = {}


    coll_dset_dict = get_training_dict(collection_names)

    mlmt_client = dsf.initialize_model_tracker()
    for collection_name in collection_names:
        dset_list = coll_dset_dict[collection_name]
        for dset_dict in dset_list:
            query_filter = {
                'training_dataset.bucket': dset_dict['bucket'],
                'training_dataset.dataset_key': dset_dict['dataset_key']
            }
            query_filter.update(filter_dict)
            query_params = {
                "match_metadata": query_filter
            }

            print('Querying models in collection %s for dataset %s, %s' % (collection_name, bucket, dset_key))
            metadata_list = mlmt_client.model.query_model_metadata(
                collection_name=collection_name,
                query_params=query_params,
                include_fields=['model_uuid']
            ).result()
            for i, metadata_dict in enumerate(metadata_list):
                if i % 50 == 0:
                    print('Processing collection %s model %d' % (collection_name, i))
                model_uuid = metadata_dict['model_uuid']
                result_dict.setdefault((dset_key,bucket), []).append((collection_name, model_uuid))

    return result_dict

#-------------------------------------------------------------------------------------------------------------------
def get_multitask_perf_from_files(result_dir, pred_type='regression'):
    """Retrieve model metadata and performance metrics stored in the filesystem from a multitask hyperparameter search.
    Format the per-task performance metrics in a table with a row for each task and columns for each model/subset
    combination.

    Args:
        result_dir (str): Path to root result directory containing output from a hyperparameter search run.

        pred_type (str): Prediction type ('classification' or 'regression') of models to query.

    Returns:
        pd.DataFrame: Table of model metadata fields and performance metrics.

    """

    model_uuid_list = []
    learning_rate_list = []
    dropouts_list = []
    layer_sizes_list = []
    best_epoch_list = []
    max_epochs_list = []

    subsets = ['train', 'valid', 'test']

    if pred_type == 'regression':
        metrics = ['num_compounds', 'r2_score', 'task_r2_scores']
    else:
        metrics = ['num_compounds', 'roc_auc_score', 'task_roc_auc_scores']
    score_dict = {}
    for subset in subsets:
        score_dict[subset] = {}
        for metric in metrics:
            score_dict[subset][metric] = []


    # Navigate the results directory tree
    model_list = []
    metrics_list = []
    for dirpath, dirnames, filenames in os.walk(result_dir):
        if ('model_metadata.json' in filenames) and ('model_metrics.json' in filenames):
            meta_path = os.path.join(dirpath, 'model_metadata.json')
            with open(meta_path, 'r') as meta_fp:
                meta_dict = json.load(meta_fp)
            model_list.append(meta_dict)
            metrics_path = os.path.join(dirpath, 'model_metrics.json')
            with open(metrics_path, 'r') as metrics_fp:
                metrics_dicts = json.load(metrics_fp)
            metrics_list.append(metrics_dicts)

    print("Found data for %d models under %s" % (len(model_list), result_dir))

    for metadata_dict, metrics_dicts in zip(model_list, metrics_list):
        model_uuid = metadata_dict['model_uuid']
        #print("Got metadata for model UUID %s" % model_uuid)

        # Get list of training run metrics for this model
        #print("Got %d metrics dicts for model %s" % (len(metrics_dicts), model_uuid))
        if len(metrics_dicts) < 3:
            raise Exception("Got no or incomplete metrics for model %s, skipping..." % model_uuid)
            #print("Got no or incomplete metrics for model %s, skipping..." % model_uuid)
            #continue
        subset_metrics = {}
        for metrics_dict in metrics_dicts:
            if metrics_dict['label'] == 'best':
                subset = metrics_dict['subset']
                subset_metrics[subset] = metrics_dict['prediction_results']

        model_uuid_list.append(model_uuid)
        model_params = metadata_dict['model_parameters']
        dset_params = metadata_dict['training_dataset']
        response_cols = dset_params['response_cols']
        nn_params = metadata_dict['nn_specific']
        max_epochs_list.append(nn_params['max_epochs'])
        best_epoch_list.append(nn_params['best_epoch'])
        learning_rate_list.append(nn_params['learning_rate'])
        layer_sizes_list.append(','.join(['%d' % s for s in nn_params['layer_sizes']]))
        dropouts_list.append(','.join(['%.2f' % d for d in nn_params['dropouts']]))
        for subset in subsets:
            for metric in metrics:
                score_dict[subset][metric].append(subset_metrics[subset][metric])


    # Format the data as a table with groups of 3 columns for each model
    num_models = len(model_uuid_list)
    if pred_type == 'regression':
        model_params = ['model_uuid', 'learning_rate', 'layer_sizes', 'dropouts', 'max_epochs', 'best_epoch',
                        'subset', 'num_compounds', 'mean_r2_score']
    else:
        model_params = ['model_uuid', 'learning_rate', 'layer_sizes', 'dropouts', 'max_epochs', 'best_epoch',
                        'subset', 'num_compounds', 'mean_roc_auc_score']
    param_list = model_params + response_cols
    perf_df = pd.DataFrame(dict(col_0=param_list))
    colnum = 0
    for i in range(num_models):
        for subset in subsets:
            vals = []
            if subset == 'train':
                vals.append(model_uuid_list[i])
                vals.append(learning_rate_list[i])
                vals.append(layer_sizes_list[i])
                vals.append(dropouts_list[i])
                vals.append('%d' % max_epochs_list[i])
                vals.append('%d' % best_epoch_list[i])
            else:
                vals = vals + ['']*6
            vals.append(subset)
            vals.append('%d' % score_dict[subset]['num_compounds'][i])
            if pred_type == 'regression':
                vals.append('%.3f' % score_dict[subset]['r2_score'][i])
                vals = vals + ['%.3f' % v for v in score_dict[subset]['task_r2_scores'][i]]
            else:
                vals.append('%.3f' % score_dict[subset]['roc_auc_score'][i])
                vals = vals + ['%.3f' % v for v in score_dict[subset]['task_roc_auc_scores'][i]]
            colnum += 1
            colname = 'col_%d' % colnum
            perf_df[colname] = vals

    return perf_df


#-------------------------------------------------------------------------------------------------------------------
def get_multitask_perf_from_files_new(result_dir, pred_type='regression', dataset_key=None, tar=False):
    """Retrieve model metadata and performance metrics stored in the filesystem from a multitask hyperparameter search.
    Format the per-task performance metrics in a table with a row for each task and columns for each model/subset
    combination.

    Args:
        result_dir (str): Path to root result directory containing output from a hyperparameter search run.

        pred_type (str): Prediction type ('classification' or 'regression') of models to query.

        dataset_key(str): Optional, the dataset_key to filter models only trained on a single dataset.

        tar (bool): Default false, whether to open tar files or only search for models in folders.

    Returns:
        pd.DataFrame: Table of model metadata fields and performance metrics.

    """

    # Navigate the results directory tree (read tar files version, which is slower)
    model_list = []
    tar_list=glob(f'{result_dir}/**/*.tar.gz', recursive=True)
    if tar:
        for tar_file in tar_list:
            with tarfile.open(tar_file, "r:gz") as tar:
                if './model_metadata.json' in tar.getnames():
                    with tar.extractfile('./model_metadata.json') as meta:
                        meta=json.loads(meta.read())
                        meta['model_path']=tar_file
                elif 'model_metadata.json' in tar.getnames():
                    with tar.extractfile('model_metadata.json') as meta:
                        meta=json.loads(meta.read())
                else:
                    continue
            if meta['model_parameters']['prediction_type']==pred_type:
                if (dataset_key is not None) and (meta['training_dataset']['dataset_key']==dataset_key):
                        model_list.append(meta)
                elif dataset_key is None:
                    model_list.append(meta)
    else:
        model_path_list=glob(f'{result_dir}/**/model_metadata.json', recursive=True)
        for model_path in model_path_list:
            with open(model_path, 'r') as model:
                meta=json.loads(model.read())
                tarfiles=[x for x in tar_list if meta['model_uuid'] in x]
                if len(tarfiles)==1:
                    meta['model_path']=tarfiles[0]
                else:
                    meta['model_path']=os.path.dirname(model_path)               
            if meta['model_parameters']['prediction_type']==pred_type:
                if (dataset_key is not None) and (meta['training_dataset']['dataset_key']==dataset_key):
                        model_list.append(meta)
                elif dataset_key is None:
                    model_list.append(meta)

    print(f'Found data for {len(model_list)} {pred_type} models under {result_dir}')
    
    # unpack metadata dicts
    metadata=pd.DataFrame(model_list)
    
    # establish initial unpacked models df
    dropcols=['model_uuid','time_built','model_path','training_metrics']
    models=metadata[['model_uuid','time_built','model_path']]
    
    # find colums to keep as dicts and extract
    dict_cols=['model_uuid','splitting_parameters']
    dict_cols.extend([x for x in metadata.columns if 'specific' in x and (x!='descriptor_specific')])
    dropcols.extend([x for x in metadata.columns if ('specific' in x) and (x!='descriptor_specific')])
    keep_dicts=metadata[dict_cols]       
    
    # extract training data
    training_metrics=metadata[['model_uuid','training_metrics']]
    
    # drop info from metadata
    metadata=metadata.drop(columns=dropcols)
    
    # unpack and re-merge simple dicts into models df
    for col in metadata.columns:
        df=pd.DataFrame(dict(zip(models.model_uuid, metadata[col]))).T.dropna(how='all').reset_index(names=['model_uuid'])
        models=models.merge(df, how='left', on='model_uuid')
    
    # manipulate dfs
    models['features']=np.where(models.featurizer=='computed_descriptors',models.descriptor_type, models.featurizer)
    keep_dicts=keep_dicts[keep_dicts.model_uuid.isin(models.model_uuid)]

    # deal with metrics
    tm=pd.DataFrame(training_metrics.training_metrics.tolist())
    preds=[]
    for col in tm.columns:
        
        # get metrics and metric label
        met=pd.DataFrame(tm[col].tolist())
        metlabel=met.label.iloc[0]+'_'+met.subset.iloc[0]
    
        # expand metrics to get scores
        pred=pd.DataFrame(met.prediction_results.tolist())
        pred=models[['model_uuid','response_cols']].join(pred)

        # check for > 1 dataset
        if len(set(models.response_cols.astype(str)))>1:
            raise Exception (f"Warning: you cannot export multitask model performances for more than one dataset at a time. Please provide the dataset_key as an additional parameter. Your {pred_type} options are: {list(set(models.dataset_key))}.")

        num_model_tasks=models.num_model_tasks.iloc[0]
        
        # get task scores - long form and rename columns
        taskcols=['response_cols']
        taskcols.extend([x for x in pred.columns if 'task' in x])    
        task_preds=pred[['model_uuid']+taskcols].set_index('model_uuid').explode(taskcols).reset_index()

        # get full model scores and rename columns
        predcols=[x for x in pred.columns if 'task' not in x]
        predcols.remove('response_cols')
        pred=pred[predcols].copy()
        pred.columns=[metlabel+'_'+col if col!='model_uuid' else col for col in predcols]
        pred['response_cols']='full_model'
        
        # rename task_pred columns to match full model names
        coldict={}
        for col in task_preds.columns:
            if col not in ['model_uuid','response_cols']:
                coldict[col]=[predcol for predcol in pred.columns if predcol.replace(metlabel+'_','').startswith(col.replace('task_','')[0:3])][0]
        task_preds=task_preds.rename(columns=coldict)

        # concatenate all scores
        if num_model_tasks>1:
            pred=pd.concat([pred,task_preds])
            
        # if single task model, rename response columns and filter out empty rows
        if num_model_tasks==1:
            pred=pred[pred.response_cols=='full_model']
            pred['response_cols']=[x[0] for x in models.response_cols]
    
        # append to list
        preds.append(pred)
        
    # trim model df columns - add compatibility for new metadata weight_transform_type
    models=models.filter(items=['model_uuid', 'time_built', 'ampl_version','dataset_key', 'model_path',
           'model_type', 'prediction_type', 'splitter',
           'split_strategy', 'split_valid_frac', 'split_test_frac', 'split_uuid', 
            'production', 'feature_transform_type','response_transform_type', 'weight_transform_type',
           'smiles_col', 'features','model_choice_score_type',])
    
    # merge model info and pred_df info
    for pred in preds:
        models=models.merge(pred)

    # deal with info left in dicts
    models=models.merge(keep_dicts)
    models['model_parameters_dict']=np.nan
    models['model_parameters_dict']=models.filter(items=['xgb_specific','nn_specific','rf_specific','model_parameters_dict']).ffill(axis=1).model_parameters_dict
    models['feat_parameters_dict']=np.nan
    models['feat_parameters_dict']=models.filter(items=['ecfp_specific','auto_featurizer_specific','autoencoder_specific','feat_parameters_dict']).ffill(axis=1).feat_parameters_dict     
    for col in ['xgb_specific','nn_specific','rf_specific','ecfp_specific','auto_featurizer_specific','autoencoder_specific']:
        try: models=models.drop(columns=col)
        except: pass

    return models


#-------------------------------------------------------------------------------------------------------------------
def get_multitask_perf_from_tracker(collection_name, response_cols=None, expand_responses=None, expand_subsets='test',
                                    exhaustive=False):
    """Retrieve full metadata and metrics from model tracker for all models in a collection and format them
    into a table, including per-task performance metrics for multitask models.

    Meant for multitask NN models, but works for single task models as well.

    By AKP. Works for model tracker as of 10/2020

    Args:
        collection_name (str): Name of model tracker collection to search for models.

        response_cols (list, str or None): Names of tasks (response columns) to query performance results for.
            If None, checks to see if the entire collection has the same response cols.
            Otherwise, should be list of strings or a comma-separated string.
            asks for clarification. Note: make sure response cols are listed in same order as in metadata.
            Recommended: None first, then clarify.

        expand_responses (list, str or None): Names of tasks / response columns you want to include results for in
            the final dataframe. Useful if you have a lot of tasks and only want to look at the performance of a
            few of them. Must also be a list or comma separated string, and must be a subset of response_cols.
            If None, will expand all responses.

        expand_subsets (list, str or None): Dataset subsets ('train', 'valid' and/or 'test') to show metrics for.
            Again, must be list or comma separated string, or None to expand all.

        exhaustive (bool): If True, return large dataframe with all model tracker metadata minus any columns not
            in expand_responses. If False, return trimmed dataframe with most relevant columns.

    Returns:
        pd.DataFrame: Table of model metadata fields and performance metrics.

    """
    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return None

    # check inputs are correct
    if collection_name.startswith('old_'):
        raise Exception("This function is not implemented for the old format of metadata.")
    if isinstance(response_cols, list):
        pass
    elif response_cols is None:
        pass
    elif isinstance(response_cols, str):
        response_cols=[x.strip() for x in response_cols.split(',')]
    else:
        raise Exception("Please input response cols as None, list or comma separated string.")
    if isinstance(expand_responses, list):
        pass
    elif expand_responses is None:
        pass
    elif isinstance(expand_responses, str):
        expand_responses=[x.strip() for x in expand_responses.split(',')]
    else:
        raise Exception("Please input expand response col(s) as list or comma separated string.")
    if isinstance(expand_subsets, list):
        pass
    elif expand_subsets is None:
        pass
    elif isinstance(expand_subsets, str):
        expand_subsets=[x.strip() for x in expand_subsets.split(',')]
    else:
        raise Exception("Please input subset(s) as list or comma separated string.")

    # get metadata
    if response_cols is not None:
        filter_dict={'training_dataset.response_cols': response_cols}
    else:
        filter_dict={}
    models = trkr.get_full_metadata(filter_dict, collection_name)
    if len(models)==0:
        raise Exception("No models found with these response cols in this collection. To get a list of possible response cols, pass response_cols=None.")
    models = pd.DataFrame.from_records(models)

    # expand model metadata - deal with NA descriptors / NA other fields
    alldat=models[['model_uuid', 'time_built']]
    models=models.drop(['model_uuid', 'time_built'], axis = 1)
    for column in models.columns:
        if column == 'training_metrics':
            continue
        nai=models[models[column].isna()].index
        nonas=models[~models[column].isna()]
        tempdf=pd.DataFrame.from_records(nonas[column].tolist(), index=nonas.index)
        tempdf=pd.concat([tempdf, pd.DataFrame(np.nan, index=nai, columns=tempdf.columns)])
        alldat=alldat.join(tempdf)

    # assign response cols
    if len(alldat.response_cols.astype(str).unique())==1:
        response_cols=alldat.response_cols[0]
        print("Response cols:", response_cols)
    else:
        raise Exception(f"There is more than one set of response cols in this collection. Please choose from these lists: {alldat.response_cols.unique()}")

    # expand training metrics - deal with NA's in columns
    metrics=pd.DataFrame.from_dict(models['training_metrics'].tolist())
    allmet=alldat[['model_uuid']]
    for column in metrics.columns:
        nai=metrics[metrics[column].isna()].index
        nonas=metrics[~metrics[column].isna()]
        tempdf=pd.DataFrame.from_records(nonas[column].tolist(), index=nonas.index)
        tempdf=pd.concat([tempdf, pd.DataFrame(np.nan, index=nai, columns=tempdf.columns)])
        label=tempdf['label'][nonas.index[0]]
        metrics_type=tempdf['metrics_type'][nonas.index[0]]
        subset=tempdf['subset'][nonas.index[0]]
        nai=tempdf[tempdf['prediction_results'].isna()].index
        nonas=tempdf[~tempdf['prediction_results'].isna()]
        tempdf=pd.DataFrame.from_records(nonas['prediction_results'].tolist(), index=nonas.index)
        tempdf=pd.concat([tempdf, pd.DataFrame(np.nan, index=nai, columns=tempdf.columns)])
        tempdf=tempdf.add_prefix(f'{label}_{subset}_')
        allmet=allmet.join(tempdf, lsuffix='', rsuffix="_2")
    alldat=alldat.merge(allmet, on='model_uuid')

    # expand task level training metrics for subset(s) of interest - deal w/ NA values
    if expand_subsets is None:
        expand_subsets=['train', 'valid', 'test']
    for sub in expand_subsets:
        listcols=alldat.columns[alldat.columns.str.contains("task")& alldat.columns.str.contains(sub)]
        for column in listcols:
            colnameslist=[]
            for task in response_cols:
                colnameslist.append(f'{column}_{task}')
            nai=alldat[alldat[column].isna()].index
            nonas=alldat[~alldat[column].isna()]
            if isinstance(nonas.loc[nonas.index[0],column], list):
                tempdf=pd.DataFrame.from_records(nonas[column].tolist(), index= nonas.index, columns=colnameslist)
                tempdf=pd.concat([tempdf, pd.DataFrame(np.nan, index=nai, columns=tempdf.columns)])
                alldat = alldat.join(tempdf)
                alldat=alldat.drop(columns=column)
            else:
                print(f"Warning: task-level metadata for {column} not in metadata.")

    # make features column
    alldat['features'] = alldat['featurizer']
    if 'descriptor_type' in alldat.columns:
        alldat.loc[alldat.featurizer == 'computed_descriptors', 'features'] = alldat.loc[alldat.featurizer == 'computed_descriptors', 'descriptor_type']

    # prune to only include expand_responses
    if expand_responses is not None:
        removecols= [x for x in response_cols if x not in expand_responses]
        for col in removecols:
            alldat=alldat.drop(columns=alldat.columns[alldat.columns.str.contains(col)])

    # return or prune further and then return
    if exhaustive:
        return alldat
    else:
        alldat=alldat.drop(columns=alldat.columns[alldat.columns.str.contains('baseline')])
        keepcols=['ampl_version','model_uuid', 'features', 'prediction_type',
                  'transformers', 'uncertainty', 'batch_size', 'bias_init_consts',
                  'dropouts', 'layer_sizes', 'learning_rate', 'max_epochs', 'optimizer_type',
                  'weight_decay_penalty', 'weight_decay_penalty_type', 'weight_init_stddevs', 'splitter',
                  'split_uuid', 'split_test_frac', 'split_valid_frac', 'smiles_col', 'id_col',
                  'feature_transform_type', 'response_cols', 'response_transform_type', 'num_model_tasks',
                  'rf_estimators', 'rf_max_depth', 'rf_max_features', 'xgb_gamma', 'xgb_learning_rate',
                  'xgb_max_depth', 'xgb_colsample_bytree', 'xgb_subsample', 'xgb_n_estimators', 'xgb_min_child_weight',
                  ]
        keepcols.extend(alldat.columns[alldat.columns.str.contains('best')])
        keepcols = list(set(alldat.columns).intersection(keepcols))
        keepcols.sort()
        alldat=alldat[keepcols]
        if sum(alldat.columns.str.contains('_2'))>0:
            print("Warning: One or more of your models has metadata for >1 best / >1 baseline epochs.")
        return alldat



#-------------------------------------------------------------------------------------------------------------------
def _aggregate_predictions(datasets, bucket, col_names, result_dir):
    """Run predictions for best dataset/model_type/split_type/featurizer (max r2 score) and save csv's in /usr/local/data/

    DEPRECATED: Will not work in current software environment. Needs to be updated

    Args:
        datasets (list): List of (dataset_key, bucket) tuples to query models for.

        bucket (str): Ignored.

       col_names (list): List of names of model tracker collections to search for models.

       result_dir (str): Ignored.

    Returns:
        None.

    Todo:
        Update for current software environment, or delete function if it's not useful.

    """
    if not mlmt_supported:
        print("Model tracker not supported in your environment; can examine models saved in filesystem only.")
        return

    results = []
    mlmt_client = dsf.initialize_model_tracker()
    for dset_key, bucket in datasets:
        for model_type in ['NN', 'RF']:
            for split_type in ['scaffold', 'random']:
                for descriptor_type in ['mordred_filtered', 'moe', 'rdkit_raw']:
                    model_filter = {"training_dataset.dataset_key" : dset_key,
                                    "training_dataset.bucket" : bucket,
                                    "ModelMetrics.TrainingRun.label" : "best",
                                    'ModelMetrics.TrainingRun.subset': 'valid',
                                    'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['max', None],
                                    'model_parameters.model_type': model_type,
                                    'model_parameters.featurizer': 'computed_descriptors',
                                    'descriptor_specific.descriptor_type': descriptor_type,
                                    'splitting_parameters.splitter': split_type
                                   }
                    for col_name in col_names:
                        model = list(trkr.get_full_metadata(model_filter, collection_name=col_name))
                        if model:
                            model = model[0]
                            result_dir = '/usr/local/data/%s/%s' % (col_name, dset_key.rstrip('.csv'))
                            result_df = mp.regenerate_results(result_dir, metadata_dict=model)
                            result_df['dset_key'] = dset_key
                            actual_col = [col for col in result_df.columns if 'actual' in col][0]
                            pred_col = [col for col in result_df.columns if 'pred' in col][0]
                            result_df['error'] = abs(result_df[actual_col] - result_df[pred_col])
                            result_df['cind'] = pd.Categorical(result_df['dset_key']).labels
                            results.append(result_df)
                    results_df = pd.concat(results).reset_index(drop=True)
                    results_df.to_csv(os.path.join(result_dir, 'predictions_%s_%s_%s_%s.csv' % (dset_key, model_type, split_type, descriptor_type)), index=False)
                for featurizer in ['graphconv', 'ecfp']:
                    model_filter = {"training_dataset.dataset_key" : dset_key,
                                    "training_dataset.bucket" : bucket,
                                    "ModelMetrics.TrainingRun.label" : "best",
                                    'ModelMetrics.TrainingRun.subset': 'valid',
                                    'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['max', None],
                                    'model_parameters.model_type': model_type,
                                    'model_parameters.featurizer': featurizer,
                                    'splitting_parameters.splitter': split_type
                                   }
                    for col_name in col_names:
                        model = list(trkr.get_full_metadata(model_filter, collection_name=col_name))
                        if model:
                            model = model[0]
                            result_dir = '/usr/local/data/%s/%s' % (col_name, dset_key.rstrip('.csv'))
                            result_df = mp.regenerate_results(result_dir, metadata_dict=model)
                            result_df['dset_key'] = dset_key
                            actual_col = [col for col in result_df.columns if 'actual' in col][0]
                            pred_col = [col for col in result_df.columns if 'pred' in col][0]
                            result_df['error'] = abs(result_df[actual_col] - result_df[pred_col])
                            result_df['cind'] = pd.Categorical(result_df['dset_key']).labels
                            results.append(result_df)
                    results_df = pd.concat(results).reset_index(drop=True)
                    results_df.to_csv(os.path.join(result_dir, 'predictions_%s_%s_%s_%s.csv' % (dset_key, model_type, split_type, featurizer)), index=False)

def num_trainable_parameters_from_file(tar_path):
    """Return number of trainable paramters from tarfile

    Given a tar file for a DeepChem model this will return the number of trainable parameters

    Args:
        tar_path (str): Path to a DeepChem model

    Returns:
        int: Number of trainable parameters.

    Raises:
        ValueError: If the model is not a DeepChem neural network model
    """
    reload_dir = tempfile.mkdtemp()
    with tarfile.open(tar_path, mode='r:gz') as tar:
        futils.safe_extract(tar, path=reload_dir)

    config_file_path = os.path.join(reload_dir, 'model_metadata.json')
    with open(config_file_path) as f:
        config = json.loads(f.read())

    # Parse the saved model metadata to obtain the parameters used to train the model
    model_params = parse.wrapper(config)

    # Is this an NN model
    if not (model_params.model_type == 'NN' or model_params.model_type in parse.model_wl):
        raise ValueError('Saved model is not a neural network. Recieved %s'%model_params.model_type)

    model_params.save_results = False
    model_params.output_dir = reload_dir

    # some models need to be 'built' (graphconv models for one) 
    # before you can count the paramters
    # The only sure way to do this with DeepChem is to make some predictions.
    # This code is adapted from predict_from_model
    pred_df = pd.DataFrame(data={model_params.id_col:['a', 'b', 'c'],
        model_params.smiles_col:['OC(CCN1CCCCC1)(c1ccccc1)C1CC2C=CC1C2',
            'COC(=O)CC1CCCCCC1N1CCN(C(=O)C(C)Cc2ccc(Cl)cc2Cl)CC1',
            'O=C(O)c1ccc2cccnc2c1N1CCN(CCc2ccc(OCCCN3CCCCCC3)cc2)CC1']})
    pipe = mp.create_prediction_pipeline_from_file(model_params, 
        reload_dir=None, model_path=tar_path)
    pred_df = pipe.predict_full_dataset(pred_df, contains_responses=False, 
                                        is_featurized=False,
                                        dset_params=model_params)

    return pipe.model_wrapper.count_params()
