"""
Functions for comparing and visualizing model performance
"""

import os
import sys
import pdb
import pandas as pd
import numpy as np
import matplotlib
import logging
import json

from collections import OrderedDict
from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.pipeline import model_tracker as trkr
import atomsci.ddm.pipeline.model_pipeline as mp
from atomsci.clients import MLMTClient

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('axes', labelsize=12)

logging.basicConfig(format='%(asctime)-15s %(message)s')

nan = np.float32('nan')

#------------------------------------------------------------------------------------------------------------------
def get_collection_datasets(collection_name):
    """
    Returns a list of unique training (dataset_key, bucket) tuples used for all models in the given collection.
    """
    dataset_set = set()
    mlmt_client = MLMTClient()
    dset_dicts = mlmt_client.model.query_datasets(collection_name=collection_name, metrics_type='training').result()
    # Convert to a list of (dataset_key, bucket) tuples
    for dset_dict in dset_dicts:
        dataset_set.add((dset_dict['dataset_key'], dset_dict['bucket']))
    return sorted(dataset_set)

#------------------------------------------------------------------------------------------------------------------
def extract_collection_perf_metrics(collection_name, output_dir, pred_type='regression'):
    """
    Obtain list of training datasets with models in the given collection. Get performance metrics for
    models on each dataset and save them as CSV files in the given output directory.
    """
    datasets = get_collection_datasets(collection_name)
    os.makedirs(output_dir, exist_ok=True)
    for dset_key, bucket in datasets:
        dset_perf_df = get_training_perf_table(dset_key, bucket, collection_name, pred_type=pred_type)
        dset_perf_file = '%s/%s_%s_model_perf_metrics.csv' % (output_dir, os.path.basename(dset_key).replace('.csv', ''), collection_name)
        dset_perf_df.to_csv(dset_perf_file, index=False)
        print('Wrote file %s' % dset_perf_file)

#------------------------------------------------------------------------------------------------------------------
def get_training_perf_table(dataset_key, bucket, collection_name, pred_type='regression', other_filters = {}):
    """
    Load performance metrics from model tracker for all models saved in the model tracker DB under
    a given collection that were trained against a particular dataset. Identify training parameters
    that vary between models, and generate plots of performance vs particular combinations of
    parameters.
    """
    print("Finding models trained on %s dataset %s" % (bucket, dataset_key))
    mlmt_client = MLMTClient()
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
                    xgb_gamma = xgb_gamma_list))
    for subset in subsets:
        metric_col = '%s_%s' % (metric_type, subset)
        perf_df[metric_col] = score_dict[subset]
    sort_metric = '%s_valid' % metric_type

    perf_df = perf_df.sort_values(sort_metric, ascending=False)
    return perf_df


# ------------------------------------------------------------------------------------------------------------------
def get_best_perf_table(col_name, metric_type, model_uuid=None, metadata_dict=None, PK_pipe=False):
    """
    Extract parameters and training run performance metrics for a single model. The model may be 
    specified either by a metadata dictionary or a model_uuid; in the latter case, the function
    queries the model tracker DB for the model metadata.

    Args:
        col_name (str): Collection name containing model, if model is specified by model_uuid.

        metric_type (str): Performance metric to include in result dictionary.

        model_uuid (str): UUID of model to query, if metadata_dict is not provided.

        metadata_dict (dict): Full metadata dictionary for a model, including training metrics and
        dataset metadata.

        PK_pipe (bool): If True, include some additional parameters in the result dictionary.

    Returns:
        model_info (dict): Dictionary of parameter or metric name - value pairs.

    """
    mlmt_client = MLMTClient()
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
    if PK_pipe:
        model_info['assay_name'] = metadata_dict['training_dataset']['dataset_metadata'][
            'assay_category']
        model_info['response_col'] = metadata_dict['training_dataset']['dataset_metadata'][
            'response_cols']
    try:
        model_info['descriptor_type'] = metadata_dict['descriptor_specific']['descriptor_type']
    except:
        model_info['descriptor_type'] = 'NA'
    try:
        model_info['num_samples'] = metadata_dict['training_dataset']['dataset_metadata']['num_row']
    except:
        tmp_df = dsf.retrieve_dataset_by_datasetkey(model_info['dataset_key'], model_info['bucket'])
        model_info['num_samples'] = tmp_df.shape[0]
    if model_info['model_type'] == 'NN':
        nn_params = metadata_dict['nn_specific']
        model_info['max_epochs'] = nn_params['max_epochs']
        model_info['best_epoch'] = nn_params['best_epoch']
        model_info['learning_rate'] = nn_params['learning_rate']
        model_info['layer_sizes'] = ','.join(['%d' % s for s in nn_params['layer_sizes']])
        model_info['dropouts'] = ','.join(['%.2f' % d for d in nn_params['dropouts']])
        model_info['rf_estimators'] = nan
        model_info['rf_max_features'] = nan
        model_info['rf_max_depth'] = nan
    if model_info['model_type'] == 'RF':
        rf_params = metadata_dict['rf_specific']
        model_info['rf_estimators'] = rf_params['rf_estimators']
        model_info['rf_max_features'] = rf_params['rf_max_features']
        model_info['rf_max_depth'] = rf_params['rf_max_depth']
        model_info['max_epochs'] = nan
        model_info['best_epoch'] = nan
        model_info['learning_rate'] = nan
        model_info['layer_sizes'] = nan
        model_info['dropouts'] = nan
    
    for metrics_dict in metrics_dicts:
        subset = metrics_dict['subset']
        metric_col = '%s_%s' % (metric_type, subset)
        model_info[metric_col] = metrics_dict['prediction_results'][metric_type]
        if (model_params['prediction_type'] == 'regression') and (metric_type != 'rms_score'):
            metric_col = 'rms_score_%s' % subset
            model_info[metric_col] = metrics_dict['prediction_results']['rms_score']
    
    return model_info


# ---------------------------------------------------------------------------------------------------------
def get_best_models_info(col_names, bucket, pred_type, PK_pipeline=False, output_dir='/usr/local/data',
                         shortlist_key=None, input_dset_keys=None, save_results=False, subset='valid',
                         metric_type=None, selection_type='max', other_filters={}):
    """
    Tabulate parameters and performance metrics for the best models, according to a given metric, for 
    each specified dataset.

    Args:
        col_names (list of str): List of model tracker collections to search.

        bucket (str): Datastore bucket for datasets.

        pred_type (str): Type of models (regression or classification).

        PK_pipeline (bool): Are we being called from PK pipeline?

        output_dir (str): Directory to write output table to.

        shortlist_key (str): Datastore key for table of datasets to query models for.

        input_dset_keys (str or list of str): List of datastore keys for datasets to query models for. Either shortlist_key 
        or input_dset_keys must be specified, but not both.

        save_results (bool): If True, write the table of results to a CSV file.

        subset (str): Input dataset subset for which metrics are used to select best models.

        metric_type (str): Type of performance metric (r2_score, roc_auc_score, etc.) to use to select best models.

        selection_type (str): Score criterion ('max' or 'min') to use to select best models.

        other_filters (dict): Additional selection criteria to include in model query.

    Returns:
        top_models_df (DataFrame): Table of parameters and metrics for best models for each dataset.
    """

    mlmt_client = MLMTClient()
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
    if type(col_names) == str:
        col_names = [col_names]
    if input_dset_keys is None:
        if shortlist_key is None:
            raise ValueError('Must specify either input_dset_keys or shortlist_key')
        dset_keys = dsf.retrieve_dataset_by_datasetkey(shortlist_key, bucket)
        # Need to figure out how to handle an unknown column name for dataset_keys
        if 'dataset_key' in dset_keys.columns:
            dset_keys = dset_keys['dataset_key']
        elif 'task_name' in dset_keys.columns:
            dset_keys = dset_keys['task_name']
        else:
            dset_keys = dset_keys.values
    else:
        if shortlist_key is not None:
            raise ValueError("You can specify either shortlist_key or input_dset_keys but not both.")
        if type(input_dset_keys) == str:
            dset_keys = [input_dset_keys]
        else:
            dset_keys = input_dset_keys
   
    # Get the best model over all collections for each dataset
    for dset_key in dset_keys:
        dset_key = dset_key.strip()
        dset_model_info = []
        for col_name in col_names:
            try:
                query_params = {
                    "match_metadata": {
                        "training_dataset.dataset_key": dset_key,
                        "training_dataset.bucket": bucket,
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
                    print('Querying collection %s for models trained on dataset %s, %s' % (col_name, bucket, dset_key))
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
                model_info = get_best_perf_table(col_name, metric_type, metadata_dict=model, PK_pipe=PK_pipeline)
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
def get_best_grouped_models_info(collection='pilot_fixed', pred_type='regression', top_n=1, subset='test'):
    """
    Get results for models in the given collection. 
    """
    
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
'''
            
#------------------------------------------------------------------------------------------------------------------
def get_umap_nn_model_perf_table(dataset_key, bucket, collection_name, pred_type='regression'):
    """
    Load performance metrics from model tracker for all NN models with the given prediction_type saved in 
    the model tracker DB under a given collection that were trained against a particular dataset. Show 
    parameter settings for UMAP transformer for models where they are available.
    """
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
    mlmt_client = MLMTClient()
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

#------------------------------------------------------------------------------------------------------------------
def get_filesystem_perf_results(result_dir, hyper_id=None, dataset_name='GSK_Amgen_Combined_BSEP_PIC50',
                                pred_type='classification'):
    """
    Retrieve model metadata and performance metrics stored in the filesystem from a hyperparameter search run.
    """
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
    best_epoch_list = []
    model_score_type_list = []
    feature_transform_type_list = []
    umap_dim_list = []
    umap_targ_wt_list = []
    umap_neighbors_list = []
    umap_min_dist_list = []

    subsets = ['train', 'valid', 'test']

    if pred_type == 'regression':
        metrics = ['r2_score', 'r2_std', 'rms_score', 'mae_score']
    else:
        metrics = ['roc_auc_score', 'roc_auc_std', 'prc_auc_score', 'precision', 'recall_score',
                   'accuracy_score', 'npv', 'matthews_cc', 'kappa', 'cross_entropy', 'confusion_matrix']
    score_dict = {}
    for subset in subsets:
        score_dict[subset] = {}
        for metric in metrics:
            score_dict[subset][metric] = []
    score_dict['valid']['model_choice_score'] = []

    
    # Navigate the results directory tree
    model_list = []
    metrics_list = []
    if hyper_id is None:
        # hyper_id not specified, so let's do all that exist under the given result_dir
        subdirs = os.listdir(result_dir)
        hyper_ids = list(set(subdirs) - {'logs', 'slurm_files'})
    else:
        hyper_ids = [hyper_id]
    for hyper_id in hyper_ids:
        topdir = os.path.join(result_dir, hyper_id, dataset_name)
        if not os.path.isdir(topdir):
            continue
        # Next component of path is a random UUID added by hyperparam script for each run. Iterate over runs.
        run_uuids = [fname for fname in os.listdir(topdir) if not fname.startswith('.')]
        for run_uuid in run_uuids:
            run_path = os.path.join(topdir, run_uuid, dataset_name)
            # Next path component is a combination of various model parameters
            param_dirs = os.listdir(run_path)
            for param_str in param_dirs:
                new_path = os.path.join(topdir, run_uuid, dataset_name, param_str)
                model_dirs = [dir for dir in os.listdir(new_path) if not dir.startswith('.')]
                model_uuid = model_dirs[0]
                meta_path = os.path.join(new_path, model_uuid, 'model_metadata.json')
                metrics_path = os.path.join(new_path, model_uuid, 'training_model_metrics.json')
                if not (os.path.exists(meta_path) and os.path.exists(metrics_path)):
                    continue
                with open(meta_path, 'r') as meta_fp:
                    meta_dict = json.load(meta_fp)
                model_list.append(meta_dict)
                with open(metrics_path, 'r') as metrics_fp:
                    metrics_dict = json.load(metrics_fp)
                metrics_list.append(metrics_dict)
    
    print("Found data for %d models under %s" % (len(model_list), result_dir))

    for metadata_dict, metrics_dict in zip(model_list, metrics_list):
        model_uuid = metadata_dict['model_uuid']
        #print("Got metadata for model UUID %s" % model_uuid)

        # Get list of prediction run metrics for this model
        pred_dicts = metrics_dict['training_metrics']
        #print("Got %d metrics dicts for model %s" % (len(pred_dicts), model_uuid))
        if len(pred_dicts) < 3:
            print("Got no or incomplete metrics for model %s, skipping..." % model_uuid)
            continue
        subset_metrics = {}
        for metrics_dict in pred_dicts:
            if metrics_dict['label'] == 'best':
                subset = metrics_dict['subset']
                subset_metrics[subset] = metrics_dict['prediction_results']

        model_uuid_list.append(model_uuid)
        model_params = metadata_dict['model_parameters']
        model_type = model_params['model_type']
        model_type_list.append(model_type)
        model_score_type = model_params['model_choice_score_type']
        model_score_type_list.append(model_score_type)
        featurizer = model_params['featurizer']
        featurizer_list.append(featurizer)
        split_params = metadata_dict['splitting_parameters']
        splitter_list.append(split_params['splitter'])
        feature_transform_type = metadata_dict['training_dataset']['feature_transform_type']
        feature_transform_type_list.append(feature_transform_type)
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
        for subset in subsets:
            for metric in metrics:
                score_dict[subset][metric].append(subset_metrics[subset][metric])
        score_dict['valid']['model_choice_score'].append(subset_metrics['valid']['model_choice_score'])
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
                    model_type=model_type_list,
                    featurizer=featurizer_list,
                    splitter=splitter_list,
                    model_score_type=model_score_type_list,
                    feature_transform_type=feature_transform_type_list,
                    umap_dim=umap_dim_list,
                    umap_targ_wt=umap_targ_wt_list,
                    umap_neighbors=umap_neighbors_list,
                    umap_min_dist=umap_min_dist_list,
                    learning_rate=learning_rate_list,
                    dropouts=dropouts_list,
                    layer_sizes=layer_sizes_list,
                    best_epoch=best_epoch_list,
                    max_epochs=max_epochs_list,
                    rf_estimators=rf_estimators_list,
                    rf_max_features=rf_max_features_list,
                    rf_max_depth=rf_max_depth_list))
    perf_df['model_choice_score'] = score_dict['valid']['model_choice_score']
    for subset in subsets:
        for metric in metrics:
            metric_col = '%s_%s' % (metric, subset)
            perf_df[metric_col] = score_dict[subset][metric]
    sort_by = 'model_choice_score'
    perf_df = perf_df.sort_values(sort_by, ascending=False)
    return perf_df

#------------------------------------------------------------------------------------------------------------------
def get_summary_perf_tables(collection_names, filter_dict={}, prediction_type='regression'):
    """
    Load model parameters and performance metrics from model tracker for all models saved in the model tracker DB under
    the given collection names with the given prediction type. Tabulate the parameters and metrics including:
        dataset (assay name, target, parameter, key, bucket)
        dataset size (train/valid/test/total)
        number of training folds
        model type (NN or RF)
        featurizer
        transformation type
        metrics: r2_score, mae_score and rms_score for regression, or ROC AUC for classification
    """


    collection_list = []
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
    rf_estimators_list = []
    rf_max_features_list = []
    rf_max_depth_list = []
    best_epoch_list = []
    max_epochs_list = []
    learning_rate_list = []
    layer_sizes_list = []
    dropouts_list = []
    umap_dim_list = []
    umap_targ_wt_list = []
    umap_neighbors_list = []
    umap_min_dist_list = []
    split_uuid_list=[]


    if prediction_type == 'regression':
        score_types = ['r2_score', 'mae_score', 'rms_score']
    else:
        # TODO: add more classification metrics later
        score_types = ['roc_auc_score', 'prc_auc_score', 'accuracy_score', 'precision', 'recall_score', 'npv', 'matthews_cc']

    subsets = ['train', 'valid', 'test']
    score_dict = {}
    ncmpd_dict = {}
    for subset in subsets:
        score_dict[subset] = {}
        for score_type in score_types:
            score_dict[subset][score_type] = []
        ncmpd_dict[subset] = []


    mlmt_client = MLMTClient()
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
        for i, metadata_dict in enumerate(metadata_list):
            if i % 10 == 0:
                print('Processing collection %s model %d' % (collection_name, i))
            # Check that model has metrics before we go on
            if not 'training_metrics' in metadata_dict:
                continue
            collection_list.append(collection_name)
            model_uuid = metadata_dict['model_uuid']
            model_uuid_list.append(model_uuid)
            time_built = metadata_dict['time_built']
            time_built_list.append(time_built)

            model_params = metadata_dict['model_parameters']
            model_type = model_params['model_type']
            model_type_list.append(model_type)
            featurizer = model_params['featurizer']
            featurizer_list.append(featurizer)
            if 'descriptor_specific' in metadata_dict:
                desc_type = metadata_dict['descriptor_specific']['descriptor_type']
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
            elif model_type == 'RF':
                rf_params = metadata_dict['rf_specific']
                rf_estimators_list.append(rf_params['rf_estimators'])
                rf_max_features_list.append(rf_params['rf_max_features'])
                rf_max_depth_list.append(rf_params['rf_max_depth'])
                max_epochs_list.append(nan)
                best_epoch_list.append(nan)
                learning_rate_list.append(nan)
                layer_sizes_list.append(nan)
                dropouts_list.append(nan)
            elif model_type == 'xgboost':
                # TODO: Add xgboost parameters
                max_epochs_list.append(nan)
                best_epoch_list.append(nan)
                learning_rate_list.append(nan)
                layer_sizes_list.append(nan)
                dropouts_list.append(nan)
                rf_estimators_list.append(nan)
                rf_max_features_list.append(nan)
                rf_max_depth_list.append(nan)
            else:
                raise Exception('Unexpected model type %s' % model_type)

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
                    model_uuid=model_uuid_list,
                    time_built=time_built_list,
                    model_type=model_type_list,
                    featurizer=featurizer_list,
                    descr_type=desc_type_list,
                    transformer=transform_list,
                    splitter=splitter_list,
                    split_strategy=split_strategy_list,
                    split_uuid=split_uuid_list,
                    umap_dim=umap_dim_list,
                    umap_targ_wt=umap_targ_wt_list,
                    umap_neighbors=umap_neighbors_list,
                    umap_min_dist=umap_min_dist_list,
                    layer_sizes=layer_sizes_list,
                    dropouts=dropouts_list,
                    learning_rate=learning_rate_list,
                    best_epoch=best_epoch_list,
                    max_epochs=max_epochs_list,
                    rf_estimators=rf_estimators_list,
                    rf_max_features=rf_max_features_list,
                    rf_max_depth=rf_max_depth_list,
                    dataset_bucket=bucket_list,
                    dataset_key=dataset_key_list,
                    dataset_size=dset_size_list,
                    parameter=param_list
                    )

    perf_df = pd.DataFrame(col_dict)
    for subset in subsets:
        ncmpds_col = '%s_size' % subset
        perf_df[ncmpds_col] = ncmpd_dict[subset]
        for score_type in score_types:
            metric_col = '%s_%s' % (subset, score_type)
            perf_df[metric_col] = score_dict[subset][score_type]

    return perf_df

#------------------------------------------------------------------------------------------------------------------
def get_summary_metadata_table(uuids, collections=None):

    if isinstance(uuids,str):
        uuids = [uuids]

    if isinstance(collections,str):
        collections = [collections] * len(uuids)

    mlist = []
    mlmt_client = MLMTClient()
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
            split_uuid = 'Not Avaliable'
        
        if mdl_params['model_type'] == 'NN':
            nn_params = model_meta['nn_specific']
            minfo = {'Name': name,
                     'Transformation': transform,
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
        else:
            architecture = 'unknown'

        mlist.append(OrderedDict(minfo))
    return pd.DataFrame(mlist).set_index('Name').transpose()

#------------------------------------------------------------------------------------------------------------------
def get_training_datasets(collection_names):
    """
    Query the model tracker DB for all the unique dataset keys and buckets used to train models in the given
    collections.
    """
    result_dict = {}

    mlmt_client = MLMTClient()
    for collection_name in collection_names:
        dset_list = mlmt_client.model.get_training_datasets(collection_name=collection_name).result()
        result_dict[collection_name] = dset_list

    return result_dict

#------------------------------------------------------------------------------------------------------------------
def get_dataset_models(collection_names, filter_dict={}):
    """
    Query the model tracker for all models saved in the model tracker DB under the given collection names. Returns a dictionary
    mapping (dataset_key,bucket) pairs to the list of (collection,model_uuid) pairs trained on the corresponding datasets.
    """

    result_dict = {}


    coll_dset_dict = get_training_dict(collection_names)

    mlmt_client = MLMTClient()
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
# TODO: Update this function
def aggregate_predictions(datasets, bucket, col_names, result_dir):
    results = []
    mlmt_client = MLMTClient()
    for dset_key, bucket in datasets:
        for model_type in ['NN', 'RF']:
            for split_type in ['scaffold', 'random']:
                for descriptor_type in ['mordred_filtered', 'moe']:
                    model_filter = {"training_dataset.dataset_key" : dset_key,
                                    "training_dataset.bucket" : bucket,
                                    "ModelMetrics.TrainingRun.label" : "best",
                                    'ModelMetrics.TrainingRun.subset': 'valid',
                                    'ModelMetrics.TrainingRun.PredictionResults.r2_score': ['max', None],
                                    'model_parameters.model_type': model_type,
                                    'model_parameters.featurizer': 'descriptors',
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
