#!/usr/bin/env python

"""Contains class ModelPipeline, which loads in a dataset, splits it, trains a model, and generates predictions and output
metrics for that model. Works for a variety of featurizers, splitters and other parameters on a generic dataset
"""

import json
import logging
import os
import io
import sys
import time
import uuid
import tempfile
import tarfile
import deepchem as dc
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import pairwise_distances
import copy

from atomsci.ddm.utils import datastore_functions as dsf
import atomsci.ddm.utils.model_version_utils as mu
import atomsci.ddm.utils.file_utils as futils

from atomsci.ddm.pipeline import model_datasets as model_datasets
from atomsci.ddm.pipeline import model_wrapper as model_wrapper
from atomsci.ddm.pipeline import featurization as feat
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline import model_tracker as trkr
from atomsci.ddm.pipeline import transformations as trans

logging.basicConfig(format='%(asctime)-15s %(message)s')

# ---------------------------------------------
def calc_AD_kmean_dist(train_dset, pred_dset, k, train_dset_pair_distance=None, dist_metric="euclidean"):
    """calculate the probability of the prediction dataset fall in the the domain of traning set. Use Euclidean distance of the K nearest neighbours.
    train_dset and pred_dset should be in 2D numpy array format where each row is a compound.
    """
    if train_dset_pair_distance is None:
        # calculate the pairwise distance of training set
        train_dset_pair_distance = pairwise_distances(X=train_dset, metric=dist_metric)
    train_kmean_dis = []
    for i in range(len(train_dset_pair_distance)):
        kn_idx = np.argpartition(train_dset_pair_distance[i], k+1)
        dis = np.mean(train_dset_pair_distance[i][kn_idx[:k+1]])
        train_kmean_dis.append(dis)
    train_dset_distribution = sp.stats.norm.fit(train_kmean_dis)
    # pairwise distance between train and pred set
    pred_size = len(pred_dset)
    train_pred_dis = pairwise_distances(X=pred_dset, Y=train_dset, metric=dist_metric)
    pred_kmean_dis_score = np.zeros(pred_size)
    for i in range(pred_size):
        pred_km_dis = np.mean(np.sort(train_pred_dis[i])[:k])
        train_dset_std = train_dset_distribution[1] if train_dset_distribution[1] != 0 else 1e-6
        pred_kmean_dis_score[i] = max(1e-6, (pred_km_dis - train_dset_distribution[0]) / train_dset_std)
    return pred_kmean_dis_score

# ---------------------------------------------
def calc_AD_kmean_local_density(train_dset, pred_dset, k, train_dset_pair_distance=None, dist_metric="euclidean"):
    """Evaluate the AD of pred data by comparing the distance betweenthe unseen object and its k nearest neighbors in the training set to the distance between these k nearest neighbors and their k nearest neighbors in the training set. Return the distance ratio. Greater than 1 means the pred data is far from the domain."""
    if train_dset_pair_distance is None:
        # calculate the pair-wise distance of training set
        train_dset_pair_distance = pairwise_distances(X=train_dset, metric=dist_metric)
    # pairwise distance between train and pred set
    pred_size = len(pred_dset)
    train_pred_dis = pairwise_distances(X=pred_dset, Y=train_dset, metric=dist_metric)
    pred_kmean_dis_local_density = np.zeros(pred_size)
    for i in range(pred_size):
        # find the index of k nearest neighbour of each prediction data
        kn_idx = np.argpartition(train_pred_dis[i], k)
        pred_km_dis = np.mean(train_pred_dis[i][kn_idx[:k]])
        # find the neighbours of each neighbour and calculate the distance
        neighbor_dis = []
        for nei_ix in kn_idx[:k]:
            nei_kn_idx = np.argpartition(train_dset_pair_distance[nei_ix], k)
            neighbor_dis.append(np.mean(train_dset_pair_distance[nei_ix][nei_kn_idx[:k]]))
        ave_nei_dis = np.mean(neighbor_dis)
        if ave_nei_dis == 0:
            ave_nei_dis = 1e-6
        pred_kmean_dis_local_density[i] = pred_km_dis / ave_nei_dis
    return pred_kmean_dis_local_density

# ---------------------------------------------
def build_tarball_name(dataset_name, model_uuid, result_dir=''):
    """format for building model tarball names
        Creates the file name for a model tarball from dataset key and model_uuid
        with optional result_dir.

    Args:
       dataset_name (str): The dataset_name used to train this model
       model_uuid (str): The model_uuid assigned to this model
       result_dir (str): Optional directory for this model

    Returns:
       The path or filename of the tarball for this model
    """
    model_tarball_path = os.path.join(str(result_dir), "{}_model_{}.tar.gz".format(dataset_name, model_uuid))
    return model_tarball_path

# ---------------------------------------------
def build_dataset_name(dataset_key):
    """Return the dataset_name when given a dataset_key. Assumes that the dataset_name is a path and ends with an extension

    Args:
       dataset_key (str): A dataset_key

    Returns:
       The dataset_name which is the base name stripped of extensions
    """
    return os.path.splitext(os.path.basename(dataset_key))[0]

# ******************************************************************************************************************************

class ModelPipeline:
    """Contains methods to load in a dataset, split and featurize the data, fit a model to the train dataset,
    generate predictions for an input dataset, and generate performance metrics for these predictions.

    Attributes:
        Set in __init__:
            params (argparse.Namespace): The argparse.Namespace parameter object

            log (log): The logger

            run_mode (str): A flag determine the mode of model pipeline (eg. training or prediction)

            params.dataset_name (argparse.Namespace): The dataset_name parameter of the dataset

            ds_client (ac.DatastoreClient): the datastore api token to interact with the datastore

            perf_dict (dict): The performance dictionary

            output_dir (str): The parent path of the model directory

            mlmt_client: The mlmt service client

            metric_type (str): Defines the type of metric (e.g. roc_auc_score, r2_score)

        set in train_model or run_predictions:
            run_mode (str): The mode to run the pipeline, set to training

            featurziation (Featurization object): The featurization argument or the featurizatioin created from the
            input parameters

            model_wrapper (ModelWrapper objct): A model wrapper created from the parameters and featurization object.

        set in create_model_metadata:
            model_metadata (dict): The model metadata dictionary that stores the model metrics and metadata

        Set in load_featurize_data
            data (ModelDataset object): A data object that featurizes and splits the dataset
    """

    def __init__(self, params, ds_client=None, mlmt_client=None):
        """Initializes ModelPipeline object.

        Args:
            params (Namespace object): contains all parameter information.

            ds_client: datastore client.

            mlmt_client: model tracker client.

        Side effects:
            Sets the following ModelPipeline attributes:
                params (argparse.Namespace): The argparse.Namespace parameter object

                log (log): The logger

                run_mode (str): A flag determine the mode of model pipeline (eg. training or prediction)

                params.dataset_name (argparse.Namespace): The dataset_name parameter of the dataset

                ds_client (ac.DatastoreClient): the datastore api token to interact with the datastore

                perf_dict (dict): The performance dictionary

                output_dir (str): The parent path of the model directory.

                mlmt_client: The mlmt service

                metric_type (str): Defines the type of metric (e.g. roc_auc_score, r2_score)
        """
        self.params = params
        self.log = logging.getLogger('ATOM')
        self.run_mode = 'training'  # default, can be overridden later
        self.start_time = time.time()

        # Default dataset_name parameter from dataset_key
        if params.dataset_name is None:
            self.params.dataset_name = build_dataset_name(self.params.dataset_key)

        self.ds_client = None
        if params.datastore:
            if ds_client is None:
                self.ds_client = dsf.config_client()
            else:
                self.ds_client = ds_client
        # Check consistency of task parameters
        if type(params.response_cols) == str:
            params.response_cols = [params.response_cols]
        if params.num_model_tasks != len(params.response_cols):
            raise ValueError("num_model_tasks parameter is inconsistent with response_cols")

        if self.params.model_uuid is None:
            self.params.model_uuid = str(uuid.uuid4())

        if self.params.save_results:
            self.mlmt_client = dsf.initialize_model_tracker()

        self.perf_dict = {}
        if self.params.prediction_type == 'regression':
            if self.params.num_model_tasks > 1:
                self.metric_type = 'mean-r2_score'
            else:
                self.metric_type = 'r2_score'
        else:
            if self.params.num_model_tasks > 1:
                self.metric_type = 'mean-roc_auc_score'
            else:
                self.metric_type = 'roc_auc_score'
        if self.params.output_dir is None:
            self.params.output_dir = os.path.join(self.params.result_dir, self.params.dataset_name, '%s_%s_%s_%s' %
                                                  (
                                                    self.params.model_type,
                                                      self.params.featurizer,
                                                      self.params.splitter, self.params.prediction_type),
                                                  self.params.model_uuid)
        if not os.path.isdir(self.params.output_dir):
            os.makedirs(self.params.output_dir, exist_ok=True)
        self.output_dir = self.params.output_dir
        if self.params.model_tarball_path is None:
            self.params.model_tarball_path = build_tarball_name(self.params.dataset_name, self.params.model_uuid, self.params.result_dir)

        # ****************************************************************************************

    def load_featurize_data(self, params=None):
        """Loads the dataset from the datastore or the file system and featurizes it. If we are training
        a new model, split the dataset into training, validation and test sets.

        The data is also split into training, validation, and test sets and saved to the filesystem or datastore.

        Assumes a ModelWrapper object has already been created.

        Args:
            params (Namespace): Optional set of parameters to be used for featurization; by default this function
            uses the parameters used when the pipeline was created.

        Side effects:
            Sets the following attributes of the ModelPipeline
                data (ModelDataset object): A data object that featurizes and splits the dataset
                    data.dataset(dc.DiskDataset): The transformed, featurized, and split dataset
        """
        if params is None:
            params = self.params
        self.data = model_datasets.create_model_dataset(params, self.featurization, self.ds_client)
        self.data.get_featurized_data(params)

        if self.run_mode == 'training':
            # Ignore prevoiusly split if in production mode
            if params.production:
                # if in production mode, make a new split do not load
                self.log.info('Training in production mode. Ignoring '
                    'previous split and creating production split. '
                    'Production split will not be saved.')
                self.data.split_dataset()
            elif not (params.previously_split and self.data.load_presplit_dataset()):
                self.data.split_dataset()
                self.data.save_split_dataset()
            if self.data.params.prediction_type == 'classification':
                self.data._validate_classification_dataset()
        # We now create transformers after splitting, to allow for the case where the transformer
        # is fitted to the training data only. The transformers are then applied to the training,
        # validation and test sets separately.
        if not params.split_only:
            self.model_wrapper.create_transformers(self.data)
        else:
            self.run_mode = ''

        if self.run_mode == 'training':
            for i, (train, valid) in enumerate(self.data.train_valid_dsets):
                train = self.model_wrapper.transform_dataset(train)
                valid = self.model_wrapper.transform_dataset(valid)
                self.data.train_valid_dsets[i] = (train, valid)
            self.data.test_dset = self.model_wrapper.transform_dataset(self.data.test_dset)

        # ****************************************************************************************

    def create_model_metadata(self):
        """Initializes a data structure describing the current model, to be saved in the model zoo.
        This should include everything necessary to reproduce a model run.

        Side effects:
            Sets self.model_metadata (dictionary): A dictionary of the model metadata required to recreate the model.
            Also contains metadata about the generating dataset.
        """

        if self.params.datastore:
            dataset_metadata = dsf.get_keyval(dataset_key=self.params.dataset_key, bucket=self.params.bucket)
        else:
            dataset_metadata = {}
        if 'dataset_hash' not in self.params:
            self.params.dataset_hash=None

        train_dset_data = dict(
            datastore=self.params.datastore,
            dataset_key=self.params.dataset_key,
            bucket=self.params.bucket,
            dataset_oid=self.data.dataset_oid,
            dataset_hash=self.params.dataset_hash,
            id_col=self.params.id_col,
            smiles_col=self.params.smiles_col,
            response_cols=self.params.response_cols,
            feature_transform_type=self.params.feature_transform_type,
            response_transform_type=self.params.response_transform_type,
            weight_transform_type=self.params.weight_transform_type,
            external_export_parameters=dict(
                result_dir=self.params.result_dir),
            dataset_metadata=dataset_metadata,
            production=self.params.production
        )

        model_params = dict(
            model_bucket=self.params.model_bucket,
            system=self.params.system,
            model_type=self.params.model_type,
            featurizer=self.params.featurizer,
            prediction_type=self.params.prediction_type,
            model_choice_score_type=self.params.model_choice_score_type,
            num_model_tasks=self.params.num_model_tasks,
            class_number=self.params.class_number,
            transformers=self.params.transformers,
            transformer_key=self.params.transformer_key,
            transformer_bucket=self.params.transformer_bucket,
            transformer_oid=self.params.transformer_oid,
            uncertainty=self.params.uncertainty,
            time_generated=time.time(),
            save_results=self.params.save_results,
            hyperparam_uuid=self.params.hyperparam_uuid,
            ampl_version=mu.get_ampl_version()
        )

        splitting_metadata = self.data.get_split_metadata()
        model_metadata = dict(
            model_uuid=self.params.model_uuid,
            time_built=time.time(),
            model_parameters=model_params,
            training_dataset=train_dset_data,
            splitting_parameters=splitting_metadata
        )

        model_spec_metadata = self.model_wrapper.get_model_specific_metadata()
        for key, data in model_spec_metadata.items():
            model_metadata[key] = data
        feature_specific_metadata = self.data.featurization.get_feature_specific_metadata(self.params)
        for key, data in feature_specific_metadata.items():
            model_metadata[key] = data
        for key, data in trans.get_transformer_specific_metadata(self.params).items():
            model_metadata[key] = data

        self.model_metadata = model_metadata

    # ****************************************************************************************
    def save_model_metadata(self, retries=5, sleep_sec=60):
        """Saves the data needed to reload the model in the model tracker DB or in a local tarball file.

        Inserts the model metadata into the model tracker DB, if self.params.save_results is True.
        Otherwise, saves the model metadata to a local .json file. Generates a gzipped tar archive
        containing the metadata file, the transformer parameters and the model checkpoint files, and
        saves it in the datastore or the filesystem according to the value of save_results.

        Args:
            retries (int): Number of times to retry saving to model tracker DB.

            sleep_sec (int): Number of seconds to sleep between retries, if saving to model tracker DB.

        Side effects:
            Saves the model metadata and parameters into the model tracker DB or a local tarball file.
        """

        # Dump the model parameters and metadata to a JSON file
        out_file = os.path.join(self.output_dir, 'model_metadata.json')

        with open(out_file, 'w') as out:
            json.dump(self.model_metadata, out, sort_keys=True, indent=4, separators=(',', ': '))
            out.write("\n")

        if self.params.save_results:
            # Model tracker saves the model state and metadata in the datastore as well as saving the metadata
            # in the model zoo.
            retry = True
            i = 0
            while retry:
                if i < retries:
                    # TODO: Try to distinguish unrecoverable exceptions (e.g., model tracker is down) from ones for
                    # which retrying is worthwhile.
                    try:
                        trkr.save_model(self, collection_name=self.params.collection_name)
                        # Best model needs to be reloaded for predictions, so does not work to remove best_model_dir
                        retry = False
                    except:
                        raise
                        #self.log.warning("Need to sleep and retry saving model")
                        #time.sleep(sleep_sec)
                        #i += 1
                else:
                    retry = False
        else:
            # If not using the model tracker, save the model state and metadata in a tarball in the filesystem
            trkr.save_model_tarball(self.output_dir, self.params.model_tarball_path)
        self.model_wrapper._clean_up_excess_files(self.model_wrapper.model_dir)

    # ****************************************************************************************
    def create_prediction_metadata(self, prediction_results):
        """Initializes a data structure to hold performance metrics from a model run on a new dataset,
        to be stored in the model tracker DB. Note that this isn't used
        for the training run metadata; the training_metrics section is created by the train_model() function.

        Returns:
            prediction_metadata (dict): A dictionary of the metadata for a model run on a new dataset.
        """
        if self.params.datastore:
            dataset_metadata = dsf.get_keyval(dataset_key=self.params.dataset_key, bucket=self.params.bucket)
        else:
            dataset_metadata = {}
        prediction_metadata = dict(
            metrics_type='prediction',
            model_uuid=self.params.model_uuid,
            time_run=time.time(),
            dataset_key=self.params.dataset_key,
            bucket=self.params.bucket,
            dataset_oid=self.data.dataset_oid,
            id_col=self.params.id_col,
            smiles_col=self.params.smiles_col,
            response_cols=self.params.response_cols,
            prediction_results=prediction_results,
            dataset_metadata=dataset_metadata
        )
        return prediction_metadata

    # ****************************************************************************************

    def get_metrics(self):
        """Retrieve the model performance metrics from any previous training and prediction runs
        from the model tracker
        """
        if self.params.save_results:
            return list(trkr.get_metrics(self, collection_name=self.params.collection_name))
            metrics = self.mlmt_client.get_model_metrics(collection_name=self.params.collection_name,
                                                         model_uuid=self.params.model_uuid).result()
            return metrics
        else:
            # TODO: Eventually, may want to allow reading metrics from the JSON files saved by
            # save_metrics(), in order to support installations without the model tracker.
            self.log.warning("ModelPipeline.get_metrics() requires params.save_results = True")
            return None

    # ****************************************************************************************

    def save_metrics(self, model_metrics, prefix=None, retries=5, sleep_sec=60):
        """Saves the given model_metrics dictionary to a JSON file on disk, and also to the model tracker
        database if we're using it.

        If writing to disk, outputs to a JSON file <prefix>_model_metrics.json in the current output directory.

        Args:
            model_metrics (dict or list): Either a dictionary containing the model performance metrics, or a
            list of dictionaries with metrics for each training label and subset.

            prefix (str): An optional prefix to include in the JSON filename

            retries (int): Number of retries to save to model tracker DB, if save_results is True.

            sleep_sec (int): Number of seconds to sleep between retries.

        Side effects:
            Saves the model_metrics dictionary to the model tracker database, or writes out a .json file
        """

        # First save the metrics to disk
        if prefix is None:
            out_file = os.path.join(self.output_dir, 'model_metrics.json')
        else:
            out_file = os.path.join(self.output_dir, '%s_model_metrics.json' % prefix)

        with open(out_file, 'w') as out:
            json.dump(model_metrics, out, sort_keys=True, indent=4, separators=(',', ': '))
            out.write("\n")

        if self.params.save_results:
            if type(model_metrics) != list:
                model_metrics = [model_metrics]
            for metrics in model_metrics:
                retry = True
                i = 0
                while retry:
                    if i < retries:
                        try:
                            self.mlmt_client.save_metrics(collection_name=self.params.collection_name,
                                                        model_uuid=metrics['model_uuid'],
                                                        model_metrics=metrics)
                            retry = False
                        except:
                            raise
                            # TODO: uncomment when debugged
                            # TODO: Need to distinguish between "temporary" exceptions that justify
                            # retries and longer-term exceptions indicating that the model tracker server
                            # is down.
                            #self.log.warning("Need to sleep and retry saving metrics")
                            #time.sleep(sleep_sec)
                            #i += 1
                    else:
                        retry = False

    # ****************************************************************************************

    def split_dataset(self, featurization=None):
        """Load, featurize and split the dataset according to the current model parameter settings,
        but don't actually train a model. Returns the split_uuid for the dataset split.

        Args:
            featurization (Featurization object): An optional featurization object.

        Returns:
            split_uuid (str): The unique identifier for the dataset split.
        """

        self.run_mode = 'training'
        self.params.split_only = True
        self.params.previously_split = False
        if featurization is None:
            featurization = feat.create_featurization(self.params)
        self.featurization = featurization
        self.load_featurize_data()
        return self.data.split_uuid


    # ****************************************************************************************

    def train_model(self, featurization=None):
        """Build model described by self.params on the training dataset described by self.params.

        Generate predictions for the training, validation, and test datasets, and save the predictions and
        performance metrics in the model results DB or in a JSON file.

        Args:
            featurization (Featurization object): An optional featurization object for creating models on a
            prefeaturized dataset

        Side effects:
            Sets the following attributes of the ModelPipeline object
                run_mode (str): The mode to run the pipeline, set to training

                featurization (Featurization object): The featurization argument or the featurization created from the
                input parameters

                model_wrapper (ModelWrapper objct): A model wrapper created from the parameters and featurization object.

                model_metadata (dict): The model metadata dictionary that stores the model metrics and metadata
        """

        self.run_mode = 'training'
        if self.params.model_type == "hybrid":
            if self.params.featurizer in ["graphconv"]:
                raise Exception("Hybrid model doesn't support GraphConv featurizer now.")
            if len(self.params.response_cols) < 2:
                raise Exception("The dataset of a hybrid model should have two response columns, one for activities, one for concentrations.")
        if featurization is None:
            featurization = feat.create_featurization(self.params)
        self.featurization = featurization

        ## create model wrapper if not split_only
        if not self.params.split_only:
            self.model_wrapper = model_wrapper.create_model_wrapper(self.params, self.featurization, self.ds_client)
            self.model_wrapper.setup_model_dirs()

        self.load_featurize_data()

        ## return if split only
        if self.params.split_only:
            return

        self.model_wrapper.train(self)

        # Create the metadata for the trained model
        self.create_model_metadata()
        # Save the performance metrics for each training data subset, for the best epoch
        training_metrics = []
        for label in ['best']:
            for subset in ['train', 'valid', 'test']:
                training_dict = dict(
                    metrics_type='training',
                    label=label,
                    subset=subset)
                training_dict['prediction_results'] = self.model_wrapper.get_pred_results(subset, label)
                training_metrics.append(training_dict)

        # Save the model metrics separately
        for training_dict in training_metrics:
            training_dict['model_uuid'] = self.params.model_uuid
            training_dict['time_run'] = time.time()
            training_dict['input_dataset'] = self.model_metadata['training_dataset']
        self.save_metrics(training_metrics)

        # Save the model metadata in the model tracker or the filesystem
        self.model_metadata['training_metrics'] = training_metrics
        self.save_model_metadata()
        self.orig_params = self.params


    # ****************************************************************************************
    def run_predictions(self, featurization=None):
        """Instantiate a previously trained model, and use it to run predictions on a new dataset.

        Generate predictions for a specified dataset, and save the predictions and performance
        metrics in the model results DB or in a JSON file.

        Args:
            featurization (Featurization Object): An optional featurization object for creating the model wrappr

        Side effects:
            Sets the following attributes of ModelPipeline:
                run_mode (str): The mode to run the pipeline, set to prediction

                featurization (Featurization object): The featurization argument or the featurization created from the
                input parameters

                model_wrapper (ModelWrapper object): A model wrapper created from the parameters and featurization object.
        """

        self.run_mode = 'prediction'
        if featurization is None:
            featurization = feat.create_featurization(self.params)
        self.featurization = featurization
        # Load the dataset to run predictions on and featurize it
        self.load_featurize_data()

        # Run predictions on the full dataset
        pred_results = self.model_wrapper.get_full_dataset_pred_results(self.data)

        # Map the predictions, and metrics if requested, to the dictionary format used by
        # the model tracker
        prediction_metadata = self.create_prediction_metadata(pred_results)

        # Get the metrics from previous prediction runs, if any, and append the new results to them
        # in the model tracker DB
        model_metrics = dict(
            model_uuid=self.params.model_uuid,
            metrics_type='prediction'
        )
        model_metrics.update(prediction_metadata)
        self.save_metrics(model_metrics, 'prediction_%s' % self.params.dataset_name)

    # ****************************************************************************************
    def calc_train_dset_pair_dis(self, metric="euclidean"):
        """Calculate the pairwise distance for training set compound feature vectors, needed for AD calculation."""
        
        self.featurization = self.model_wrapper.featurization
        self.load_featurize_data()
        if len(self.data.train_valid_dsets) > 1:
            # combine train and valid set for k-fold cv models
            train_data = np.concatenate((self.data.train_valid_dsets[0][0].X, self.data.train_valid_dsets[0][1].X))
        else:
            train_data = self.data.train_valid_dsets[0][0].X
        self.train_pair_dis = pairwise_distances(X=train_data, metric=metric)
        self.train_pair_dis_metric = metric
    
    # ****************************************************************************************
    def predict_on_dataframe(self, dset_df, is_featurized=False, contains_responses=False, AD_method=None, k=5, dist_metric="euclidean"):
        """DEPRECATED
        Call predict_full_dataset instead.
        """
        self.log.warning("predict_on_dataframe is deprecated. Please call predict_full_dataset instead.")
        result_df = self.predict_full_dataset(dset_df, is_featurized=is_featurized, 
                contains_responses=contains_responses, AD_method=AD_method, k=k,
                dist_metric=dist_metric)

	    # Inside predict_full_dataset, prediction columns are generated using something like:
        # for i, colname in enumerate(self.params.response_cols):
        #     result_df['%s_pred'%colname] = preds[:,i,0]
        # predict_on_dataframe was only meant to handle single task models and so output
        # columns were not prefixed with the response_col. Thus we need to remove the prefix
        # for backwards compatibility
        if len(self.params.response_cols)==1:
            # currently the only columns that could have a response_col prefix
            suffixes = ['pred', 'std', 'actual', 'prob']
            rename_map = {}
            colname = self.params.response_cols[0]
            for suff in suffixes:
                for c in result_df.columns:
                    if c.startswith('%s_%s'%(colname, suff)):
                        rename_map[c] = c[len(colname+'_'):] # chop off response_col_ prefix

            # rename columns for backwards compatibility
            result_df.rename(columns=rename_map, inplace=True)

        return result_df

    # ****************************************************************************************
    def predict_on_smiles(self, smiles, verbose=False, AD_method=None, k=5, dist_metric="euclidean"):
        """Compute predicted responses from a pretrained model on a set of compounds given as a list of SMILES strings.

        Args:
            smiles (list): A list containting valid SMILES strings

            verbose (boolean): A switch for disabling informational messages

            AD_method (str or None): Method to use to compute applicability domain (AD) index; may be
            'z_score', 'local_density' or None (the default). With the default value, AD indices
            will not be calculated.

            k (int): Number of nearest neighbors of each training data point used to evaluate the AD index.

            dist_metric (str): Metric used to compute distances between feature vectors for AD index calculation.
            Valid values are 'cityblock', 'cosine', 'euclidean', 'jaccard', and 'manhattan'. If binary
            features such as fingerprints are used in model, 'jaccard' (equivalent to Tanimoto distance) may
            be a better choice than the other metrics which operate on continuous features.

        Returns:
            res (DataFrame): Data frame indexed by compound IDs containing a column of SMILES
            strings, with additional columns containing the predicted values for each response variable.
            If the model was trained to predict uncertainties, the returned data frame will also
            include standard deviation columns (named <response_col>_std) for each response variable.
            The result data frame may not include all the compounds in the input dataset, because
            the featurizer may not be able to featurize all of them.
        """

        if not verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
            logger = logging.getLogger('ATOM')
            logger.setLevel(logging.CRITICAL)
            sys.stdout = io.StringIO()
            import warnings
            warnings.simplefilter("ignore")

        if len(self.params.response_cols) > 1:
            raise Exception('Currently only single task models supported')
        else:
            task = self.params.response_cols[0]

        df = pd.DataFrame({'compound_id': np.linspace(0, len(smiles) - 1, len(smiles), dtype=int),
                        self.params.smiles_col: smiles,
                        task: np.zeros(len(smiles))})
        res = self.predict_on_dataframe(df, AD_method=AD_method, k=k, dist_metric=dist_metric)

        sys.stdout = sys.__stdout__

        return res

    # ****************************************************************************************
    def predict_full_dataset(self, dset_df, is_featurized=False, contains_responses=False, dset_params=None, AD_method=None, k=5, dist_metric="euclidean",
                             max_train_records_for_AD=1000):
        """Compute predicted responses from a pretrained model on a set of compounds listed in
        a data frame. The data frame should contain, at minimum, a column of compound IDs; if
        SMILES strings are needed to compute features, they should be provided as well. Feature
        columns may be provided as well. If response columns are included in the input, they will
        be included in the output as well to facilitate performance metric calculations.

        This function is similar to predict_on_dataframe, except that it supports multitask models,
        and includes class probabilities in the output for classifier models.

        Args:
            dset_df (DataFrame): A data frame containing compound IDs (if the compounds are to be
            featurized using descriptors) and/or SMILES strings (if the compounds are to be
            featurized using ECFP fingerprints or graph convolution) and/or precomputed features.
            The column names for the compound ID and SMILES columns should match id_col and smiles_col,
            respectively, in the model parameters.

            is_featurized (bool): True if dset_df contains precomputed feature columns. If so,
            dset_df must contain *all* of the feature columns defined by the featurizer that was
            used when the model was trained.

            contains_responses (bool): True if dataframe contains response values

            dset_params (Namespace):  Parameters used to interpret dataset, including id_col, smiles_col,
            and optionally, response_cols. If not provided, id_col, smiles_col and response_cols are
            assumed to be same as in the pretrained model.

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
            result_df (DataFrame): Data frame indexed by compound IDs containing a column of SMILES
            strings, with additional columns containing the predicted values for each response variable.
            If the model was trained to predict uncertainties, the returned data frame will also
            include standard deviation columns (named <response_col>_std) for each response variable.
            The result data frame may not include all the compounds in the input dataset, because
            the featurizer may not be able to featurize all of them.
        """

        self.run_mode = 'prediction'
        self.featurization = self.model_wrapper.featurization

        # Change the dataset ID, SMILES and response columns to match the ones in the current model
        dset_df = dset_df.copy()
        if dset_params is not None:
            coldict = {
                        dset_params.id_col: self.params.id_col,
                        dset_params.smiles_col: self.params.smiles_col}
            if contains_responses and (set(dset_params.response_cols) != set(self.params.response_cols)):
                for i, col in enumerate(dset_params.response_cols):
                    coldict[col] = self.params.response_cols[i]
            dset_df = dset_df.rename(columns=coldict)

        # assign unique ids to each row
        old_ids = dset_df[self.params.id_col].values
        new_ids = [str(i) for i in range(len(dset_df))]
        id_map = dict([(i, id) for i, id in zip(new_ids, old_ids)])
        dset_df[self.params.id_col] = new_ids

        self.data = model_datasets.create_minimal_dataset(self.params, self.featurization, contains_responses)

        if not self.data.get_dataset_tasks(dset_df):
            # Shouldn't happen
            raise Exception("response_cols missing from model params")
        # Get features for each compound and construct a DeepChem Dataset from them
        self.data.get_featurized_data(dset_df, is_featurized)
        # Transform the features and responses if needed
        self.data.dataset = self.model_wrapper.transform_dataset(self.data.dataset)

        # Note that at this point, the dataset may contain fewer rows than the input. Typically this happens because
        # of invalid SMILES strings. Remove any rows from the input dataframe corresponding to SMILES strings that were
        # dropped.
        dset_df = dset_df[dset_df[self.params.id_col].isin(self.data.dataset.ids.tolist())]

        # Get the predictions and standard deviations, if calculated, as numpy arrays
        preds, stds = self.model_wrapper.generate_predictions(self.data.dataset)
        result_df = pd.DataFrame({self.params.id_col: self.data.attr.index.values,
                                  self.params.smiles_col: self.data.attr[self.params.smiles_col].values})

        if self.params.model_type != "hybrid":
            if contains_responses:
                for i, colname in enumerate(self.params.response_cols):
                    result_df["%s_actual" % colname] = self.data.vals[:,i]
            for i, colname in enumerate(self.params.response_cols):
                if self.params.prediction_type == 'regression':
                    result_df["%s_pred" % colname] = preds[:,i,0]
                else:
                    class_probs = preds[:,i,:]
                    nclass = preds.shape[2]
                    if nclass == 2:
                        result_df["%s_prob" % colname] = class_probs[:,1]
                    else:
                        for k in range(nclass):
                            result_df["%s_prob_%d" % (colname, k)] = class_probs[:,k]
                    result_df["%s_pred" % colname] = np.argmax(class_probs, axis=1)
            if self.params.uncertainty and self.params.prediction_type == 'regression':
                for i, colname in enumerate(self.params.response_cols):
                    std_colname = '%s_std' % colname
                    result_df[std_colname] = stds[:,i,0]
        else:
            # hybrid model should handled differently
            if contains_responses:
                result_df["actual_activity"] = self.data.vals[:, 0]
                result_df["concentration"] = self.data.vals[:, 1]
            result_df["pred"] = preds[:, 0]

        if AD_method is not None:
            # Calculate applicability domain index

            if self.featurization.feat_type == "graphconv":
                # For graphconv models, compute embeddings and treat them as features
                pred_data = self.predict_embedding(dset_df, dset_params=dset_params)
            else:
                pred_data = copy.deepcopy(self.data.dataset.X)

            try:
                if not hasattr(self, 'featurized_train_data'):
                    self.log.debug("Featurizing training data for AD calculation.")
                    self.run_mode = 'training'
                    # If training data is too big to compute distances in a reasonable time, use a sample of the data
                    train_data_params = copy.deepcopy(self.orig_params)
                    train_data_params.max_dataset_rows = max_train_records_for_AD

                    self.load_featurize_data(params=train_data_params)
                    self.run_mode = 'prediction'
                    if len(self.data.train_valid_dsets) > 1:
                        # combine train and valid set for k-fold CV models
                        train_X = np.concatenate((self.data.train_valid_dsets[0][0].X, self.data.train_valid_dsets[0][1].X))
                    else:
                        train_X = self.data.train_valid_dsets[0][0].X
    
                    if self.featurization.feat_type == "graphconv":
                        self.log.debug("Computing training data embeddings for AD calculation.")
                        train_dset = dc.data.NumpyDataset(train_X)
                        self.featurized_train_data = self.model_wrapper.generate_embeddings(train_dset)
                    else:
                        self.featurized_train_data = train_X

                if not hasattr(self, "train_pair_dis") or not hasattr(self, "train_pair_dis_metric") or self.train_pair_dis_metric != dist_metric:
                    self.train_pair_dis = pairwise_distances(X=self.featurized_train_data, metric=dist_metric)
                    self.train_pair_dis_metric = dist_metric

                self.log.debug("Calculating AD index.")


                if AD_method == "local_density":
                    result_df["AD_index"] = calc_AD_kmean_local_density(self.featurized_train_data, pred_data, k, train_dset_pair_distance=self.train_pair_dis, dist_metric=dist_metric)
                else:
                    result_df["AD_index"] = calc_AD_kmean_dist(self.featurized_train_data, pred_data, k, train_dset_pair_distance=self.train_pair_dis, dist_metric=dist_metric)

            except:
                self.log.warning("AD index calculation failed")
                # xxx re-raise for debugging
                raise

        # insert any missing ids
        missing_ids = list(set(new_ids).difference(result_df[self.params.id_col]))
        if len(missing_ids) > 0:
            missing_df = pd.DataFrame({self.params.id_col: missing_ids})
            result_df = pd.concat([result_df, missing_df], ignore_index=True)
        # sort in ascending order, recovering the original order, keeping in mind that string representations
        # of ints don't sort in the same order as the corresponding ints.
        result_df['original_sort_order'] = [int(s) for s in result_df[self.params.id_col].values]
        result_df.sort_values(by='original_sort_order', ascending=True, inplace=True)
        result_df = result_df.drop(columns=['original_sort_order'])
        # map back to original id values
        result_df[self.params.id_col] = result_df[self.params.id_col].map(id_map)

        return result_df

    # ****************************************************************************************
    def predict_embedding(self, dset_df, dset_params=None):
        """Compute embeddings from a pretrained model on a set of compounds listed in a data frame. The data
        frame should contain, at minimum, a column of compound IDs and a column of SMILES strings.
        """

        self.run_mode = 'prediction'
        self.featurization = self.model_wrapper.featurization

        # Change the dataset ID, SMILES and response columns to match the ones in the current model
        dset_df = dset_df.copy()
        if dset_params is not None:
            coldict = {
                        dset_params.id_col: self.params.id_col,
                        dset_params.smiles_col: self.params.smiles_col}
            dset_df = dset_df.rename(columns=coldict)

        self.data = model_datasets.create_minimal_dataset(self.params, self.featurization)
        self.data.get_featurized_data(dset_df, is_featurized=False)
        # Not sure the following is necessary
        self.data.dataset = self.model_wrapper.transform_dataset(self.data.dataset)

        # Get the embeddings as a numpy array
        embeddings = self.model_wrapper.generate_embeddings(self.data.dataset)
        # Truncate the embeddings array to the length of the input dataset. The array returned by the DeepChem 
        # predict_embedding function is padded to multiples of the batch size.
        embeddings = embeddings[:len(dset_df),:]

        return embeddings


# ****************************************************************************************
def run_models(params, shared_featurization=None, generator=False):
    """Query the model tracker for models matching the criteria in params.model_filter. Run
    predictions with each model using the dataset specified by the remaining parameters.

    Args:
        params (Namespace): Parsed parameters

        shared_featurization (Featurization): Object to map compounds to features, shared across models.
        User is responsible for ensuring that shared_featurization is compatible with all matching models.

        generator (bool): True if run as a generator
    """
    mlmt_client = dsf.initialize_model_tracker()
    ds_client = dsf.config_client()
    log = logging.getLogger('ATOM')

    exclude_fields = [
        "training_metrics",
        "time_built",
        "training_dataset.dataset_metadata"
    ]

    query_params = {
        'match_metadata': params.model_filter
    }

    metadata_iter = mlmt_client.get_models(
        collection_name=params.collection_name,
        query_params=query_params,
        exclude_fields=exclude_fields,
        count=True
    )

    model_count = next(metadata_iter)

    if not model_count:
        log.error("No matching models returned")
        return

    for metadata_dict in metadata_iter:
        model_uuid = metadata_dict['model_uuid']

        log.info("Got metadata for model UUID %s" % model_uuid)

        # Parse the saved model metadata to obtain the parameters used to train the model
        model_params = parse.wrapper(metadata_dict)

        # Override selected model training data parameters with parameters for current dataset

        model_params.model_uuid = model_uuid
        model_params.collection_name = params.collection_name
        model_params.datastore = True
        model_params.save_results = True
        model_params.dataset_key = params.dataset_key
        model_params.bucket = params.bucket
        model_params.dataset_oid = params.dataset_oid
        model_params.system = params.system
        model_params.id_col = params.id_col
        model_params.smiles_col = params.smiles_col
        model_params.result_dir = params.result_dir
        model_params.model_filter = params.model_filter

        # Create a separate output_dir under model_params.result_dir for each model. For lack of a better idea, use the model UUID
        # to name the output dir, to ensure uniqueness.
        model_params.output_dir = os.path.join(params.result_dir, model_uuid)

        # Allow descriptor featurizer to use a different descriptor table than was used for the training data.
        # This could be needed e.g. when a model was trained with GSK compounds and tested with ChEMBL data.
        model_params.descriptor_key = params.descriptor_key
        model_params.descriptor_bucket = params.descriptor_bucket
        model_params.descriptor_oid = params.descriptor_oid

        # If there is no shared featurization object, create one for this model
        if shared_featurization is None:
            featurization = feat.create_featurization(model_params)
        else:
            featurization = shared_featurization


        # Create a ModelPipeline object
        pipeline = ModelPipeline(model_params, ds_client, mlmt_client)

        # Create the ModelWrapper object.
        pipeline.model_wrapper = model_wrapper.create_model_wrapper(pipeline.params, featurization,
                                                                    pipeline.ds_client)

        # Get the tarball containing the saved model from the datastore, and extract it into model_dir.
        model_dataset_oid = metadata_dict['model_parameters']['model_dataset_oid']
        # TODO: Should we catch exceptions from retrieve_dataset_by_dataset_oid, or let them propagate?
        model_dir = dsf.retrieve_dataset_by_dataset_oid(model_dataset_oid, client=ds_client, return_metadata=False,
                                                        nrows=None, print_metadata=False, sep=False,
                                                        tarpath=pipeline.model_wrapper.model_dir)
        pipeline.log.info("Extracted model tarball to %s" % model_dir)

        # If that worked, reload the saved model training state
        pipeline.model_wrapper.reload_model(pipeline.model_wrapper.model_dir)

        # Run predictions on the specified dataset
        pipeline.run_predictions(featurization)

        # Return the pipeline to the calling function, if run as a generator
        if generator:
            yield pipeline


# ****************************************************************************************
def regenerate_results(result_dir, params=None, metadata_dict=None, shared_featurization=None, system='twintron-blue'):
    """Query the model tracker for models matching the criteria in params.model_filter. Run
    predictions with each model using the dataset specified by the remaining parameters.

    Args:
        result_dir (str): Parent of directory where result files will be written

        params (Namespace): Parsed parameters

        metadata_dict (dict): Model metadata

        shared_featurization (Featurization): Object to map compounds to features, shared across models.
        User is responsible for ensuring that shared_featurization is compatible with all matching models.

        system (str): System name

    Returns:
        result_dict (dict): Results from predictions
    """

    mlmt_client = dsf.initialize_model_tracker()
    ds_client = dsf.config_client()
    log = logging.getLogger('ATOM')

    if metadata_dict is None:
        if params is None:
            log.error("Must either provide params or metadata_dict")
            return
        metadata_dict = trkr.get_metadata_by_uuid(params.model_uuid,
                                                  collection_name=params.collection_name)
        if metadata_dict is None:
            log.error("No matching models returned")
            return

    # Parse the saved model metadata to obtain the parameters used to train the model
    model_params = parse.wrapper(metadata_dict)
    model_params.model_uuid = metadata_dict['model_uuid']
    model_params.datastore = True

    dset_df = model_datasets.create_split_dataset_from_metadata(model_params, ds_client)
    test_df = dset_df[dset_df.subset == 'test']

    model_uuid = model_params.model_uuid

    log.info("Got metadata for model UUID %s" % model_uuid)

    model_params.result_dir = result_dir

    # Create a separate output_dir under model_params.result_dir for each model. For lack of a better idea, use the model UUID
    # to name the output dir, to ensure uniqueness.
    model_params.output_dir = os.path.join(model_params.result_dir, model_uuid)

    # Allow descriptor featurizer to use a different descriptor table than was used for the training data.
    # This could be needed e.g. when a model was trained with GSK compounds and tested with ChEMBL data, or
    # when running a model that was trained on LC on a non-LC system.
    model_params.system = system

    # Create a ModelPipeline object
    pipeline = ModelPipeline(model_params, ds_client, mlmt_client)

    # If there is no shared featurization object, create one for this model
    if shared_featurization is None:
        featurization = feat.create_featurization(model_params)
    else:
        featurization = shared_featurization

    log.info("Featurization = %s" % str(featurization))
    # Create the ModelWrapper object.

    pipeline.model_wrapper = model_wrapper.create_model_wrapper(pipeline.params, featurization,
                                                                pipeline.ds_client)
    # Get the tarball containing the saved model from the datastore, and extract it into model_dir (old format)
    # or output_dir (new format) according to the format of the tarball contents.

    extract_dir = trkr.extract_datastore_model_tarball(model_uuid, model_params.model_bucket, model_params.output_dir, 
                                         pipeline.model_wrapper.model_dir)

    # If that worked, reload the saved model training state

    pipeline.model_wrapper.reload_model(pipeline.model_wrapper.model_dir)
    # Run predictions on the specified dataset
    result_dict = pipeline.predict_on_dataframe(test_df, contains_responses=True)
    result_dict['model_type'] = model_params.model_type
    result_dict['featurizer'] = model_params.featurizer
    result_dict['splitter'] = model_params.splitter
    if 'descriptor_type' in model_params:
        result_dict['descriptor_type'] = model_params.descriptor_type

    return result_dict


# ****************************************************************************************
def create_prediction_pipeline(params, model_uuid, collection_name=None, featurization=None, alt_bucket='CRADA'):
    """Create a ModelPipeline object to be used for running blind predictions on datasets
    where the ground truth is not known, given a pretrained model in the model tracker database.

    Args:
        params (Namespace or dict): A parsed parameters namespace, containing parameters describing how input
        datasets should be processed. If a dictionary is passed, it will be parsed to fill in default values
        and convert it to a Namespace object.

        model_uuid (str): The UUID of a trained model.

        collection_name (str): The collection where the model is stored in the model tracker DB.

        featurization (Featurization): An optional featurization object to be used for featurizing the input data.
        If none is provided, one will be created based on the stored model parameters.

        alt_bucket (str): Alternative bucket to search for model tarball and transformer files, if
        original bucket no longer exists.

    Returns:
        pipeline (ModelPipeline): A pipeline object to be used for making predictions.
    """
    mlmt_client = dsf.initialize_model_tracker()
    ds_client = dsf.config_client()

    if collection_name is None:
        collection_name = trkr.get_model_collection_by_uuid(model_uuid, mlmt_client)

    if type(params) == dict:
        params = parse.wrapper(params)

    metadata_dict = trkr.get_metadata_by_uuid(model_uuid, collection_name=collection_name)
    if not metadata_dict:
        raise Exception("No model found with UUID %s in collection %s" % (model_uuid, collection_name))

    print("Got metadata for model UUID %s" % model_uuid)

    model_ampl_version = metadata_dict['model_parameters']['ampl_version']
    # check the model version to make sure it's compatible with the running ampl version
    mu.check_version_compatible(model_ampl_version)
    # Parse the saved model metadata to obtain the parameters used to train the model
    model_params = parse.wrapper(metadata_dict)
    orig_params = copy.deepcopy(model_params)

    # Override selected model training data parameters with parameters for current dataset

    model_params.model_uuid = model_uuid
    model_params.save_results = True
    model_params.id_col = params.id_col
    model_params.smiles_col = params.smiles_col
    model_params.result_dir = params.result_dir
    model_params.system = params.system


    # Check that buckets where model tarball and transformers were saved still exist. If not, try alt_bucket.
    model_bucket_meta = ds_client.ds_buckets.get_buckets(buckets=[model_params.model_bucket]).result()
    if len(model_bucket_meta) == 0:
        model_params.model_bucket = alt_bucket
    if (model_params.transformer_bucket != model_params.model_bucket):
        trans_bucket_meta = ds_client.ds_buckets.get_buckets(buckets=[model_params.transformer_bucket]).result()
        if len(trans_bucket_meta) == 0:
            model_params.transformer_bucket = alt_bucket
    else:
        if len(model_bucket_meta) == 0:
            model_params.transformer_bucket = alt_bucket

    # Create a separate output_dir under model_params.result_dir for each model. For lack of a better idea, use the model UUID
    # to name the output dir, to ensure uniqueness.
    model_params.output_dir = os.path.join(params.result_dir, model_uuid)

    # Allow using computed_descriptors featurizer for a model trained with the descriptors featurizer, and vice versa
    if (model_params.featurizer == 'descriptors' and params.featurizer == 'computed_descriptors') or (
            model_params.featurizer == 'computed_descriptors' and params.featurizer == 'descriptors'):
        model_params.featurizer = params.featurizer

    # Allow descriptor featurizer to use a different descriptor table than was used for the training data.
    # This could be needed e.g. when a model was trained with GSK compounds and tested with ChEMBL data.
    model_params.descriptor_key = params.descriptor_key
    model_params.descriptor_bucket = params.descriptor_bucket
    model_params.descriptor_oid = params.descriptor_oid

    # If the caller didn't provide a featurization object, create one for this model
    if featurization is None:
        featurization = feat.create_featurization(model_params)

    # Create a ModelPipeline object
    pipeline = ModelPipeline(model_params, ds_client, mlmt_client)
    pipeline.orig_params = orig_params

    # Create the ModelWrapper object.
    pipeline.model_wrapper = model_wrapper.create_model_wrapper(pipeline.params, featurization,
                                                                pipeline.ds_client)

    if params.verbose:
        pipeline.log.setLevel(logging.DEBUG)
    else:
        pipeline.log.setLevel(logging.CRITICAL)

    # Get the tarball containing the saved model from the datastore, and extract it into model_dir or output_dir,
    # depending on what style of tarball it is (old or new respectively)
    extract_dir = trkr.extract_datastore_model_tarball(model_uuid, model_params.model_bucket, model_params.output_dir, 
                                         pipeline.model_wrapper.model_dir)

    if extract_dir == model_params.output_dir:
        # Model came from new style tarball
        pipeline.model_wrapper.model_dir = os.path.join(model_params.output_dir, 'best_model')

    # Reload the saved model training state
    pipeline.model_wrapper.reload_model(pipeline.model_wrapper.model_dir)

    return pipeline


# ****************************************************************************************
def create_prediction_pipeline_from_file(params, reload_dir, model_path=None, model_type='best_model', featurization=None,
                                         verbose=True):
    """Create a ModelPipeline object to be used for running blind predictions on datasets, given a pretrained model stored
    in the filesystem. The model may be stored either as a gzipped tar archive or as a directory.

    Args:
        params (Namespace): A parsed parameters namespace, containing parameters describing how input
        datasets should be processed.

        reload_dir (str): The path to the parent directory containing the various model subdirectories
          (e.g.: '/home/cdsw/model/delaney-processed/delaney-processed/pxc50_NN_graphconv_scaffold_regression/').
        If reload_dir is None, then model_path must be specified. If both are specified, then the tar archive given
        by model_path will be unpacked into reload_dir, possibly overwriting existing files in that directory.

        model_path (str): Path to a gzipped tar archive containing the saved model metadata and parameters. If specified,
        the tar archive is unpacked into reload_dir if that directory is given, or to a temporary directory otherwise.

        model_type (str): Name of the subdirectory in reload_dir or in the tar archive where the trained model state parameters
        should be loaded from.

        featurization (Featurization): An optional featurization object to be used for featurizing the input data.
        If none is provided, one will be created based on the stored model parameters.

    Returns:
        pipeline (ModelPipeline): A pipeline object to be used for making predictions.
    """
    log = logging.getLogger('ATOM')

    # Unpack the model tar archive if one is specified
    if model_path is not None:
        # if mismatch, it will raise an exception
        matched = mu.check_version_compatible(model_path)
        if reload_dir is None:
            # Create a temporary directory
            reload_dir = tempfile.mkdtemp()
        else:
            os.makedirs(reload_dir, exist_ok=True)
        with tarfile.open(model_path, mode='r:gz') as tar:
            futils.safe_extract(tar, path=reload_dir)
    elif reload_dir is None:
        raise ValueError("Either reload_dir or model_path must be specified.")

    # Opens the model_metadata.json file containing the reloaded model parameters
    config_file_path = os.path.join(reload_dir, 'model_metadata.json')
    with open(config_file_path) as f:
        config = json.loads(f.read())
    # Set the transformer_key parameter to point to the transformer pickle file we just extracted
    try:
        has_transformers = config['model_parameters']['transformers']
        if has_transformers:
            config['model_parameters']['transformer_key'] = "%s/transformers.pkl" % reload_dir
    except KeyError:
        pass

    # Parse the saved model metadata to obtain the parameters used to train the model
    model_params = parse.wrapper(config)
    orig_params = copy.deepcopy(model_params)

    # Override selected model training data parameters with parameters for current dataset

    model_params.save_results = False
    model_params.output_dir = reload_dir
    if params is not None:
        model_params.id_col = params.id_col
        model_params.smiles_col = params.smiles_col
        model_params.result_dir = params.result_dir
        model_params.system = params.system
        verbose = params.verbose

        # Allow using computed_descriptors featurizer for a model trained with the descriptors featurizer, and vice versa
        if (model_params.featurizer == 'descriptors' and params.featurizer == 'computed_descriptors') or (
                model_params.featurizer == 'computed_descriptors' and params.featurizer == 'descriptors'):
            model_params.featurizer = params.featurizer

        # Allow descriptor featurizer to use a different descriptor table than was used for the training data.
        # This could be needed e.g. when a model was trained with GSK compounds and tested with ChEMBL data.
        model_params.descriptor_key = params.descriptor_key
        model_params.descriptor_bucket = params.descriptor_bucket
        model_params.descriptor_oid = params.descriptor_oid

    # If the caller didn't provide a featurization object, create one for this model
    if featurization is None:
        featurization = feat.create_featurization(model_params)

    log.info("Featurization = %s" % str(featurization))
    # Create a ModelPipeline object
    pipeline = ModelPipeline(model_params)
    pipeline.orig_params = orig_params

    # Create the ModelWrapper object.
    pipeline.model_wrapper = model_wrapper.create_model_wrapper(pipeline.params, featurization)

    if verbose:
        pipeline.log.setLevel(logging.DEBUG)
    else:
        pipeline.log.setLevel(logging.CRITICAL)

    # Reload the saved model training state
    model_dir = os.path.join(reload_dir, model_type)

    # If that worked, reload the saved model training state
    pipeline.model_wrapper.reload_model(model_dir)

    return pipeline


# ****************************************************************************************

def load_from_tracker(model_uuid, collection_name=None, client=None, verbose=False, alt_bucket='CRADA'):
    """DEPRECATED. Use the function create_prediction_pipeline() directly, or use the higher-level function
    predict_from_model.predict_from_tracker_model().

    Create a ModelPipeline object using the metadata in the  model tracker.

    Args:
        model_uuid (str): The UUID of a trained model.

        collection_name (str): The collection where the model is stored in the model tracker DB.

        client : Ignored, for backward compatibility only

        verbose (bool): A switch for disabling informational messages

        alt_bucket (str): Alternative bucket to search for model tarball and transformer files, if
        original bucket no longer exists.

    Returns:
        tuple of:
            pipeline (ModelPipeline): A pipeline object to be used for making predictions.

            pparams (Namespace): Parsed parameter namespace from the requested model.
    """

    logger = logging.getLogger('ATOM')
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        logger.setLevel(logging.CRITICAL)
        sys.stdout = io.StringIO()
        import warnings
        warnings.simplefilter("ignore")

    if collection_name is None:
        collection_name = trkr.get_model_collection_by_uuid(model_uuid)

    metadata_dict = trkr.get_metadata_by_uuid(model_uuid, collection_name=collection_name)
    if not metadata_dict:
        raise Exception("No model found with UUID %s in collection %s" % (model_uuid, collection_name))

    logger.info("Got metadata for model UUID %s" % model_uuid)

    # Parse the saved model metadata to obtain the parameters used to train the model
    pparams = parse.wrapper(metadata_dict)
    # pparams.uncertainty   = False
    pparams.verbose = verbose
    pparams.result_dir = tempfile.mkdtemp()  # Redirect the untaring of the model to a temporary directory

    model = create_prediction_pipeline(pparams, model_uuid, collection_name, alt_bucket=alt_bucket)
    # model.params.uncertainty = False

    if not verbose:
        sys.stdout = sys.__stdout__

    return (model, pparams)


# ****************************************************************************************
def ensemble_predict(model_uuids, collections, dset_df, labels=None, dset_params=None, splitters=None,
                     mt_client=None, aggregate="mean", contains_responses=False):
    """Load a series of pretrained models and predict responses with each model; then aggregate
    the predicted responses into one prediction per compound.

    Args:
        model_uuids (iterable of str): Sequence of UUIDs of trained models.

        collections (str or iterable of str): The collection(s) where the models are stored in the
        model tracker DB. If a single string, the same collection is assumed to contain all the models.
        Otherwise, collections should be of the same length as model_uuids.

        dset_df (DataFrame): Dataset to perform predictions on. Should contain compound IDs and
        SMILES strings. May contain features.

        labels (iterable of str): Optional suffixes for model-specific prediction column names.
        If not provided, the columns are labeled 'pred_<uuid>' where <uuid> is the model UUID.

        dset_params (Namespace): Parameters used to interpret dataset, including id_col and smiles_col.
        If not provided, id_col and smiles_col are assumed to be same as in the pretrained model and
        the same for all models.

        mt_client: Ignored, for backward compatibility only.

        aggregate (str): Method to be used to combine predictions.

    Returns:
        pred_df (DataFrame): Table with predicted responses from each model, plus the ensemble prediction.

    """

    # Get the singleton MLMTClient instance
    mlmt_client = dsf.initialize_model_tracker()
    log = logging.getLogger('ATOM')

    pred_df = None

    if type(collections) == str:
        collections = [collections] * len(model_uuids)

    if labels is None:
        labels = model_uuids

    ok_labels = []
    for i, (model_uuid, collection_name, label) in enumerate(zip(model_uuids, collections, labels)):
        log.info("Loading model %s from collection %s" % (model_uuid, collection_name))
        metadata_dict = trkr.get_metadata_by_uuid(model_uuid, collection_name=collection_name)
        if not metadata_dict:
            raise Exception("No model found with UUID %s in collection %s" % (model_uuid, collection_name))

        log.info("Got metadata for model UUID %s" % model_uuid)

        # Parse the saved model metadata to obtain the parameters used to train the model
        model_pparams = parse.wrapper(metadata_dict)

        # Override selected parameters
        model_pparams.result_dir = tempfile.mkdtemp()

        if splitters is not None:
            if model_pparams.splitter != splitters[i]:
                log.info("Replacing %s splitter in stored model with %s" % (model_pparams.splitter, splitters[i]))
                model_pparams.splitter = splitters[i]

        if dset_params is not None:
            model_pparams.id_col = dset_params.id_col
            model_pparams.smiles_col = dset_params.smiles_col
            if contains_responses:
                model_pparams.response_cols = dset_params.response_cols
        pipe = create_prediction_pipeline(model_pparams, model_uuid, collection_name)

        if pred_df is None:
            initial_cols = [model_pparams.id_col, model_pparams.smiles_col]
            if contains_responses:
                initial_cols.extend(model_pparams.response_cols)
            pred_df = dset_df[initial_cols].copy()
            if contains_responses:
                # Assume singletask model for now
                pred_df = pred_df.rename(columns={model_pparams.response_cols[0]: 'actual'})

        pipe.run_mode = 'prediction'
        pipe.featurization = pipe.model_wrapper.featurization
        pipe.data = model_datasets.create_minimal_dataset(pipe.params, pipe.featurization, contains_responses)

        if not pipe.data.get_dataset_tasks(dset_df):
            # Shouldn't happen - response_cols should already be set in saved model parameters
            raise Exception("response_cols missing from model params")
        is_featurized = (len(set(pipe.featurization.get_feature_columns()) - set(dset_df.columns.values)) == 0)
        pipe.data.get_featurized_data(dset_df, is_featurized)
        pipe.data.dataset = pipe.model_wrapper.transform_dataset(pipe.data.dataset)

        # Create a temporary data frame to hold the compound IDs and predictions. The model may not
        # return predictions for all the requested compounds, so we have to outer join the predictions
        # to the existing data frame.
        result_df = pd.DataFrame({model_pparams.id_col: pipe.data.attr.index.values})

        # Get the predictions and standard deviations, if calculated, as numpy arrays
        try:
            preds, stds = pipe.model_wrapper.generate_predictions(pipe.data.dataset)
        except ValueError:
            log.error("\n***** Prediction failed for model %s %s\n" % (label, model_uuid))
            continue
        i = 0
        if pipe.params.prediction_type == 'regression':
            result_df["pred_%s" % label] = preds[:, i, 0]
        else:
            # Assume binary classifier for now. We're going to aggregate the probabilities for class 1.
            result_df["pred_%s" % label] = preds[:, i, 1]
        if pipe.params.uncertainty and pipe.params.prediction_type == 'regression':
            std_colname = 'std_%s' % label
            result_df[std_colname] = stds[:, i, 0]
        pred_df = pred_df.merge(result_df, how='left', on=model_pparams.id_col)
        ok_labels.append(label)

    # Aggregate the ensemble of predictions
    pred_cols = ["pred_%s" % label for label in ok_labels]
    pred_vals = pred_df[pred_cols].values
    if aggregate == 'mean':
        agg_pred = np.nanmean(pred_vals, axis=1)
    elif aggregate == 'median':
        agg_pred = np.nanmedian(pred_vals, axis=1)
    elif aggregate == 'max':
        agg_pred = np.nanmax(pred_vals, axis=1)
    elif aggregate == 'min':
        agg_pred = np.nanmin(pred_vals, axis=1)
    elif aggregate == 'weighted':
        std_cols = ["std_%s" % label for label in ok_labels]
        std_vals = pred_df[std_cols].values
        if len(set(std_cols) - set(pred_df.columns.values)) > 0:
            raise Exception("Weighted ensemble needs uncertainties for all component models.")
        if np.any(std_vals == 0.0):
            raise Exception("Can't compute weighted ensemble because some standard deviations are zero")
        agg_pred = np.nansum(pred_vals / std_vals, axis=1) / np.nansum(1.0 / std_vals, axis=1)
    else:
        raise ValueError("Unknown aggregate value %s" % aggregate)

    if pipe.params.prediction_type == 'regression':
        pred_df["ensemble_pred"] = agg_pred
    else:
        pred_df["ensemble_class_prob"] = agg_pred
        pred_df["ensemble_pred"] = [int(p >= 0.5) for p in agg_pred]

    log.info("Done with ensemble prediction")
    return pred_df


# ****************************************************************************************
def retrain_model(model_uuid, collection_name=None, result_dir=None, mt_client=None, verbose=True):
    """Obtain model parameters from the metadata in the model tracker, given the model_uuid,
    and train a new model using exactly the same parameters (except for result_dir). Returns
    the resulting ModelPipeline object. The pipeline object can then be used as input for
    performance plots and other analyses that can't be done using just the metrics stored
    in the model tracker; or to make predictions on new data.

    Args:
        model_uuid (str): The UUID of a trained model.

        collection_name (str): The collection where the model is stored in the model tracker DB.

        result_dir (str): The directory of model results when the model tracker is not available.

        mt_client : Ignored

        verbose (bool): A switch for disabling informational messages

    Returns:
        pipeline (ModelPipeline): A pipeline object containing data from the model training.
    """

    log = logging.getLogger('ATOM')
    if not result_dir:
        mlmt_client = dsf.initialize_model_tracker()

        log.info("Loading model %s from collection %s" % (model_uuid, collection_name))
        metadata_dict = trkr.get_metadata_by_uuid(model_uuid, collection_name=collection_name)
        if not metadata_dict:
            raise Exception("No model found with UUID %s in collection %s" % (model_uuid, collection_name))
    else:
        for dirpath, dirnames, filenames in os.walk(result_dir):
            if model_uuid in dirnames:
                model_dir = os.path.join(dirpath, model_uuid)
                break

        with open(os.path.join(model_dir, 'model_metadata.json')) as f:
            metadata_dict = json.load(f)

    log.info("Got metadata for model UUID %s" % model_uuid)

    # Parse the saved model metadata to obtain the parameters used to train the model
    model_pparams = parse.wrapper(metadata_dict)

    model_pparams.result_dir = tempfile.mkdtemp()
    # TODO: This is a hack; possibly the datastore parameter isn't being stored in the metadata?
    model_pparams.datastore = True if not result_dir else False
    pipe = ModelPipeline(model_pparams)
    pipe.train_model()

    return pipe

# ****************************************************************************************
def main():
    """Entry point when script is run from a shell"""

    params = parse.wrapper(sys.argv[1:])
    # model_filter parameter determines whether you are loading pretrained models and running
    # predictions on them, or training a new model
    if 'model_filter' in params.__dict__ and params.model_filter is not None:
        # DEPRECATED: This feature isn't used by anyone as far as I know; it will be removed in
        # the near future.
        run_models(params)
    elif params.split_only:
        params.verbose = False
        mp = ModelPipeline(params)
        split_uuid = mp.split_dataset()
        print(split_uuid)
    else:
        print("Running model pipeline")
        logging.basicConfig(format='%(asctime)-15s %(message)s')
        logger = logging.getLogger('ATOM')
        if params.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.CRITICAL)
        mp = ModelPipeline(params)
        mp.train_model()
        mp.log.warn("Dataset size: {}".format(mp.data.dataset.get_shape()[0][0]))


# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__' and len(sys.argv) > 1:
    main()
    sys.exit(0)
