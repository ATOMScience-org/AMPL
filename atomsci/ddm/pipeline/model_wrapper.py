#!/usr/bin/env python

"""Contains class ModelWrapper and its subclasses, which are wrappers for DeepChem and scikit-learn model classes."""

import logging
import os
import shutil
import joblib

import deepchem as dc
import numpy as np
import tensorflow as tf
if dc.__version__.startswith('2.1'):
    from deepchem.models.tensorgraph.fcnet import MultitaskRegressor, MultitaskClassifier
else:
    from deepchem.models.fcnet import MultitaskRegressor, MultitaskClassifier
from collections import OrderedDict
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


try:
    import dgl
    import dgllife
    import deepchem.models as dcm
    from deepchem.models import AttentiveFPModel
    afp_supported = True
except (ImportError, OSError):
    afp_supported = False

try:
    import xgboost as xgb
    xgboost_supported = True
except ImportError:
    xgboost_supported = False

import pickle
import yaml
import glob
import time
from packaging import version

from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.utils import llnl_utils
from atomsci.ddm.pipeline import transformations as trans
from atomsci.ddm.pipeline import perf_data as perf
import atomsci.ddm.pipeline.parameter_parser as pp

from tensorflow.python.keras.utils.layer_utils import count_params

logging.basicConfig(format='%(asctime)-15s %(message)s')

def get_latest_pytorch_checkpoint(model, model_dir=None):
    """Gets the latest torch model

    Calls get_checkpoints(), sorts them, and returns the
    latest model (the one with the biggets number)

    Args:
        model: A model that inherits DeepChem TorchModel

        model_dir: Optional argument that directs the model
            to a specific folder

    Returns:
        path relative to model_dir, if provided, to the latest
        checkpoint

    """
    chkpts = model.get_checkpoints(model_dir)
    print(chkpts)
    if len(chkpts) == 0:
        raise ValueError("No 'best' checkpoint found. deepchem.Model.get_checkpoints() is empty")
    # checkpoint with the highest number is the best one
    latest_chkpt = max(chkpts, key=os.path.getctime)
    print(latest_chkpt)

    return latest_chkpt
 

def dc_restore(model, checkpoint=None, model_dir=None, session=None):
    """Reload the values of all variables from a checkpoint file.

    copied from DeepChem 2.3 keras_model.py to silence warnings caused
    when a model is loaded in inference mode.

    Args:
        model (DeepChem.KerasModel: keras model to restore

        checkpoint (str): the path to the checkpoint file to load.  If this is None, the most recent
            checkpoint will be chosen automatically.  Call get_checkpoints() to get a
            list of all available checkpoints.

        model_dir (str): default None
            Directory to restore checkpoint from. If None, use model.model_dir.

        session (tf.Session()) default None
            Session to run restore ops under. If None, model.session is used.

    Returns:
        None
    """
    model._ensure_built()
    if model_dir is None:
        model_dir = model.model_dir
    if checkpoint is None:
        checkpoint = tf.train.latest_checkpoint(model_dir)
    if checkpoint is None:
        raise ValueError('No checkpoint found')
    if tf.executing_eagerly():
        # expect_partial() silences warnings when this model is restored for
        # inference only.
        model._checkpoint.restore(checkpoint).expect_partial()
    else:
        if session is None:
            session = model.session
        # expect_partial() silences warnings when this model is restored for
        # inference only.
        model._checkpoint.restore(checkpoint).expect_partial().run_restore_ops(session)

def dc_torch_restore(model, checkpoint= None, model_dir = None):
    """Reload the values of all variables from a checkpoint file.

    This is copied from deepchem 2.6.1 to explicitly set the torch loading device to
    'cpu' if cuda devices aren't available when reloading the model.

    Args:
        model: the torch model to restore

        checkpoint (str):
            the path to the checkpoint file to load.  If this is None, the most recent
            checkpoint will be chosen automatically.  Call get_checkpoints() to get a
            list of all available checkpoints.

        model_dir (str): default None
            Directory to restore checkpoint from. If None, use self.model_dir.  If
            checkpoint is not None, this is ignored.
    """
    model._ensure_built()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set torch device
    if checkpoint is None:
      checkpoints = sorted(model.get_checkpoints(model_dir))
      if len(checkpoints) == 0:
        raise ValueError('No checkpoint found')
      checkpoint = checkpoints[0]
    data = torch.load(checkpoint, map_location=device) # include map_location to transfer a gpu model to cpu
    model.model.load_state_dict(data['model_state_dict'])
    model._pytorch_optimizer.load_state_dict(data['optimizer_state_dict'])
    model._global_step = data['global_step']


def all_bases(model):
    """Given a model

    Returns:
        all bases
    """
    result = [model]
    current_funcs = [model]
    while len(current_funcs)>0:
        current_func = current_funcs.pop(0)
        current_funcs = current_funcs + list(current_func.__bases__)
        result = result + list(current_func.__bases__)

    return result

# ****************************************************************************************
def create_model_wrapper(params, featurizer, ds_client=None):
    """Factory function for creating Model objects of the correct subclass for params.model_type.

    Args:
        params (Namespace): Parameters passed to the model pipeline

        featurizer (Featurization): Object managing the featurization of compounds

        ds_client (DatastoreClient): Interface to the file datastore

    Returns:
        model (pipeline.Model): Wrapper for DeepChem, sklearn or other model.

    Raises:
        ValueError: Only params.model_type = 'NN', 'RF' or 'xgboost' is supported.
    """
    if params.model_type == 'NN':
        if params.featurizer == 'graphconv':
            return GraphConvDCModelWrapper(params, featurizer, ds_client)
        else:
            return MultitaskDCModelWrapper(params, featurizer, ds_client)
    elif params.model_type == 'RF':
        return DCRFModelWrapper(params, featurizer, ds_client)
    elif params.model_type == 'xgboost':
        if not xgboost_supported:
            raise Exception("Unable to import xgboost. \
                             xgboost package needs to be installed to use xgboost model. \
                             Installatin: \
                             from pip: pip3 install xgboost==0.90.\
                             livermore compute (lc): /usr/mic/bio/anaconda3/bin/pip install xgboost==0.90 --user \
                             twintron-blue (TTB): /opt/conda/bin/pip install xgboost==0.90 --user "
                            )
        elif version.parse(xgb.__version__) < version.parse('0.9'):
            raise Exception("xgboost required to be = 0.9 for GPU support. \
                             current version = xgb.__version__ \
                             installation: \
                             from pip: pip install xgboost==0.90")
        else:
            return DCxgboostModelWrapper(params, featurizer, ds_client)
    elif params.model_type == 'hybrid':
        return HybridModelWrapper(params, featurizer, ds_client)
    elif params.model_type in pp.model_wl:
        requested_model = pp.model_wl[params.model_type]
        bases = all_bases(requested_model)
        # keras and torch models have slightly different interfaces. They also save their models
        # differently. These two wrappers implement the same interface.
        if any(['TorchModel' in str(b) for b in bases]):
            if not afp_supported:
                raise Exception("dgl and dgllife packages must be installed to use attentive_fp model.")
            return PytorchDeepChemModelWrapper(params, featurizer, ds_client)
        elif any(['KerasModel' in str(b) for b in bases]):
            return KerasDeepChemModelWrapper(params, featurizer, ds_client)
    else:
        raise ValueError("Unknown model_type %s" % params.model_type)

# ****************************************************************************************

class ModelWrapper(object):
    """Wrapper for DeepChem and sklearn model objects. Provides methods to train and test a model,
    generate predictions for an input dataset, and generate performance metrics for these predictions.

    Attributes:
        Set in __init__
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information

            featurziation (Featurization object): The featurization object created outside of model_wrapper

            log (log): The logger

            output_dir (str): The parent path of the model directory

            transformers (list): Initialized as an empty list, stores the transformers on the response cols

            transformers_x (list): Initialized as an empty list, stores the transformers on the features

            transformers_w (list): Initialized as an empty list, stores the transformers on the weights

        set in setup_model_dirs:
            best_model_dir (str): The subdirectory under output_dir that contains the best model. Created in setup_model_dirs

    """
    def __init__(self, params, featurizer, ds_client):
        """Initializes ModelWrapper object.

        Args:
            params (Namespace object): contains all parameter information.

            featurizer (Featurization object): initialized outside of model_wrapper

            ds_client (DatastoreClient): Interface to the file datastore

        Side effects:
            Sets the following attributes of ModelWrapper:
                params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information

                featurziation (Featurization object): The featurization object created outside of model_wrapper

                log (log): The logger

                output_dir (str): The parent path of the model directory

                transformers (list): Initialized as an empty list, stores the transformers on the response cols

                transformers_x (list): Initialized as an empty list, stores the transformers on the features

                transformers_w (list): Initialized as an empty list, stores the transformers on the weights

        """
        self.params = params
        self.featurization = featurizer
        self.ds_client = ds_client
        self.log = logging.getLogger('ATOM')
        self.output_dir = self.params.output_dir
        self.model_dir = os.path.join(self.output_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.transformers = []
        self.transformers_x = []
        self.transformers_w = []

        # ****************************************************************************************

    def setup_model_dirs(self):
        """Sets up paths and directories for persisting models at particular training epochs, used by
        the DeepChem model classes.

        Side effects:
            Sets the following attributes of ModelWrapper:
                best_model_dir (str): The subdirectory under output_dir that contains the best model. Created in setup_model_dirs
        """
        self.best_model_dir = os.path.join(self.output_dir, 'best_model')

        # ****************************************************************************************

    def train(self, pipeline):
        """Trains a model (for multiple epochs if applicable), and saves the tuned model.

        Args:
            pipeline (ModelPipeline): The ModelPipeline instance for this model run.

        Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

        # ****************************************************************************************

    def get_model_specific_metadata(self):
        """Returns a dictionary of parameter settings for this ModelWrapper object that are specific
        to the model type.

        Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

        # ****************************************************************************************
    def _create_output_transformers(self, model_dataset):
        """Initialize transformers for responses and persist them for later.

        Args:
            model_dataset: The ModelDataset object that handles the current dataset

        Side effects:
            Overwrites the attributes:
                transformers: A list of deepchem transformation objects on response_col, only if conditions are met
        """
        # TODO: Just a warning, we may have response transformers for classification datasets in the future
        if self.params.prediction_type=='regression' and self.params.transformers==True:
            self.transformers = [trans.NormalizationTransformerMissingData(transform_y=True, dataset=model_dataset.dataset)]

        # ****************************************************************************************

    def _create_feature_transformers(self, model_dataset):
        """Initialize transformers for features, and persist them for later.

        Args:
            model_dataset: The ModelDataset object that handles the current dataset

        Side effects:
            Overwrites the attributes:
                transformers_x: A list of deepchem transformation objects on featurizers, only if conditions are met.
        """
        # Set up transformers for features, if needed
        self.transformers_x = trans.create_feature_transformers(self.params, model_dataset)

        # ****************************************************************************************

    def create_transformers(self, model_dataset):
        """Initialize transformers for responses, features and weights, and persist them for later.

        Args:
            model_dataset: The ModelDataset object that handles the current dataset

        Side effects:
            Overwrites the attributes:
                transformers: A list of deepchem transformation objects on responses, only if conditions are met

                transformers_x: A list of deepchem transformation objects on features, only if conditions are met.

                transformers_w: A list of deepchem transformation objects on weights, only if conditions are met.

                params.transformer_key: A string pointing to the dataset key containing the transformer in the datastore, or the path to the transformer

        """
        self._create_output_transformers(model_dataset)

        self._create_feature_transformers(model_dataset)

        # Set up transformers for weights, if needed
        self.transformers_w = trans.create_weight_transformers(self.params, model_dataset)

        if len(self.transformers) + len(self.transformers_x) + len(self.transformers_w) > 0:

            # Transformers are no longer saved as separate datastore objects; they are included in the model tarball
            self.params.transformer_key = os.path.join(self.output_dir, 'transformers.pkl')
            with open(self.params.transformer_key, 'wb') as txfmrpkl:
                pickle.dump((self.transformers, self.transformers_x, self.transformers_w), txfmrpkl)
            self.log.info("Wrote transformers to %s" % self.params.transformer_key)
            self.params.transformer_oid = ""
            self.params.transformer_bucket = ""

        # ****************************************************************************************

    def reload_transformers(self):
        """Load response, feature and weight transformers from datastore objects or files. Before AMPL v1.2 these
        were persisted as separate datastore objects when the model tracker was used; subsequently they
        are included in model tarballs, which should have been unpacked before this function gets called.
        """

        # Try local path first to check for transformers unpacked from model tarball
        if not trans.transformers_needed(self.params):
            return
        local_path = f"{self.output_dir}/transformers.pkl"
        if os.path.exists(local_path):
            self.log.info(f"Reloading transformers from model tarball {local_path}")
            with open(local_path, 'rb') as txfmr:
                transformers_tuple = pickle.load(txfmr)
        else:
            if self.params.transformer_key is not None:
                if self.params.save_results:
                    self.log.info(f"Reloading transformers from datastore key {self.params.transformer_key}")
                    transformers_tuple = dsf.retrieve_dataset_by_datasetkey(
                        dataset_key = self.params.transformer_key,
                        bucket = self.params.transformer_bucket,
                        client = self.ds_client )
                else:
                    self.log.info(f"Reloading transformers from file {self.params.transformer_key}")
                    with open(self.params.transformer_key, 'rb') as txfmr:
                        transformers_tuple = pickle.load(txfmr)
            else:
                # Shouldn't happen
                raise Exception("Transformers needed to reload model, but no transformer_key specified.")


        if len(transformers_tuple) == 3:
            self.transformers, self.transformers_x, self.transformers_w = transformers_tuple
        else:
            self.transformers, self.transformers_x = transformers_tuple
            self.transformers_w = []

        # ****************************************************************************************

    def transform_dataset(self, dataset):
        """Transform the responses and/or features in the given DeepChem dataset using the current transformers.

        Args:
            dataset: The DeepChem DiskDataset that contains a dataset

        Returns:
            transformed_dataset: The transformed DeepChem DiskDataset

        """
        transformed_dataset = dataset
        if len(self.transformers) > 0:
            self.log.info("Transforming response data")
            for transformer in self.transformers:
                transformed_dataset = transformer.transform(transformed_dataset)
        if len(self.transformers_x) > 0:
            self.log.info("Transforming feature data")
            for transformer in self.transformers_x:
                transformed_dataset = transformer.transform(transformed_dataset)
        if len(self.transformers_w) > 0:
            self.log.info("Transforming weights")
            for transformer in self.transformers_w:
                transformed_dataset = transformer.transform(transformed_dataset)

        return transformed_dataset
        # ****************************************************************************************

    def get_num_features(self):
        """Get the number of features.

        Returns:
           the number of dimensions of the feature space, taking both featurization method
        and transformers into account.
        """
        if self.params.feature_transform_type == 'umap':
            return self.params.umap_dim
        else:
            return self.featurization.get_feature_count()

        # ****************************************************************************************

    def get_train_valid_pred_results(self, perf_data):
        """Returns predicted values and metrics for the training, validation or test set
        associated with the PerfData object perf_data. Results are returned as a dictionary
        of parameter, value pairs in the format expected by the model tracker.

        Args:
            perf_data: A PerfData object that stores the predicted values and metrics

        Returns:
            dict: A dictionary of the prediction results

        """
        return perf_data.get_prediction_results()

        # ****************************************************************************************
    def get_test_perf_data(self, model_dir, model_dataset):
        """Returns the predicted values and metrics for the current test dataset against
        the version of the model stored in model_dir, as a PerfData object.

        Args:
            model_dir (str): Directory where the saved model is stored
            model_dataset (DiskDataset): Stores the current dataset and related methods

        Returns:
            perf_data: PerfData object containing the predicted values and metrics for the current test dataset
        """
        # Load the saved model from model_dir
        self.reload_model(model_dir)

        # Create a PerfData object, which knows how to format the prediction results in the structure
        # expected by the model tracker.

        # We pass transformed=False to indicate that the preds and uncertainties we get from
        # generate_predictions are already untransformed, so that perf_data.get_prediction_results()
        # doesn't untransform them again.
        if hasattr(self.transformers[0], "ishybrid"):
            # indicate that we are training a hybrid model
            perf_data = perf.create_perf_data("hybrid", model_dataset, self.transformers, 'test', is_ki=self.params.is_ki, ki_convert_ratio=self.params.ki_convert_ratio, transformed=False)
        else:
            perf_data = perf.create_perf_data(self.params.prediction_type, model_dataset, self.transformers, 'test', transformed=False)
        test_dset = model_dataset.test_dset
        test_preds, test_stds = self.generate_predictions(test_dset)
        _ = perf_data.accumulate_preds(test_preds, test_dset.ids, test_stds)
        return perf_data

        # ****************************************************************************************
    def get_test_pred_results(self, model_dir, model_dataset):
        """Returns predicted values and metrics for the current test dataset against the version
        of the model stored in model_dir, as a dictionary in the format expected by the model tracker.

        Args:
            model_dir (str): Directory where the saved model is stored
            model_dataset (DiskDataset): Stores the current dataset and related methods

        Returns:
            dict: A dictionary containing the prediction values and metrics for the current dataset.
        """
        perf_data = self.get_test_perf_data(model_dir, model_dataset)
        return perf_data.get_prediction_results()

        # ****************************************************************************************
    def get_full_dataset_perf_data(self, model_dataset):
        """Returns the predicted values and metrics from the current model for the full current dataset,
        as a PerfData object.

        Args:
            model_dataset (DiskDataset): Stores the current dataset and related methods

        Returns:
            perf_data: PerfData object containing the predicted values and metrics for the current full dataset
        """

        # Create a PerfData object, which knows how to format the prediction results in the structure
        # expected by the model tracker.

        # We pass transformed=False to indicate that the preds and uncertainties we get from
        # generate_predictions are already untransformed, so that perf_data.get_prediction_results()
        # doesn't untransform them again.
        if hasattr(self.transformers[0], "ishybrid"):
            # indicate that we are training a hybrid model
            perf_data = perf.create_perf_data("hybrid", model_dataset, self.transformers, 'full', is_ki=self.params.is_ki, ki_convert_ratio=self.params.ki_convert_ratio, transformed=False)
        else:
            perf_data = perf.create_perf_data(self.params.prediction_type, model_dataset, self.transformers, 'full', transformed=False)
        full_preds, full_stds = self.generate_predictions(model_dataset.dataset)
        _ = perf_data.accumulate_preds(full_preds, model_dataset.dataset.ids, full_stds)
        return perf_data

        # ****************************************************************************************
    def get_full_dataset_pred_results(self, model_dataset):
        """Returns predicted values and metrics from the current model for the full current dataset,
        as a dictionary in the format expected by the model tracker.

        Args:
            model_dataset (DiskDataset): Stores the current dataset and related methods

        Returns:
            dict: A dictionary containing predicted values and metrics for the current full dataset

        """
        self.data = model_dataset
        perf_data = self.get_full_dataset_perf_data(model_dataset)
        return perf_data.get_prediction_results()

    def generate_predictions(self, dataset):
        """Generate predictions,.

        Args:
            dataset:

        Returns:
            None.

        """
        raise NotImplementedError

    def generate_embeddings(self, dataset):
        """Generate embeddings.

        Args:
            dataset:

        Returns:
            None

        """
        raise NotImplementedError

    def reload_model(self, reload_dir):
        """Args:
            reload_dir:

        Returns:

        """
        raise NotImplementedError


    # ****************************************************************************************
    def model_save(self):
        """A wrapper function to save a model  due to the `DeepChem model.save()` has inconsistent implementation.

        The `SKlearnModel()` class and xgboost model in DeepChem use `model.save()`,
        while the `MultitaskRegressor` class uses `model.save_checkpoint()`. The
        workaround is to try `model.save()` first. If failed, then try `model.save_checkpoint()`
        """
        try:
            self.model.save()
        except Exception:
          try:
            self.model.save_checkpoint()
          except Exception as e:
            self.log.error("Error when saving model:\n%s" % str(e))

class LCTimerIterator:
    """This creates an iterator that keeps track of time limits

    Side Effects:
        Sets the following attributes in input argument params:
            max_epochs (int): The max_epochs attribute will be updated to the current epoch
                index if it is projected that training will exceed the time limit
                given.
    """
    # ****************************************************************************************
    def __init__(self, params, pipeline, log):
        """Args:
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all
               parameter information

            pipeline (ModelPipeline): The ModelPipeline instance for this model run.

            log (log): The logger used in the ModelWrapper
        """
        self.max_epochs = int(params.max_epochs)
        self.time_limit = int(params.slurm_time_limit)
        self.start_time = pipeline.start_time
        self.training_start = time.time()
        self.params = params
        self.ei = -1
        self.log = log

    # ****************************************************************************************
    def __iter__(self):
        return self
        
    # ****************************************************************************************
    def __next__(self):
        """Returns epoch index or stops when the have been enough iterations."""
        self.ei = self.ei + 1
        if self.ei >= self.max_epochs:
            raise StopIteration
        elif llnl_utils.is_lc_system() and (self.ei > 0):
            # If we're running on an LC system, check that we have enough time to complete another epoch
            # before the current job finishes, by extrapolating from the time elapsed so far.

            now = time.time() 
            elapsed_time = now - self.start_time
            training_time = now - self.training_start
            time_remaining = self.time_limit * 60 - elapsed_time
            time_needed = self._time_needed(training_time)

            if time_needed > 0.9 * time_remaining:
                self.log.warn("Projected time to finish one more epoch exceeds time left in job; cutting training to %d epochs" %
                                self.ei)
                self.params.max_epochs = self.ei
                raise StopIteration
            else:
                return self.ei
        else:
            return self.ei

    # ****************************************************************************************
    def _time_needed(self, training_time):
        """Calculates the time needed to complete another epoch given the training time

        Args:
            training_time (int): Time the model has spent training so far

        Returns:
            int: Amount of time needed to complete another epoch
        """

        return training_time/self.ei

class LCTimerKFoldIterator(LCTimerIterator):
    """This creates an iterator that keeps track of time limits for kfold training

    Side Effects:
        Sets the following attributes in input argument params:
            max_epochs (int): The max_epochs attribute will be updated to the current epoch
                index if it is projected that training will exceed the time limit
                given.
    """

    # ****************************************************************************************
    def _time_needed(self, training_time):
        """Calculates the time needed to complete another epoch given the training time.

        epochs_remaining is how many epochs we have to run if we do one more across all folds,
        then do self.best_epoch+1 epochs on the combined training & validation set,
        allowing for the possibility that the next epoch may be the best one.

        Args:
            training_time (int): Time the model has spent training so far

        Returns:
            int: Amount of time needed to complete another epoch
        """

        epochs_remaining = self.ei + 2
        time_per_epoch = training_time/self.ei
        time_needed = epochs_remaining * time_per_epoch

        return time_needed

# ****************************************************************************************
class NNModelWrapper(ModelWrapper):
    """Wrapper for NN models.

        Many NN models share similar functions. This class aggregates those similar functions
        to reduce copied code

    """

    # ****************************************************************************************
    def get_perf_data(self, subset, epoch_label=None):
        """Returns predicted values and metrics from a training, validation or test subset
        of the current dataset, or the full dataset. subset may be 'train', 'valid', 'test' or 'full',
        epoch_label indicates the training epoch we want results for; currently the
        only option for this is 'best'. Results are returned as a PerfData object of the appropriate class
        for the model's split strategy and prediction type.

        Args:
            subset (str): Label for the current subset of the dataset (choices ['train','valid','test','full'])

            epoch_label (str): Label for the training epoch we want results for (choices ['best'])

        Returns:
            PerfData object: Performance object pulled from the appropriate subset

        Raises:
            ValueError: if epoch_label not in ['best']

            ValueError: If subset not in ['train','valid','test','full']
        """

        if subset == 'full':
            return self.get_full_dataset_perf_data(self.data)
        if epoch_label == 'best':
            epoch = self.best_epoch
            model_dir = self.best_model_dir
        else:
            raise ValueError("Unknown epoch_label '%s'" % epoch_label)

        if subset == 'train':
            return self.train_perf_data[epoch]
        elif subset == 'valid':
            return self.valid_perf_data[epoch]
        elif subset == 'test':
            #return self.get_test_perf_data(model_dir, self.data)
            return self.test_perf_data[epoch]
        else:
            raise ValueError("Unknown dataset subset '%s'" % subset)

    # ****************************************************************************************
    def get_pred_results(self, subset, epoch_label=None):
        """Returns predicted values and metrics from a training, validation or test subset
        of the current dataset, or the full dataset. subset may be 'train', 'valid', 'test'
        accordingly.  epoch_label indicates the training epoch we want results for; currently the
        only option for this is 'best'.  Results are returned as a dictionary of parameter, value pairs.

        Args:
            subset (str): Label for the current subset of the dataset (choices ['train','valid','test','full'])

            epoch_label (str): Label for the training epoch we want results for (choices ['best'])

        Returns:
            dict: A dictionary of parameter/ value pairs of the prediction values and results of the dataset subset

        Raises:
            ValueError: if epoch_label not in ['best']

            ValueError: If subset not in ['train','valid','test','full']
        """
        if subset == 'full':
            return self.get_full_dataset_pred_results(self.data)
        if epoch_label == 'best':
            epoch = self.best_epoch
            model_dir = self.best_model_dir
        else:
            raise ValueError("Unknown epoch_label '%s'" % epoch_label)
        if subset == 'train':
            return self.get_train_valid_pred_results(self.train_perf_data[epoch])
        elif subset == 'valid':
            return self.get_train_valid_pred_results(self.valid_perf_data[epoch])
        elif subset == 'test':
            return self.get_train_valid_pred_results(self.test_perf_data[epoch])
        else:
            raise ValueError("Unknown dataset subset '%s'" % subset)

    # ****************************************************************************************
    def _clean_up_excess_files(self, dest_dir):
        """Function to clean up extra model files left behind in the training process.
        Only removes self.model_dir
        """
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.mkdir(dest_dir)

    # ****************************************************************************************
    def train(self, pipeline):
        """Trains a neural net model for multiple epochs, choose the epoch with the best validation
        set performance, refits the model for that number of epochs, and saves the tuned model.

        Args:
            pipeline (ModelPipeline): The ModelPipeline instance for this model run.

        Side effects:
            Sets the following attributes for NNModelWrapper:
                data (ModelDataset): contains the dataset, set in pipeline

                best_epoch (int): Initialized as None, keeps track of the epoch with the best validation score

                train_perf_data (list of PerfData): Initialized as an empty array,
                    contains the predictions and performance of the training dataset

                valid_perf_data (list of PerfData): Initialized as an empty array,
                    contains the predictions and performance of the validation dataset

                train_epoch_perfs (np.array): Initialized as an empty array,
                    contains a list of dictionaries of predicted values and metrics on the training dataset

                valid_epoch_perfs (np.array of dicts): Initialized as an empty array,
                    contains a list of dictionaries of predicted values and metrics on the validation dataset
        """
        # TODO: Fix docstrings above
        num_folds = len(pipeline.data.train_valid_dsets)
        if num_folds > 1:
            self.train_kfold_cv(pipeline)
        else:
            self.train_with_early_stopping(pipeline)

    # ****************************************************************************************
    def train_kfold_cv(self, pipeline):
        """Trains a neural net model with K-fold cross-validation for a specified number of epochs.
        Finds the epoch with the best validation set performance averaged over folds, then refits
        a model for the same number of epochs to the combined training and validation data.

        Args:
            pipeline (ModelPipeline): The ModelPipeline instance for this model run.

        Side effects:
            Sets the following attributes for NNModelWrapper:
                data (ModelDataset): contains the dataset, set in pipeline

                best_epoch (int): Initialized as None, keeps track of the epoch with the best validation score

                train_perf_data (list of PerfData): Initialized as an empty array,
                    contains the predictions and performance of the training dataset

                valid_perf_data (list of PerfData): Initialized as an empty array,
                    contains the predictions and performance of the validation dataset

                train_epoch_perfs (np.array): Contains a standard training set performance metric (r2_score or roc_auc), averaged over folds,
                    at the end of each epoch.

                valid_epoch_perfs (np.array): Contains a standard validation set performance metric (r2_score or roc_auc), averaged over folds,
                    at the end of each epoch.
        """
        # TODO: Fix docstrings above
        num_folds = len(pipeline.data.train_valid_dsets)
        self.data = pipeline.data

        # Create PerfData structures for computing cross-validation metrics
        em = perf.EpochManagerKFold(self,
                                subsets={'train':'train_valid', 'valid':'valid', 'test':'test'},
                                prediction_type=self.params.prediction_type, 
                                model_dataset=pipeline.data, 
                                production=self.params.production,
                                transformers=self.transformers)
        em.set_make_pred(lambda x: self.model.predict(x, []))
        em.on_new_best_valid(lambda : 1+1) # does not need to take any action

        test_dset = pipeline.data.test_dset

        # Train a separate model for each fold
        models = []
        for k in range(num_folds):
            models.append(self.recreate_model())

        for ei in LCTimerKFoldIterator(self.params, pipeline, self.log):
            # Create PerfData structures that are only used within loop to compute metrics during initial training
            train_perf_data = perf.create_perf_data(self.params.prediction_type, pipeline.data, self.transformers, 'train')
            test_perf_data = perf.create_perf_data(self.params.prediction_type, pipeline.data, self.transformers, 'test')
            for k in range(num_folds):
                self.model = models[k]
                train_dset, valid_dset = pipeline.data.train_valid_dsets[k]

                # We turn off automatic checkpointing - we only want to save a checkpoints for the final model.
                self.model.fit(train_dset, nb_epoch=1, checkpoint_interval=0, restore=False)
                train_pred = self.model.predict(train_dset, [])
                test_pred = self.model.predict(test_dset, [])

                train_perf = train_perf_data.accumulate_preds(train_pred, train_dset.ids)
                test_perf = test_perf_data.accumulate_preds(test_pred, test_dset.ids)

                valid_perf = em.accumulate(ei, subset='valid', dset=valid_dset)
                self.log.info("Fold %d, epoch %d: training %s = %.3f, validation %s = %.3f, test %s = %.3f" % (
                              k, ei, pipeline.metric_type, train_perf, pipeline.metric_type, valid_perf,
                              pipeline.metric_type, test_perf))

            # Compute performance metrics for current epoch across validation sets for all folds, and update
            # the best_epoch and best score if the new score exceeds the previous best score by a specified
            # threshold.
            em.compute(ei, 'valid')
            em.update_valid(ei)
            if em.should_stop():
                break
            self.num_epochs_trained = ei + 1

        # Train a new model for best_epoch epochs on the combined training/validation set. Compute the training and test
        # set metrics at each epoch.
        fit_dataset = pipeline.data.combined_training_data()
        retrain_start = time.time()
        self.model = self.recreate_model()
        self.log.info(f"Best epoch was {self.best_epoch}, retraining with combined training/validation set")

        for ei in range(self.best_epoch+1):
            self.model.fit(fit_dataset, nb_epoch=1, checkpoint_interval=0, restore=False)
            train_perf, test_perf = em.update_epoch(ei, train_dset=fit_dataset, test_dset=test_dset)

            self.log.info(f"Combined folds: Epoch {ei}, training {pipeline.metric_type} = {train_perf:.3},"
                         + f"test {pipeline.metric_type} = {test_perf:.3}")

        self.model.save_checkpoint()
        self.model_save()

        # Only copy the model files we need, not the entire directory
        self._copy_model(self.best_model_dir)
        retrain_time = time.time() - retrain_start
        self.log.info("Time to retrain model for %d epochs: %.1f seconds, %.1f sec/epoch" % (self.best_epoch, retrain_time, 
                       retrain_time/self.best_epoch))

    # ****************************************************************************************
    def train_with_early_stopping(self, pipeline):
        """Trains a neural net model for up to self.params.max_epochs epochs, while tracking the validation
        set metric given by params.model_choice_score_type. Saves a model checkpoint each time the metric
        is improved over its previous saved value by more than a threshold percentage. If the metric fails to
        improve for more than a specified 'patience' number of epochs, stop training and revert the model state
        to the last saved checkpoint.

        Args:
            pipeline (ModelPipeline): The ModelPipeline instance for this model run.

        Side effects:
            Sets the following attributes for NNModelWrapper:
                data (ModelDataset): contains the dataset, set in pipeline

                best_epoch (int): Initialized as None, keeps track of the epoch with the best validation score

                best_validation_score (float): The best validation model choice score attained during training.

                train_perf_data (list of PerfData): Initialized as an empty array,
                    contains the predictions and performance of the training dataset

                valid_perf_data (list of PerfData): Initialized as an empty array,
                    contains the predictions and performance of the validation dataset

                train_epoch_perfs (np.array): A standard training set performance metric (r2_score or roc_auc), at the end of each epoch.

                valid_epoch_perfs (np.array): A standard validation set performance metric (r2_score or roc_auc), at the end of each epoch.
        """
        self.data = pipeline.data

        em = perf.EpochManager(self,
                                prediction_type=self.params.prediction_type, 
                                model_dataset=pipeline.data, 
                                production=self.params.production,
                                transformers=self.transformers)
        em.set_make_pred(lambda x: self.model.predict(x, []))
        em.on_new_best_valid(lambda : self.model.save_checkpoint())

        test_dset = pipeline.data.test_dset
        train_dset, valid_dset = pipeline.data.train_valid_dsets[0]
        for ei in LCTimerIterator(self.params, pipeline, self.log):
            # Train the model for one epoch. We turn off automatic checkpointing, so the last checkpoint
            # saved will be the one we created intentionally when we reached a new best validation score.
            self.model.fit(train_dset, nb_epoch=1, checkpoint_interval=0)
            train_perf, valid_perf, test_perf = em.update_epoch(ei,
                                train_dset=train_dset, valid_dset=valid_dset, test_dset=test_dset)

            self.log.info("Epoch %d: training %s = %.3f, validation %s = %.3f, test %s = %.3f" % (
                          ei, pipeline.metric_type, train_perf, pipeline.metric_type, valid_perf,
                          pipeline.metric_type, test_perf))

            self.num_epochs_trained = ei + 1
            # Compute performance metrics for each subset, and check if we've reached a new best validation set score
            if em.should_stop():
                break

        # Revert to last checkpoint
        self.restore()
        self.model_save()

        # Only copy the model files we need, not the entire directory
        self._copy_model(self.best_model_dir)
        self.log.info(f"Best model from epoch {self.best_epoch} saved to {self.best_model_dir}")

    def restore(self, checkpoint=None, model_dir=None):
        """Restores this model"""
        dc_torch_restore(self.model, checkpoint, model_dir)

    # ****************************************************************************************
    def _copy_model(self, dest_dir):
        """Copies the files needed to recreate a DeepChem NN model from the current model
        directory to a destination directory.

        Looks at self.model.get_checkpoints() and assumes the last checkpoint saved is the best

        Args:
            dest_dir (str): The destination directory for the model files
        """
        chkpt_file = get_latest_pytorch_checkpoint(self.model)

        self._clean_up_excess_files(dest_dir)

        shutil.copy2(chkpt_file, dest_dir)
        self.log.info("Saved model files to '%s'" % dest_dir)


    # ****************************************************************************************
    def generate_predictions(self, dataset):
        """Generates predictions for specified dataset with current model, as well as standard deviations
        if params.uncertainty=True

        Args:
            dataset: the deepchem DiskDataset to generate predictions for

        Returns:
            (pred, std): tuple of predictions for compounds and standard deviation estimates, if requested.
            Each element of tuple is a numpy array of shape (ncmpds, ntasks, nclasses), where nclasses = 1 for regression
            models.
        """
        pred, std = None, None
        self.log.info("Predicting values for current model")

        # For deepchem's predict_uncertainty function, you are not allowed to specify transformers. That means that the
        # predictions are being made in the transformed space, not the original space. We call undo_transforms() to generate
        # the transformed predictions. To transform the standard deviations, we rely on the fact that at present we only use
        # dc.trans.NormalizationTransformer (which centers and scales the data).

        # Uncertainty is now supported by DeepChem's GraphConv, at least for regression models.
        # if self.params.uncertainty and self.params.prediction_type == 'regression' and self.params.featurizer != 'graphconv':

        # Current (2.1) DeepChem neural net classification models don't support uncertainties.
        if self.params.uncertainty and self.params.prediction_type == 'classification':
            self.log.warning("Warning: DeepChem neural net models support uncertainty for regression only.")
 
        if self.params.uncertainty and self.params.prediction_type == 'regression':
            # For multitask, predict_uncertainty returns a list of (pred, std) tuples, one for each task.
            # For singletask, it returns one tuple. Convert the result into a pair of ndarrays of shape (ncmpds, ntasks, nclasses).
            pred_std = self.model.predict_uncertainty(dataset)
            if type(pred_std) == tuple:
                #JEA
                #ntasks = 1
                ntasks = len(pred_std[0][0])
                pred, std = pred_std
                pred = pred.reshape((pred.shape[0], 1, pred.shape[1]))
                std = std.reshape(pred.shape)
            else:
                ntasks = len(pred_std)
                pred0, std0 = pred_std[0]
                ncmpds = pred0.shape[0]
                nclasses = pred0.shape[1]
                pred = np.concatenate([p.reshape((ncmpds, 1, nclasses)) for p, s in pred_std], axis=1)
                std = np.concatenate([s.reshape((ncmpds, 1, nclasses)) for p, s in pred_std], axis=1)

            if self.params.transformers and self.transformers is not None:
                  # Transform the standard deviations, if we can. This is a bit of a hack, but it works for
                # NormalizationTransformer, since the standard deviations used to scale the data are
                # stored in the transformer object.

                # =-=ksm: The second 'isinstance' shouldn't be necessary since NormalizationTransformerMissingData
                # is a subclass of dc.trans.NormalizationTransformer.
                if len(self.transformers) == 1 and (isinstance(self.transformers[0], dc.trans.NormalizationTransformer) 
                                                 or isinstance(self.transformers[0],trans.NormalizationTransformerMissingData)):
                    y_stds = self.transformers[0].y_stds.reshape((1,ntasks,1))
                    std = std / y_stds
                pred = dc.trans.undo_transforms(pred, self.transformers)
        else:
            txform = [] if (not self.params.transformers or self.transformers is None) else self.transformers
            pred = self.model.predict(dataset, txform)
            if self.params.prediction_type == 'regression':
                if type(pred) == list and len(pred) == 0:
                    # DeepChem models return empty list if no valid predictions
                    pred = np.array([]).reshape((0,0,1))
                else:
                    pred = pred.reshape((pred.shape[0], pred.shape[1], 1))
        return pred, std

# ****************************************************************************************
class HybridModelWrapper(NNModelWrapper):
    """A wrapper for hybrid models, contains methods to load in a dataset, split and featurize the data, fit a model to the train dataset,
    generate predictions for an input dataset, and generate performance metrics for these predictions.

    Attributes:
        Set in __init__
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information
            featurziation (Featurization object): The featurization object created outside of model_wrapper

            log (log): The logger

            output_dir (str): The parent path of the model directory

            transformers (list): Initialized as an empty list, stores the transformers on the response cols

            transformers_x (list): Initialized as an empty list, stores the transformers on the features

            transformers_w (list): Initialized as an empty list, stores the transformers on the weights

            model_dir (str): The subdirectory under output_dir that contains the model. Created in setup_model_dirs.

            best_model_dir (str): The subdirectory under output_dir that contains the best model. Created in setup_model_dirs

            model: The PyTorch NN sequential model.
        Created in train:
            data (ModelDataset): contains the dataset, set in pipeline

            best_epoch (int): Initialized as None, keeps track of the epoch with the best validation score

            train_perf_data (np.array of PerfData): Initialized as an empty array,
                contains the predictions and performance of the training dataset

            valid_perf_data (np.array of PerfData): Initialized as an empty array,
                contains the predictions and performance of the validation dataset

            train_epoch_perfs (np.array of dicts): Initialized as an empty array,
                contains a list of dictionaries of predicted values and metrics on the training dataset

            valid_epoch_perfs (np.array of dicts): Initialized as an empty array,
                contains a list of dictionaries of predicted values and metrics on the validation dataset

    """

    def __init__(self, params, featurizer, ds_client):
        """Initializes HybridModelWrapper object.

        Args:
            params (Namespace object): contains all parameter information.

            featurizer (Featurizer object): initialized outside of model_wrapper

        Side effects:
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information

            featurziation (Featurization object): The featurization object created outside of model_wrapper

            log (log): The logger

            output_dir (str): The parent path of the model directory

            transforsamers (list): Initialized as an empty list, stores the transformers on the response cols

            transformers_x (list): Initialized as an empty list, stores the transformers on the features

            transformers_w (list): Initialized as an empty list, stores the transformers on the weights

            model: dc.models.TorchModel
        """
        super().__init__(params, featurizer, ds_client)
        if self.params.layer_sizes is None:
            if self.params.featurizer == 'ecfp':
                self.params.layer_sizes = [1000, 500]
            elif self.params.featurizer in ['descriptors', 'computed_descriptors']:
                self.params.layer_sizes = [200, 100]
            else:
                # Shouldn't happen
                self.log.warning("You need to define default layer sizes for featurizer %s" %
                                    self.params.featurizer)
                self.params.layer_sizes = [1000, 500]

        if self.params.dropouts is None:
            self.params.dropouts = [0.4] * len(self.params.layer_sizes)

        n_features = self.get_num_features()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.params.prediction_type == 'regression':
            model_dict = OrderedDict([
                ("layer1", torch.nn.Linear(n_features, self.params.layer_sizes[0]).to(self.dev)),
                ("dp1", torch.nn.Dropout(p=self.params.dropouts[0]).to(self.dev)),
                ("relu1", torch.nn.ReLU().to(self.dev))
            ])
            
            if len(self.params.layer_sizes) > 1:
                for i in range(1, len(self.params.layer_sizes)):
                    model_dict[f"layer{i+1}"] = torch.nn.Linear(self.params.layer_sizes[i-1], self.params.layer_sizes[i]).to(self.dev)
                    model_dict[f"dp{i+1}"] = torch.nn.Dropout(p=self.params.dropouts[i]).to(self.dev)
                    model_dict[f"relu{i+1}"] = torch.nn.ReLU().to(self.dev)
            
            model_dict["last_layer"] = torch.nn.Linear(self.params.layer_sizes[-1], 1).to(self.dev)
            
            self.model_dict = model_dict
            self.model = torch.nn.Sequential(model_dict).to(self.dev)
        else:
            raise Exception("Hybrid model only support regression prediction.")

    def _predict_binding(self, activity, conc):
        """Predict measurements of fractional binding/inhibition of target receptors by a compound with the given activity,
        in -Log scale, at the specified concentration in nM. If the given activity is pKi, a ratio to convert Ki into IC50
        is needed. It can be the ratio of concentration and Kd of the radioligand in a competitive binding assay, or the concentration
        of the substrate and Michaelis constant (Km) of enzymatic inhibition assay.
        """
        
        if self.params.is_ki:
            if self.params.ki_convert_ratio is None:
                raise Exception("Ki converting ratio is missing. Cannot convert Ki into IC50")
            Ki = 10**(9-activity)
            IC50 = Ki * (1 + self.params.ki_convert_ratio)
        else:
            IC50 = 10**(9-activity)
        pred_frac = 1.0/(1.0 + IC50/conc)
        
        return pred_frac

    def _l2_loss(self, yp, yr):
        """Da's loss function, based on L2 terms for both pKi and percent binding values
        This function is not appropriate for model fitting, but can be used for R^2 calculation.
        """
        yreal = yr.to("cpu").numpy()
        pos_ki = np.where(np.isnan(yreal[:,1]))[0]
        pos_bind = np.where(~np.isnan(yreal[:,1]))[0]
        loss_ki = torch.sum((yp[pos_ki, 0] - yr[pos_ki, 0]) ** 2)
        if len(pos_bind[0]) == 0:
            return loss_ki, torch.tensor(0.0, dtype=torch.float32)
        # convert Ki to % binding
        y_stds = self.transformers[0].y_stds
        y_means = self.transformers[0].y_means
        if self.params.is_ki:
            bind_pred = self._predict_binding(y_means + y_stds * yp[pos_bind, 0], conc=yr[pos_bind, 1])
        else:
            bind_pred = self._predict_binding(y_means + y_stds * yp[pos_bind, 0], conc=yr[pos_bind, 1])
        # calculate the loss_bind
        loss_bind = torch.sum((bind_pred - yr[pos_bind, 0]) ** 2)
        return loss_ki, loss_bind

    def _poisson_hybrid_loss(self, yp, yr):
        """Hybrid loss function based on L2 losses for deviations of predicted and measured pKi values
        and Poisson losses for predicted vs measured binding values. The idea is to choose loss terms
        that when minimized maximize the likelihood.

        Note that we compute both pKi and binding loss terms for compounds that have both kinds of data, since they are
        independent measurements. Therefore, pos_ki and pos_bind index sets may overlap.
        """

        # Get indices of non-missing pKi values
        yreal = yr.to("cpu").numpy()
        pos_ki = np.where(np.isnan(yreal[:,1]))[0]
        # Get indices of non-missing binding values
        pos_bind = np.where(~np.isnan(yreal[:,1]))[0]

        # Compute L2 loss for pKi predictions
        loss_ki = torch.sum((yp[pos_ki, 0] - yr[pos_ki, 0]) ** 2)
        #convert the ki prediction back to Ki scale
        y_stds = self.transformers[0].y_stds
        y_means = self.transformers[0].y_means
        # Compute fraction bound to *radioligand* (not drug) from predicted pKi
        if self.params.is_ki:
            rl_bind_pred = 1 - self._predict_binding(y_means + y_stds * yp[pos_bind, 0], conc=yr[pos_bind, 1])
        else:
            rl_bind_pred = 1 - self._predict_binding(y_means + y_stds * yp[pos_bind, 0], conc=yr[pos_bind, 1])
        rl_bind_real = 1 - yr[pos_bind, 0]
        # Compute Poisson loss for radioligand binding
        loss_bind = torch.sum(rl_bind_pred - rl_bind_real * torch.log(rl_bind_pred))

        if np.isnan(loss_ki.item()):
            raise Exception("Ki loss is NaN")
        if np.isnan(loss_bind.item()):
            raise Exception("Binding loss is NaN")
        return loss_ki, loss_bind

    def _loss_batch(self, loss_func, xb, yb, opt=None):
        """Compute loss_func for the batch xb, yb. If opt is provided, perform a training
        step on the model weights.
        """

        loss_ki, loss_bind = self.loss_func(self.model(xb), yb)
        loss = loss_ki + loss_bind
        
        if opt is not None:
            loss.backward()
            opt.step()   
            opt.zero_grad()

        return loss_ki.item(), loss_bind.item(), len(xb)

    class SubsetData(object):
        """Container for DataLoader object and attributes of a dataset subset"""
        def __init__(self, ds, dl, n_ki, n_bind):
            self.ds = ds
            self.dl = dl
            self.n_ki = n_ki
            self.n_bind = n_bind
    
    def _tensorize(self, x):
            return torch.tensor(x, dtype=torch.float32)

    def _load_hybrid_data(self, data):
        """Convert the DeepChem dataset into the SubsetData for hybrid model."""
        self.train_valid_dsets = []
        test_dset = data.test_dset
        num_folds = len(data.train_valid_dsets)

        for k in range(num_folds):
            train_dset, valid_dset = data.train_valid_dsets[k]
            # datasets were normalized in previous steps
            x_train, y_train, x_valid, y_valid = map(
                self._tensorize, (train_dset.X, train_dset.y, valid_dset.X, valid_dset.y)
            )
            # train
            train_ki_pos = np.where(np.isnan(y_train[:,1].numpy()))[0]
            train_bind_pos = np.where(~np.isnan(y_train[:,1].numpy()))[0]
            
            # valid
            valid_ki_pos = np.where(np.isnan(y_valid[:,1].numpy()))[0]
            valid_bind_pos = np.where(~np.isnan(y_valid[:,1].numpy()))[0]
            
            train_ds = TensorDataset(x_train, y_train)
            train_dl = DataLoader(train_ds, batch_size=self.params.batch_size, shuffle=True, pin_memory=True)
            train_data = self.SubsetData(train_ds, 
                                        train_dl, 
                                        len(train_ki_pos), 
                                        len(train_bind_pos))

            valid_ds = TensorDataset(x_valid, y_valid)
            valid_dl = DataLoader(valid_ds, batch_size=self.params.batch_size * 2, pin_memory=True)
            valid_data = self.SubsetData(valid_ds, 
                                        valid_dl, 
                                        len(valid_ki_pos), 
                                        len(valid_bind_pos))

            self.train_valid_dsets.append((train_data, valid_data))

        x_test, y_test = map(
            self._tensorize, (test_dset.X, test_dset.y)
        )
        test_ki_pos = np.where(np.isnan(y_test[:,1].numpy()))[0]
        test_bind_pos = np.where(~np.isnan(y_test[:,1].numpy()))[0]

        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=self.params.batch_size * 2, pin_memory=True)
        test_data = self.SubsetData(test_ds, 
                                    test_dl, 
                                    len(test_ki_pos), 
                                    len(test_bind_pos))

        self.test_data = test_data
    
    def save_model(self, checkpoint_file, model, opt, epoch, model_dict):
        """Save a model to a checkpoint file.
        Include epoch, model_dict in checkpoint dict.
        """
        checkpoint = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            opt_state_dict=opt.state_dict(),
            model_dict=model_dict
            )
        
        torch.save(checkpoint, checkpoint_file)

    def train(self, pipeline):
        if self.params.loss_func.lower() == "poisson":
            self.loss_func = self._poisson_hybrid_loss
        else:
            self.loss_func = self._l2_loss

        # load hybrid data
        self._load_hybrid_data(pipeline.data)

        checkpoint_file = os.path.join(self.model_dir, 
            f"{self.params.dataset_name}_model_{self.params.model_uuid}.pt")

        opt = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)

        em = perf.EpochManager(self,
                                prediction_type="hybrid",
                                model_dataset=pipeline.data,
                                transformers=self.transformers,
                                is_ki=self.params.is_ki,
                                production=self.params.production,
                                ki_convert_ratio=self.params.ki_convert_ratio)

        em.set_make_pred(lambda x: self.generate_predictions(x)[0])
        # initialize ei here so we can use it in the closure
        ei = 0
        em.on_new_best_valid(lambda : self.save_model(checkpoint_file, self.model, 
            opt, ei, self.model_dict))

        train_dset, valid_dset = pipeline.data.train_valid_dsets[0]
        if len(pipeline.data.train_valid_dsets) > 1:
            raise Exception("Currently the hybrid model  doesn't support K-fold cross validation splitting.")
        test_dset = pipeline.data.test_dset
        train_data, valid_data = self.train_valid_dsets[0]
        for ei in LCTimerIterator(self.params, pipeline, self.log):
            # Train the model for one epoch. We turn off automatic checkpointing, so the last checkpoint
            # saved will be the one we created intentionally when we reached a new best validation score.
            train_loss_ep = 0
            self.model.train()
            for i, (xb, yb) in enumerate(train_data.dl):
                xb = xb.to(self.dev)
                yb = yb.to(self.dev)
                train_loss_ki, train_loss_bind, train_count = self._loss_batch(self.loss_func, xb, yb, opt)
                train_loss_ep += (train_loss_ki + train_loss_bind)
            train_loss_ep /= (train_data.n_ki + train_data.n_bind)

            # validation set
            with torch.no_grad():
                valid_loss_ep = 0
                for xb, yb in valid_data.dl:
                    xb = xb.to(self.dev)
                    yb = yb.to(self.dev)
                    valid_loss_ki, valid_loss_bind, valid_count = self._loss_batch(self.loss_func, xb, yb)
                    valid_loss_ep += (valid_loss_ki + valid_loss_bind)
                valid_loss_ep /= (valid_data.n_ki + valid_data.n_bind)

            train_perf, valid_perf, test_perf = em.update_epoch(ei,
                                train_dset=train_dset, valid_dset=valid_dset, test_dset=test_dset)

            self.log.info("Epoch %d: training %s = %.3f, training loss = %.3f, validation %s = %.3f, validation loss = %.3f, test %s = %.3f" % (
                          ei, pipeline.metric_type, train_perf, train_loss_ep, pipeline.metric_type, valid_perf, valid_loss_ep,
                          pipeline.metric_type, test_perf))

            # Compute performance metrics for each subset, and check if we've reached a new best validation set score
            self.num_epochs_trained = ei + 1
            if em.should_stop():
                break
 
        # Revert to last checkpoint
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['opt_state_dict'])

        # copy the best model checkpoint file
        self._clean_up_excess_files(self.best_model_dir)
        shutil.copy2(checkpoint_file, self.best_model_dir)
        self.log.info(f"Best model from epoch {self.best_epoch} saved to {self.model_dir}")

    # ****************************************************************************************
    def reload_model(self, reload_dir):
        """Loads a saved neural net model from the specified directory.

        Args:
            reload_dir (str): Directory where saved model is located.
            model_dataset (ModelDataset Object): contains the current full dataset

        Side effects:
            Resets the value of model, transformers, and transformers_x
        """
        
        checkpoint_file = os.path.join(reload_dir, f"{self.params.dataset_name}_model_{self.params.model_uuid}.pt")
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.best_epoch = checkpoint["epoch"]
            self.model = torch.nn.Sequential(checkpoint["model_dict"]).to(self.dev)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            raise Exception(f"Checkpoint file doesn't exist in the reload_dir {reload_dir}")
        
        # Restore the transformers from the datastore or filesystem
        self.reload_transformers()

    # ****************************************************************************************
    def generate_predictions(self, dataset):
        """Generates predictions for specified dataset with current model, as well as standard deviations
        if params.uncertainty=True

        Args:
            dataset: the deepchem DiskDataset to generate predictions for

        Returns:
            (pred, std): tuple of predictions for compounds and standard deviation estimates, if requested.
            Each element of tuple is a numpy array of shape (ncmpds, ntasks, nclasses), where nclasses = 1 for regression
            models.
        """
        pred, std = None, None
        self.log.info("Predicting values for current model")

        x_data, y_data = map(
            self._tensorize, (dataset.X, dataset.y)
        )
        has_conc = len(y_data.shape) > 1 and y_data.shape[1] > 1 and np.nan_to_num(y_data[:,1]).max() > 0
        data_ki_pos = np.where(np.isnan(y_data[:,1].numpy()))[0] if has_conc else np.where(y_data[:,0].numpy())[0]
        data_bind_pos = np.where(~np.isnan(y_data[:,1].numpy()))[0] if has_conc else np.array([])

        data_ds = TensorDataset(x_data, y_data)
        data_dl = DataLoader(data_ds, batch_size=self.params.batch_size * 2, pin_memory=True)
        data_data = self.SubsetData(data_ds, 
                                    data_dl, 
                                    len(data_ki_pos), 
                                    len(data_bind_pos))
        pred = []
        real = []
        for xb, yb in data_dl:
            xb = xb.to(self.dev)
            yb = yb.to(self.dev)
            yp = self.model(xb)
            for i in range(len(yb)):
                real.append(yb.to("cpu").numpy()[i])
                pred.append(yp.detach().to("cpu").numpy()[i])
        real = np.array(real)
        pred = np.array(pred)

        if self.params.transformers and self.transformers is not None:
            if has_conc:
                pred = np.concatenate((pred, real[:, [1]]), axis=1)
                pred = self.transformers[0].untransform(pred, isreal=False)
                pred_bind_pos = np.where(~np.isnan(pred[:, 1]))[0]
                pred[pred_bind_pos, 0] = self._predict_binding(pred[pred_bind_pos, 0], pred[pred_bind_pos, 1])
            else:
                pred = self.transformers[0].untransform(pred, isreal=False)
        else:
            if has_conc:
                pred = np.concatenate((pred, real[:, [1]]), axis=1)
        return pred, std

    # ****************************************************************************************
    def get_model_specific_metadata(self):
        """Returns a dictionary of parameter settings for this ModelWrapper object that are specific
        to hybrid models.

        Returns:
            model_spec_metdata (dict): A dictionary of the parameter sets for the HybridModelWrapper object.
                Parameters are saved under the key 'hybrid_specific' as a subdictionary.
        """
        nn_metadata = dict(
                    best_epoch = self.best_epoch,
                    max_epochs = self.params.max_epochs,
                    batch_size = self.params.batch_size,
                    layer_sizes = self.params.layer_sizes,
                    dropouts = self.params.dropouts,
                    learning_rate = self.params.learning_rate,
        )
        model_spec_metadata = dict(hybrid_specific = nn_metadata)
        return model_spec_metadata

    # ****************************************************************************************
    def _create_output_transformers(self, model_dataset):
        """Initialize transformers for responses and persist them for later.

        Args:
            model_dataset: The ModelDataset object that handles the current dataset

        Side effects:
            Overwrites the attributes:
                transformers: A list of deepchem transformation objects on response_col, only if conditions are met
        """
        # TODO: Just a warning, we may have response transformers for classification datasets in the future
        if self.params.prediction_type=='regression' and self.params.transformers==True:
            self.transformers = [trans.NormalizationTransformerHybrid(transform_y=True, dataset=model_dataset.dataset)]

# ****************************************************************************************
class ForestModelWrapper(ModelWrapper):
    """Wrapper class for DCRFModelWrapper and DCxgboostModelWrapper

    contains code that is similar between the two tree based classes
    """
    def __init__(self, params, featurizer, ds_client):
        """Initializes DCRFModelWrapper object.

        Args:
            params (Namespace object): contains all parameter information.

            featurizer (Featurization): Object managing the featurization of compounds
            ds_client: datastore client.
        """
        super().__init__(params, featurizer, ds_client)
        self.best_model_dir = os.path.join(self.output_dir, 'best_model')
        self.model_dir = self.best_model_dir
        os.makedirs(self.best_model_dir, exist_ok=True)

        self.model = self.make_dc_model(self.best_model_dir)

    # ****************************************************************************************
    def train(self, pipeline):
        """Trains a forest model and saves the trained model.

        Args:
            pipeline (ModelPipeline): The ModelPipeline instance for this model run.

        Returns:
            None

        Side effects:
            data (ModelDataset): contains the dataset, set in pipeline

            best_epoch (int): Set to 0, not applicable to deepchem random forest models

            train_perf_data (PerfData): Contains the predictions and performance of the training dataset

            valid_perf_data (PerfData): Contains the predictions and performance of the validation dataset

            train_perfs (dict): A dictionary of predicted values and metrics on the training dataset

            valid_perfs (dict): A dictionary of predicted values and metrics on the training dataset
        """

        self.data = pipeline.data
        self.best_epoch = None
        self.train_perf_data = perf.create_perf_data(self.params.prediction_type, pipeline.data, self.transformers,'train')
        self.valid_perf_data = perf.create_perf_data(self.params.prediction_type, pipeline.data, self.transformers, 'valid')
        self.test_perf_data = perf.create_perf_data(self.params.prediction_type, pipeline.data, self.transformers, 'test')

        test_dset = pipeline.data.test_dset

        num_folds = len(pipeline.data.train_valid_dsets)
        for k in range(num_folds):
            train_dset, valid_dset = pipeline.data.train_valid_dsets[k]
            self.model.fit(train_dset)

            train_pred = self.model.predict(train_dset, [])
            train_perf = self.train_perf_data.accumulate_preds(train_pred, train_dset.ids)

            valid_pred = self.model.predict(valid_dset, [])
            valid_perf = self.valid_perf_data.accumulate_preds(valid_pred, valid_dset.ids)

            test_pred = self.model.predict(test_dset, [])
            test_perf = self.test_perf_data.accumulate_preds(test_pred, test_dset.ids)
            self.log.info("Fold %d: training %s = %.3f, validation %s = %.3f, test %s = %.3f" % (
                          k, pipeline.metric_type, train_perf, pipeline.metric_type, valid_perf,
                             pipeline.metric_type, test_perf))


        # Compute mean and SD of performance metrics across validation sets for all folds
        self.train_perf, self.train_perf_std = self.train_perf_data.compute_perf_metrics()
        self.valid_perf, self.valid_perf_std = self.valid_perf_data.compute_perf_metrics()
        self.test_perf, self.test_perf_std = self.test_perf_data.compute_perf_metrics()

        # Compute score to be used for ranking model hyperparameter sets
        self.model_choice_score = self.valid_perf_data.model_choice_score(self.params.model_choice_score_type)

        if num_folds > 1:
            # For k-fold CV, retrain on the combined training and validation sets
            fit_dataset = self.data.combined_training_data()
            self.model.fit(fit_dataset)
        self.model_save()
        # The best model is just the single RF training run.
        self.best_epoch = 0

    # ****************************************************************************************
    def make_dc_model(self, model_dir):
        """Build a DeepChem model.

        Builds a model, wraps it in DeepChem's wrapper and returns it

        Args:
            model_dir (str): Directory where saved model is located.

        returns:
            A DeepChem model
        """
        raise NotImplementedError

    # ****************************************************************************************
    def reload_model(self, reload_dir):
        """Loads a saved random forest model from the specified directory. Also loads any transformers that
        were saved with it.

        Args:
            reload_dir (str): Directory where saved model is located.

            model_dataset (ModelDataset Object): contains the current full dataset

        Side effects:
            Resets the value of model, transformers, transformers_x and transformers_w

        """
        # Restore the transformers from the datastore or filesystem
        self.reload_transformers()
        self.model = self.make_dc_model(reload_dir)
        self.model.reload()

    # ****************************************************************************************
    def get_pred_results(self, subset, epoch_label=None):
        """Returns predicted values and metrics from a training, validation or test subset
        of the current dataset, or the full dataset.

        Args:
            subset: 'train', 'valid', 'test' or 'full' accordingly.

            epoch_label: ignored; this function always returns the results for the current model.

        Returns:
            A dictionary of parameter, value pairs, in the format expected by the
            prediction_results element of the ModelMetrics data.

        Raises:
            ValueError: if subset not in ['train','valid','test','full']

        """
        if subset == 'train':
            return self.get_train_valid_pred_results(self.train_perf_data)
        elif subset == 'valid':
            return self.get_train_valid_pred_results(self.valid_perf_data)
        elif subset == 'test':
            return self.get_train_valid_pred_results(self.test_perf_data)
        elif subset == 'full':
            return self.get_full_dataset_pred_results(self.data)
        else:
            raise ValueError("Unknown dataset subset '%s'" % subset)

    # ****************************************************************************************
    def get_perf_data(self, subset, epoch_label=None):
        """Returns predicted values and metrics from a training, validation or test subset
        of the current dataset, or the full dataset.

        Args:
            subset (str): may be 'train', 'valid', 'test' or 'full'

            epoch_label (not used in random forest, but kept as part of the method structure)

        Results:
            PerfData object: Subclass of perfdata object associated with the appropriate subset's split strategy and prediction type.

        Raises:
            ValueError: if subset not in ['train','valid','test','full']
        """

        if subset == 'train':
            return self.train_perf_data
        elif subset == 'valid':
            return self.valid_perf_data
        elif subset == 'test':
            #return self.get_test_perf_data(self.best_model_dir, self.data)
            return self.test_perf_data
        elif subset == 'full':
            return self.get_full_dataset_perf_data(self.data)
        else:
            raise ValueError("Unknown dataset subset '%s'" % subset)

    # ****************************************************************************************
    def _clean_up_excess_files(self, dest_dir):
        """Function to clean up extra model files left behind in the training process.
        Does not apply to Forest models.
        """
        return

# ****************************************************************************************
class DCRFModelWrapper(ForestModelWrapper):
    """Contains methods to load in a dataset, split and featurize the data, fit a model to the train dataset,
    generate predictions for an input dataset, and generate performance metrics for these predictions.

    Attributes:
        Set in __init__
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information
            featurization (Featurization object): The featurization object created outside of model_wrapper
            log (log): The logger
            output_dir (str): The parent path of the model directory
            transformers (list): Initialized as an empty list, stores the transformers on the response col
            transformers_x (list): Initialized as an empty list, stores the transformers on the featurizers
            model_dir (str): The subdirectory under output_dir that contains the model. Created in setup_model_dirs.
            best_model_dir (str): The subdirectory under output_dir that contains the best model. Created in setup_model_dirs
            model: The dc.models.sklearn_models.SklearnModel as specified by the params attribute

        Created in train:
            data (ModelDataset): contains the dataset, set in pipeline
            best_epoch (int): Set to 0, not applicable to deepchem random forest models
            train_perf_data (PerfData): Contains the predictions and performance of the training dataset
            valid_perf_data (PerfData): Contains the predictions and performance of the validation dataset
            train_perfs (dict): A dictionary of predicted values and metrics on the training dataset
            valid_perfs (dict): A dictionary of predicted values and metrics on the training dataset

    """

    def __init__(self, params, featurizer, ds_client):
        """Initializes DCRFModelWrapper object.

        Args:
            params (Namespace object): contains all parameter information.

            featurizer (Featurization): Object managing the featurization of compounds
            ds_client: datastore client.
        """
        super().__init__(params, featurizer, ds_client)

    # ****************************************************************************************
    def make_dc_model(self, model_dir):
        """Build a DeepChem model.

        Builds a model, wraps it in DeepChem's wrapper and returns it

        Args:
            model_dir (str): Directory where saved model is located.

        returns:
            A DeepChem model
        """
        if self.params.prediction_type == 'regression':
            rf_model = RandomForestRegressor(n_estimators=self.params.rf_estimators,
                                             max_features=self.params.rf_max_features,
                                             max_depth=self.params.rf_max_depth,
                                             n_jobs=-1)
        else:
            rf_model = RandomForestClassifier(n_estimators=self.params.rf_estimators,
                                              max_features=self.params.rf_max_features,
                                              max_depth=self.params.rf_max_depth,
                                              n_jobs=-1)

        return dc.models.sklearn_models.SklearnModel(rf_model, model_dir=model_dir)

    # ****************************************************************************************
    def train(self, pipeline):
        """Trains a random forest model and saves the trained model.

        Args:
            pipeline (ModelPipeline): The ModelPipeline instance for this model run.

        Returns:
            None

        Side effects:
            data (ModelDataset): contains the dataset, set in pipeline

            best_epoch (int): Set to 0, not applicable to deepchem random forest models

            train_perf_data (PerfData): Contains the predictions and performance of the training dataset

            valid_perf_data (PerfData): Contains the predictions and performance of the validation dataset

            train_perfs (dict): A dictionary of predicted values and metrics on the training dataset

            valid_perfs (dict): A dictionary of predicted values and metrics on the training dataset
        """
        self.log.info("Fitting random forest model")
        super().train(pipeline)

    # ****************************************************************************************
    def generate_predictions(self, dataset):
        """Generates predictions for specified dataset, as well as uncertainty values if params.uncertainty=True

        Args:
            dataset: the deepchem DiskDataset to generate predictions for

        Returns:
            (pred, std): numpy arrays containing predictions for compounds and the standard error estimates.

        """
        pred, std = None, None
        self.log.info("Evaluating current model")

        pred = self.model.predict(dataset, self.transformers)
        ncmpds = pred.shape[0]
        pred = pred.reshape((ncmpds,1,-1))

        if self.params.uncertainty:
            if self.params.prediction_type == 'regression':
                rf_model = joblib.load(os.path.join(self.best_model_dir, 'model.joblib'))
                ## s.d. from forest
                if self.params.transformers and self.transformers is not None:
                    RF_per_tree_pred = [dc.trans.undo_transforms(
                        tree.predict(dataset.X), self.transformers) for tree in rf_model.estimators_]
                else:
                    RF_per_tree_pred = [tree.predict(dataset.X) for tree in rf_model.estimators_]

                # Don't need to "untransform" standard deviations here, since they're calculated from
                # the untransformed per-tree predictions.
                std = np.array([np.std(col) for col in zip(*RF_per_tree_pred)]).reshape((ncmpds,1,-1))
            else:
                # We can estimate uncertainty for binary classifiers, but not multiclass (yet)
                nclasses = pred.shape[2]
                if nclasses == 2:
                    ntrees = self.params.rf_estimators
                    # Use normal approximation to binomial sampling error. Later we can do Jeffrey's interval if we
                    # want to get fancy.
                    std = np.sqrt(pred * (1-pred) / ntrees)
                else:
                    self.log.warning("Warning: Random forest only supports uncertainties for binary classifiers.")

        return pred, std

    # ****************************************************************************************
    def get_model_specific_metadata(self):
        """Returns a dictionary of parameter settings for this ModelWrapper object that are specific
        to random forest models.

        Returns:
            model_spec_metadata (dict): Returns random forest specific metadata as a subdict under the key 'rf_specific'

        """
        rf_metadata = {
            'rf_estimators': self.params.rf_estimators,
            'rf_max_features': self.params.rf_max_features,
            'rf_max_depth': self.params.rf_max_depth
        }
        model_spec_metadata = dict(rf_specific = rf_metadata)
        return model_spec_metadata
    
# ****************************************************************************************
class DCxgboostModelWrapper(ForestModelWrapper):
    """Contains methods to load in a dataset, split and featurize the data, fit a model to the train dataset,
    generate predictions for an input dataset, and generate performance metrics for these predictions.

    Attributes:
        Set in __init__
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information
            featurization (Featurization object): The featurization object created outside of model_wrapper
            log (log): The logger
            output_dir (str): The parent path of the model directory
            transformers (list): Initialized as an empty list, stores the transformers on the response cols
            transformers_x (list): Initialized as an empty list, stores the transformers on the features
            transformers_w (list): Initialized as an empty list, stores the transformers on the weights
            model_dir (str): The subdirectory under output_dir that contains the model. Created in setup_model_dirs.
            best_model_dir (str): The subdirectory under output_dir that contains the best model. Created in setup_model_dirs
            model: The dc.models.sklearn_models.SklearnModel as specified by the params attribute

        Created in train:
            data (ModelDataset): contains the dataset, set in pipeline
            best_epoch (int): Set to 0, not applicable
            train_perf_data (PerfObjects): Contains the predictions and performance of the training dataset
            valid_perf_data (PerfObjects): Contains the predictions and performance of the validation dataset
            train_perfs (dict): A dictionary of predicted values and metrics on the training dataset
            valid_perfs (dict): A dictionary of predicted values and metrics on the validation dataset

    """

    def __init__(self, params, featurizer, ds_client):
        """Initializes RunModel object.

        Args:
            params (Namespace object): contains all parameter information.

            featurizer (Featurization): Object managing the featurization of compounds
            ds_client: datastore client.
        """
        super().__init__(params, featurizer, ds_client)

    # ****************************************************************************************
    def make_dc_model(self, model_dir):
        """Build a DeepChem model.

        Builds a model, wraps it in DeepChem's wrapper and returns it

        Args:
            model_dir (str): Directory where saved model is located.

        returns:
            A DeepChem model
        """
        if self.params.prediction_type == 'regression':
            xgb_model = xgb.XGBRegressor(max_depth=self.params.xgb_max_depth,
                                         learning_rate=self.params.xgb_learning_rate,
                                         n_estimators=self.params.xgb_n_estimators,
                                         silent=True,
                                         objective='reg:squarederror',
                                         booster='gbtree',
                                         gamma=self.params.xgb_gamma,
                                         min_child_weight=self.params.xgb_min_child_weight,
                                         max_delta_step=0,
                                         subsample=self.params.xgb_subsample,
                                         colsample_bytree=self.params.xgb_colsample_bytree,
                                         colsample_bylevel=1,
                                         reg_alpha=0,
                                         reg_lambda=1,
                                         scale_pos_weight=1,
                                         base_score=0.5,
                                         random_state=0,
                                         missing=np.nan,
                                         importance_type='gain',
                                         n_jobs=-1,
                                         gpu_id = -1,
                                         n_gpus = 0,
                                         max_bin = 16,
                                         )
        else:
            xgb_model = xgb.XGBClassifier(max_depth=self.params.xgb_max_depth,
                                         learning_rate=self.params.xgb_learning_rate,
                                         n_estimators=self.params.xgb_n_estimators,
                                          silent=True,
                                          objective='binary:logistic',
                                          booster='gbtree',
                                          gamma=self.params.xgb_gamma,
                                          min_child_weight=self.params.xgb_min_child_weight,
                                          max_delta_step=0,
                                          subsample=self.params.xgb_subsample,
                                          colsample_bytree=self.params.xgb_colsample_bytree,
                                          colsample_bylevel=1,
                                          reg_alpha=0,
                                          reg_lambda=1,
                                          scale_pos_weight=1,
                                          base_score=0.5,
                                          random_state=0,
                                          importance_type='gain',
                                          missing=np.nan,
                                          gpu_id = -1,
                                          n_jobs=-1,                                          
                                          n_gpus = 0,
                                          max_bin = 16,
                                         )

        return dc.models.sklearn_models.SklearnModel(xgb_model, model_dir=model_dir)

    # ****************************************************************************************
    def train(self, pipeline):
        """Trains a xgboost model and saves the trained model.

        Args:
            pipeline (ModelPipeline): The ModelPipeline instance for this model run.

        Returns:
            None

        Side effects:
            data (ModelDataset): contains the dataset, set in pipeline

            best_epoch (int): Set to 0, not applicable to deepchem xgboost models

            train_perf_data (PerfData): Contains the predictions and performance of the training dataset

            valid_perf_data (PerfData): Contains the predictions and performance of the validation dataset

            train_perfs (dict): A dictionary of predicted values and metrics on the training dataset

            valid_perfs (dict): A dictionary of predicted values and metrics on the training dataset
        """
        self.log.info("Fitting xgboost model")

        self.data = pipeline.data
        self.best_epoch = None
        self.train_perf_data = perf.create_perf_data(self.params.prediction_type, pipeline.data, self.transformers,'train')
        self.valid_perf_data = perf.create_perf_data(self.params.prediction_type, pipeline.data, self.transformers, 'valid')
        self.test_perf_data = perf.create_perf_data(self.params.prediction_type, pipeline.data, self.transformers, 'test')

        test_dset = pipeline.data.test_dset

        num_folds = len(pipeline.data.train_valid_dsets)
        for k in range(num_folds):
            train_dset, valid_dset = pipeline.data.train_valid_dsets[k]
            self.model.fit(train_dset)

            train_pred = self.model.predict(train_dset, [])
            train_perf = self.train_perf_data.accumulate_preds(train_pred, train_dset.ids)

            valid_pred = self.model.predict(valid_dset, [])
            valid_perf = self.valid_perf_data.accumulate_preds(valid_pred, valid_dset.ids)

            test_pred = self.model.predict(test_dset, [])
            test_perf = self.test_perf_data.accumulate_preds(test_pred, test_dset.ids)
            self.log.info("Fold %d: training %s = %.3f, validation %s = %.3f, test %s = %.3f" % (
                          k, pipeline.metric_type, train_perf, pipeline.metric_type, valid_perf,
                             pipeline.metric_type, test_perf))

        # Compute mean and SD of performance metrics across validation sets for all folds
        self.train_perf, self.train_perf_std = self.train_perf_data.compute_perf_metrics()
        self.valid_perf, self.valid_perf_std = self.valid_perf_data.compute_perf_metrics()
        self.test_perf, self.test_perf_std = self.test_perf_data.compute_perf_metrics()

        # Compute score to be used for ranking model hyperparameter sets
        self.model_choice_score = self.valid_perf_data.model_choice_score(self.params.model_choice_score_type)

        if num_folds > 1:
            # For k-fold CV, retrain on the combined training and validation sets
            fit_dataset = self.data.combined_training_data()
            self.model.fit(fit_dataset)
        self.model_save()
        # The best model is just the single xgb training run.
        self.best_epoch = 0

    # ****************************************************************************************
    def reload_model(self, reload_dir):

        """Loads a saved xgboost model from the specified directory. Also loads any transformers that
        were saved with it.

        Args:
            reload_dir (str): Directory where saved model is located.

            model_dataset (ModelDataset Object): contains the current full dataset

        Side effects:
            Resets the value of model, transformers, transformers_x and transformers_w

        """

        if self.params.prediction_type == 'regression':
            xgb_model = xgb.XGBRegressor(max_depth=self.params.xgb_max_depth,
                                         learning_rate=self.params.xgb_learning_rate,
                                         n_estimators=self.params.xgb_n_estimators,
                                         silent=True,
                                         objective='reg:squarederror',
                                         booster='gbtree',
                                         gamma=self.params.xgb_gamma,
                                         min_child_weight=self.params.xgb_min_child_weight,
                                         max_delta_step=0,
                                         subsample=self.params.xgb_subsample,
                                         colsample_bytree=self.params.xgb_colsample_bytree,
                                         colsample_bylevel=1,
                                         reg_alpha=0,
                                         reg_lambda=1,
                                         scale_pos_weight=1,
                                         base_score=0.5,
                                         random_state=0,
                                         missing=np.nan,
                                         importance_type='gain',
                                         n_jobs=-1,
                                         gpu_id = -1,
                                         n_gpus = 0,
                                         max_bin = 16,
                                         )
        else:
            xgb_model = xgb.XGBClassifier(max_depth=self.params.xgb_max_depth,
                                         learning_rate=self.params.xgb_learning_rate,
                                         n_estimators=self.params.xgb_n_estimators,
                                         silent=True,
                                         objective='binary:logistic',
                                         booster='gbtree',
                                         gamma=self.params.xgb_gamma,
                                         min_child_weight=self.params.xgb_min_child_weight,
                                         max_delta_step=0,
                                         subsample=self.params.xgb_subsample,
                                         colsample_bytree=self.params.xgb_colsample_bytree,
                                         colsample_bylevel=1,
                                         reg_alpha=0,
                                         reg_lambda=1,
                                         scale_pos_weight=1,
                                         base_score=0.5,
                                         random_state=0,
                                         importance_type='gain',
                                         missing=np.nan,
                                         gpu_id = -1,
                                         n_jobs=-1,                                          
                                         n_gpus = 0,
                                         max_bin = 16,
                                         )

        # Restore the transformers from the datastore or filesystem
        self.reload_transformers()

        self.model = dc.models.GBDTModel(xgb_model, model_dir=self.best_model_dir)
        self.model.reload()

    # ****************************************************************************************
    def get_pred_results(self, subset, epoch_label=None):
        """Returns predicted values and metrics from a training, validation or test subset
        of the current dataset, or the full dataset.

        Args:
            subset: 'train', 'valid', 'test' or 'full' accordingly.

            epoch_label: ignored; this function always returns the results for the current model.

        Returns:
            A dictionary of parameter, value pairs, in the format expected by the
            prediction_results element of the ModelMetrics data.

        Raises:
            ValueError: if subset not in ['train','valid','test','full']

        """
        if subset == 'train':
            return self.get_train_valid_pred_results(self.train_perf_data)
        elif subset == 'valid':
            return self.get_train_valid_pred_results(self.valid_perf_data)
        elif subset == 'test':
            return self.get_train_valid_pred_results(self.test_perf_data)
        elif subset == 'full':
            return self.get_full_dataset_pred_results(self.data)
        else:
            raise ValueError("Unknown dataset subset '%s'" % subset)

    # ****************************************************************************************
    def get_perf_data(self, subset, epoch_label=None):
        """Returns predicted values and metrics from a training, validation or test subset
        of the current dataset, or the full dataset.

        Args:
            subset (str): may be 'train', 'valid', 'test' or 'full'

            epoch_label (not used in random forest, but kept as part of the method structure)

        Results:
            PerfData object: Subclass of perfdata object associated with the appropriate subset's split strategy and prediction type.

        Raises:
            ValueError: if subset not in ['train','valid','test','full']
        """

        if subset == 'train':
            return self.train_perf_data
        elif subset == 'valid':
            return self.valid_perf_data
        elif subset == 'test':
            #return self.get_test_perf_data(self.best_model_dir, self.data)
            return self.test_perf_data
        elif subset == 'full':
            return self.get_full_dataset_perf_data(self.data)
        else:
            raise ValueError("Unknown dataset subset '%s'" % subset)

    # ****************************************************************************************
    def generate_predictions(self, dataset):
        """Generates predictions for specified dataset, as well as uncertainty values if params.uncertainty=True

        Args:
            dataset: the deepchem DiskDataset to generate predictions for

        Returns:
            (pred, std): numpy arrays containing predictions for compounds and the standard error estimates.

        """
        pred, std = None, None
        self.log.warning("Evaluating current model")

        pred = self.model.predict(dataset, self.transformers)
        ncmpds = pred.shape[0]
        pred = pred.reshape((ncmpds, 1, -1))
        self.log.warning("uncertainty not supported by xgboost models")

        return pred, std

    # ****************************************************************************************
    def get_model_specific_metadata(self):
        """Returns a dictionary of parameter settings for this ModelWrapper object that are specific
        to xgboost models.

        Returns:
            model_spec_metadata (dict): Returns xgboost specific metadata as a subdict under the key 'xgb_specific'

        """
        xgb_metadata = {"xgb_max_depth" : self.params.xgb_max_depth,
                       "xgb_learning_rate" : self.params.xgb_learning_rate,
                       "xgb_n_estimators" : self.params.xgb_n_estimators,
                       "xgb_gamma" : self.params.xgb_gamma,
                       "xgb_min_child_weight" : self.params.xgb_min_child_weight,
                       "xgb_subsample" : self.params.xgb_subsample,
                       "xgb_colsample_bytree"  :self.params.xgb_colsample_bytree
                        }
        model_spec_metadata = dict(xgb_specific=xgb_metadata)
        return model_spec_metadata

    # ****************************************************************************************
    def _clean_up_excess_files(self, dest_dir):
        """Function to clean up extra model files left behind in the training process.
        Does not apply to xgboost
        """
        return

# ****************************************************************************************
class PytorchDeepChemModelWrapper(NNModelWrapper):
    """Implementation of AttentiveFP model from Xiong et al. [1]_. It uses a graph attention model
    to propagate information from bond and neighboring atom features across a molecule represented as
    a graph.

    References:
        .. [1] Xiong, Zhaoping et al. "Pushing the Boundaries of Molecular Representation for Drug Discovery
           with the Graph Attention Mechanism." Journal of Medicinal Chemistry (2019) doi: 10.1021/acs.jmedchem.0b00959

    Attributes:
        Set in __init__
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information
            featurization (Featurization object): The featurization object created outside of model_wrapper
            log (log): The logger
            output_dir (str): The parent path of the model directory
            transformers (list): Initialized as an empty list, stores the transformers on the response col
            transformers_x (list): Initialized as an empty list, stores the transformers on the featurizers
            model_dir (str): The subdirectory under output_dir that contains the model. Created in setup_model_dirs.
            best_model_dir (str): The subdirectory under output_dir that contains the best model. Created in setup_model_dirs
            model: The dc.models.sklearn_models.SklearnModel as specified by the params attribute

        Created in train:
            data (ModelDataset): contains the dataset, set in pipeline
            best_epoch (int): Set to 0, not applicable
            train_perf_data (PerfObjects): Contains the predictions and performance of the training dataset
            valid_perf_data (PerfObjects): Contains the predictions and performance of the validation dataset
            train_perfs (dict): A dictionary of predicted values and metrics on the training dataset
            valid_perfs (dict): A dictionary of predicted values and metrics on the validation dataset

    """
    def __init__(self, params, featurizer, ds_client):
        """Initializes AttentiveFPModelWrapper object. Creates the underlying DeepChem AttentiveFPModel instance.

        Args:
            params (Namespace object): contains all parameter information.

            featurizer (Featurization): Object managing the featurization of compounds
            ds_client: datastore client.
        """
        # use NNModelWrapper init. 
        super().__init__(params, featurizer, ds_client)
        self.num_epochs_trained = 0

        self.model = self.recreate_model()

    # ****************************************************************************************
    def recreate_model(self, **kwargs):
        """Creates a new DeepChem Model object of the correct type for the requested featurizer and prediction type
        and returns it.

        Args:
            kwargs: These arguments are used to overwrite parameters set in self.params and
                are passed to the underlying deepchem model object. e.g. model_dir is set in self.reload_model
        """
        # extract parameters specific to this model
        extracted_features = pp.extract_model_params(self.params)

        # model_dir is set and handled by AMPL.
        extracted_features.update({'model_dir': self.model_dir})

        # parameters can be overwritten by passing them explicitly
        extracted_features.update(kwargs)

        chosen_model = pp.model_wl[self.params.model_type]
        self.log.info(f'Args passed to {chosen_model}:{str(extracted_features)}')

        # build the model
        model = chosen_model(
                **extracted_features
            ) 

        return model

    # ****************************************************************************************
    def reload_model(self, reload_dir):
        """Loads a saved neural net model from the specified directory.

        Args:
            reload_dir (str): Directory where saved model is located.
            model_dataset (ModelDataset Object): contains the current full dataset

        Side effects:
            Resets the value of model, transformers, and transformers_x
        """
        self.model = self.recreate_model(model_dir=reload_dir)
        # checkpoint with the highest number is the best one
        best_chkpt = get_latest_pytorch_checkpoint(self.model)
        self.restore(best_chkpt, reload_dir)

        # Restore the transformers from the datastore or filesystem
        self.reload_transformers()

    # ****************************************************************************************
    def get_model_specific_metadata(self):
        """Returns a dictionary of parameter settings for this ModelWrapper object that are specific
        to neural network models.

        Returns:
            model_spec_metdata (dict): A dictionary of the parameter sets for the NNModelWrapper object.
                Parameters are saved under the key 'nn_specific' as a subdictionary.
        """
        nn_metadata = pp.extract_model_params(self.params, strip_prefix=False)
        nn_metadata['max_epochs'] = self.params.max_epochs
        nn_metadata['best_epoch'] = self.best_epoch
        model_spec_metadata = dict(nn_specific = nn_metadata)
        return model_spec_metadata

    def restore(self, checkpoint=None, model_dir=None):
        """Restores this model"""
        dc_torch_restore(self.model, checkpoint, model_dir)

    def count_params(self):
        """Returns the number of trainable parameters

        There's no function implemented in Pytorch that does this so I'm using the
        solution found here:https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/25

        Args:
            None

        Returns
            Int- the number of trainable parameters
        """
        pytorch_total_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        return pytorch_total_params

# ****************************************************************************************
class MultitaskDCModelWrapper(PytorchDeepChemModelWrapper):
    """Contains methods to load in a dataset, split and featurize the data, fit a model to the train dataset,
    generate predictions for an input dataset, and generate performance metrics for these predictions.

    Attributes:
        Set in __init__
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information
            featurziation (Featurization object): The featurization object created outside of model_wrapper

            log (log): The logger

            output_dir (str): The parent path of the model directory

            transformers (list): Initialized as an empty list, stores the transformers on the response col

            transformers_x (list): Initialized as an empty list, stores the transformers on the featurizers

            model_dir (str): The subdirectory under output_dir that contains the model. Created in setup_model_dirs.

            best_model_dir (str): The subdirectory under output_dir that contains the best model. Created in setup_model_dirs

            g: The tensorflow graph object

            sess: The tensor flow graph session

            model: The dc.models.GraphConvModel, MultitaskRegressor, or MultitaskClassifier object, as specified by the params attribute

        Created in train:
            data (ModelDataset): contains the dataset, set in pipeline

            best_epoch (int): Initialized as None, keeps track of the epoch with the best validation score

            train_perf_data (np.array of PerfData): Initialized as an empty array,
                contains the predictions and performance of the training dataset

            valid_perf_data (np.array of PerfData): Initialized as an empty array,
                contains the predictions and performance of the validation dataset

            train_epoch_perfs (np.array of dicts): Initialized as an empty array,
                contains a list of dictionaries of predicted values and metrics on the training dataset

            valid_epoch_perfs (np.array of dicts): Initialized as an empty array,
                contains a list of dictionaries of predicted values and metrics on the validation dataset

    """

    def recreate_model(self, model_dir=None):
        """Creates a new DeepChem Model object of the correct type for the requested featurizer and prediction type
        and returns it.

        reload_dir (str): Directory where saved model is located.
        """
        if model_dir is None:
            model_dir = self.model_dir

        n_features = self.get_num_features()
        if self.params.layer_sizes is None:
            if self.params.featurizer == 'ecfp':
                self.params.layer_sizes = [1000, 500]
            elif self.params.featurizer in ['descriptors', 'computed_descriptors']:
                self.params.layer_sizes = [200, 100]
            else:
                # Shouldn't happen
                self.log.warning("You need to define default layer sizes for featurizer %s" %
                                    self.params.featurizer)
                self.params.layer_sizes = [1000, 500]

        if self.params.dropouts is None:
            self.params.dropouts = [0.4] * len(self.params.layer_sizes)
        if self.params.weight_init_stddevs is None:
            self.params.weight_init_stddevs = [0.02] * len(self.params.layer_sizes)
        if self.params.bias_init_consts is None:
            self.params.bias_init_consts = [1.0] * len(self.params.layer_sizes)

        if self.params.prediction_type == 'regression':

            # TODO: Need to check that MultitaskRegressor params are actually being used
            model = MultitaskRegressor(
                self.params.num_model_tasks,
                n_features,
                layer_sizes=self.params.layer_sizes,
                dropouts=self.params.dropouts,
                weight_init_stddevs=self.params.weight_init_stddevs,
                bias_init_consts=self.params.bias_init_consts,
                learning_rate=self.params.learning_rate,
                weight_decay_penalty=self.params.weight_decay_penalty,
                weight_decay_penalty_type=self.params.weight_decay_penalty_type,
                batch_size=self.params.batch_size,
                verbosity='low',
                model_dir=model_dir,
                learning_rate_decay_time=1000,
                beta1=0.9,
                beta2=0.999,
                mode=self.params.prediction_type,
                tensorboard=False,
                uncertainty=self.params.uncertainty)
        else:
            # TODO: Need to check that MultitaskClassifier params are actually being used
            model = MultitaskClassifier(
                self.params.num_model_tasks,
                n_features,
                layer_sizes=self.params.layer_sizes,
                dropouts=self.params.dropouts,
                weight_init_stddevs=self.params.weight_init_stddevs,
                bias_init_consts=self.params.bias_init_consts,
                learning_rate=self.params.learning_rate,
                weight_decay_penalty=self.params.weight_decay_penalty,
                weight_decay_penalty_type=self.params.weight_decay_penalty_type,
                batch_size=self.params.batch_size,
                verbosity='low',
                model_dir=model_dir,
                learning_rate_decay_time=1000,
                beta1=.9,
                beta2=.999,
                mode=self.params.prediction_type,
                tensorboard=False,
                n_classes=self.params.class_number)

        return model

    # ****************************************************************************************
    def generate_embeddings(self, dataset):
        """Generate the output of the final embedding layer of a fully connected NN model for the given dataset.

        Args:
            dataset:

        Returns:
            embedding (np.ndarray): An array of outputs from the nodes in the embedding layer.

        """
        return self.model.predict_embedding(dataset)


    # ****************************************************************************************
    def get_model_specific_metadata(self):
        """Returns a dictionary of parameter settings for this ModelWrapper object that are specific
        to neural network models.

        Returns:
            model_spec_metdata (dict): A dictionary of the parameter sets for the MultitaskDCModelWrapper object.
                Parameters are saved under the key 'nn_specific' as a subdictionary.
        """
        
        nn_metadata = dict(
                    best_epoch = self.best_epoch,
                    max_epochs = self.params.max_epochs,
                    batch_size = self.params.batch_size,
                    optimizer_type = self.params.optimizer_type,
                    layer_sizes = self.params.layer_sizes,
                    dropouts = self.params.dropouts,
                    weight_init_stddevs = self.params.weight_init_stddevs,
                    bias_init_consts = self.params.bias_init_consts,
                    learning_rate = self.params.learning_rate,
                    weight_decay_penalty=self.params.weight_decay_penalty,
                    weight_decay_penalty_type=self.params.weight_decay_penalty_type
        )
        model_spec_metadata = dict(nn_specific = nn_metadata)
        return model_spec_metadata

# ****************************************************************************************
class KerasDeepChemModelWrapper(PytorchDeepChemModelWrapper):
    def _copy_model(self, dest_dir):
        """Copies the files needed to recreate a DeepChem NN model from the current model
        directory to a destination directory.

        Args:
            dest_dir (str): The destination directory for the model files
        """

        chkpt_file = os.path.join(self.model_dir, 'checkpoint')
        with open(chkpt_file, 'r') as chkpt_in:
            chkpt_dict = yaml.load(chkpt_in.read())
        chkpt_prefix = chkpt_dict['model_checkpoint_path']
        files = [chkpt_file]
        # files.append(os.path.join(self.model_dir, 'model.pickle'))
        files.append(os.path.join(self.model_dir, '%s.index' % chkpt_prefix))
        # files.append(os.path.join(self.model_dir, '%s.meta' % chkpt_prefix))
        files = files + glob.glob(os.path.join(self.model_dir, '%s.data-*' % chkpt_prefix))
        self._clean_up_excess_files(dest_dir)
        for file in files:
            shutil.copy2(file, dest_dir)
        self.log.info("Saved model files to '%s'" % dest_dir)

    def reload_model(self, reload_dir):
        """Loads a saved neural net model from the specified directory.

        Args:
            reload_dir (str): Directory where saved model is located.
            model_dataset (ModelDataset Object): contains the current full dataset

        Side effects:
            Resets the value of model, transformers, and transformers_x
        """
        self.model = self.recreate_model(model_dir=reload_dir)

        # Get latest checkpoint path transposed to current model dir
        ckpt = tf.train.get_checkpoint_state(reload_dir)
        if os.path.exists(f"{ckpt.model_checkpoint_path}.index"):
            checkpoint = ckpt.model_checkpoint_path
        else:
            checkpoint = os.path.join(reload_dir, os.path.basename(ckpt.model_checkpoint_path))
        self.restore(checkpoint=checkpoint)

        # Restore the transformers from the datastore or filesystem
        self.reload_transformers()

    def restore(self, checkpoint=None, model_dir=None, session=None):
        """Restores this model"""
        dc_restore(self.model, checkpoint, model_dir, session)

    def count_params(self):
        """Returns
            the number of trainable parameters using Keras' count_params function

        Args:
            None

        Returns
            Int- the number of trainable parameters using Keras' count_params function
        """

        return count_params(self.model.model.trainable_weights)

# ****************************************************************************************
class GraphConvDCModelWrapper(KerasDeepChemModelWrapper):
    """Contains methods to load in a dataset, split and featurize the data, fit a model to the train dataset,
    generate predictions for an input dataset, and generate performance metrics for these predictions.

    Attributes:
        Set in __init__
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information
            featurziation (Featurization object): The featurization object created outside of model_wrapper

            log (log): The logger

            output_dir (str): The parent path of the model directory

            transformers (list): Initialized as an empty list, stores the transformers on the response col

            transformers_x (list): Initialized as an empty list, stores the transformers on the featurizers

            model_dir (str): The subdirectory under output_dir that contains the model. Created in setup_model_dirs.

            best_model_dir (str): The subdirectory under output_dir that contains the best model. Created in setup_model_dirs

            g: The tensorflow graph object

            sess: The tensor flow graph session

            model: The dc.models.GraphConvModel, MultitaskRegressor, or MultitaskClassifier object, as specified by the params attribute

        Created in train:
            data (ModelDataset): contains the dataset, set in pipeline

            best_epoch (int): Initialized as None, keeps track of the epoch with the best validation score

            train_perf_data (np.array of PerfData): Initialized as an empty array,
                contains the predictions and performance of the training dataset

            valid_perf_data (np.array of PerfData): Initialized as an empty array,
                contains the predictions and performance of the validation dataset

            train_epoch_perfs (np.array of dicts): Initialized as an empty array,
                contains a list of dictionaries of predicted values and metrics on the training dataset

            valid_epoch_perfs (np.array of dicts): Initialized as an empty array,
                contains a list of dictionaries of predicted values and metrics on the validation dataset

    """

    def __init__(self, params, featurizer, ds_client):
        """Initializes GraphConvDCModelWrapper object.

        Args:
            params (Namespace object): contains all parameter information.

            featurizer (Featurizer object): initialized outside of model_wrapper

        Side effects:
            params (argparse.Namespace): The argparse.Namespace parameter object that contains all parameter information

            featurziation (Featurization object): The featurization object created outside of model_wrapper

            log (log): The logger

            output_dir (str): The parent path of the model directory

            transformers (list): Initialized as an empty list, stores the transformers on the response col

            transformers_x (list): Initialized as an empty list, stores the transformers on the featurizers

            g: The tensorflow graph object

            sess: The tensor flow graph session

            model: The dc.models.GraphConvModel, MultitaskRegressor, or MultitaskClassifier object, as specified by the params attribute

        """
        super().__init__(params, featurizer, ds_client)
        # TODO (ksm): The next two attributes aren't used; suggest we drop them.
        self.g = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.g)
        self.num_epochs_trained = 0

        self.model = self.recreate_model(model_dir=self.model_dir)

    # ****************************************************************************************
    def recreate_model(self, model_dir=None):
        """Creates a new DeepChem Model object of the correct type for the requested featurizer and prediction type
        and returns it.

        reload_dir (str): Directory where saved model is located.
        """
        if model_dir is None:
            model_dir = self.model_dir

        # Set defaults for layer sizes and dropouts, if not specified by caller. Note that
        if self.params.layer_sizes is None:
            self.params.layer_sizes = [64, 64, 128]
        if self.params.dropouts is None:
            if self.params.uncertainty:
                self.params.dropouts = [0.25] * len(self.params.layer_sizes)
            else:
                self.params.dropouts = [0.0] * len(self.params.layer_sizes)

        model = dc.models.GraphConvModel(
            self.params.num_model_tasks,
            batch_size=self.params.batch_size,
            learning_rate=self.params.learning_rate,
            learning_rate_decay_time=1000,
            optimizer_type=self.params.optimizer_type,
            beta1=0.9,
            beta2=0.999,
            model_dir=model_dir,
            mode=self.params.prediction_type,
            tensorboard=False,
            uncertainty=self.params.uncertainty,
            graph_conv_layers=self.params.layer_sizes[:-1],
            dense_layer_size=self.params.layer_sizes[-1],
            dropout=self.params.dropouts,
            penalty=self.params.weight_decay_penalty,
            penalty_type=self.params.weight_decay_penalty_type)
        return model

    # ****************************************************************************************
    def generate_embeddings(self, dataset):
        """Generate the output of the final embedding layer of a GraphConv model for the given dataset.

        Args:
            dataset:

        Returns:
            embedding (np.ndarray): An array of outputs from the nodes in the embedding layer.

        """
        return self.model.predict_embedding(dataset)


    # ****************************************************************************************
    def get_model_specific_metadata(self):
        """Returns a dictionary of parameter settings for this ModelWrapper object that are specific
        to neural network models.

        Returns:
            model_spec_metdata (dict): A dictionary of the parameter sets for the GraphConvDCModelWrapper object.
                Parameters are saved under the key 'nn_specific' as a subdictionary.
        """
        nn_metadata = dict(
                    best_epoch = self.best_epoch,
                    max_epochs = self.params.max_epochs,
                    batch_size = self.params.batch_size,
                    optimizer_type = self.params.optimizer_type,
                    layer_sizes = self.params.layer_sizes,
                    dropouts = self.params.dropouts,
                    weight_init_stddevs = self.params.weight_init_stddevs,
                    bias_init_consts = self.params.bias_init_consts,
                    learning_rate = self.params.learning_rate,
                    weight_decay_penalty=self.params.weight_decay_penalty,
                    weight_decay_penalty_type=self.params.weight_decay_penalty_type
        )
        model_spec_metadata = dict(nn_specific = nn_metadata)
        return model_spec_metadata


