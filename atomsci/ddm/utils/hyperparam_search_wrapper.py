#!/usr/bin/env python

# noinspection SpellCheckingInspection
"""Script to generate hyperparameter combinations based on input params and send off jobs to a slurm system.
Author: Amanda Minnich
"""

# from __future__ import unicode_literals

import argparse
import os
import os.path
import sys
import numpy as np
import logging
import itertools
from collections.abc import Iterable
import pandas as pd
import uuid

import subprocess
import shutil
import time

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from atomsci.ddm.pipeline import featurization as feat
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline import model_datasets as model_datasets
from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.pipeline import model_tracker as trkr
logging.basicConfig(format='%(asctime)-15s %(message)s')

import logging
import traceback
import copy
import pickle


def run_command(shell_script, python_path, script_dir, params):
    """Function to submit jobs on a slurm system

    Args:
        shell_script: Name of shell script to run

        python_path: Path to python version

        script_dir: Directory where script lives

        params: parameters in dictionary format

    Returns:
        None

    """
    # dataset_hash sneaks into params.
    new_params = argparse.Namespace(**parse.remove_unrecognized_arguments(params))
    # It's necessary to make this call here becausae it makes sense for 
    # relative paths to be calucated relative to the .json file, not to 
    # wherever maestro will eventually run the model_pipeline script
    parse.make_dataset_key_absolute(new_params)
    params_str = parse.to_str(new_params)
    slurm_command = 'sbatch {0} {1} {2} "{3}"'.format(shell_script, python_path, script_dir, params_str)
    print(slurm_command)
    os.system(slurm_command)

def gen_maestro_command(python_path, script_dir, params):
    """Generates a string that can be fed into a command line.

    Side Effects:
        Dataset key will be converted to an absolute path before
            returned. It's difficult to predict the working directory
            used when maestro runs the script.

    Args:
        shell_script: Name of shell script to run

        python_path: Path to python version

        script_dir: Directory where script lives

        params: parameters in dictionary format

    Returns:
        str: Formatted command in the form of a string

    """
    # Converts dataset_key to an aboslute path
    new_params = argparse.Namespace(**parse.remove_unrecognized_arguments(params))
    # It's necessary to make this call here becausae it makes sense for 
    # relative paths to be calucated relative to the .json file, not to 
    # wherever maestro will eventually run the model_pipeline script
    parse.make_dataset_key_absolute(new_params)
    params_str = parse.to_str(new_params)
    slurm_command = '{0} {1}/pipeline/model_pipeline.py {2}'.format(python_path, script_dir, params_str)

    return slurm_command

def run_cmd(cmd):
    """Function to submit a job using subprocess

    Args:
        cmd: Command to run

    Returns:
        output: Output of command

    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p.wait()
    return output


def reformat_filter_dict(filter_dict):
    """Function to reformat a filter dictionary to match the Model Tracker metadata structure. Updated 9/2020 by A. Paulson
    for new LC model tracker.

    Args:
        filter_dict: Dictionary containing metadata for model of interest

    Returns:
        new_filter_dict: Filter dict reformatted

    """
    rename_dict = {'model_parameters':
                       {'dependencies', 'featurizer', 'git_hash_code','model_bucket', 'model_choice_score_type',
                        'model_dataset_oid','model_type', 'num_model_tasks', 'prediction_type', 'save_results',
                        'system', 'task_type', 'time_generated', 'transformer_bucket', 'transformer_key',
                        'transformer_oid', 'transformers', 'uncertainty'},
               'splitting_parameters':
                       {'base_splitter', 'butina_cutoff', 'cutoff_date', 'date_col','num_folds', 'split_strategy',
                        'split_test_frac', 'split_uuid', 'split_valid_frac', 'splitter'},
               'training_dataset':
                       {'bucket', 'dataset_key', 'dataset_oid', 'num_classes','feature_transform_type',
                        'response_transform_type', 'id_col', 'smiles_col', 'response_cols'},
               'umap_specific':
                       {'umap_dim', 'umap_metric', 'umap_min_dist', 'umap_neighbors','umap_targ_wt'}
              }
    if filter_dict['model_type'] == 'NN':
        rename_dict['nn_specific'] = {'baseline_epoch', 'batch_size', 'best_epoch', 'bias_init_consts','dropouts',
                                      'layer_sizes', 'learning_rate', 'max_epochs','optimizer_type', 'weight_decay_penalty',
                                      'weight_decay_penalty_type', 'weight_init_stddevs'}
    elif filter_dict['model_type'] == 'RF':
        rename_dict['rf_specific'] = {'rf_estimators', 'rf_max_depth', 'rf_max_features'}
    elif filter_dict['model_type'] == 'xgboost':
        rename_dict['xgb_specific'] = {'xgb_colsample_bytree', 'xgb_gamma', 'xgb_learning_rate','xgb_max_depth',
                               'xgb_min_child_weight', 'xgb_n_estimators','xgb_subsample'}
    if filter_dict['featurizer'] == 'ecfp':
        rename_dict['ecfp_specific'] = {'ecfp_radius', 'ecfp_size'}
    elif (filter_dict['featurizer'] == 'descriptor') | (filter_dict['featurizer'] == 'computed_descriptors'):
        rename_dict['descriptor_specific'] = {'descriptor_key', 'descriptor_bucket', 'descriptor_oid', 'descriptor_type'}
    elif filter_dict['featurizer'] == 'molvae':
        rename_dict['autoencoder_specific'] = {'autoencoder_model_key', 'autoencoder_model_bucket', 'autoencoder_model_oid', 'autoencoder_type'}
    new_filter_dict = {}
    for key, values in rename_dict.items():
        for value in values:
            if value in filter_dict:
                filter_val = filter_dict[value]
                if type(filter_val) == np.int64:
                    filter_dict[value] = int(filter_val)
                elif type(filter_val) == np.float64:
                    filter_dict[value] = float(filter_val)
                elif type(filter_val) == list:
                    for i, item in enumerate(filter_val):
                        if type(item) == np.int64:
                            filter_dict[value][i] = int(item)
                        elif type(filter_val) == np.float64:
                            filter_dict[value][i] = float(item)
                new_filter_dict['%s.%s' % (key, value)] = filter_dict[value]
    return new_filter_dict


def permutate_NNlayer_combo_params(layer_nums, node_nums, dropout_list, max_final_layer_size):
    """Generate combos of layer_sizes(str) and dropouts(str) params from the layer_nums (list), node_nums (list), dropout_list (list).

    The permutation will make the NN funnel shaped, so that the next layer can only be smaller or of the same size of the current layer.

    Example:
        permutate_NNlayer_combo_params([2], [4,8,16], [0], 16)
        returns [[16, 4], [16, 8], [8,4]] [[0,0],[0,0],[0,0]]

    If there are duplicates of the same size, it will create consecutive layers of the same size.

    Example:
        permutate_NNlayer_combo_params([2], [4,8,8], [0], 16)
        returns [[8, 8], [8, 4]] [[0,0],[0,0]]

    Args:
        layer_nums: specify numbers of layers.

        node_nums: specify numbers of nodes per layer.

        dropout_list: specify the dropouts.

        max_last_layer_size: sets the max size of the last layer. It will be set to the smallest node_num if needed.

    Returns:
        layer_sizes, dropouts: the layer sizes and dropouts generated based on the input parameters

    """
    import itertools
    import numpy as np
    layer_sizes = []
    dropouts = []
    node_nums = np.sort(np.array(node_nums))[::-1]
    max_final_layer_size = int(max_final_layer_size)
    # set to the smallest node_num in the provided list, if necessary.
    if node_nums[-1] > max_final_layer_size:
        max_final_layer_size = node_nums[-1]

    for dropout in dropout_list:
        _repeated_layers =[]
        for layer_num in layer_nums:
            for layer in itertools.combinations(node_nums, layer_num):
                layer = [i for i in layer]
                if (layer[-1] <= max_final_layer_size) and (layer not in _repeated_layers):
                    _repeated_layers.append(layer)
                    layer_sizes.append(layer)
                    dropouts.append([(dropout) for i in layer])
    return layer_sizes, dropouts


def get_num_params(combo):
    """Calculates the number of parameters in a fully-connected neural networ

    Args:
        combo: Model parameters

    Returns:
        tmp_sum: Calculated number of parameters

    """
    layers = combo['layer_sizes']
    # All layers multiplied by adjacent layers, summed, plus the final layer times the number of samples. Extra addition is for bias terms
    tmp_sum = layers[0] + sum(layers[i] * layers[i + 1] + layers[i+1] for i in range(len(layers) - 1))
    # Add in first layer times the feature vector size. Estimate 300 for descriptors.
    #TODO: Update for moe vs mordred
    if combo['featurizer'] == 'ecfp':
        return tmp_sum + layers[0]*1024
    if combo['featurizer'] == 'descriptors':
        if combo['descriptor_type'] == 'moe':
            return tmp_sum + layers[0]*306
        if combo['descriptor_type'] == 'mordred_filtered':
            return tmp_sum + layers[0]*1555
    else:
        return tmp_sum


# Global variable with keys that should not be used to generate hyperparameters
excluded_keys = {'shortlist_key', 'use_shortlist', 'dataset_key', 'object_oid', 'script_dir',
                  'python_path', 'config_file', 'hyperparam', 'search_type', 'split_only', 'layer_nums',
                  'node_nums', 'dropout_list', 'max_final_layer_size', 'splitter', 'nn_size_scale_factor',
                  'rerun', 'max_jobs'}


class HyperparameterSearch(object):
    """The class for generating and running all hyperparameter combinations based on the input params given
    """
    def __init__(self, params):
        """

        Args:
            params: The input hyperparameter parameters

            hyperparam_uuid: Optional, UUID for hyperparameter run if you want to group this run with a previous run.
            We ended up mainly doing this via collections, so not really used
        """
        self.hyperparam_layers = {'layer_sizes', 'dropouts', 'weight_init_stddevs', 'bias_init_consts'}
        self.hyperparam_keys = {'model_type', 'featurizer', 'splitter', 'learning_rate', 'weight_decay_penalty',
                                'rf_estimators', 'rf_max_features', 'rf_max_depth',
                                'umap_dim', 'umap_targ_wt', 'umap_metric', 'umap_neighbors', 'umap_min_dist',
                                'xgb_learning_rate',
                                'xgb_gamma'}
        self.nn_specific_keys = {'learning_rate', 'layers','weight_decay_penalty'}
        self.rf_specific_keys = {'rf_estimators', 'rf_max_features', 'rf_max_depth'}
        self.xgboost_specific_keys = {'xgb_learning_rate', 'xgb_gamma'}
        self.hyperparam_keys |= self.hyperparam_layers
        self.excluded_keys = excluded_keys
        self.convert_to_float = parse.convert_to_float_list
        self.convert_to_int = parse.convert_to_int_list
        self.params = params
        # simplify NN layer construction
        if (params.layer_nums != None) and (params.node_nums != None) and (params.dropout_list != None):

            self.params.layer_sizes, self.params.dropouts = permutate_NNlayer_combo_params(params.layer_nums,
                                                                                           params.node_nums,
                                                                                           params.dropout_list,
                                                                                           params.max_final_layer_size)
        if params.hyperparam_uuid is None:
            self.hyperparam_uuid = str(uuid.uuid4())
        else:
            self.hyperparam_uuid = params.hyperparam_uuid
        self.hyperparams = {}
        self.new_params = {}
        self.layers = {}
        self.param_combos = []
        self.num_rows = {}
        self.log = logging.getLogger("hyperparam_search")
        # Create handlers
        c_handler = logging.StreamHandler()
        log_path = os.path.join(self.params.result_dir, 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        f_handler = logging.FileHandler(os.path.join(log_path, '{0}.log'.format(self.hyperparam_uuid)))
        self.out_file = open(os.path.join(log_path, '{0}.json'.format(self.hyperparam_uuid)), 'a')
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.INFO)
        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        # Add handlers to the logger
        self.log.addHandler(c_handler)
        self.log.addHandler(f_handler)


        slurm_path = os.path.join(self.params.result_dir, 'slurm_files')
        if not os.path.exists(slurm_path):
            os.makedirs(slurm_path)
        self.shell_script = os.path.join(self.params.result_dir, 'run.sh')
        with open(self.shell_script, 'w') as f:
            f.write("#!/bin/bash\n")

            f.write("#SBATCH -D {0}\n".format(slurm_path))

            # If any of these properties == None, that property is not set
            if self.params.slurm_account:
                f.write("#SBATCH -A {0}\n".format(self.params.slurm_account))
            elif self.params.lc_account:
                f.write("#SBATCH -A {0}\n".format(self.params.lc_account))

            if self.params.slurm_export:
                f.write("#SBATCH --export={0}\n".format(self.params.slurm_export))

            if self.params.slurm_nodes:
                f.write("#SBATCH -N {0}\n".format(self.params.slurm_nodes))

            if self.params.slurm_partition:
                f.write("#SBATCH -p {0}\n".format(self.params.slurm_partition))

            if self.params.slurm_time_limit:
                f.write("#SBATCH -t {0}\n".format(self.params.slurm_time_limit))

            if self.params.slurm_options:
                f.write('{0}\n'.format(self.params.slurm_options))

            f.write('start=`date +%s`\necho $3\n$1 $2/pipeline/model_pipeline.py $3\nend=`date +%s`\n'
                    'runtime=$((end-start))\necho "runtime: " $runtime')

    def generate_param_combos(self):
        """Performs additional parsing of parameters and generates all combinations

        Returns:
            None

        """
        for key, value in vars(self.params).items():
            if (value is None) or (key in self.excluded_keys):
                continue
            elif key == 'result_dir' or key == 'output_dir':
                self.new_params[key] = os.path.join(value, self.hyperparam_uuid)
            # Need to zip together layers in special way
            elif key in self.hyperparam_layers and type(value[0]) == list:
                self.layers[key] = value
            # Parses the hyperparameter keys depending on the size of the key list
            elif key in self.hyperparam_keys:
                if type(value) != list:
                    self.new_params[key] = value
                    self.hyperparam_keys.remove(key)
                elif len(value) == 1:
                    self.new_params[key] = value[0]
                    self.hyperparam_keys.remove(key)
                else:
                    self.hyperparams[key] = value
            else:
                self.new_params[key] = value
        # Adds layers to the parameter combos
        if self.layers:
            self.assemble_layers()
        # setting up the various hyperparameter combos for each model type.
        if type(self.params.model_type) == str:
            self.params.model_type = [self.params.model_type]
        if type(self.params.featurizer) == str:
            self.params.featurizer = [self.params.featurizer]
        if type(self.params.descriptor_type) == str:
            self.params.descriptor_type = [self.params.descriptor_type]

        for model_type in self.params.model_type:
            if model_type == 'NN':
                # if the model type is NN, loops through the featurizer to check for GraphConv.
                for featurizer in self.params.featurizer:
                    if featurizer == 'computed_descriptors':
                        for desc in self.params.descriptor_type:
                            subcombo = {k: val for k, val in self.hyperparams.items() if k in
                                        self.hyperparam_keys - self.rf_specific_keys - self.xgboost_specific_keys}
                            # could put in list
                            subcombo['model_type'] = [model_type]
                            subcombo['featurizer'] = [featurizer]
                            subcombo['descriptor_type'] = [desc]
                            self.param_combos.extend(self.generate_combos(subcombo))
                    else:
                        subcombo = {k: val for k, val in self.hyperparams.items() if k in
                                    self.hyperparam_keys - self.rf_specific_keys - self.xgboost_specific_keys}
                        # could put in list
                        subcombo['model_type'] = [model_type]
                        subcombo['featurizer'] = [featurizer]
                        subcombo['descriptor_type'] = ['moe']
                        if (featurizer == 'graphconv') & (self.params.prediction_type=='classification'):
                            subcombo['uncertainty'] = [False]
                            
                        self.param_combos.extend(self.generate_combos(subcombo))
            elif model_type == 'RF':
                for featurizer in self.params.featurizer:
                    if featurizer == 'graphconv':
                        continue
                    elif featurizer == 'computed_descriptors':
                        for desc in self.params.descriptor_type:
                            # Adds the subcombo for RF
                            subcombo = {k: val for k, val in self.hyperparams.items() if k in
                                        self.hyperparam_keys - self.nn_specific_keys - self.xgboost_specific_keys}
                            subcombo['model_type'] = [model_type]
                            subcombo['featurizer'] = [featurizer]
                            subcombo['descriptor_type'] = [desc]
                            self.param_combos.extend(self.generate_combos(subcombo))
                    else:
                        # Adds the subcombo for RF
                        subcombo = {k: val for k, val in self.hyperparams.items() if k in
                                    self.hyperparam_keys - self.nn_specific_keys - self.xgboost_specific_keys}
                        subcombo['model_type'] = [model_type]
                        subcombo['featurizer'] = [featurizer]
                        subcombo['descriptor_type'] = ['moe']

                        self.param_combos.extend(self.generate_combos(subcombo))
            elif model_type == 'xgboost':
                for featurizer in self.params.featurizer:
                    if featurizer == 'graphconv':
                        continue
                    elif featurizer == 'computed_descriptors':
                        for desc in self.params.descriptor_type:
                            # Adds the subcombo for xgboost
                            subcombo = {k: val for k, val in self.hyperparams.items() if k in
                                        self.hyperparam_keys - self.nn_specific_keys - self.rf_specific_keys}
                            subcombo['model_type'] = [model_type]
                            subcombo['featurizer'] = [featurizer]
                            subcombo['descriptor_type'] = [desc]
                            self.param_combos.extend(self.generate_combos(subcombo))
                    else:
                        # Adds the subcombo for xgboost
                        subcombo = {k: val for k, val in self.hyperparams.items() if k in
                                    self.hyperparam_keys - self.nn_specific_keys - self.rf_specific_keys}
                        subcombo['model_type'] = [model_type]
                        subcombo['featurizer'] = [featurizer]
                        subcombo['descriptor_type'] = ['moe']
                        self.param_combos.extend(self.generate_combos(subcombo))

    def generate_combos(self, params_dict):
        """Calls sub-function generate_combo and then uses itertools.product to generate all desired combinations

        Args:
            params_dict:

        Returns:
            None

        """
        new_dict = self.generate_combo(params_dict)
        hyperparam_combos = []
        hyperparams = new_dict.keys()
        hyperparam_vals = new_dict.values()
        for ind, hyperparameter_tuple in enumerate(itertools.product(*hyperparam_vals)):
            model_params = {}
            for hyperparam, hyperparam_val in zip(hyperparams, hyperparameter_tuple):
                model_params[hyperparam] = hyperparam_val
            hyperparam_combos.append(model_params)
        return hyperparam_combos

    def assemble_layers(self):
        """Reformats layer parameters

        Returns:
            None

        """
        tmp_list = []
        for i in range(min([len(x) for x in list(self.layers.values())])):
            tmp_dict = {}
            for key, value in self.layers.items():
                tmp_dict[key] = value[i]
            x = [len(y) for y in tmp_dict.values()]
            try:
                assert x.count(x[0]) == len(x)
            except:
                continue
            tmp_list.append(tmp_dict)
        self.hyperparams['layers'] = tmp_list
        self.hyperparam_keys.add('layers')

    def generate_assay_list(self):
        """Generates the list of datasets to build models for, with their key, bucket, split, and split uuid

        Returns:
           None

        """
        # Creates the assay list with additional options for use_shortlist
        if not self.params.use_shortlist:
            if type(self.params.splitter) == str:
                splitters = [self.params.splitter]
            else:
                splitters = self.params.splitter
            self.assays = []
            for splitter in splitters:
                if 'previously_split' in self.params.__dict__.keys() and 'split_uuid' in self.params.__dict__.keys() \
                    and self.params.previously_split and self.params.split_uuid is not None:
                    self.assays.append((self.params.dataset_key, self.params.bucket, self.params.response_cols, self.params.collection_name, self.params.splitter, self.params.split_uuid))
                else:
                    try:
                        split_uuid = self.return_split_uuid(self.params.dataset_key, splitter=splitter)
                        self.assays.append((self.params.dataset_key, self.params.bucket, self.params.response_cols, self.params.collection_name, splitter, split_uuid))
                    except Exception as e:
                        print(e)
                        print(traceback.print_exc())
                        sys.exit(1)
        else:
            self.assays = self.get_shortlist_df(split_uuids=True)
        self.assays = [(t[0].strip(), t[1].strip(), t[2], t[3].strip(), t[4].strip(), t[5].strip()) for t in self.assays]

    def get_dataset_metadata(self, assay_params, retry_time=60):
        """Gather the required metadata for a dataset

        Args:
            assay_params: dataset metadata

        Returns:
            None

        """
        if not self.params.datastore:
            return
        print(assay_params['dataset_key'])
        retry = True
        i = 0
        #TODO: need to catch if dataset doesn't exist versus 500 failure
        while retry:
            try:
                metadata = dsf.get_keyval(dataset_key=assay_params['dataset_key'], bucket=assay_params['bucket'])
                retry = False
            except Exception as e:
                if i < 5:
                    print("Could not get metadata from datastore for dataset %s because of exception %s, sleeping..."
                            % (assay_params['dataset_key'], e))
                    time.sleep(retry_time)
                    i += 1
                else:
                    print("Could not get metadata from datastore for dataset %s because of exception %s, exiting"
                            % (assay_params['dataset_key'], e))
                    return None
        if 'id_col' in metadata.keys():
            assay_params['id_col'] = metadata['id_col']
        if 'response_cols' not in assay_params or assay_params['response_cols'] is None:
            if 'param' in metadata.keys():
                assay_params['response_cols'] = [metadata['param']]
            if 'response_col' in metadata.keys():
                assay_params['response_cols'] = [metadata['response_col']]
            if 'response_cols' in metadata.keys():
                assay_params['response_cols'] = metadata['response_cols']
        if 'smiles_col' in metadata.keys():
            assay_params['smiles_col'] = metadata['smiles_col']
        if 'class_name' in metadata.keys():
            assay_params['class_name'] = metadata['class_name']
        if 'class_number' in metadata.keys():
            assay_params['class_number'] = metadata['class_number']
        if 'num_row' in metadata.keys():
            self.num_rows[assay_params['dataset_key']] = metadata['num_row']
        assay_params['dataset_name'] = assay_params['dataset_key'].split('/')[-1].rstrip('.csv')
        assay_params['hyperparam_uuid'] = self.hyperparam_uuid

    def split_and_save_dataset(self, assay_params):
        """Splits a given dataset, saves it, and sets the split_uuid in the metadata

        Args:
            assay_params: Dataset metadata

        Returns:
            None

        """
        self.get_dataset_metadata(assay_params)
        # TODO: check usage with defaults
        namespace_params = parse.wrapper(assay_params)
        # TODO: Don't want to recreate each time
        featurization = feat.create_featurization(namespace_params)
        data = model_datasets.create_model_dataset(namespace_params, featurization)
        data.get_featurized_data()
        data.split_dataset()
        data.save_split_dataset()
        assay_params['previously_split'] = True
        assay_params['split_uuid'] = data.split_uuid

    def return_split_uuid(self, dataset_key, bucket=None, splitter=None, split_combo=None, retry_time=60):
        """Loads a dataset, splits it, saves it, and returns the split_uuid

        Args:
            dataset_key: key for dataset to split

            bucket: datastore-specific user group bucket

            splitter: Type of splitter to use to split the dataset

            split_combo: tuple of form (split_valid_frac, split_test_frac)

        Returns:
            None

        """
        if bucket is None:
            bucket = self.params.bucket
        if splitter is None:
            splitter=self.params.splitter
        if split_combo is None:
            split_valid_frac = self.params.split_valid_frac
            split_test_frac = self.params.split_test_frac
        else:
            split_valid_frac = split_combo[0]
            split_test_frac = split_combo[1]
        retry = True
        i = 0
        #TODO: need to catch if dataset doesn't exist versus 500 failure
        while retry:
            try:
                metadata = dsf.get_keyval(dataset_key=dataset_key, bucket=bucket)
                retry = False
            except Exception as e:
                if i < 5:
                    print("Could not get metadata from datastore for dataset %s because of exception %s, sleeping..." % (dataset_key, e))
                    time.sleep(retry_time)
                    i += 1
                else:
                    print("Could not get metadata from datastore for dataset %s because of exception %s, exiting" % (dataset_key, e))
                    return None
        assay_params = {'dataset_key': dataset_key, 'bucket': bucket, 'splitter': splitter,
                        'split_valid_frac': split_valid_frac, 'split_test_frac': split_test_frac}
        #Need a featurizer type to split dataset, but since we only care about getting the split_uuid, does not matter which featurizer you use
        if type(self.params.featurizer) == list:
            assay_params['featurizer'] = self.params.featurizer[0]
        else:
            assay_params['featurizer'] = self.params.featurizer
        if 'id_col' in metadata.keys():
            assay_params['id_col'] = metadata['id_col']
        if 'response_cols' not in assay_params or assay_params['response_cols'] is None:
            if 'param' in metadata.keys():
                assay_params['response_cols'] = [metadata['param']]
            if 'response_col' in metadata.keys():
                assay_params['response_cols'] = [metadata['response_col']]
            if 'response_cols' in metadata.keys():
                assay_params['response_cols'] = metadata['response_cols']
        if 'smiles_col' in metadata.keys():
            assay_params['smiles_col'] = metadata['smiles_col']
        if 'class_name' in metadata.keys():
            assay_params['class_name'] = metadata['class_name']
        if 'class_number' in metadata.keys():
            assay_params['class_number'] = metadata['class_number']
        assay_params['dataset_name'] = assay_params['dataset_key'].split('/')[-1].rstrip('.csv')
        assay_params['datastore'] = True
        assay_params['previously_featurized'] = self.params.previously_featurized
        try:
            assay_params['descriptor_key'] = self.params.descriptor_key
            assay_params['descriptor_bucket'] = self.params.descriptor_bucket
        except:
            print("")
        #TODO: check usage with defaults
        namespace_params = parse.wrapper(assay_params)
        # TODO: Don't want to recreate each time
        featurization = feat.create_featurization(namespace_params)
        data = model_datasets.create_model_dataset(namespace_params, featurization)
        retry = True
        i = 0
        while retry:
            try:
                data.get_featurized_data()
                data.split_dataset()
                data.save_split_dataset()
                return data.split_uuid
            except Exception as e:
                if i < 5:
                    print("Could not get metadata from datastore for dataset %s because of exception %s, sleeping" % (dataset_key, e))
                    time.sleep(retry_time)
                    i += 1
                else:
                    print("Could not save split dataset for dataset %s because of exception %s" % (dataset_key, e))
                    return None

    def return_split_uuid_file(self, dataset_key, response_cols, bucket=None, splitter=None, split_combo=None, retry_time=60):
        """Loads a dataset, splits it, saves it, and returns the split_uuid.

        Args:
            dataset_key: key for dataset to split

            bucket: datastore-specific user group bucket

            splitter: Type of splitter to use to split the dataset

            split_combo: tuple of form (split_valid_frac, split_test_frac)

        Returns:
            None

        """
        
        if bucket is None:
            bucket = self.params.bucket
        if splitter is None:
            splitter=self.params.splitter
        if split_combo is None:
            split_valid_frac = self.params.split_valid_frac
            split_test_frac = self.params.split_test_frac
        else:
            split_valid_frac = split_combo[0]
            split_test_frac = split_combo[1]
        
        assay_params = {'dataset_key': dataset_key, 'bucket': bucket, 'splitter': splitter,
                        'split_valid_frac': split_valid_frac, 'split_test_frac': split_test_frac}
        if 'id_col' in self.params.__dict__.keys():
            assay_params['id_col']=self.params.id_col
        if 'smiles_col' in self.params.__dict__.keys():
            assay_params['smiles_col']=self.params.smiles_col
        if isinstance(response_cols, list):
            assay_params['response_cols']=",".join(response_cols)
        elif isinstance(response_cols,str):
            assay_params['response_cols']=response_cols
            
        assay_params['dataset_name'] = assay_params['dataset_key'].split('/')[-1].replace('.csv','')
        # rdkit_raw b/c it's the fastest and won't have to be redone every split 
        assay_params['featurizer'] = 'computed_descriptors'
        assay_params['descriptor_type'] = 'rdkit_raw'
        assay_params['previously_featurized'] = True
        assay_params['datastore'] = False
        
        namespace_params = parse.wrapper(assay_params)
        # TODO: Don't want to recreate each time
        featurization = feat.create_featurization(namespace_params)
        data = model_datasets.create_model_dataset(namespace_params, featurization)
        
        data.get_featurized_data()
        data.split_dataset()
        data.save_split_dataset()
        return data.split_uuid
            
    
    def generate_split_shortlist(self, retry_time=60):
        """Processes a shortlist, generates splits for each dataset on the list, and uploads a new shortlist file with the
        split_uuids included. Generates splits for the split_combos [[0.1,0.1], [0.1,0.2],[0.2,0.2]], [random, scaffold]

        Returns:
            None

        """
        retry = True
        i = 0
        while retry:
            try:
                shortlist_metadata = dsf.retrieve_dataset_by_datasetkey(
                    bucket=self.params.bucket, dataset_key=self.params.shortlist_key, return_metadata=True)
                retry = False
            except Exception as e:
                if i < 5:
                    print("Could not retrieve shortlist %s from datastore because of exception %s, sleeping..." %
                          (self.params.shortlist_key, e))
                    time.sleep(retry_time)
                    i += 1
                else:
                    print("Could not retrieve shortlist %s from datastore because of exception %s, exiting" %
                          (self.params.shortlist_key, e))
                    return None

        datasets = self.get_shortlist_df()
        rows = []
        for assay, bucket, response_cols, collection in datasets:
            split_uuids = {'dataset_key': assay, 'bucket': bucket, 'response_cols':response_cols, 'collection':collection}
            for splitter in ['random', 'scaffold', 'fingerprint']:
                for split_combo in [[0.1,0.1], [0.15,0.15],[0.1,0.2],[0.2,0.2]]:
                    split_name = "%s_%d_%d" % (splitter, split_combo[0]*100, split_combo[1]*100)
                    try:
                        split_uuids[split_name] = self.return_split_uuid(assay, bucket, splitter, split_combo)
                    except Exception as e:
                        print(e)
                        print("Splitting failed for dataset %s" % assay)
                        split_uuids[split_name] = None
                        continue
            rows.append(split_uuids)
        df = pd.DataFrame(rows)
        new_metadata = {}
        new_metadata['dataset_key'] = shortlist_metadata['dataset_key'].strip('.csv') + '_with_uuids.csv'
        new_metadata['has_uuids'] = True
        new_metadata['description'] = '%s, with UUIDs' % shortlist_metadata['description']
        retry = True
        i = 0
        while retry:
            try:
                dsf.upload_df_to_DS(df,
                                    bucket=self.params.bucket,
                                    filename=new_metadata['dataset_key'],
                                    title=new_metadata['dataset_key'].replace('_', ' '),
                                    description=new_metadata['description'],
                                    tags=[],
                                    key_values={},
                                    dataset_key=new_metadata['dataset_key'])
                retry=False
            except Exception as e:
                if i < 5:
                    print("Could not save new shortlist because of exception %s, sleeping..." % e)
                    time.sleep(retry_time)
                    i += 1
                else:
                    #TODO: Add save to disk.
                    print("Could not save new shortlist because of exception %s, exiting" % e)
                    retry = False
    
    def generate_split_shortlist_file(self):
        """Processes a shortlist, generates splits for each dataset on the list, and uploads a new shortlist file with the
        split_uuids included. Generates splits for the split_combos [[0.1,0.1], [0.15,0.15], [0.1,0.2], [0.2,0.2]], [random, scaffold]

        Returns:
            None

        """

        datasets = self.get_shortlist_df()
        rows = []
        for assay, bucket, response_cols, collection in datasets:
            split_uuids = {'dataset_key': assay, 'bucket': bucket, 'response_cols':response_cols, 'collection':collection}
            for splitter in ['random', 'scaffold','fingerprint']:
                for split_combo in [[0.1,0.1], [0.15,0.15],[0.1,0.2],[0.2,0.2]]:
                    split_name = "%s_%d_%d" % (splitter, split_combo[0]*100, split_combo[1]*100)
                    try:
                        split_uuids[split_name] = self.return_split_uuid_file(assay, response_cols, bucket, splitter, split_combo)
                    except Exception as e:
                        print(e)
                        print("Splitting failed for dataset %s" % assay)
                        split_uuids[split_name] = None
                        continue
            rows.append(split_uuids)
        df = pd.DataFrame(rows)
        fname = self.params.shortlist_key.replace('.csv','_with_uuids.csv')
        df.to_csv(fname, index=False)

    def get_shortlist_df(self, split_uuids=False, retry_time=60):
        """Get dataframe short list

        Args:
            split_uuids: Boolean value saying if you want just datasets returned or the split_uuids as well

        Returns:
            The list of dataset_keys, along with their accompanying bucket, split type, and split_uuid if split_uuids is True
        """
        if self.params.datastore:
            retry = True
            i = 0
            while retry:
                try:
                    df = dsf.retrieve_dataset_by_datasetkey(self.params.shortlist_key, self.params.bucket)
                    retry=False
                except Exception as e:
                    if i < 5:
                        print("Could not retrieve shortlist %s because of exception %s, sleeping..." % (self.params.shortlist_key, e))
                        time.sleep(retry_time)
                        i += 1
                    else:
                        print("Could not retrieve shortlist %s because of exception %s, exiting" % (self.params.shortlist_key, e))
                        sys.exit(1)
        else:
            if not os.path.exists(self.params.shortlist_key):
                return None
            df = pd.read_csv(self.params.shortlist_key, index_col=False)
        if df is None:
            sys.exit(1)
        if len(df.columns) == 1:
            assays = df[df.columns[0]].values.tolist()
        else:
            if 'task_name' in df.columns:
                col_name = 'task_name'
            else:
                col_name = 'dataset_key'
            assays = df[col_name].values.tolist()
        if 'bucket' in df.columns:
            buckets = df['bucket'].values.tolist()
        elif 'bucket_name' in df.columns:
            buckets = df['bucket_name'].values.tolist()
        else:
            buckets=[self.params.bucket]*len(df)
        if 'response_cols' in df.columns:
            responses= df.response_cols.str.split(',').tolist()
        else:
            responses=[self.params.response_cols]*len(df)
        if 'collection' in df.columns:
            collections=df.collection.values.tolist()
        else:
            collections=[self.params.collection_name]*len(df)
        datasets=list(zip(assays,buckets,responses,collections))
        datasets = [(d[0].strip(), d[1].strip(), ",".join(d[2]), d[3].strip()) for d in datasets]
            
        if not split_uuids:
            return datasets
        if type(self.params.splitter) == str:
            splitters = [self.params.splitter]
        else:
            splitters = self.params.splitter
        assays = []
        for splitter in splitters:
            split_name = '%s_%d_%d' % (splitter, self.params.split_valid_frac*100, self.params.split_test_frac*100)
            if split_name in df.columns:
                for i, row in df.iterrows():
                    try:
                        assays.append((datasets[i][0], datasets[i][1], datasets[i][2], datasets[i][3], splitter, row[split_name]))
                    except:
                        print("dataset_key, bucket, response_cols, & collecion_name must be specified in shortlist or config file, not neither.")
            else:
                print(f"Warning: {split_name} not found in shortlist. Creating default split scaffold_10_10 now.")
                for assay, bucket, response_cols, collection in datasets:
                    try:
                    # do we want to move this into loop so we ignore ones it failed for?
                        if self.params.datastore:
                            split_uuid = self.return_split_uuid(assay, bucket)
                        else:
                            split_uuid = self.return_split_uuid_file(assay, response_cols, bucket)
                        assays.append((assay, bucket, response_cols, collection, splitter, split_uuid))
                    except Exception as e:
                        print("Splitting failed for dataset %s, skipping..." % assay)
                        print(e)
                        print(traceback.print_exc())
                        continue
        return assays

    def build_jobs(self):
        """Builds jobs.
        Reformats parameters as necessary

        Returns:
            None

        """
        result_assay_params = []
        for assay, bucket, response_cols, collection, splitter, split_uuid in self.assays:
            # Writes the series of command line arguments for scripts without a hyperparameter combo
            assay_params = copy.deepcopy(self.new_params)
            assay_params['dataset_key'] = assay
            assay_params['dataset_name'] = os.path.splitext(os.path.basename(assay))[0]
            assay_params['bucket'] = bucket
            assay_params['response_cols'] = response_cols
            assay_params['collection_name'] = collection
            assay_params['split_uuid'] = split_uuid
            assay_params['previously_split'] = True
            assay_params['splitter'] = splitter
            print(f"prediction_type: {assay_params['prediction_type']}")
            try:
                self.get_dataset_metadata(assay_params)
            except Exception as e:
                print(e)
                print(traceback.print_exc())
                continue
            # creates output directory
            base_result_dir = os.path.join(assay_params['result_dir'], assay_params['dataset_name'])

            if not self.param_combos:
                assay_params['result_dir'] = os.path.join(base_result_dir, str(uuid.uuid4()))
                result_assay_params.append(assay_params)
            else:
                for combo in self.param_combos:
                    # For a temporary parameter list, appends and modifies parameters for each hyperparameter combo.
                    combo_params = copy.deepcopy(assay_params)
                    for key, value in combo.items():
                        if key == 'layers':
                            for k, v in value.items():
                                combo_params[k] = v
                        else:
                            combo_params[key] = value

                    combo_params['result_dir'] = os.path.join(base_result_dir, str(uuid.uuid4()))
                    result_assay_params.append(combo_params)

        return result_assay_params

    def filter_jobs(self, job_list):
        """Removes jobs that should not be run

        Returns:
            None
        """
        result_list = []
        for assay_params in job_list:
            if assay_params['model_type'] == 'NN' and assay_params['featurizer'] != 'graphconv':
                if assay_params['dataset_key'] in self.num_rows:
                    num_params = get_num_params(assay_params)
                    if num_params*self.params.nn_size_scale_factor >= self.num_rows[assay_params['dataset_key']]:
                        continue
            if not self.params.rerun and self.already_run(assay_params):
                continue

            result_list.append(assay_params)

        return result_list

    def submit_jobs(self, job_list, retry_time=60):
        """Reformats parameters as necessary and then calls run_command in a loop to submit a job for each param combo

        Returns:
            None
        """
        for assay_params in job_list: 
            if len(self.filter_jobs([assay_params]))==1:
                i = int(run_cmd('squeue | grep $(whoami) | wc -l').decode("utf-8"))
                while i >= self.params.max_jobs:
                    print("%d jobs in queue, sleeping" % i)
                    time.sleep(retry_time)
                    i = int(run_cmd('squeue | grep $(whoami) | wc -l').decode("utf-8"))
                self.log.info(assay_params)
                self.out_file.write(str(assay_params))
                run_command(self.shell_script, self.params.python_path, self.params.script_dir, assay_params)

    def already_run(self, assay_params, retry_time=10):
        """Checks to see if a model with a given metadata combination has already been built

        Args:
            assay_params: model metadata information

        Returns:
            Boolean specifying if model has been previously built

        """
        if not self.params.save_results:
            return False
        filter_dict = copy.deepcopy(assay_params)
        for key in ['result_dir', 'previously_featurized', 'collection_name', 'time_generated', 'hyperparam_uuid', 'model_uuid']:
            if key in filter_dict:
                del filter_dict[key]
        filter_dict = reformat_filter_dict(filter_dict)
        retry = True
        i = 0
        while retry:
            try:
                print(f"Checking model tracker DB for existing model with parameter combo in {assay_params['collection_name']} collection.")
                models = list(trkr.get_full_metadata(filter_dict, collection_name=assay_params['collection_name']))
                retry = False
            except Exception as e:
                if i < 5:
                    time.sleep(retry_time)
                    i += 1
                else:
                    print("Could not check Model Tracker for existing model at this time because of exception %s" % e)
                    return False
        if models:
            print("Already created model for this param combo")
            return True
        print("No existing model found")
        return False

    def generate_combo(self, params_dict):
        """This is implemented in the specific sub-classes

        """
        raise NotImplementedError

    def run_search(self):
        """The driver code for generating hyperparameter combinations and submitting jobs

        Returns:
            None

        """
        job_list = self.generate_searches()
        print("Submitting jobs")
        self.submit_jobs(job_list)

    def generate_searches(self):
        """Generate a list of training jobs

        Generates a list of model training jobs that spans
        the hyperparameter search space. This function
        filters out jobs that are redundant by calling filter_jobs

        Args:
            None

        Returns:
            list(tuple): A list of tuples that contain assay parameters
        """

        print("Generating param combos") 
        self.generate_param_combos()
        print("Generating assay list")
        self.generate_assay_list()
        print("build_ jobs")
        job_list = self.build_jobs()
#         print("filter redundant jobs")
#         job_list = self.filter_jobs(job_list)

        return job_list

    def generate_maestro_commands(self):
        """Generates commands that can be used by maestro

        Generates a list of commands that can be put directly into
        the shell to run model training.

        Args:
            None

        Returns:
            list: A list of shell commands
        """

        job_list = self.generate_searches()
        commands = []
        for assay_params in job_list:
            commands.append(gen_maestro_command(self.params.python_path, self.params.script_dir, assay_params))
        return commands

class GridSearch(HyperparameterSearch):
    """Generates fixed steps on a grid for a given hyperparameter range"""

    def __init__(self, params):
        super().__init__(params)

    def split_and_save_dataset(self, assay_params):
        self.split_and_save_dataset(assay_params)

    def generate_param_combos(self):
        super().generate_param_combos()

    def generate_assay_list(self):
        super().generate_assay_list()

    def generate_combo(self, params_dict):
        """Method to generate all combinations from a given set of key-value pairs

        Args:
            params_dict: Set of key-value pairs with the key being the param name and the value being the list of values
            you want to try for that param

        Returns:
            new_dict: The list of all combinations of parameters

        """
        if not params_dict:
            return None

        new_dict = {}
        for key, value in params_dict.items():
            assert isinstance(value, Iterable)
            if key == 'layers':
                new_dict[key] = value
            elif type(value[0]) != str:
                tmp_list = list(np.linspace(value[0], value[1], value[2]))
                if key in self.convert_to_int:
                    new_dict[key] = [int(x) for x in tmp_list]
                else:
                    new_dict[key] = tmp_list
            else:
                new_dict[key] = value
        return new_dict

class RandomSearch(HyperparameterSearch):
    """Generates the specified number of random parameter values for within the specified range"""

    def __init__(self, params):
        super().__init__(params)

    def split_and_save_dataset(self, assay_params):
        self.split_and_save_dataset(assay_params)

    def generate_param_combos(self):
        super().generate_param_combos()

    def generate_assay_list(self):
        super().generate_assay_list()

    def generate_combo(self, params_dict):
        """Method to generate all combinations from a given set of key-value pairs

        Args:
            params_dict: Set of key-value pairs with the key being the param name and the value being the list of values
            you want to try for that param

        Returns:
            new_dict: The list of all combinations of parameters

        """
        if not params_dict:
            return None
        new_dict = {}
        for key, value in params_dict.items():
            assert isinstance(value, Iterable)
            if key == 'layers':
                new_dict[key] = value
            elif type(value[0]) != str:
                tmp_list = list(np.random.uniform(value[0], value[1], value[2]))
                if key in self.convert_to_int:
                    new_dict[key] = [int(x) for x in tmp_list]
                else:
                    new_dict[key] = tmp_list
            else:
                new_dict[key] = value
        return new_dict

class GeometricSearch(HyperparameterSearch):
    """Generates parameter values in logistic steps, rather than linear like GridSearch does"""

    def __init__(self, params):
        super().__init__(params)

    def split_and_save_dataset(self, assay_params):
        self.split_and_save_dataset(assay_params)

    def generate_param_combos(self):
        super().generate_param_combos()

    def generate_assay_list(self):
        super().generate_assay_list()

    def generate_combo(self, params_dict):
        """Method to generate all combinations from a given set of key-value pairs

        Args:
            params_dict: Set of key-value pairs with the key being the param name and the value being the list of values
            you want to try for that param

        Returns:
            new_dict: The list of all combinations of parameters

        """
        if not params_dict:
            return None

        new_dict = {}
        for key, value in params_dict.items():
            assert isinstance(value, Iterable)
            if key == 'layers':
                new_dict[key] = value
            elif type(value[0]) != str:
                tmp_list = list(np.geomspace(value[0], value[1], int(value[2])))
                if key in self.convert_to_int:
                    new_dict[key] = [int(x) for x in tmp_list]
                else:
                    new_dict[key] = tmp_list
            else:
                new_dict[key] = value
        return new_dict

class UserSpecifiedSearch(HyperparameterSearch):
    """Generates combinations using the user-specified steps"""

    def __init__(self, params):
        super().__init__(params)

    def split_and_save_dataset(self, assay_params):
        self.split_and_save_dataset(assay_params)

    def generate_param_combos(self):
        super().generate_param_combos()

    def generate_assay_list(self):
        super().generate_assay_list()

    def generate_combo(self, params_dict):
        """Method to generate all combinations from a given set of key-value pairs

        Args:
            params_dict: Set of key-value pairs with the key being the param name and the value being the list of values
            you want to try for that param

        Returns:
            new_dict: The list of all combinations of parameters

        """

        if not params_dict:
            return None
        new_dict = {}
        for key, value in params_dict.items():
            assert isinstance(value, Iterable)
            if key == 'layers':
                new_dict[key] = value
            elif key in self.convert_to_int:
                new_dict[key] = [int(x) for x in value]
            elif key in self.convert_to_float:
                new_dict[key] = [float(x) for x in value]
            else:
                new_dict[key] = value
        return new_dict

def build_hyperopt_search_domain(label, method, param_list):
    """Generate HyperOpt search domain object from method and parameters, layer_nums is only for NN models.
    This function is used by the HyperOptSearch class, not intended for standalone usage.
    """
    if method == "choice":
        return hp.choice(label, param_list)
    elif method == "uniform":
        return hp.uniform(label, param_list[0], param_list[1])
    elif method == "loguniform":
        return hp.loguniform(label, param_list[0], param_list[1])
    elif method == "uniformint":
        return hp.uniformint(label, param_list[0], param_list[1])
    else:
        raise Exception(f"Method {method} is not supported, choose from 'choice, uniform, loguniform, uniformint'.")

class HyperOptSearch():
    """Perform hyperparameter search with Bayesian Optmization (Tree Parzen Estimator)

    To use HyperOptSearch, modify the config json file as follows:

        serach_type: use "hyperopt"

        result_dir: use two directories (recommended), separated by comma, 1st one will be used to save the best model tarball, 2nd one will be used to store all models during the process.  e.g. "result_dir": "/path/of/the/final/dir,/path/of/the/temp/dir"

        model_type: RF or NN, also add max number of HyperOptSearch evaluations, e.g. "model_type": "RF|100".  If no max number provide, the default 100 will be used.  #For NN models only

        lr: specify learning rate searching method and related parameters as the following scheme.
            method|parameter1,parameter2...

            method: supported searching schemes in HyperOpt include: choice, uniform, loguniform, and uniformint, see https://github.com/hyperopt/hyperopt/wiki/FMin for details.

    parameters:
        choice: all values to search from, separated by comma, e.g. choice|0.0001,0.0005,0.0002,0.001

        uniform: low and high bound of the interval to serach, e.g. uniform|0.00001,0.001

        loguniform: low and high bound (in natural log) of the interval to serach, e.g. loguniform|-13.8,-6.9

        uniformint: low and high bound of the interval to serach, e.g. uniformint|8,256

        ls: similar as learning_rate, specify number of layers and size of each one.

            method|num_layers|parameter1,parameter2...
            e.g. choice|2|8,16,32,64,128,256,512  #this will generate a two-layer config, each layer takes size from the list "8,16,32,64,128,256,512"
            e.g. uniformint|3|8,512  #this will generate a three-layer config, each layer takes size from the uniform interval [8,512]

        dp: similar as layer_sizes, just make sure dropouts and layer_sizes should have the same number of layers.

            e.g. uniform|3|0,0.4   #this will generate a three-layer config, each layer takes size from the uniform interval [0,0.4]

        #For RF models only
        rfe: rf_estimator, same structure as the learning rate above, e.g. uniformint|64,512  #take integer values from a uniform interval [64,512]

        rfd: rf_max_depth, e.g. uniformint|8,256

        rff: rf_max_feature, e.g. uniformint|8,128
    """
    def __init__(self, params):
        self.params = params
        #separate temp output dir and final output dir
        result_dir_list = params.result_dir.split(",")
        if len(result_dir_list) > 1:
            self.params.result_dir = result_dir_list[1]
            self.final_dir = result_dir_list[0]
        else:
            self.params.result_dir = result_dir_list[0]
            self.final_dir = result_dir_list[0]

        if len(self.params.model_type.split("|")) > 1:
            self.max_eval = int(self.params.model_type.split("|")[1])
            self.params.model_type = self.params.model_type.split("|")[0]
        else:
            self.max_eval = 100

        #define the searching space
        self.space = {}
        if isinstance(self.params.featurizer, list):
            self.space["featurizer"] = build_hyperopt_search_domain("featurizer", "choice", self.params.featurizer)
        if isinstance(self.params.descriptor_type, list):
            self.space["descriptor_type"] = build_hyperopt_search_domain("descriptor_type", "choice", self.params.descriptor_type)
        if self.params.model_type == "RF":
            #build searching domain for RF parameters
            if self.params.rfe:
                domain_list = self.params.rfe.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["rf_estimators"] = build_hyperopt_search_domain("rf_estimators", method, par_list)

            if self.params.rfd:
                domain_list = self.params.rfd.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["rf_max_depth"] = build_hyperopt_search_domain("rf_max_depth", method, par_list)

            if self.params.rff:
                domain_list = self.params.rff.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["rf_max_features"] = build_hyperopt_search_domain("rf_max_features", method, par_list)
        elif self.params.model_type == "NN":
            #build searching domain for NN parameters
            if self.params.lr:
                domain_list = self.params.lr.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["learning_rate"] = build_hyperopt_search_domain("learning_rate", method, par_list)

            # for layer sizes, use a different method if the ls_ratio is provided
            if self.params.ls:
                domain_list = self.params.ls.split("|")
                method = domain_list[0]
                num_layer = int(domain_list[1])
                par_list = [float(e) for e in domain_list[2].split(",")]
                if not self.params.ls_ratio:
                    for i in range(num_layer):
                        self.space[f"ls{i}"] = build_hyperopt_search_domain(f"ls{i}", method, par_list)
                else:
                    self.space["ls"] = build_hyperopt_search_domain("ls", method, par_list)
                    domain_list = self.params.ls_ratio.split("|")
                    method = domain_list[0]
                    par_list = [float(e) for e in domain_list[-1].split(",")]
                    for i in range(1,num_layer):
                        self.space[f"ratio{i}"] = build_hyperopt_search_domain(f"ratio{i}", method, par_list)

            if self.params.dp:
                domain_list = self.params.dp.split("|")
                method = domain_list[0]
                num_layer = int(domain_list[1])
                par_list = [float(e) for e in domain_list[2].split(",")]
                for i in range(num_layer):
                    self.space[f"dp{i}"] = build_hyperopt_search_domain(f"dp{i}", method, par_list)
        elif self.params.model_type == "xgboost":
            #build searching domain for XGBoost parameters
            if self.params.xgbg:
                domain_list = self.params.xgbg.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["xgbg"] = build_hyperopt_search_domain("xgbg", method, par_list)

            if self.params.xgbl:
                domain_list = self.params.xgbl.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["xgbl"] = build_hyperopt_search_domain("xgbl", method, par_list)

            if self.params.xgbd:
                domain_list = self.params.xgbd.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["xgbd"] = build_hyperopt_search_domain("xgbd", method, par_list)

            if self.params.xgbc:
                domain_list = self.params.xgbc.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["xgbc"] = build_hyperopt_search_domain("xgbc", method, par_list)

            if self.params.xgbs:
                domain_list = self.params.xgbs.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["xgbs"] = build_hyperopt_search_domain("xgbs", method, par_list)

            if self.params.xgbn:
                domain_list = self.params.xgbn.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["xgbn"] = build_hyperopt_search_domain("xgbn", method, par_list)

            if self.params.xgbw:
                domain_list = self.params.xgbw.split("|")
                method = domain_list[0]
                par_list = [float(e) for e in domain_list[1].split(",")]
                self.space["xgbw"] = build_hyperopt_search_domain("xgbw", method, par_list)


    def run_search(self):
        #name of the results
        feat = "_".join(self.params.featurizer) if isinstance(self.params.featurizer, list) else self.params.featurizer
        desc = "_".join(self.params.descriptor_type) if isinstance(self.params.descriptor_type, list) else self.params.descriptor_type
        if "_" not in feat or feat in ["computed_descriptors", "descriptors"]:
            fd = feat if feat in ["graphconv", "ecfp"] else desc
        else:
            fd = f"{feat}_{desc}"

        def lossfn(p):
            if "featurizer" in p:
                self.params.featurizer = p["featurizer"]

            if "descriptor_type" in p:
                self.params.descriptor_type = p["descriptor_type"]

            if self.params.model_type == "RF":
                if self.params.rfe:
                    self.params.rf_estimators =  p["rf_estimators"]
                if self.params.rfd:
                    self.params.rf_max_depth = p["rf_max_depth"]
                if self.params.rff:
                    self.params.rf_max_features = p["rf_max_features"]
                hp_params = f'{self.params.rf_estimators}_{self.params.rf_max_depth}_{self.params.rf_max_features}'
                print(f'rf_estimators: {self.params.rf_estimators}, rf_max_depth: {self.params.rf_max_depth}, rf_max_feature: {self.params.rf_max_features}')
            elif self.params.model_type == "NN":
                if self.params.lr:
                    self.params.learning_rate = p["learning_rate"]
                if self.params.dp:
                    self.params.dropouts = ",".join([str(p[e]) for e in p if e[:2] == "dp"])
                if self.params.ls:
                    if not self.params.ls_ratio:
                        self.params.layer_sizes = ",".join([str(p[e]) for e in p if e[:2] == "ls"])
                    else:
                        list_layer_sizes = [p["ls"]]
                        for i in range(1,len([e for e in p if e[:5] == "ratio"])+1):
                            list_layer_sizes.append(int(list_layer_sizes[-1] * p[f"ratio{i}"]))
                        self.params.layer_sizes = ",".join([str(e) for e in list_layer_sizes])
                hp_params = f'{self.params.learning_rate}_{self.params.layer_sizes}_{self.params.dropouts}'
                print(f"learning_rate: {self.params.learning_rate}, layer_sizes: {self.params.layer_sizes}, dropouts: {self.params.dropouts}")
            elif self.params.model_type == "xgboost":
                if self.params.xgbg:
                    self.params.xgb_gamma = p["xgbg"]
                if self.params.xgbl:
                    self.params.xgb_learning_rate = p["xgbl"]
                if self.params.xgbd:
                    self.params.xgb_max_depth = p["xgbd"]
                if self.params.xgbc:
                    self.params.xgb_colsample_bytree = p["xgbc"]
                if self.params.xgbs:
                    self.params.xgb_subsample = p["xgbs"]
                if self.params.xgbn:
                    self.params.xgb_n_estimators = p["xgbn"]
                if self.params.xgbw:
                    self.params.xgb_min_child_weight = p["xgbw"]
                hp_params = f'{self.params.xgb_gamma}_{self.params.xgb_learning_rate}_{self.params.xgb_max_depth}_{self.params.xgb_colsample_bytree}_{self.params.xgb_subsample}_{self.params.xgb_n_estimators}_{self.params.xgb_min_child_weight}'
                print(f"xgb_gamma: {self.params.xgb_gamma}, "
                      f"xgb_learning_rate: {self.params.xgb_learning_rate}, "
                      f"xgb_max_depth: {self.params.xgb_max_depth}, "
                      f"xgb_colsample_bytree: {self.params.xgb_colsample_bytree}, "
                      f"xgb_subsample: {self.params.xgb_subsample}, "
                      f"xgb_n_estimators: {self.params.xgb_n_estimators}, "
                      f"xgb_min_child_weight: {self.params.xgb_min_child_weight}")

            # set hyperparam to False to make sure the layer_sizes and dropouts are not lists if not optimized.
            self.params.hyperparam = False
            if isinstance(self.params.layer_sizes, list):
                if isinstance(self.params.layer_sizes[0], list):
                    self.params.layer_sizes = ",".join([str(e) for e in self.params.layer_sizes[0]])
                else:
                    self.params.layer_sizes = ",".join([str(e) for e in self.params.layer_sizes])
                hp_params = f'{self.params.learning_rate}_{self.params.layer_sizes}_{self.params.dropouts}'
            if isinstance(self.params.dropouts, list):
                if isinstance(self.params.dropouts[0], list):
                    self.params.dropouts = ",".join([str(e) for e in self.params.dropouts[0]])
                else:
                    self.params.dropouts = ",".join([str(e) for e in self.params.dropouts])
                hp_params = f'{self.params.learning_rate}_{self.params.layer_sizes}_{self.params.dropouts}'

            tparam = parse.wrapper(self.params.__dict__)
            print(f"{self.params.model_type} model with {self.params.featurizer} and {self.params.descriptor_type}")
            # make sure classification model has uncertainty as False. 
            if tparam.prediction_type != "regression":
                tparam.uncertainty = False
            pl = mp.ModelPipeline(tparam)

            model_failed = False
            try:
                pl.train_model()
            except:
                model_failed = True

            subsets = ["train", "valid", "test"]
            pred_results = dict(zip(subsets, [{} for _ in subsets]))
            for subset in subsets:
                if not model_failed:
                    perf_data = pl.model_wrapper.get_perf_data(subset=subset, epoch_label="best")
                    sub_pred_results = perf_data.get_prediction_results()
                else:
                    if tparam.prediction_type == "regression":
                        sub_pred_results = {"r2_score": 0, "rms_score": 100}
                    else:
                        sub_pred_results = {"roc_auc_score": 0, "accuracy_score": 0}

                if tparam.prediction_type == "regression":
                    pred_results[subset]["r2"] = sub_pred_results['r2_score']
                    pred_results[subset]["rms"] = sub_pred_results['rms_score']
                else:
                    pred_results[subset]["roc_auc"] = sub_pred_results["roc_auc_score"]
                    pred_results[subset]["acc"] = sub_pred_results["accuracy_score"]
            if tparam.prediction_type == "regression":
                res_dict = {'loss': 1-pred_results["valid"]["r2"], 'status': STATUS_OK, 'model': tparam.model_tarball_path, 'featurizer': tparam.featurizer, 'desc': tparam.descriptor_type}
                for subset in subsets:
                    res_dict[f"{subset}_r2"] = pred_results[subset]["r2"]
                    res_dict[f"{subset}_rms"] = pred_results[subset]["rms"]
            else:
                res_dict = {'loss': 100-pred_results["valid"]["roc_auc"], 'status': STATUS_OK, 'model': tparam.model_tarball_path, 'featurizer': tparam.featurizer, 'desc': tparam.descriptor_type}
                for subset in subsets:
                    res_dict[f"{subset}_roc_auc"] = pred_results[subset]["roc_auc"]
                    res_dict[f"{subset}_acc"] = pred_results[subset]["acc"]
            res_dict["hp_params"] = hp_params

            # print the model metrics as logs
            print()
            if tparam.prediction_type == "regression":
                print(f'model_performance|{res_dict["train_r2"]:.3f}|{res_dict["train_rms"]:.3f}|{res_dict["valid_r2"]:.3f}|{res_dict["valid_rms"]:.3f}|{res_dict["test_r2"]:.3f}|{res_dict["test_rms"]:.3f}|{res_dict["hp_params"]}|{res_dict["model"]}\n')
            else:
                print(f'model_performance|{res_dict["train_roc_auc"]:.3f}|{res_dict["train_acc"]:.3f}|{res_dict["valid_roc_auc"]:.3f}|{res_dict["valid_acc"]:.3f}|{res_dict["test_roc_auc"]:.3f}|{res_dict["test_acc"]:.3f}|{res_dict["hp_params"]}|{res_dict["model"]}\n')

            return res_dict

        if self.params.prediction_type == "regression":
            print('model_performance|train_r2|train_rms|valid_r2|valid_rms|test_r2|test_rms|model_params|model\n')
        else:
            print('model_performance|train_roc_auc|train_acc|valid_roc_auc|valid_acc|test_roc_auc|test_acc|model_params|model\n')

        if self.params.hp_checkpoint_load is not None and os.path.isfile(self.params.hp_checkpoint_load):
            print(f"load hpo trial object from {self.params.hp_checkpoint_load}")
            with open(self.params.hp_checkpoint_load, "rb") as f:
                trials = pickle.load(f)
        else:
            trials = Trials()

        if self.params.hp_checkpoint_save is not None:
            print("hp_checkpoint_save provided, save a checkpoint file every 5 trials.")
            max_evals = 5
            while True:
                if os.path.isfile(self.params.hp_checkpoint_save):
                    print(f"load hpo trial object from {self.params.hp_checkpoint_save}")
                    with open(self.params.hp_checkpoint_save, "rb") as f:
                        trials = pickle.load(f)
                    max_evals = min(len(trials) + 5, self.max_eval)
                else:
                    max_evals = min(max_evals, self.max_eval)

                best = fmin(lossfn, self.space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

                print(f"Save HPO trial object to {self.params.hp_checkpoint_save}")
                with open(self.params.hp_checkpoint_save, "wb") as f:
                    pickle.dump(trials, f)

                if max_evals == self.max_eval:
                    break
        else:
            best = fmin(lossfn, self.space, algo=tpe.suggest, max_evals=self.max_eval, trials=trials)

        print("Generating the performance -- iteration table and Copy the best model tarball.")

        feat_list = [trials.trials[i]["result"]["featurizer"] for i in range(len(trials.trials))]
        desc_list = [trials.trials[i]["result"]["desc"] for i in range(len(trials.trials))]
        hp_params_list = [trials.trials[i]["result"]["hp_params"] for i in range(len(trials.trials))]
        trial_data = {"trial": list(range(len(trials.trials))), "featurizer": feat_list, "descriptor": desc_list, "model_params": hp_params_list}
        subsets = ["train", "valid", "test"]
        for subset in subsets:
            if self.params.prediction_type == "regression":
                trial_data[f"{subset}_r2"] = [trials.trials[i]["result"][f"{subset}_r2"] for i in range(len(trials.trials))]
                trial_data[f"{subset}_rms"] = [trials.trials[i]["result"][f"{subset}_rms"] for i in range(len(trials.trials))]
            else:
                trial_data[f"{subset}_roc_auc"] = [trials.trials[i]["result"][f"{subset}_roc_auc"] for i in range(len(trials.trials))]
                trial_data[f"{subset}_acc"] = [trials.trials[i]["result"][f"{subset}_acc"] for i in range(len(trials.trials))]
        perf = pd.DataFrame(trial_data)

        if self.params.prediction_type == "regression":
            best_trial = perf.sort_values(by="valid_r2", ascending=False)["trial"].iloc[0]
            best_model = trials.trials[best_trial]["result"]["model"]
            print(f'Best model: {best_model}, valid R2: {perf.sort_values(by="valid_r2", ascending=False)["valid_r2"].iloc[0]}')
        else:
            best_trial = perf.sort_values(by="valid_roc_auc", ascending=False)["trial"].iloc[0]
            best_model = trials.trials[best_trial]["result"]["model"]
            print(f'Best model: {best_model}, valid ROC_AUC: {perf.sort_values(by="valid_roc_auc", ascending=False)["valid_roc_auc"].iloc[0]}')

        bmodel_prefix = "_".join(os.path.basename(best_model).split("_")[:-1])
        bmodel_uuid = os.path.basename(best_model).split(".")[0].split("_")[-1]

        perf.to_csv(os.path.join(self.final_dir, f"performance_{self.params.prediction_type}_{bmodel_prefix}_{self.params.model_type}_{fd}_{bmodel_uuid}.csv"), index=False)
        if os.path.isfile(best_model):
            # if the model tracker is used, the model won't be saved to the result_dir
            shutil.copy2(best_model, os.path.join(self.final_dir,
                                              f"best_{self.params.prediction_type}_{bmodel_prefix}_{self.params.model_type}_{fd}_{bmodel_uuid}.tar.gz"))

def parse_params(param_list):
    """Parse paramters

    Parses parameters using parameter_parser.wrapper and
    filters out unnecessary parameters. Returns what an
    argparse.Namespace

    Args:
        *any_arg: any single input of a str, dict, argparse.Namespace, or list

    Returns:
        argparse.Namespace
    """
    params = parse.wrapper(param_list)
    keep_params = {'prediction_type',
                   'model_type',
                   'featurizer',
                   'hyperparam_uuid',
                   'splitter',
                   'datastore',
                   'save_results',
                   'previously_featurized',
                   'previously_split',
                   'prediction_type',
                   'descriptor_key',
                   'descriptor_type',
                   'split_valid_frac',
                   'split_test_frac',
                   'split_uuid',
                   'bucket',
                   'lc_account',
                   'slurm_account',
                   'slurm_export',
                   'slurm_nodes',
                   'slurm_options',
                   'slurm_partition',
                   'slurm_time_limit'} | excluded_keys
    if params.search_type == 'hyperopt':
        # keep more parameters
        keep_params = keep_params | {'lr', 'learning_rate','ls', 'layer_sizes','ls_ratio','dp', 'dropouts','rfe', 'rf_estimators','rfd', 'rf_max_depth','rff', 'rf_max_features','xgbg', 'xgb_gamma','xgbl', 'xgb_learning_rate', 'xgbd', 'xgb_max_depth', 'xgbc', 'xgb_colsample_bytree', 'xgbs', 'xgb_subsample', 'xgbn', 'xgb_n_estimators', 'xgbw', 'xgb_min_child_weight', 'hp_checkpoint_load', 'hp_checkpoint_save'}

    params.__dict__ = parse.prune_defaults(params, keep_params=keep_params)

    return params

def build_search(params):
    """Builds HyperparamterSearch object

       Looks at params.search_type and builds a HyperparamSearch object
       of the correct flavor. Will exit if the search_type is not
       recognized.

    Args:
       params (Namespace): Namespace returned by
           atomsci.ddm.pipeline.parameter_parser.wrapper()

    Returns:
       HyperparameterSearch
    """
    if params.search_type == 'grid':
        hs = GridSearch(params)
    elif params.search_type == 'random':
        hs = RandomSearch(params)
    elif params.search_type == 'geometric':
        hs = GeometricSearch(params)
    elif params.search_type == 'user_specified':
        hs = UserSpecifiedSearch(params)
    elif params.search_type == 'hyperopt':
        hs = HyperOptSearch(params)
    else:
        print("Incorrect search type specified")
        sys.exit(1)

    return hs

def main():
    """Entry point when script is run

    Args:
        None

    Returns:
        None

    """

    params = parse_params(sys.argv[1:])
    hs = build_search(params)

    if params.split_only and params.datastore:
        hs.generate_split_shortlist()
    elif params.split_only and not params.datastore:
        hs.generate_split_shortlist_file()
    else:
        hs.run_search()

if __name__ == '__main__' and len(sys.argv) > 1:
    main()
    sys.exit(0)
