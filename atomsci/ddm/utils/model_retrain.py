#!/usr/bin/env python

# Purpose:
#
#  Script to take the existing model_metadata.json file or a directory and scans for 
#  model_metadata.json files and retrain, save them to DC 2.3 models. 
#
# usage: model_retrain.py [-h] -i INPUT [-o OUTPUT]
#
# optional arguments:
#   -h, --help            show this help message and exit
#
#  -i INPUT, --input INPUT     input directory/file
#  -o OUTPUT, --output OUTPUT  output result directory

import argparse
from datetime import timedelta
import json
from pathlib import Path
import os
import sys
import tempfile
import time

import tarfile
import logging
import pandas as pd

logging.basicConfig()

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_tracker as mt
import atomsci.ddm.utils.datastore_functions as dsf
from atomsci.ddm.pipeline import compare_models as cmp
import atomsci.ddm.utils.file_utils as futils

import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

mlmt_supported = True
try:
    from atomsci.clients import MLMTClient
except (ModuleNotFoundError, ImportError):
    logger.warning("Model tracker client not supported in your environment; will save models in filesystem only.")
    mlmt_supported = False


def train_model(input, output, dskey='', production=False):
    """Retrain a model saved in a model_metadata.json file
    
    Args:
       input (str): path to model_metadata.json file

       output (str): path to output directory

       dskey (str): new dataset key if file location has changed

       production (bool): retrain the model using production mode

    Returns:
       the model pipeline object with trained model
    """
    # Train model
    # -----------
    # Read parameter JSON file
    with open(input) as f:
        config = json.loads(f.read())

    # set a new dataset key if necessary
    if not dskey == '':
        config['dataset_key'] = dskey

    # Parse parameters
    params = parse.wrapper(config)
    params.result_dir = output
    # otherwise this will have the same uuid as the source model
    params.model_uuid = None
    # use the same split
    params.previously_split = True
    params.split_uuid = config['splitting_parameters']['split_uuid']
    # use production mode to train
    params.production = production
    if params.production and 'nn_specific' in config:
        params.max_epochs = config['nn_specific']['best_epoch']+1
    # change save mode if retraining elsewhere
    if not mlmt_supported:
        params.save_results=False
    # specify collection
    logger.debug("model params %s" % str(params))
    logger.debug(params.__dict__.items())

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    return model

def train_model_from_tar(input, output, dskey='', production=False):
    """Retrain a model saved in a tar.gz file
    
    Args:
       input (str): path to a tar.gz file

       output (str): path to output directory

       dskey (str): new dataset key if file location has changed

    Returns:
       the model pipeline object with trained model
    """
    tmpdir = tempfile.mkdtemp()

    with tarfile.open(input, mode='r:gz') as tar:
        futils.safe_extract(tar, path=tmpdir)

    # make metadata path
    metadata_path = os.path.join(tmpdir, 'model_metadata.json')
    
    return train_model(metadata_path, output, dskey=dskey, production=production)

def train_model_from_tracker(model_uuid, output_dir, production=False):
    """Retrain a model saved in the model tracker, but save it to output_dir and don't insert it into the model tracker

    Args:
       model_uuid (str): model tracker model_uuid file

       output_dir (str): path to output directory

    Returns:
       the model pipeline object with trained model
    """
    
    if not mlmt_supported:
        logger.debug("Model tracker not supported in your environment; can load models from filesystem only.")
        return None

    mlmt_client = dsf.initialize_model_tracker()
    
    collection_name = mt.get_model_collection_by_uuid(model_uuid, mlmt_client=mlmt_client)
        
    # get metadata from tracker
    config = mt.get_metadata_by_uuid(model_uuid)
    
    # check if datastore dataset
    try:
        result = dsf.retrieve_dataset_by_datasetkey(config['training_dataset']['dataset_key'], bucket=config['training_dataset']['bucket'])
        if result is not None:
            config['datastore']=True
    except:
        pass
    # fix weird old parameters
    #if config[]
    # Parse parameters
    params = parse.wrapper(config)
    params.result_dir = output_dir
    # otherwise this will have the same uuid as the source model
    params.model_uuid = None
    # use the same split
    params.previously_split = True
    params.split_uuid = config['splitting_parameters']['split_uuid']
    # specify collection
    params.collection_name = collection_name
    # use production mode to train
    params.production = production
    if params.production and 'nn_specific' in config:
        params.max_epochs = config['nn_specific']['best_epoch']+1

    logger.debug("model params %s" % str(params))

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    return model

def train_models_from_dataset_keys(input, output, pred_type='regression', production=False):
    """Retrain a list of models from an input file

    Args:
        input (str): path to an Excel or csv file. the required columns are 'dataset_key' and 'bucket' (public, private_file or Filesystem).

        output (str): path to output directory

        pred_type (str, optional): set the model prediction type. if not, uses the default 'regression'

    Returns:
        None
    """
    df = pd.DataFrame()
    # parse the input file
    logger.debug("Parsing %s file." % input)

    try:
        df = pd.read_excel(input)
    except:
        try:
            df = pd.read_csv(input)
        except:
            Exception('Unable to parse input %s. Only Excel or csv file is accepted.' % input)

    # extract the public bucket, then dataset keys
    public_list = df.loc[df['bucket'] == 'public']
    dataset_keys = public_list['dataset_key'].tolist()
    logger.debug('Found %d public dataset keys' % len(dataset_keys))

    client = MLMTClient()
    collections = client.get_collection_names()
    bucket = 'public'

    # find the collections 
    colls_w_dset = []
    for dset in dataset_keys:
        for coll in collections:
            datasets = cmp.get_collection_datasets(coll)
            if (dset, bucket) in datasets:
                colls_w_dset.append(coll)
    
    logger.debug('Found the dataset_keys in %d collections' % len(colls_w_dset))

    logger.debug("Train the model using prediction type %s." % pred_type)

    metric_type = 'r2_score'
    
    if (pred_type == 'classification'):
        metric_type = 'roc_auc_score'
    
    try:
        # find the best models
        best_mods = cmp.get_best_models_info(col_names=colls_w_dset, bucket=bucket, pred_type=pred_type, 
                                     result_dir=None, PK_pipeline=False, output_dir=output,
                                     shortlist_key=None, input_dset_keys=dataset_keys, save_results=False, subset='valid',
                                     metric_type=metric_type, selection_type='max', other_filters={})

        # retrain with uuid
        for model_uuid in best_mods.model_uuid.sort_values():
            try:
                logger.debug('Training %s in %s' % (model_uuid, output))
                train_model_from_tracker(model_uuid, output, production=production)
            except:
                Exception(f'Error for model_uuid {model_uuid}')
                pass
    except Exception as e:
        Exception('Error: %s' % str(e) )

#----------------
# main
#----------------
def main(argv):
    start_time = time.time()

    # input file/dir (required)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input directory, file or model_uuid')
    parser.add_argument('-o', '--output', help='output result directory')
    parser.add_argument('-dk', '--dataset_key', default='', help='Sometimes dataset keys get moved. Specify new location of dataset. Only works when passing in one model at time.')
    parser.add_argument('-pd_type', '--pred_type', default='regression', help='Specify the prediction type used for model retrain. The default is set to regression.')
    parser.add_argument('-prod', '--production', action='store_true', default=False, help='Retrain the model in production mode')

    args = parser.parse_args()

    input = args.input
    output = args.output

    # if not specified, default to temp dir
    if not (output and output.strip()):
        output = tempfile.mkdtemp()

    # 1 check if it's a directory
    if os.path.isdir(input):
        # loop
        for path in Path(input).rglob('model_metadata.json'):
            train_model(path.absolute(), output, production=args.production)
    elif os.path.isfile(input):
        # 2 if it's a file, check if it's a json or tar.gz or file that contains list of dataset keys
        if input.endswith('.json'):
            train_model(input, output, dskey=args.dataset_key, production=args.production)
        elif input.endswith('.tar.gz'):
            train_model_from_tar(input, output, dskey=args.dataset_key, production=args.production)
        else:
            train_models_from_dataset_keys(input, output, pred_type=args.pred_type, production=args.production)
    else:
        try:
            # 3 try to process 'input' as uuid
            train_model_from_tracker(input, output, production=args.production)
        except:
            Exception('Unrecognized input %s'%input)

    elapsed_time_secs = time.time() - start_time
    logger.info("Execution took: %s secs" % timedelta(seconds=round(elapsed_time_secs)))

if __name__ == "__main__":
   main(sys.argv[1:])
