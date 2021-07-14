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
from datetime import datetime
import glob
import json
from pathlib import Path
import os
import sys
import tempfile
import time

import tarfile
import logging

logger = logging.getLogger(__name__)

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.utils.curate_data as curate_data
import atomsci.ddm.utils.struct_utils as struct_utils
import atomsci.ddm.pipeline.model_tracker as mt
import atomsci.ddm.utils.datastore_functions as dsf

mlmt_supported = True
try:
    from atomsci.clients import MLMTClient
except (ModuleNotFoundError, ImportError):
    logger.debug("Model tracker client not supported in your environment; will save models in filesystem only.")
    mlmt_supported = False
    


def train_model(input, output):
    """ Retrain a model saved in a model_metadata.json file

    Args:
        input (str): path to model_metadata.json file

        output (str): path to output directory

    Returns:
        None
    """
    # Train model
    # -----------
    # Read parameter JSON file
    with open(input) as f:
        config = json.loads(f.read())

    # Parse parameters
    params = parse.wrapper(config)
    params.result_dir = output
    # otherwise this will have the same uuid as the source model
    params.model_uuid = None
    # use the same split
    params.previously_split = True
    params.split_uuid = config['splitting_parameters']['split_uuid']
    # specify collection
    
    
    logger.debug("model params %s" % str(params))

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    return model

def train_model_from_tar(input, output):
    """ Retrain a model saved in a tar.gz file

    Args:
        input (str): path to a tar.gz file

        output (str): path to output directory

    Returns:
        None
    """
    tmpdir = tempfile.mkdtemp()

    model_fp = tarfile.open(input, mode='r:gz')
    model_fp.extractall(path=tmpdir)
    model_fp.close()

    # make metadata path
    metadata_path = os.path.join(tmpdir, 'model_metadata.json')

    return train_model(metadata_path, output)

def train_model_from_tracker(model_uuid, output_dir):
    """ Retrain a model saved in the model tracker, but save it to output_dir and don't insert it into the model tracker

    Args:
        model_uuid (str): model tracker model_uuid file

        output_dir (str): path to output directory

    Returns:
        the model pipeline object with trained model
    """
    
    if not mlmt_supported:
        print("Model tracker not supported in your environment; can load models from filesystem only.")
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
    if config[]
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

    logger.debug("model params %s" % str(params))

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    return model


#----------------
# main
#----------------
def main(argv):
    start_time = time.time()

    # input file/dir (required)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input directory, file or model_uuid')
    parser.add_argument('-o', '--output', help='output result directory')

    args = parser.parse_args()

    input = args.input
    output = args.output

    # if not specified, default to temp dir
    if not (output and output.strip()):
        output = tempfile.mkdtemp()

    if os.path.isdir(input):
    # loop
        for path in Path(input).rglob('model_metadata.json'):
            train_model(path.absolute(), output)
    elif input.endswith('.json'):
        train_model(input, output)
    elif input.endswith('.tar.gz'):
        train_model_from_tar(input, output)
    else:
        try:
            train_model_from_tracker(input, output)
        except:
            Exception('Unrecognized input %s'%input)

    elapsed_time_secs = time.time() - start_time
    logger.info("Execution took: %s secs" % timedelta(seconds=round(elapsed_time_secs)))

if __name__ == "__main__":
   main(sys.argv[1:])
