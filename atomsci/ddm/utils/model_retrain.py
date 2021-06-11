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

def train_model(input, output, dskey=''):
    """ Retrain a model saved in a model_metadata.json file

    Args:
        input (str): path to model_metadata.json file

        output (str): path to output directory

        dskey (str): new dataset key if file location has changed

    Returns:
        None
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


    logger.debug("model params %s" % str(params))

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    return model

def train_model_from_tar(input, output, dskey=''):
    """ Retrain a model saved in a tar.gz file

    Args:
        input (str): path to a tar.gz file

        output (str): path to output directory

        dskey (str): new dataset key if file location has changed

    Returns:
        None
    """
    tmpdir = tempfile.mkdtemp()

    model_fp = tarfile.open(input, mode='r:gz')
    model_fp.extractall(path=tmpdir)
    model_fp.close()

    # make metadata path
    metadata_path = os.path.join(tmpdir, 'model_metadata.json')

    return train_model(metadata_path, output, dskey=dskey)


#----------------
# main
#----------------
def main(argv):
    start_time = time.time()

    # input file/dir (required)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input directory/file')
    parser.add_argument('-o', '--output', help='output result directory')
    parser.add_argument('-dk', '--dataset_key', default='', help='Sometimes dataset keys get moved. Specify new location of dataset. Only works when passing in one model at time.')

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
        train_model(input, output, dskey=args.dataset_key)
    elif input.endswith('.tar.gz'):
        train_model_from_tar(input, output, dskey=args.dataset_key)
    else:
        raise Exception('Unrecoganized input %s'%input)

    elapsed_time_secs = time.time() - start_time
    logger.info("Execution took: %s secs" % timedelta(seconds=round(elapsed_time_secs)))

if __name__ == "__main__":
   main(sys.argv[1:])
