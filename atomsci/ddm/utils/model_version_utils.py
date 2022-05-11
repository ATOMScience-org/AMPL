"""
model_version_utils.py

Misc utilities to get the AMPL model version:

To check the model version

 usage: model_version_utils.py [-h] -i INPUT

 optional arguments:
   -h, --help            show this help message and exit

  -i INPUT, --input INPUT     input directory/file (required)

"""

import argparse
import traceback
import tarfile
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import logging
import pandas as pd
import pdb

logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from atomsci.ddm.utils import datastore_functions as dsf

import pkg_resources

# get the ampl run version
if ('site-packages' in dsf.__file__) or ('dist-packages' in dsf.__file__): # install_dev.sh points to github directory
    import subprocess
    import json
    data = subprocess.check_output(["pip", "list", "--format", "json"])
    parsed_results = json.loads(data)
    ampl_version = next(item for item in parsed_results if item["name"] == "atomsci-ampl")['version']
else:
    try:
        VERSION_fn = os.path.join(
            os.path.dirname(pkg_resources.resource_filename('atomsci', '')),
            'VERSION')
    except:
        VERSION_fn = dsf.__file__.rsplit('/', maxsplit=4)[0]+'/VERSION'

    f = open(VERSION_fn, 'r')
    ampl_version = f.read().strip()
    f.close()

def get_ampl_version_from_dir(dirname):
    """
    Get the AMPL versions from a directory

    Args:
        dirname (str): directory
    
    Returns:
        returns list of AMPL versions
    """
    versions = []
    # loop
    for path in Path(dirname).rglob('*.tar.gz'):
        try:
            version = get_ampl_version(path.absolute())
            versions.append('{}, {}'.format(path.absolute(), version))
        except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
            logger.exception("Exception message: {}".format(e))
            pass
            
    return '\n'.join(versions)

def get_ampl_version(filename):
    """
    Get the AMPL version from the tar file's model_metadata.json

    Args:
        filename (str): tar file
    
    Returns:
        returns the AMPL version number
    """
    tmpdir = tempfile.mkdtemp()
        
    model_fp = tarfile.open(filename, mode='r:gz')
    model_fp.extractall(path=tmpdir)
    model_fp.close()
        
    # make metadata path
    metadata_path = os.path.join(tmpdir, 'model_metadata.json')
    version = get_version_from_json(metadata_path)
    logger.info('{}, {}'.format(filename, version))
    return version

def get_version_from_json(metadata_path):
    """
    Parse model_metadata.json to get the AMPL version

    Args:
        filename (str): tar file
    
    Returns:
        returns the AMPL version number
        
    """
    with open(metadata_path, 'r') as data_file:
        metadata_dict = json.load(data_file)
        version = metadata_dict.get("model_parameters").get("ampl_version", 'probably 1.0.0')
        return version

def check_version_compatible(input_tarfile, ignore_check=True):
    """
    Compare the input file's version against the running AMPL version to see if
    they are compatible

    Args:
        filename (str): tar file
    
    Returns:
        returns True if the second digits match
    
    """
    version = get_ampl_version(input_tarfile).strip()
    version_tokens = version.split('.')
    current_version_tokens = ampl_version.split('.')
    logger.info('Version compatible check: {} version = "{}", AMPL version = "{}"'.format(input_tarfile, version, ampl_version))
    match = (version_tokens[1] == current_version_tokens[1])
    
    # raise an exception if not match and we don't want to ignore
    if not match:
        if not ignore_check:
            my_error = ValueError('Version compatible check: {} version: "{}" not matching AMPL version: "{}"'.format(input_tarfile, version, ampl_version))
            raise my_error
    return match

#----------------
# main
#----------------
def main(argv):

    # input file/dir (required)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input model directory/file')

    args = parser.parse_args()

    finput = args.input

    # check if it's a directory
    if os.path.isdir(finput):
        get_ampl_version_from_dir(finput)
    elif os.path.isfile(finput):
        get_ampl_version(finput)

if __name__ == "__main__":
   main(sys.argv[1:])