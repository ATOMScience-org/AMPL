"""checksum_utils.py

Utilities for checksum related functions
"""

import hashlib
import tarfile
import json
import logging

log = logging.getLogger(__name__)

def create_checksum(filename):
    """
    Calculates hash of the file

    Args:
        filename (str): path to the dataset file
    
    Returns:
        returns the checksum

    """
    # https://docs.python.org/3/library/hashlib.html#hash-algorithms
    hash = hashlib.md5()
    with open(filename, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hash.update(chunk)
    return hash.hexdigest()

def uses_same_training_data_by_datasets(ds1, ds2):
    """Checks if the two input files' checksums match.

    Args:
        ds1 (str): dataset 1

        ds2 (str): dataset 2

    Returns:
        True if the checksums of the two input match

    """
    hash1 = create_checksum(ds1)
    hash2 = create_checksum(ds2)

    return hash1 == hash2

def uses_same_training_data_by_tarballs(tar1, tar2):
    """Checks if the two input files' checksums match.

    Args:
        tar1 (str): path to the first tar file

        tar2 (str):  path to the second tar file

    Returns:
        True if the checksums of the two input files match
        False if
               1) the two checksums don't match
               2) one of the file doesn't have checksums (will generate a warning)

    """
    ds_hash1 = get_dataset_hash_from_tar(tar1)
    ds_hash2 = get_dataset_hash_from_tar(tar2)
    
    log.info('Compare two tars hashes. ds_hash1 = %s ds_dash2 = %s', ds_hash1, ds_hash2)

    if ds_hash1 is None:
        log.warning("%s does not have a dataset hash.", tar1)
        return False
    
    if ds_hash2 is None:
        log.warning("%s does not have a dataset hash.", tar2)
        return False

    return ds_hash1 == ds_hash2

def get_dataset_hash_from_tar(tar):
    # extract the model_metadata.json from tar
    model_fp = tarfile.open(tar, mode='r:gz')
    metadata_file = model_fp.getmember("./model_metadata.json")
    ext_metadata = model_fp.extractfile(metadata_file)
      
    meta_json = json.load(ext_metadata)
    model_fp.close()

    return meta_json.get('training_dataset').get('dataset_hash')