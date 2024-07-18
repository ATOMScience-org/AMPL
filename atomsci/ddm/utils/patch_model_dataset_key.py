#!/usr/bin/env python

"""Script to create a new version of a model tarball file with the training dataset dataset_key
parameter (in model_metadata.json) pointing to a different file path. This may be needed when the
model was trained on a different computer system or with data in a private directory, and the model
is subsequently made publicly available. In this case the training data needs to be copied to an
accessible directory on the target computer so that it can be used for AD index computations when
predicting from the model.
"""

import os
import sys
import tarfile
import tempfile
import json
import glob

from atomsci.ddm.utils import checksum_utils as cu
from atomsci.ddm.utils import file_utils as futils

def check_data_accessibility(model_path, verbose=True):
    """Check the dataset_key parameters in one or more AMPL model tarball files
    to see if the associated dataset files are readable. If `model_path` is a directory, the files
    within it with .tar.gz extensions are checked; otherwise it is interpreted as a path to a model
    tarball. Returns a dictionary with tarball paths as keys and tuples as values, with the first
    element of the tuple being the training dataset path and the second a boolean indicating whether
    it is readable.
    """
    if os.path.isdir(model_path):
        model_paths = glob.glob(f"{model_path}/*.tar.gz")
    else:
        model_paths = [model_path]
    dataset_info = {}
    for path in model_paths:
        with tarfile.open(path, mode='r:gz') as tarball:
            try:
                meta_info = tarball.getmember('./model_metadata.json')
            except KeyError:
                print(f"{path} is not an AMPL model tarball")
                continue
            with tarball.extractfile(meta_info) as meta_fd:
                meta_dict = json.loads(meta_fd.read())
                dataset_path = meta_dict['training_dataset']['dataset_key']
                try:
                    dset_fp = open(dataset_path, 'r')
                    dset_fp.close()
                    dataset_info[path] = (dataset_path, True)
                except:
                    dataset_info[path] = (dataset_path, False)
                    if verbose:
                        print(f"{os.path.basename(path)} trained on unreadable file:\n\t{dataset_path}")
    return dataset_info


def patch_model_dataset_key(model_path, new_model_path, dataset_path, require_hash_match=True):
    """Create a new version of the model tarball given by `model_path` with its dataset_key parameter
    replaced with `dataset_path`, and write it to `new_model_path`.

    Args:
        model_path (str): Path to existing model tarball.

        new_model_path (str): Path to write new model to.

        dataset_path (str): New path for training dataset.

        require_hash_match (bool): If True, do not create a new tarball if the checksum for the file
        at `dataset_path` doesn't match the checksum stored in the model metadata.
    """

    # Compute a checksum for the new dataset.
    try:
        new_hash = cu.create_checksum(dataset_path)
    except Exception:
        print(f"Error reading dataset {dataset_path}")
        raise

    # Extract the model tarball into a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(model_path, 'rb') as tarfile_fp:
            with tarfile.open(fileobj=tarfile_fp, mode='r:gz') as tfile:
                tar_contents = tfile.getnames()
                if './model_metadata.json' not in tar_contents:
                    raise ValueError(f"{model_path} is not an AMPL model tarball")
                futils.safe_extract(tfile, path=tmp_dir)
        meta_path = os.path.join(tmp_dir, 'model_metadata.json')
        with open(meta_path, 'r') as meta_fp:
            meta_dict = json.load(meta_fp)
        old_hash = meta_dict['training_dataset']['dataset_hash']
        if new_hash != old_hash:
            print(f"Warning: {dataset_path} is different from the original model training dataset at:")
            print(meta_dict['training_dataset']['dataset_key'])
            if require_hash_match:
                print('New model tarball not created')
                return 1
            meta_dict['training_dataset']['dataset_hash'] = new_hash
        meta_dict['training_dataset']['dataset_key'] = os.path.realpath(dataset_path)
        # TODO: Check that it's OK to leave the original dataset_key and hash in the various training_metrics
        # elements.

        # Write the modified metadata.json file
        with open(meta_path, 'w') as meta_fp:
            json.dump(meta_dict, meta_fp, sort_keys=True, indent=4, separators=(',', ': '))
            meta_fp.write("\n")

        # Create the new tarball
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        with tarfile.open(new_model_path, mode='w:gz') as tarball:
            tarball.add(tmp_dir, arcname='.')
            print(f"Wrote {new_model_path}")
        return 0

if (__file__ == '__main__') and (len(sys.argv) > 3):
    model_path = sys.argv[1]
    new_model_path = sys.argv[2]
    dataset_path = sys.argv[3]
    require_hash_match = True
    if len(sys.argv) > 4:
        require_hash_match = (sys.argv[4].lower().startswith('n'))
    retval = patch_model_dataset_key(model_path, new_model_path, dataset_path, require_hash_match)
    sys.exit(retval)


