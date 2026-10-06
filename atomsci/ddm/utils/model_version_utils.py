"""model_version_utils.py

Misc utilities to get the AMPL version(s) used to train one or more models and check them
for compatibility with the currently running version of AMPL.:

To check the model version

 usage: model_version_utils.py [-h] -i INPUT

 optional arguments:
   -h, --help            show this help message and exit

  -i INPUT, --input INPUT     input directory/file (required)

"""

import argparse
import json
import logging
import os
import re
import sys
import tarfile
from pathlib import Path

logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata  # python<=3.7


# ampl versions compatible groups
comp_dict = { '1.2': 'group1', '1.3': 'group1', '1.4': 'group2', '1.5': 'group3', '1.6': 'group3', '1.7': 'group3', '1.8': 'group3' }
version_pattern = re.compile(r"[\d.]+")

def get_ampl_version():
    """Get the running ampl version

    Returns:
         the AMPL version
    """
    return metadata.version("atomsci-ampl")

def get_ampl_version_from_dir(dirname):
    """Get the AMPL versions for all the models stored under the given directory and its subdirectories,
    recursively.

    Args:
        dirname (str): directory

    Returns:
        list of AMPL versions
    """
    versions = []
    # loop
    for path in Path(dirname).rglob('*.tar.gz'):
        try:
            version = get_ampl_version_from_model(path.absolute())
            versions.append(f'{path.absolute()}, {version}')
        except (json.decoder.JSONDecodeError, FileNotFoundError):
            logger.exception("Failed to get AMPL version from model")
            
    return '\n'.join(versions)

def get_ampl_version_from_model(filename):
    """Get the AMPL version from the tar file's model_metadata.json

    Args:
        filename (str): tar file

    Returns:
        the AMPL version number
    """
    with tarfile.open(filename, mode='r:gz') as tar:
        try:
            meta_info = tar.getmember('./model_metadata.json')
        except KeyError:
            print(f"{filename} is not an AMPL model tarball")
            return None
        with tar.extractfile(meta_info) as meta_fd:
            metadata_dict = json.loads(meta_fd.read())
            version = metadata_dict.get("model_parameters").get("ampl_version", 'probably 1.0.0')
    logger.info(f'{filename}, {version}')
    return version

def get_major_version(full_version):
    return '.'.join(full_version.split('.')[:2])

def get_ampl_version_from_json(metadata_path):
    """Parse model_metadata.json to get the AMPL version

    Args:
        filename (str): tar file

    Returns:
        the AMPL version number

    """
    with open(metadata_path, 'r') as data_file:
        metadata_dict = json.load(data_file)
        version = metadata_dict.get("model_parameters").get("ampl_version", 'probably 1.0.0')
        return version

def validate_version(input):
    valid = re.fullmatch(version_pattern, input)
    if valid is None:
        raise ValueError(f"Input {input} is not valid version format.")
    return True

def check_version_compatible(input_value, ignore_check=False):
    """Check whether an AMPL version string or model file is compatible.

    The input is first validated as an AMPL version string. If validation
    fails, the input is treated as a model file path. The file must exist
    and contain readable AMPL model metadata.

    Args:
        input_value (str or pathlib.Path): AMPL version string or model file.
        ignore_check (bool): If True, return compatibility without raising
            an exception when versions are incompatible.

    Returns:
        bool: True if the versions are compatible, otherwise False when
            ``ignore_check`` is True.

    Raises:
        ValueError: If the input is neither a valid version string nor an
            existing file, if the file cannot be read, or if the versions
            are incompatible and ``ignore_check`` is False.
    """
    try:
        validate_version(str(input_value))
        model_version = str(input_value)

    except (ValueError, TypeError):
        input_path = Path(input_value)

        if not input_path.exists():
            raise ValueError(
                f"Input was neither a valid AMPL version string nor an "
                f"existing file: {input_value!r}"
            )

        if not input_path.is_file():
            raise ValueError(
                f"Input exists but is not a file: {input_value!r}"
            )

        try:
            model_version = get_ampl_version_from_model(input_path)
        except (
            OSError,
            tarfile.TarError,
            json.JSONDecodeError,
            KeyError,
            AttributeError,
            TypeError,
        ) as exc:
            raise ValueError(
                f"Unable to read a valid AMPL version from file: "
                f"{input_value!r}"
            ) from exc

        if not model_version:
            raise ValueError(
                f"File does not contain an AMPL version: {input_value!r}"
            )

    model_ampl_version = get_major_version(str(model_version).strip())
    ampl_version = get_major_version(get_ampl_version())

    match = (
        comp_dict.get(ampl_version, ampl_version)
        == comp_dict.get(model_ampl_version, model_ampl_version)
    )

    if not match and not ignore_check:
        raise ValueError(
            f"AMPL version {model_ampl_version!r} from {input_value!r} "
            f"is not compatible with running AMPL version {ampl_version!r}"
        )

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
        get_ampl_version_from_model(finput)

if __name__ == "__main__":
   main(sys.argv[1:])
