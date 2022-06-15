import argparse
import json
import logging
import os
import shutil
import sys

import pytest
import inspect
import warnings
import atomsci.ddm.utils.model_version_utils as mu

from pathlib import Path
import pdb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# test models
example_model_file = os.path.abspath('../../../ddm/examples/tutorials/models/aurka_union_trainset_base_smiles_model_2fb9ae80-26d4-4f27-af28-2e9e9db594de.tar.gz')
example_models_dir = os.path.abspath('../../../ddm/examples/tutorials/models')

def test_get_ampl_version_by_file():
    version = mu.get_ampl_version(example_model_file)
    assert version == '1.2.0'

def test_get_ampl_version_by_dir():
   versions = mu.get_ampl_version_from_dir(example_models_dir)
   version_tokens = versions.split('\n')
   assert len(version_tokens) >= 2
   result_tokens = version_tokens[0].split(',')
   assert result_tokens[1].strip() == '1.2.0'

def test_check_versions_compatible():
    matched = mu.check_version_compatible(example_model_file)
    assert not matched
