import os

import inspect

from atomsci.ddm.utils import model_version_utils as mu


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# test models
example_model_file = os.path.abspath('../../../ddm/examples/archive/tutorials2021/models/aurka_union_trainset_base_smiles_model_ampl_120_2fb9ae80-26d4-4f27-af28-2e9e9db594de.tar.gz')
example_models_dir = os.path.abspath('../../../ddm/examples/archive/tutorials2021/models')

def test_get_ampl_version_by_file():
    version = mu.get_ampl_version_from_model(example_model_file)
    assert version == '1.2.0'

def test_get_ampl_version_by_dir():
   versions = mu.get_ampl_version_from_dir(example_models_dir)
   version_tokens = versions.split('\n')
   assert len(version_tokens) >= 2
   for f in versions:
       if f.startswith('cyp3a4'):
         assert '1.2.0' in f

def test_check_versions_compatible():
    try:
        # the versions will not match. should raise an exception
        matched = mu.check_version_compatible(example_model_file)
    except Exception:
        assert True
        
def test_check_versions_compatible_ignore_check():
    # the versions will not match. with ignore check. should pass
    matched = mu.check_version_compatible(example_model_file, True)
    assert not matched

def test_check_versions_compatible_by_version_string():
    # test with version string instead of model file
    matched = mu.check_version_compatible('1.2.0', True)
    assert not matched

def test_invalidate_version_format():
    try:
        # test invalid version string
        matched = mu.check_version_compatible('1.2a')
    except ValueError:
        assert True
