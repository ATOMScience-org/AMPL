import os

import pytest
import inspect

from atomsci.ddm.utils.model_file_reader import ModelFileReader
from atomsci.ddm.utils import model_file_reader as mfr


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

model_path = '../../examples/BSEP/models/bsep_classif_scaffold_split.tar.gz'
tar_model = ModelFileReader(model_path)

def test_model_split_uuid():
    split_uuid = tar_model.get_split_uuid()

    assert split_uuid == '162b11b7-da6a-49bd-b85e-2971a0b0a949'

def test_model_uuid():
    model_uuid = tar_model.get_model_uuid()

    assert model_uuid == 'f12a02d3-9238-48b4-883d-3f3775d227a2'

def test_model_type():
    model_type = tar_model.get_model_type()

    assert model_type == 'NN'

def test_no_medata_json_in_dir():
    with pytest.raises(Exception) as e:
        ModelFileReader('..') # should raise error
    assert e.type == IOError

def test_multiple_models_metadata():
    data_list =  mfr.get_multiple_models_metadata('../../examples/BSEP/models/bsep_classif_random_split.tar.gz', '../../examples/BSEP/models/bsep_classif_scaffold_split_graphconv.tar.gz', '../..//examples/BSEP/models/bsep_classif_scaffold_split.tar.gz')
    # should be parsed fine 
    assert len(data_list) == 3
    
def test_incorrect_model_file():
     with pytest.raises(Exception) as e:
         data_list =  mfr.get_multiple_models_metadata('../../examples/BSEP/models/bsep_classif_random_split.tar')
     assert e.type == IOError