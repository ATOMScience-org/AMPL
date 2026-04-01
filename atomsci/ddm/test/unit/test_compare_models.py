"""
Test suite for model performance comparison functionality
Written by Perplexity Deep Research AI bot
Edited by AKP, 2025-02-22
"""

import os
import json
import tarfile
import pytest
import pandas as pd
from glob import glob
from pathlib import Path
from atomsci.ddm.pipeline import compare_models
import atomsci.ddm.utils.file_utils as futils



# --------------------------
# Fixtures and Test Data
# --------------------------

@pytest.fixture
def sample_result_dir():
    """Fixture to use an existing sample result directory."""
    # Resolve the path to the existing directory
    result_dir = Path(__file__).parent / "../../examples/tutorials/dataset/"
    assert result_dir.exists(), f"Directory {result_dir} does not exist."
    return result_dir

# @pytest.fixture
def unpack_tar_files(sample_result_dir):
    """Fixture to create a model directories with unpacked tar files"""

    # get .tar.gz files
    tar_list=glob(f'{sample_result_dir}/**/*.tar.gz', recursive=True)

    # unpack tar files into model directories
    delete_dirs = []

    for tar_file in tar_list:

        # create a directory for the extracted files
        extract_location = tar_file.replace('.tar.gz', '')
        os.makedirs(extract_location, exist_ok=True)
        delete_dirs.append(extract_location)

        # extract the tar file
        with tarfile.open(tar_file, mode='r:gz') as tar:
            futils.safe_extract(tar, path=extract_location)

    return delete_dirs

# @pytest.fixture
def delete_model_dirs(delete_dirs):
    """Fixture to clean up model directories after tests"""
    yield
    # Cleanup
    for dir in delete_dirs:
        if os.path.exists(dir):
            os.rmdir(dir)


# --------------------------
# Core Functionality Tests
# --------------------------

def test_basic_directory_processing(sample_result_dir):
    """Test basic JSON model discovery and processing"""

    delete_dirs=unpack_tar_files(sample_result_dir)

    df = compare_models.get_multitask_perf_from_files_new(
        str(sample_result_dir),
        pred_type='regression'
    )

    delete_model_dirs(delete_dirs)

    assert df.model_uuid.nunique() == 5, f"Expected 5 unique models, but got {df.model_uuid.nunique()}"
    assert all(df.prediction_type == 'regression'), "Not all rows have 'regression' as prediction_type"


def test_tar_file_processing(sample_result_dir):
    """Test TAR archive handling"""
    df = compare_models.get_multitask_perf_from_files_new(
        str(sample_result_dir),
        tar=True
    )
    assert df.model_uuid.nunique() == 5, f"Expected 5 unique models, but got {df.model_uuid.nunique()}"
    assert all(df.model_path.str.endswith('.tar.gz')), "Not all model paths end with '.tar.gz'"
    

# --------------------------
# Error Condition Tests
# --------------------------

# def test_invalid_json_handling(tmp_path):
#     """Test corrupted JSON file handling"""
#     bad_dir = tmp_path / "bad_model"
#     bad_dir.mkdir()
#     (bad_dir / "model_metadata.json").write_text("{invalid_json}")
    
#     with pytest.raises(json.JSONDecodeError):
#         compare_models.get_multitask_perf_from_files_new(str(tmp_path))

# def test_empty_directory(tmp_path):
#     """Test empty input directory handling"""
#     df = compare_models.get_multitask_perf_from_files_new(str(tmp_path))
#     assert df.empty

# --------------------------
# DataFrame Integrity Tests
# --------------------------

def test_dataframe_structure(sample_result_dir):
    """Validate DataFrame schema and data types"""
    df = compare_models.get_multitask_perf_from_files_new(str(sample_result_dir))
    
    # Required columns
    assert {'model_uuid', 'time_built', 'ampl_version','dataset_key', 'model_path',
           'model_type','seed','prediction_type', 'splitter',
           'split_strategy', 'split_valid_frac', 'split_test_frac', 'split_uuid', 
            'production', 'feature_transform_type','response_transform_type', 'weight_transform_type',
           'smiles_col', 'features','model_choice_score_type',}.issubset(df.columns)
    
    # Type validation - columns that are merged end up as objects??
    assert pd.api.types.is_string_dtype(df.model_uuid), f"Expected model_uuid to be string, but got {df.model_uuid.dtype}"
    assert pd.api.types.is_float_dtype(df.time_built), f"Expected time_built to be float, but got {df.time_built.dtype}"
    assert pd.api.types.is_string_dtype(df.ampl_version), f"Expected ampl_version to be string, but got {df.ampl_version.dtype}"
    assert pd.api.types.is_string_dtype(df.dataset_key), f"Expected dataset_key to be string, but got {df.dataset_key.dtype}"
    assert pd.api.types.is_string_dtype(df.model_path), f"Expected model_path to be string, but got {df.model_path.dtype}"
    assert pd.api.types.is_string_dtype(df.model_type), f"Expected model_type to be string, but got {df.model_type.dtype}"
    assert pd.api.types.is_numeric_dtype(df.seed),  f"Expected seed to be numeric, but got {df.seed.dtype}"
    assert pd.api.types.is_string_dtype(df.prediction_type), f"Expected prediction_type to be string, but got {df.prediction_type.dtype}"
    assert pd.api.types.is_string_dtype(df.splitter), f"Expected splitter to be string, but got {df.splitter.dtype}"
    assert pd.api.types.is_string_dtype(df.split_strategy), f"Expected split_strategy to be string, but got {df.split_strategy.dtype}"
    # assert pd.api.types.is_float_dtype(df.split_valid_frac), f"Expected split_valid_frac to be float, but got {df.split_valid_frac.dtype}"
    # assert pd.api.types.is_float_dtype(df.split_test_frac), f"Expected split_test_frac to be float, but got {df.split_test_frac.dtype}"
    assert pd.api.types.is_string_dtype(df.split_uuid), f"Expected split_uuid to be string, but got {df.split_uuid.dtype}"
    # assert pd.api.types.is_bool_dtype(df.production),   f"Expected production to be boolean, but got {df.production.dtype}"
    # assert pd.api.types.is_string_dtype(df.feature_transform_type), f"Expected feature_transform_type to be string, but got {df.feature_transform_type.dtype}"
    # assert pd.api.types.is_string_dtype(df.response_transform_type), f"Expected response_transform_type to be string, but got {df.response_transform_type.dtype}"
    # assert pd.api.types.is_string_dtype(df.weight_transform_type), f"Expected weight_transform_type to be string, but got {df.weight_transform_type.dtype}"
    # assert pd.api.types.is_string_dtype(df.smiles_col), f"Expected smiles_col to be string, but got {df.smiles_col.dtype}"
    # assert pd.api.types.is_string_dtype(df.features), f"Expected features to be string, but got {df.features.dtype}"
    # assert pd.api.types.is_string_dtype(df.model_choice_score_type), f"Expected model_choice_score_type to be string, but got {df.model_choice_score_type.dtype}"

# --------------------------
# Special Case Tests
# --------------------------

def test_mixed_model_types(sample_result_dir):
    """Test handling directories with both MT and ST model files"""
    
    df = compare_models.get_multitask_perf_from_files_new(
        str(sample_result_dir),
        tar=True
    )
    
    assert len(df.multitask.unique()) > 1, "Expected multiple multitask types"
    assert df.multitask.max() == 1, "Expected at least one multitask model"

# --------------------------
# main
# --------------------------
if __name__ == "__main__":
    pytest.main([__file__])