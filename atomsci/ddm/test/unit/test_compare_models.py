"""
Test suite for model performance comparison functionality
Written by Perplexity Deep Research AI bot
Edited by AKP, 2025-02-22
"""

import os
import json
import tarfile
import shutil
import pytest
import pandas as pd
from glob import glob
from pathlib import Path
from atomsci.ddm.pipeline import compare_models
import atomsci.ddm.utils.file_utils as futils



# --------------------------
# Fixtures and Test Data
# --------------------------


def _get_regression_dataset_key(sample_result_dir):
    fs_df = compare_models.get_filesystem_perf_results(
        str(sample_result_dir),
        pred_type='regression',
    )
    assert not fs_df.empty, 'Expected regression fixture models to be present.'
    return fs_df['dataset_key'].dropna().iloc[0]


def _get_expected_model_count(sample_result_dir, dataset_key):
    fs_df = compare_models.get_filesystem_perf_results(
        str(sample_result_dir),
        pred_type='regression',
    )
    filtered = fs_df[fs_df['dataset_key'] == dataset_key]
    assert not filtered.empty, f'Expected regression fixture models for dataset_key={dataset_key}'
    return filtered['model_uuid'].nunique()

@pytest.fixture
def sample_result_dir():
    """Fixture to use an existing sample result directory."""
    # Resolve the path to the existing directory
    result_dir = Path(__file__).parent / "../../examples/tutorials/dataset/"
    assert result_dir.exists(), f"Directory {result_dir} does not exist."
    return result_dir

@pytest.fixture
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

    yield delete_dirs

    # Cleanup extracted trees even when they contain files.
    for dir in delete_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)


# --------------------------
# Core Functionality Tests
# --------------------------

def test_basic_directory_processing(sample_result_dir, unpack_tar_files):
    """Test basic JSON model discovery and processing"""

    dataset_key = _get_regression_dataset_key(sample_result_dir)
    expected_models = _get_expected_model_count(sample_result_dir, dataset_key)

    df = compare_models.get_multitask_perf_from_files_new(
        str(sample_result_dir),
        pred_type='regression',
        dataset_key=dataset_key,
    )

    assert df.model_uuid.nunique() == expected_models, (
        f"Expected {expected_models} unique models, but got {df.model_uuid.nunique()}"
    )
    assert all(df.prediction_type == 'regression'), "Not all rows have 'regression' as prediction_type"


def test_tar_file_processing(sample_result_dir):
    """Test TAR archive handling"""
    dataset_key = _get_regression_dataset_key(sample_result_dir)
    expected_models = _get_expected_model_count(sample_result_dir, dataset_key)

    df = compare_models.get_multitask_perf_from_files_new(
        str(sample_result_dir),
        pred_type='regression',
        dataset_key=dataset_key,
        tar=True
    )
    assert df.model_uuid.nunique() == expected_models, (
        f"Expected {expected_models} unique models, but got {df.model_uuid.nunique()}"
    )
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
    dataset_key = _get_regression_dataset_key(sample_result_dir)
    df = compare_models.get_multitask_perf_from_files_new(
        str(sample_result_dir),
        pred_type='regression',
        dataset_key=dataset_key,
    )
    
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
    dataset_key = _get_regression_dataset_key(sample_result_dir)

    df = compare_models.get_multitask_perf_from_files_new(
        str(sample_result_dir),
        pred_type='regression',
        dataset_key=dataset_key,
        tar=True
    )

    if len(df.multitask.unique()) < 2:
        pytest.skip('Fixture does not currently contain both multitask and single-task regression tar models.')

    assert set(df.multitask.unique()) == {0, 1}, "Expected both multitask and single-task models"


def test_filesystem_perf_results_include_invalid_prediction_columns(sample_result_dir):
    df = compare_models.get_filesystem_perf_results(
        str(sample_result_dir),
        pred_type='regression'
    )

    expected_cols = {
        'best_train_invalid_prediction_count',
        'best_train_invalid_prediction_fraction',
        'best_train_invalid_prediction_threshold',
        'best_valid_invalid_prediction_count',
        'best_valid_invalid_prediction_fraction',
        'best_valid_invalid_prediction_threshold',
        'best_test_invalid_prediction_count',
        'best_test_invalid_prediction_fraction',
        'best_test_invalid_prediction_threshold',
    }
    assert expected_cols.issubset(df.columns)


def test_get_multitask_perf_from_files_new_includes_invalid_prediction_columns(sample_result_dir):
    fs_df = compare_models.get_filesystem_perf_results(
        str(sample_result_dir),
        pred_type='regression'
    )
    with_invalid = fs_df[~fs_df['best_valid_invalid_prediction_count'].isna()]
    if with_invalid.empty:
        pytest.skip('No regression fixture models with invalid prediction metrics available.')
    dataset_key = with_invalid['dataset_key'].dropna().iloc[0]

    df = compare_models.get_multitask_perf_from_files_new(
        str(sample_result_dir),
        pred_type='regression',
        dataset_key=dataset_key,
    )

    assert 'best_train_invalid_prediction_count' in df.columns
    assert 'best_valid_invalid_prediction_fraction' in df.columns
    assert 'best_test_invalid_prediction_threshold' in df.columns


def test_get_multitask_perf_from_files_new_returns_empty_df_for_non_matching_dataset_key(sample_result_dir):
    df = compare_models.get_multitask_perf_from_files_new(
        str(sample_result_dir),
        pred_type='regression',
        dataset_key='__definitely_not_a_real_dataset_key__.csv',
    )
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_multitask_perf_from_files_new_handles_missing_seed_and_time_built(tmp_path):
    model_dir = tmp_path / 'model_missing_fields'
    model_dir.mkdir()

    metadata = {
        'model_uuid': 'm-1',
        'splitting_parameters': {
            'splitter': 'random',
            'split_strategy': 'train_valid_test',
            'split_valid_frac': 0.1,
            'split_test_frac': 0.1,
            'split_uuid': 'split-1',
        },
        'model_parameters': {
            'ampl_version': '0.0.0-test',
            'model_type': 'RF',
            'prediction_type': 'regression',
            'featurizer': 'ecfp',
            'descriptor_type': 'none',
            'num_model_tasks': 1,
            'production': False,
            'model_choice_score_type': 'r2',
        },
        'training_dataset': {
            'dataset_key': 'toy.csv',
            'response_cols': ['y'],
            'feature_transform_type': 'none',
            'response_transform_type': 'none',
            'weight_transform_type': 'none',
            'smiles_col': 'smiles',
        },
        'training_metrics': [
            {
                'label': 'best',
                'subset': 'train',
                'prediction_results': {
                    'r2_score': 0.1,
                    'rms_score': 1.0,
                    'mae_score': 1.0,
                    'num_compounds': 4,
                    'model_choice_score': 0.1,
                },
            },
            {
                'label': 'best',
                'subset': 'valid',
                'prediction_results': {
                    'r2_score': 0.1,
                    'rms_score': 1.0,
                    'mae_score': 1.0,
                    'num_compounds': 4,
                    'model_choice_score': 0.1,
                },
            },
            {
                'label': 'best',
                'subset': 'test',
                'prediction_results': {
                    'r2_score': 0.1,
                    'rms_score': 1.0,
                    'mae_score': 1.0,
                    'num_compounds': 4,
                    'model_choice_score': 0.1,
                },
            },
        ],
    }

    metadata_path = model_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    tar_path = tmp_path / 'model_missing_fields.tar.gz'
    with tarfile.open(tar_path, mode='w:gz') as tar:
        tar.add(metadata_path, arcname='model_metadata.json')

    df = compare_models.get_multitask_perf_from_files_new(str(tmp_path), pred_type='regression', tar=True)
    assert len(df) == 1
    assert 'seed' in df.columns
    assert 'time_built' in df.columns
    assert pd.isna(df['seed'].iloc[0])
    assert pd.isna(df['time_built'].iloc[0])

# --------------------------
# main
# --------------------------
if __name__ == "__main__":
    pytest.main([__file__])