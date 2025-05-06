"""
This module contains unit tests for the functions in the 
atomsci.ddm.utils.split_response_dist_plots module.

The tests cover the following functionalities:
- Generating labeled datasets based on split information.
- Plotting response distributions for different subsets of data.
- Computing Wasserstein distances between response distributions of different subsets.

"""
from contextlib import contextmanager
import pytest
import pandas as pd

from atomsci.ddm.utils.split_response_dist_plots import plot_split_subset_response_distrs, compute_split_subset_wasserstein_distances, get_split_labeled_dataset
from matplotlib import pyplot as plt

# --- Fixtures ---
@pytest.fixture
def params():
    """Fixture to provide parameters for the tests.

    Returns:
        dict: A dictionary containing parameters for the tests.
    """
    return {
        'dataset_key': 'test_dataset.csv',
        'split_uuid': '1234',
        'split_strategy': 'train_valid_test',
        'splitter': 'random',
        'split_valid_frac': 0.2,
        'split_test_frac': 0.2,
        'num_folds': 5,
        'smiles_col': 'smiles',
        'response_cols': ['response1', 'response2'],
        'prediction_type': 'regression'
    }

@pytest.fixture
def dataset():
    """Fixture to provide a sample dataset for the tests.

    Returns:
        pd.DataFrame: A DataFrame containing sample data.
    """
    data = {
        'compound_id': ['cmpd1', 'cmpd2', 'cmpd3', 'cmpd4', 'cmpd5'],
        'smiles': ['C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
        'response1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'response2': [5.0, 4.0, 3.0, 2.0, 1.0],
        'split_subset': ['train', 'valid', 'test', 'train', 'valid']
    }
    return pd.DataFrame(data)

@pytest.fixture
def split_file():
    """Fixture to provide a sample split file for the tests.

    Returns:
        pd.DataFrame: A DataFrame containing sample split data.
    """
    data = {
        'compound_id': ['cmpd1', 'cmpd2', 'cmpd3', 'cmpd4', 'cmpd5'],
        'subset': ['train', 'valid', 'test', 'train', 'valid']
    }
    return pd.DataFrame(data)

# Create a context manager for the figure
@contextmanager
def mock_plot_context():
    """
    Provides a mock context for matplotlib plotting during testing.

    This function creates a new matplotlib figure, yields it for testing purposes,
    and ensures that the figure is properly closed after the test is complete.

    Yields:
        matplotlib.figure.Figure: The current matplotlib figure object.
    """
    plt.figure()  # Create a new figure
    try:
        yield plt.gcf()  # Yield the current figure for testing
    finally:
        plt.close()  # Close the figure after the test

@pytest.fixture
def mock_plot():
    return mock_plot_context  # Return the context manager
   
# --- Test Cases ---     
def test_get_split_labeled_dataset(params, dataset, split_file, mocker):
    """Test the get_split_labeled_dataset function.

    Args:
        params (dict): Parameters for the test.
        dataset (pd.DataFrame): Sample dataset.
        split_file (pd.DataFrame): Sample split file.
        mocker (pytest_mock.MockerFixture): Mocking fixture.
    """
    mocker.patch('pandas.read_csv', side_effect=[dataset, split_file])
    dset_df, split_label = get_split_labeled_dataset(params)
    assert 'split_subset' in dset_df.columns
    assert split_label == 'random split'

# def test_plot_split_subset_response_distrs(params, dataset, split_file, mocker):
#     """Test the plot_split_subset_response_distrs function.

#     Args:
#         params (dict): Parameters for the test.
#         dataset (pd.DataFrame): Sample dataset.
#         split_file (pd.DataFrame): Sample split file.
#         mocker (pytest_mock.MockerFixture): Mocking fixture.
#     """
#     mocker.patch('pandas.read_csv', side_effect=[dataset, split_file])
#     fig, ax = plt.subplots(1, len(params['response_cols']))
#     plot_split_subset_response_distrs(params, axes=ax)
#     for axis in ax:
#         plot_tester = PlotTester(axis)
#         if params['prediction_type'] == 'regression':
#             plot_tester.assert_title_contains('distribution by subset under')
        
# def test_plot_split_subset_response_distrs_regression(mock_plot, params, dataset, split_file, mocker):
#     """Test the plot_split_subset_response_distrs function for regression.

#     Args:
#         mock_plot (matplotlib.figure.Figure): Mock plot fixture.
#         params (dict): Parameters for the test.
#         dataset (pd.DataFrame): Sample dataset.
#         split_file (pd.DataFrame): Sample split file.
#         mocker (pytest_mock.MockerFixture): Mocking fixture.
#     """
#     with mock_plot() as fig:
#         ax = fig.subplots(1, len(params['response_cols']))
#         mocker.patch('pandas.read_csv', side_effect=[dataset, split_file])
        
#         plot_split_subset_response_distrs(params, axes=ax)
#         for colnum, col in enumerate(params['response_cols']):
#             plot_tester = PlotTester(ax[colnum])
#             if params['prediction_type'] == 'regression':
#                 plot_tester.assert_title_contains(f"{col} distribution by subset under")

# def test_plot_split_subset_response_distrs_classification(mock_plot, params, dataset, split_file, mocker):
#     """Test the plot_split_subset_response_distrs function for classification.

#     Args:
#         mock_plot (matplotlib.figure.Figure): Mock plot fixture.
#         params (dict): Parameters for the test.
#         dataset (pd.DataFrame): Sample dataset.
#         split_file (pd.DataFrame): Sample split file.
#         mocker (pytest_mock.MockerFixture): Mocking fixture.
#     """
#     params['prediction_type'] = 'classification'
#     mocker.patch('pandas.read_csv', side_effect=[dataset, split_file])
#     with mock_plot() as fig:
#         ax = fig.subplots(1, len(params['response_cols']))
#         plot_split_subset_response_distrs(params, axes=ax)
#         for colnum, col in enumerate(params['response_cols']):
#             plot_tester = PlotTester(ax[colnum])
#             if params['prediction_type'] != 'regression':
#                 plot_tester.assert_title_contains(f"Percent of {col} = 1 by subset under")
            
def test_compute_split_subset_wasserstein_distances(params, dataset, split_file, mocker):
    """Test the compute_split_subset_wasserstein_distances function.

    Args:
        params (dict): Parameters for the test.
        dataset (pd.DataFrame): Sample dataset.
        split_file (pd.DataFrame): Sample split file.
        mocker (pytest_mock.MockerFixture): Mocking fixture.
    """
    mocker.patch('pandas.read_csv', side_effect=[dataset, split_file])
    dist_df = compute_split_subset_wasserstein_distances(params)
    assert not dist_df.empty
    assert 'distance' in dist_df.columns