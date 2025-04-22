"""
This module contains unit tests for the SplitStats class from the atomsci.ddm.utils.compare_splits_plots module.
The tests ensure that the various plotting functions in the SplitStats class work correctly and produce the expected plots.
"""
import pytest
import pandas as pd

from matplotlib import pyplot as plt
from atomsci.ddm.utils.compare_splits_plots import SplitStats, split, parse_args
from matplotcheck.base import PlotTester
from contextlib import contextmanager

# --- Fixtures ---
@pytest.fixture
def mock_data():
    """
    Generate mock data for testing purposes.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - total_df: DataFrame with columns 'cmpd_id', 'smiles', and 'response'.
            - split_df: DataFrame with columns 'cmpd_id' and 'subset'.
    """
    total_data = {
        'cmpd_id': ['cmpd1', 'cmpd2', 'cmpd3', 'cmpd4', 'cmpd5'],
        'smiles': ['CCO', 'CCN', 'CCC', 'CCCl', 'CCBr'],
        'response': [1, 0, 1, 0, 1]
    }
    split_data = {
        'cmpd_id': ['cmpd1', 'cmpd2', 'cmpd3', 'cmpd4', 'cmpd5'],
        'subset': ['train', 'test', 'train', 'valid', 'test']
    }
    total_df = pd.DataFrame(total_data)
    split_df = pd.DataFrame(split_data)
    return total_df, split_df

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
def test_dist_hist_train_v_test_plot(mock_plot, mock_data):
    """
    Test the `dist_hist_train_v_test_plot` method of the `SplitStats` class.
    This test verifies that the histogram plot comparing the distribution of 
    Tanimoto distances between training and test sets is correctly generated.
    Args:
        mock_data (tuple): A tuple containing two DataFrames, `total_df` and 
                           `split_df`, which represent the complete dataset 
                           and the split dataset respectively.
    Asserts:
        - The plot axis (`ax`) is not None.
        - The histogram has been plotted with at least one bin.
        - The x-axis label contains the text "Tanimoto distance".
        - The y-axis label contains the text "Proportion of compounds".
    """
    with mock_plot() as fig:
        ax = fig.subplots()
        total_df, split_df = mock_data
        ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
    
        ax = ss.dist_hist_train_v_test_plot(ax=ax)
    
        pt = PlotTester(ax)
     
        assert ax is not None
        pt.assert_num_bins(1)   # Check if histogram has been plotted
        pt.assert_axis_label_contains("x", "Tanimoto distance")
        pt.assert_axis_label_contains("y", "Proportion of compounds")
    
def test_dist_hist_train_v_valid_plot(mock_plot, mock_data):
    """
    Test the `dist_hist_train_v_valid_plot` method of the `SplitStats` class.
    This test verifies that the `dist_hist_train_v_valid_plot` method correctly
    generates a histogram plot comparing the distribution of Tanimoto distances
    between training and validation sets.
    Args:
        mock_data (tuple): A tuple containing two DataFrames:
            - total_df: The complete dataset DataFrame.
            - split_df: The DataFrame containing the split information.
    Asserts:
        - The axis object `ax` is not None.
        - The histogram has been plotted with the correct number of bins.
        - The x-axis label contains the text "Tanimoto distance".
        - The y-axis label contains the text "Proportion of compounds".
    """
    with mock_plot() as fig:
        ax = fig.subplots()
        total_df, split_df = mock_data
        ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
        
        ax = ss.dist_hist_train_v_valid_plot(ax=ax)
    
        pt = PlotTester(ax)
          
        assert ax is not None
        pt.assert_num_bins(1)   # Check if histogram has been plotted
        pt.assert_axis_label_contains("x", "Tanimoto distance")
        pt.assert_axis_label_contains("y", "Proportion of compounds")
        
def test_dist_hist_plot_train_v_test(mock_plot, mock_data):
    """
    Test the distribution histogram plot for training vs. testing data.
    This test function takes mock data, creates a SplitStats object, and generates
    a histogram plot comparing the training and testing data distributions. It then
    uses a PlotTester object to verify the plot's properties.
    Args:
        mock_data (tuple): A tuple containing the total dataframe and the split dataframe.
    Asserts:
        Asserts that the axis object is not None.
        Asserts that the histogram has been plotted by checking the number of bins.
        Asserts that the x-axis label contains "Tanimoto distance".
        Asserts that the y-axis label contains "Proportion of compounds".
    """
    with mock_plot() as fig:
        ax = fig.subplots()
        total_df, split_df = mock_data
        ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
            
        ax = ss.dist_hist_train_v_test_plot(ax=ax)
    
        pt = PlotTester(ax)
          
        assert ax is not None
        pt.assert_num_bins(1)   # Check if histogram has been plotted
        pt.assert_axis_label_contains("x", "Tanimoto distance")
        pt.assert_axis_label_contains("y", "Proportion of compounds")

def test_dist_hist_plot_train_v_valid(mock_plot, mock_data):
    """
    Test the distribution histogram plot for training vs validation data.
    This test function verifies the following:
    1. The histogram plot is created without errors.
    2. The number of bins in the histogram is as expected.
    3. The x-axis label contains "Tanimoto distance".
    4. The y-axis label contains "Proportion of compounds".
    Args:
        mock_data (tuple): A tuple containing the total dataframe and the split dataframe.
    Raises:
        AssertionError: If any of the assertions fail.
    """
    with mock_plot() as fig:
        ax = fig.subplots()
        total_df, split_df = mock_data
        ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
            
        ax = ss.dist_hist_train_v_valid_plot(ax=ax)
            
        pt = PlotTester(ax)
          
        assert ax is not None
        pt.assert_num_bins(1)   # Check if histogram has been plotted
        pt.assert_axis_label_contains("x", "Tanimoto distance")
        pt.assert_axis_label_contains("y", "Proportion of compounds")
    
def test_split_function(mock_data):
    """
    Test the `split` function to ensure it correctly splits the dataset into training, test, and validation sets.
    Args:
        mock_data (tuple): A tuple containing the total dataframe and the split dataframe.
    Asserts:
        - The length of the training, test, and validation dataframes are as expected.
        - The 'cmpd_id' values in each subset are correct.
    """
    total_df, split_df = mock_data
    train_df, test_df, valid_df = split(total_df, split_df, id_col='cmpd_id')
        
    assert len(train_df) == 2
    assert len(test_df) == 2
    assert len(valid_df) == 1
        
    assert set(train_df['cmpd_id']) == {'cmpd1', 'cmpd3'}
    assert set(test_df['cmpd_id']) == {'cmpd2', 'cmpd5'}
    assert set(valid_df['cmpd_id']) == {'cmpd4'}

def test_split_stats_initialization(mock_data):
    """
    Test the initialization of the `SplitStats` class to ensure it correctly processes the input data.
    Args:
        mock_data (tuple): A tuple containing the total dataframe and the split dataframe.
    Asserts:
        - The attributes of the `SplitStats` object are correctly initialized.
    """
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
        
    assert ss.smiles_col == 'smiles'
    assert ss.id_col == 'cmpd_id'
    assert ss.response_cols == ['response']
    assert ss.total_df.equals(total_df)
    assert ss.split_df.equals(split_df)
    assert len(ss.train_df) == 2
    assert len(ss.test_df) == 2
    assert len(ss.valid_df) == 1

def test_print_stats(mock_data, capsys):
    """
    Test the `print_stats` method to ensure it correctly prints the statistics.
    Args:
        mock_data (tuple): A tuple containing the total dataframe and the split dataframe.
        capsys: Pytest fixture to capture stdout and stderr.
    Asserts:
        - The printed output contains the expected statistics.
    """
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
    ss.print_stats()
        
    captured = capsys.readouterr()
    assert "dist tvt mean" in captured.out
    assert "dist tvv mean" in captured.out
    assert "train frac mean" in captured.out
    assert "test frac mean" in captured.out
    assert "valid frac mean" in captured.out

def test_split(mock_data):
    """
    Test the `split` function to ensure it correctly splits the dataset into training, test, and validation sets.
        
    Args:
        mock_data (tuple): A tuple containing two DataFrames, `total_df` and `split_df`, which represent the complete dataset 
                        and the split dataset respectively.
    Asserts:
        - The training set contains the correct compounds.
        - The test set contains the correct compounds.
        - The validation set contains the correct compounds.
    """
    total_df, split_df = mock_data
    train_df, test_df, valid_df = split(total_df, split_df, id_col='cmpd_id')
        
    # Check training set
    assert set(train_df['cmpd_id']) == {'cmpd1', 'cmpd3'}
    # Check test set
    assert set(test_df['cmpd_id']) == {'cmpd2', 'cmpd5'}
    # Check validation set
    assert set(valid_df['cmpd_id']) == {'cmpd4'}
        
    # Check that the splits are mutually exclusive and collectively exhaustive
    all_ids = set(total_df['cmpd_id'])
    split_ids = set(train_df['cmpd_id']).union(set(test_df['cmpd_id'])).union(set(valid_df['cmpd_id']))
    assert all_ids == split_ids

def test_parse_args(mocker):
    """
    Test the `parse_args` function to ensure it correctly parses command-line arguments.
    Args:
        mocker: Pytest mocker fixture to mock command-line arguments.
    Asserts:
        - The parsed arguments match the expected values.
    """
    mocker.patch('sys.argv', [
        'compare_splits_plots.py', 'dataset.csv', 'id', 'smiles', 'split_a.csv', 'split_b.csv', 'output_dir'
    ])
    args = parse_args()
        
    assert args.csv == 'dataset.csv'
    assert args.id_col == 'id'
    assert args.smiles_col == 'smiles'
    assert args.split_a == 'split_a.csv'
    assert args.split_b == 'split_b.csv'
    assert args.output_dir == 'output_dir'