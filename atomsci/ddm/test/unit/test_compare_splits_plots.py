"""
This module contains unit tests for the SplitStats class from the atomsci.ddm.utils.compare_splits_plots module.
The tests ensure that the various plotting functions in the SplitStats class work correctly and produce the expected plots.
"""
import pytest
import pandas as pd

from matplotlib import pyplot as plt
from atomsci.ddm.utils.compare_splits_plots import SplitStats
from matplotcheck.base import PlotTester

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

@pytest.fixture
def mock_plot():
    """
    Fixture to create a mock plot.

    This fixture sets up a new matplotlib figure for testing purposes. It yields the current figure
    object to the test function and ensures that the figure is closed after the test is completed.

    Yields:
        matplotlib.figure.Figure: The current figure object for testing.
    """
    plt.figure()  # Create a new figure
    yield plt.gcf()  # Yield the current figure for testing
    plt.close()  # Close the figure after the test

# --- Test Cases ---    
def test_dist_hist_train_v_test_plot(mock_data):
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
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
    
    fig, ax = plt.subplots()
    ax = ss.dist_hist_train_v_test_plot(ax=ax)
    
    pt = PlotTester(ax)
     
    assert ax is not None
    pt.assert_num_bins(1)   # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")
    
def test_dist_hist_train_v_valid_plot(mock_data):
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
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
        
    fig, ax = plt.subplots()
    ax = ss.dist_hist_train_v_valid_plot(ax=ax)
    
    pt = PlotTester(ax)
          
    assert ax is not None
    pt.assert_num_bins(1)   # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")
        
def test_dist_hist_plot_train_v_test(mock_data):
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
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
            
    fig, ax = plt.subplots()
    ax = ss.dist_hist_train_v_test_plot(ax=ax)
    
    pt = PlotTester(ax)
          
    assert ax is not None
    pt.assert_num_bins(1)   # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")

def test_dist_hist_plot_train_v_valid(mock_data):
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
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
            
    fig, ax = plt.subplots()
    ax = ss.dist_hist_train_v_valid_plot(ax=ax)
            
    pt = PlotTester(ax)
          
    assert ax is not None
    pt.assert_num_bins(1)   # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")