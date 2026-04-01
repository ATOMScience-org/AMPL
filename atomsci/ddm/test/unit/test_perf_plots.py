"""This module contains unit tests for performance plots in the AMPL project"""
from contextlib import contextmanager
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

import pytest
import pandas as pd

from matplotcheck.base import PlotTester
from atomsci.ddm.pipeline.perf_plots import plot_pred_vs_actual_from_df

# --- Fixtures ---
@pytest.fixture
def mock_data():
    """ 
    A pytest fixture that provides mock data for testing.
    
    Returns:
        pd.DataFrame: A DataFrame containing mock data with columns 'avg_pIC50_actual', 
                  'avg_pIC50_pred', and 'avg_pIC50_std'.
    """
    data = {
        'avg_pIC50_actual': [5.0, 6.0, 7.0, 8.0, 9.0],
        'avg_pIC50_pred': [5.1, 5.9, 7.2, 7.8, 9.1],
        'avg_pIC50_std': [0.1, 0.2, 0.1, 0.2, 0.1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_model_pipeline():
    """ 
    A pytest fixture that provides a mock model pipeline object with predefined parameters and data.

    Returns:
        MagicMock: A mock model pipeline object with predefined parameters and data.
    """
    mock_mp = MagicMock()
    mock_mp.params.prediction_type = 'regression'
    mock_mp.params.featurizer = 'ecfp'
    mock_mp.params.split_strategy = 'train_valid_test'
    mock_mp.params.dataset_name = 'test_dataset'
    mock_mp.params.model_type = 'test_model'
    mock_mp.params.splitter = 'test_splitter'
    mock_mp.params.descriptor_type = 'test_descriptor'
    mock_mp.params.dataset_key = "../test_datasets/H1_hybrid.csv"
    mock_mp.data.train_valid_dsets = [(MagicMock(), MagicMock())]
    mock_mp.data.split_uuid = "c63c6d89-8832-4434-b27a-17213bd6ef8f"
    mock_mp.data.test_dset = MagicMock()
    mock_mp.data.dataset = MagicMock()
    mock_mp.model_wrapper.get_perf_data = MagicMock()
    return mock_mp

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
def test_plot_pred_vs_actual_from_df_basic(mock_data, mock_plot):
    """
    Tests basic functionality of the plot_pred_vs_actual_from_df function.
    This test verifies that the plot_pred_vs_actual_from_df function correctly 
    generates a scatter plot with the specified actual and predicted columns 
    from the provided DataFrame. It checks the following:
    - The plot is created and the axis object is not None.
    - The x-axis label contains the string "avg_pIC50_actual".
    - The y-axis label contains the string "avg_pIC50_pred".
    - The plot type is a scatter plot.
    Args:
        mock_data (pd.DataFrame): A mock DataFrame containing the data to be plotted.
        mock_plot (Mock): A mock object for the plot.
    Raises:
        AssertionError: If any of the assertions fail.
    """
    df = mock_data

    with mock_plot() as mock_scatter:
        ax = plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred')
        
        # Verify the plot was created
        assert ax is not None

        # Additional assertions using PlotTester
        pt = PlotTester(ax)
        pt.assert_axis_label_contains("x", "avg_pIC50_actual")
        pt.assert_axis_label_contains("y", "avg_pIC50_pred")
        pt.assert_plot_type("scatter")

def test_plot_pred_vs_actual_from_df_with_std(mock_data, mock_plot):
    """
    Tests plot_pred_vs_actual_from_df with standard deviation.

    This test verifies that the plot_pred_vs_actual_from_df function correctly 
    generates a scatter plot with the specified actual, predicted, and standard 
    deviation columns from the provided DataFrame. It checks the following:
    - The plot is created and the axis object is not None.
    - The x-axis label contains the string "avg_pIC50_actual".
    - The y-axis label contains the string "avg_pIC50_pred".
    - The plot type is a scatter plot.

    Args:
        mock_data (pd.DataFrame): A mock DataFrame containing the data to be plotted.
        mock_plot (Mock): A mock object for the plot.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    df = mock_data
    with mock_plot() as mock_scatter:
        ax = plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred', std_col='avg_pIC50_std')
        
        pt = PlotTester(ax)
        assert ax is not None
        pt.assert_axis_label_contains("x", "avg_pIC50_actual")
        pt.assert_axis_label_contains("y", "avg_pIC50_pred")
        pt.assert_plot_type("scatter")

def test_plot_pred_vs_actual_from_df_with_label(mock_data, mock_plot):
    """
    Tests the plot_pred_vs_actual_from_df function with a label.
    This test verifies that the plot_pred_vs_actual_from_df function correctly plots
    the predicted vs actual values from a DataFrame and includes the specified label.
    Args:
        mock_data (DataFrame): Mock DataFrame containing the data to be plotted.
        mock_plot (Mock): Mock object for the plot.
    Asserts:
        The plot axis is not None.
        The x-axis label contains "avg_pIC50_actual".
        The y-axis label contains "avg_pIC50_pred".
        The plot type is "scatter".
        The plot title is "Test Label".
    """
    df = mock_data
    with mock_plot() as mock_scatter:
        ax = plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred', label='Test Label')
        
        pt = PlotTester(ax)
        assert ax is not None
        pt.assert_axis_label_contains("x", "avg_pIC50_actual")
        pt.assert_axis_label_contains("y", "avg_pIC50_pred")
        pt.assert_plot_type("scatter")
        assert ax.get_title() == 'Test Label'

def test_plot_pred_vs_actual_from_df_with_threshold(mock_data, mock_plot):
    """
    Test the plot_pred_vs_actual_from_df function with a threshold.
    This test checks if the plot generated by the plot_pred_vs_actual_from_df function
    correctly plots the predicted vs actual values from a DataFrame and includes a threshold line.
    Args:
        mock_data (pd.DataFrame): Mock DataFrame containing the data to be plotted.
        mock_plot (Mock): Mock object for the plot.
    Asserts:
        The axis object is not None.
        The x-axis label contains "avg_pIC50_actual".
        The y-axis label contains "avg_pIC50_pred".
        The plot type is a scatter plot.
        There is at least one line in the plot with a dashed linestyle ('--').
    """

    df = mock_data
    with mock_plot() as mock_scatter:
        ax = plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred', threshold=7.0)
        
        pt = PlotTester(ax)
        assert ax is not None
        pt.assert_axis_label_contains("x", "avg_pIC50_actual")
        pt.assert_axis_label_contains("y", "avg_pIC50_pred")
        pt.assert_plot_type("scatter")
        assert any(line.get_linestyle() == '--' for line in ax.get_lines())

def test_plot_pred_vs_actual_from_df_with_all_options(mock_data, mock_plot):
    """
    Test the plot_pred_vs_actual_from_df function with all options.
    This test verifies that the plot_pred_vs_actual_from_df function correctly plots
    the predicted vs actual values from a DataFrame with the specified options.
    Args:
        mock_data (pd.DataFrame): Mock DataFrame containing the test data.
        mock_plot (Mock): Mock object for the plot.
    Asserts:
        - The plot axis is not None.
        - The x-axis label contains "avg_pIC50_actual".
        - The y-axis label contains "avg_pIC50_pred".
        - The plot type is a scatter plot.
        - The plot title is 'Test Label'.
        - At least one line in the plot has a dashed linestyle ('--').
    """
    df = mock_data
    with mock_plot() as mock_scatter:
        ax = plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred', std_col='avg_pIC50_std', label='Test Label', threshold=7.0)
        
        pt = PlotTester(ax)
        assert ax is not None
        pt.assert_axis_label_contains("x", "avg_pIC50_actual")
        pt.assert_axis_label_contains("y", "avg_pIC50_pred")
        pt.assert_plot_type("scatter")
        assert ax.get_title() == 'Test Label'
        assert any(line.get_linestyle() == '--' for line in ax.get_lines())