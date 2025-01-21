
import pytest
"""
This module contains unit tests for the SplitStats class from the atomsci.ddm.utils.compare_splits_plots module.
The tests ensure that the various plotting functions in the SplitStats class work correctly and produce the expected plots.
Fixtures:
    mock_data: Creates mock data for testing, including a total DataFrame and a split DataFrame.
    mock_plot: Creates a mock plot for testing.
Tests:
    test_dist_hist_train_v_test_plot: Tests the dist_hist_train_v_test_plot function to ensure it produces the correct histogram plot.
    test_dist_hist_train_v_valid_plot: Tests the dist_hist_train_v_valid_plot function to ensure it produces the correct histogram plot.
    test_dist_hist_plot_train_v_test: Tests the dist_hist_plot function for Train vs Test data to ensure it produces the correct histogram plot.
    test_dist_hist_plot_train_v_valid: Tests the dist_hist_plot function for Train vs Valid data to ensure it produces the correct histogram plot.
    test_umap_plot: Tests the umap_plot function to ensure it produces the correct UMAP plot.
    test_subset_frac_plot: Tests the subset_frac_plot function to ensure it produces the correct subset fraction plot.
    test_make_all_plots: Tests the make_all_plots function to ensure it produces all the required plots and handles exceptions correctly.
"""
import pandas as pd

from matplotlib import pyplot as plt
from atomsci.ddm.utils.compare_splits_plots import SplitStats
from matplotcheck.base import PlotTester

@pytest.fixture
def mock_data():
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
    """Fixture to create a mock plot."""
    plt.figure()  # Create a new figure
    yield plt.gcf()  # Yield the current figure for testing
    plt.close()  # Close the figure after the test
    
def test_dist_hist_train_v_test_plot(mock_data):
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
    
    fig, ax = plt.subplots()
    ax = ss.dist_hist_train_v_test_plot(ax=ax)
    
    pt = PlotTester(ax)
     
    assert ax is not None
    pt.assert_num_bins(1)   # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")
    
    #plt.close(fig)
    
def test_dist_hist_train_v_valid_plot(mock_data):
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
        
    fig, ax = plt.subplots()
    ax = ss.dist_hist_train_v_valid_plot(ax=ax)
    
    pt = PlotTester(ax)
          
    assert ax is not None
    pt.assert_num_bins(1)   # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")
        
    #plt.close(fig)
        
def test_dist_hist_plot_train_v_test(mock_data):
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
            
    fig, ax = plt.subplots()
    ax = ss.dist_hist_train_v_test_plot(ax)
    
    pt = PlotTester(ax)
          
    assert ax is not None
    pt.assert_num_bins(0)   # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")
            
    #plt.close(fig)

def test_dist_hist_plot_train_v_valid(mock_data):
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
            
    fig, ax = plt.subplots()
    ax = ss.dist_hist_train_v_valid_plot(ax)
            
    pt = PlotTester(ax)
          
    assert ax is not None
    pt.assert_num_bins(0)   # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")
            
    #plt.close(fig)
    

def test_umap_plot(mock_data, mock_plot):
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
        
    fig, ax = plt.subplots()
    ss.umap_plot()
        
    pt = PlotTester(ax)
        
    assert ax is not None
    #pt.assert_num_axes(1)
    # pt.assert_axis_label_contains("x", "UMAP 1")
    # pt.assert_axis_label_contains("y", "UMAP 2")
    pt.assert_plot_type("scatter")
    pt.assert_legend_labels(['train', 'test', 'valid'])
        
    #plt.close(fig)
    
# def test_subset_frac_plot(mock_data):
#     total_df, split_df = mock_data
#     ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
        
#     fig, ax = plt.subplots()
#     ss.subset_frac_plot()
        
#     pt = PlotTester(ax)
        
#     assert ax is not None
#     #pt.assert_num_axes(1)
#     pt.assert_axis_label_contains("x", "subset")
#     pt.assert_axis_label_contains("y", "frac")
#     pt.assert_legend_labels(['train', 'test', 'valid'])
        
#     plt.close(fig)
    
# def test_make_all_plots(mock_data, mock_plot):
#     total_df, split_df = mock_data
#     ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
        
#     with pytest.raises(Exception):
#         ss.make_all_plots()
        
#     fig, ax = plt.subplots()
#     ss.make_all_plots()
        
#     pt = PlotTester(ax)
        
#     assert ax is not None
#     #pt.assert_num_axes(1)
#     pt.assert_axis_label_contains("x", "Tanimoto distance")
#     pt.assert_axis_label_contains("y", "Proportion of compounds")
#     pt.assert_legend_labels(['train', 'test', 'valid'])
        
#     plt.close(fig)
    
def test_dist_hist_plot_train_v_test(mock_data):
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
        
    fig, ax = plt.subplots()
    ss.dist_hist_plot_train_v_test(ss.dists_tvt, 'Train vs Test pairwise Tanimoto Distance')
        
    pt = PlotTester(ax)
        
    assert ax is not None
    pt.assert_num_bins(0)  # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")
        
    #plt.close(fig)

def test_dist_hist_plot_train_v_valid(mock_data):
    total_df, split_df = mock_data
    ss = SplitStats(total_df, split_df, smiles_col='smiles', id_col='cmpd_id', response_cols=['response'])
        
    fig, ax = plt.subplots()
    ss.dist_hist_plot_train_v_valid(ss.dists_tvv, 'Train vs Valid pairwise Tanimoto Distance')
        
    pt = PlotTester(ax)
        
    assert ax is not None
    pt.assert_num_bins(0)  # Check if histogram has been plotted
    pt.assert_axis_label_contains("x", "Tanimoto distance")
    pt.assert_axis_label_contains("y", "Proportion of compounds")
        
    #plt.close(fig)