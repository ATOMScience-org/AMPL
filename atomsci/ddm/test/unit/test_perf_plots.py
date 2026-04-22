"""This module contains unit tests for performance plots in the AMPL project"""
from contextlib import contextmanager
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
import tarfile
import json

import pytest
import pandas as pd

from matplotcheck.base import PlotTester
import atomsci.ddm.pipeline.perf_plots as perf_plots
import io
import numpy as np

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
        ax = perf_plots.plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred')
        
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
        ax = perf_plots.plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred', std_col='avg_pIC50_std')
        
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
        ax = perf_plots.plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred', label='Test Label')
        
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
        ax = perf_plots.plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred', threshold=7.0)
        
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
        ax = perf_plots.plot_pred_vs_actual_from_df(df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred', std_col='avg_pIC50_std', label='Test Label', threshold=7.0)
        
        pt = PlotTester(ax)
        assert ax is not None
        pt.assert_axis_label_contains("x", "avg_pIC50_actual")
        pt.assert_axis_label_contains("y", "avg_pIC50_pred")
        pt.assert_plot_type("scatter")
        assert ax.get_title() == 'Test Label'
        assert any(line.get_linestyle() == '--' for line in ax.get_lines())


def _make_model_tarball(tmp_path, metadata: dict, name="model.tar.gz"):
    """Create a minimal tar.gz containing model_metadata.json."""
    model_path = tmp_path / name
    meta_bytes = json.dumps(metadata).encode("utf-8")

    with tarfile.open(model_path, mode="w:gz") as tar:
        ti = tarfile.TarInfo(name="model_metadata.json")
        ti.size = len(meta_bytes)
        tar.addfile(ti, io.BytesIO(meta_bytes))
    return str(model_path)


def _write_csv(path, df: pd.DataFrame):
    df.to_csv(path, index=False)
    return str(path)


# -------------------------------
# Tests: merge_response_cols_from_original
# -------------------------------

def test_merge_response_cols_happy_path_adds_missing_response_cols():
    feat_df = pd.DataFrame(
        {
            "cmpd_id": ["1", "2", "3"],
            "feat1": [0.1, 0.2, 0.3],
        }
    )
    orig_df = pd.DataFrame(
        {
            "cmpd_id": ["1", "2", "3"],
            "y": [10.0, 20.0, 30.0],
        }
    )

    out = perf_plots.merge_response_cols_from_original(
        feat_df, orig_df, id_col="cmpd_id", response_cols=["y"]
    )

    assert "y" in out.columns
    assert out.loc[out.cmpd_id == "2", "y"].iloc[0] == 20.0


def test_merge_response_cols_does_not_overwrite_existing_columns():
    feat_df = pd.DataFrame(
        {
            "cmpd_id": ["1", "2"],
            "y": [999.0, 999.0],  # pre-existing, should not be overwritten
        }
    )
    orig_df = pd.DataFrame({"cmpd_id": ["1", "2"], "y": [1.0, 2.0]})

    out = perf_plots.merge_response_cols_from_original(
        feat_df, orig_df, id_col="cmpd_id", response_cols=["y"]
    )

    assert out["y"].tolist() == [999.0, 999.0]


def test_merge_response_cols_raises_when_id_col_missing_in_feat():
    feat_df = pd.DataFrame({"not_id": [1, 2], "feat": [0.1, 0.2]})
    orig_df = pd.DataFrame({"cmpd_id": [1, 2], "y": [1.0, 2.0]})

    with pytest.raises(KeyError):
        perf_plots.merge_response_cols_from_original(
            feat_df, orig_df, id_col="cmpd_id", response_cols=["y"]
        )


def test_merge_response_cols_missing_frac_exceeds_threshold_raises():
    # orig has 4 unique IDs, feat only contains 1 => 75% missing
    feat_df = pd.DataFrame({"cmpd_id": ["1"], "feat": [0.1]})
    orig_df = pd.DataFrame({"cmpd_id": ["1", "2", "3", "4"], "y": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="exceeds allowed"):
        perf_plots.merge_response_cols_from_original(
            feat_df,
            orig_df,
            id_col="cmpd_id",
            response_cols=["y"],
            max_missing_frac=0.10,
            sample_n=2,
        )


def test_merge_response_cols_extra_ids_warns_by_default():
    feat_df = pd.DataFrame({"cmpd_id": ["1", "X"], "feat": [0.1, 0.9]})
    orig_df = pd.DataFrame({"cmpd_id": ["1"], "y": [1.0]})

    # Keep missing_frac low: orig_set - feat_set = {} so missing_frac=0
    with pytest.warns(UserWarning, match="contains 1 unique compound IDs not found"):
        perf_plots.merge_response_cols_from_original(
            feat_df,
            orig_df,
            id_col="cmpd_id",
            response_cols=["y"],
            max_missing_frac=0.50,
            error_on_extra_feat_ids=False,
        )


def test_merge_response_cols_extra_ids_raises_if_configured():
    feat_df = pd.DataFrame({"cmpd_id": ["1", "X"], "feat": [0.1, 0.9]})
    orig_df = pd.DataFrame({"cmpd_id": ["1"], "y": [1.0]})

    with pytest.raises(ValueError, match="not found in original"):
        perf_plots.merge_response_cols_from_original(
            feat_df,
            orig_df,
            id_col="cmpd_id",
            response_cols=["y"],
            max_missing_frac=0.50,
            error_on_extra_feat_ids=True,
        )


def test_merge_response_cols_coerces_id_to_str_by_default():
    feat_df = pd.DataFrame({"cmpd_id": [1, 2], "feat": [0.1, 0.2]})
    orig_df = pd.DataFrame({"cmpd_id": ["1", "2"], "y": [10.0, 20.0]})

    out = perf_plots.merge_response_cols_from_original(
        feat_df, orig_df, id_col="cmpd_id", response_cols=["y"], coerce_id_to_str=True
    )
    assert out["y"].tolist() == [10.0, 20.0]


def test_merge_response_cols_returns_feat_unchanged_if_no_responses_available():
    feat_df = pd.DataFrame({"cmpd_id": ["1"], "feat": [0.1]})
    orig_df = pd.DataFrame({"cmpd_id": ["1"], "other": [9]})

    out = perf_plots.merge_response_cols_from_original(
        feat_df, orig_df, id_col="cmpd_id", response_cols=["y"]
    )
    assert list(out.columns) == ["cmpd_id", "feat"]


# -------------------------------
# Tests: plot_pred_vs_actual_from_file
# -------------------------------

def test_plot_pred_vs_actual_from_file_raises_for_classification(tmp_path):
    # Minimal config for classification should raise early
    dataset_csv = tmp_path / "data.csv"
    _write_csv(dataset_csv, pd.DataFrame({"cmpd_id": ["1"], "smiles": ["C"], "y": [1]}))

    cfg = {
        "model_parameters": {"prediction_type": "classification", "model_type": "rf", "featurizer": "ecfp"},
        "training_dataset": {"dataset_key": str(dataset_csv), "response_cols": ["y"], "id_col": "cmpd_id", "smiles_col": "smiles"},
        "splitting_parameters": {"split_uuid": "uuid", "splitter": "random", "split_strategy": "train_valid_test"},
        "training_metrics": [],
    }
    model_path = _make_model_tarball(tmp_path, cfg)

    with pytest.raises(ValueError, match="only be called for regression models"):
        perf_plots.plot_pred_vs_actual_from_file(model_path)


def test_plot_pred_vs_actual_from_file_builds_figure_and_r2_titles(tmp_path, monkeypatch):
    # Prepare dataset and split file
    dataset_csv = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "cmpd_id": ["1", "2", "3", "4", "5", "6"],
            "smiles": ["C", "CC", "CCC", "CCCC", "CCO", "CO"],
            "y": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    _write_csv(dataset_csv, df)

    split_uuid = "split-uuid-123"
    split_csv = tmp_path / f"anything_{split_uuid}_split.csv"
    split_df = pd.DataFrame(
        {
            "cmpd_id": ["1", "2", "3", "4", "5", "6"],
            "subset": ["train", "train", "valid", "valid", "test", "test"],
        }
    )
    _write_csv(split_csv, split_df)

    cfg = {
        "model_parameters": {
            "prediction_type": "regression",
            "model_type": "rf",
            "featurizer": "ecfp",
            # keep uncertainty absent so error_bars becomes False even if requested
        },
        "training_dataset": {
            "dataset_key": str(dataset_csv),
            "response_cols": ["y"],
            "id_col": "cmpd_id",
            "smiles_col": "smiles",
        },
        "splitting_parameters": {
            "split_uuid": split_uuid,
            "splitter": "random",
            "split_strategy": "train_valid_test",
        },
        "training_metrics": [],
    }
    model_path = _make_model_tarball(tmp_path, cfg)

    # Patch safe_extract to allow tar extraction without depending on atomsci futils implementation
    def _safe_extract_passthrough(tar, path):
        tar.extractall(path=path)

    monkeypatch.setattr(perf_plots.futils, "safe_extract", _safe_extract_passthrough)

    # Patch prediction to return a dataframe with required columns: y_actual, y_pred and subset
    def _fake_predict_from_model_file(model_path_in, input_df, **kwargs):
        out = input_df.copy()
        out["y_actual"] = out["y"].astype(float)
        out["y_pred"] = out["y_actual"]  # perfect predictions => r2 = 1.0 on each subset
        return out

    monkeypatch.setattr(perf_plots.pfm, "predict_from_model_file", _fake_predict_from_model_file)

    fig = perf_plots.plot_pred_vs_actual_from_file(model_path, plot_size=3, error_bars=True)
    assert fig is not None

    # Expect 1 response * 3 subsets = 3 axes
    assert len(fig.axes) == 3

    titles = [ax.get_title() for ax in fig.axes]
    # First plot has "y train, R^2 = 1.000" style title, others "valid, R^2 = 1.000", "test, ..."
    assert any("R^2" in t for t in titles)
    assert any("1.000" in t for t in titles)

    plt.close(fig)


def test_plot_pred_vs_actual_from_file_uses_external_training_data(tmp_path, monkeypatch):
    # Original dataset_key in metadata points to one CSV, but we override via external_training_data.
    dataset_csv_original = tmp_path / "data_original.csv"
    dataset_csv_external = tmp_path / "data_external.csv"

    df_orig = pd.DataFrame({"cmpd_id": ["1"], "smiles": ["C"], "y": [1.0]})
    _write_csv(dataset_csv_original, df_orig)

    df_ext = pd.DataFrame({
        "cmpd_id": ["1", "2", "3"],
        "smiles": ["C", "CC", "CCC"],
        "y": [2.0, 3.0, 4.0],
    })
    _write_csv(dataset_csv_external, df_ext)

    split_uuid = "uuid2"
    split_csv = tmp_path / f"split_{split_uuid}.csv"
    _write_csv(split_csv, pd.DataFrame({
        "cmpd_id": ["1", "2", "3"],
        "subset": ["train", "valid", "test"],
    }))

    cfg = {
        "model_parameters": {"prediction_type": "regression", "model_type": "rf", "featurizer": "ecfp"},
        "training_dataset": {"dataset_key": str(dataset_csv_original), "response_cols": ["y"], "id_col": "cmpd_id", "smiles_col": "smiles"},
        "splitting_parameters": {"split_uuid": split_uuid, "splitter": "random", "split_strategy": "train_valid_test"},
        "training_metrics": [],
    }
    model_path = _make_model_tarball(tmp_path, cfg)

    def _safe_extract_passthrough(tar, path):
        tar.extractall(path=path)

    monkeypatch.setattr(perf_plots.futils, "safe_extract", _safe_extract_passthrough)

    seen = {"used_y": None}

    def _fake_predict_from_model_file(model_path_in, input_df, **kwargs):
        # Record which dataset was used by checking y
        seen["used_y"] = float(input_df["y"].iloc[0])
        out = input_df.copy()
        out["y_actual"] = out["y"].astype(float)
        out["y_pred"] = out["y_actual"]
        return out

    monkeypatch.setattr(perf_plots.pfm, "predict_from_model_file", _fake_predict_from_model_file)

    fig = perf_plots.plot_pred_vs_actual_from_file(
        model_path,
        external_training_data=str(dataset_csv_external),
        plot_size=3,
    )

    assert seen["used_y"] == 2.0  # proves override worked
    plt.close(fig)


def test_plot_pred_vs_actual_from_file_descriptor_path_rewrite(tmp_path, monkeypatch):
    # If featurizer is descriptors/computed_descriptors, function rewrites dataset path to scaled_descriptors/..._with_<desc>_descriptors.csv
    data_dir = tmp_path
    orig = data_dir / "data.csv"
    _write_csv(orig, pd.DataFrame({"cmpd_id": ["1"], "smiles": ["C"], "y": [1.0]}))

    # Create the expected rewritten file
    scaled_dir = data_dir / "scaled_descriptors"
    scaled_dir.mkdir()
    rewritten = scaled_dir / "data_with_mordred_descriptors.csv"

    df_rewritten = pd.DataFrame({
        "cmpd_id": ["1", "2", "3"],
        "smiles": ["C", "CC", "CCC"],
        "y": [2.0, 3.0, 4.0],
    })
    _write_csv(rewritten, df_rewritten)

    split_uuid = "uuid3"
    split_csv = data_dir / f"split_{split_uuid}.csv"
    _write_csv(split_csv, pd.DataFrame({
        "cmpd_id": ["1", "2", "3"],
        "subset": ["train", "valid", "test"],
    }))

    cfg = {
        "model_parameters": {"prediction_type": "regression", "model_type": "rf", "featurizer": "descriptors"},
        "descriptor_specific": {"descriptor_type": "mordred"},
        "training_dataset": {"dataset_key": str(orig), "response_cols": ["y"], "id_col": "cmpd_id", "smiles_col": "smiles"},
        "splitting_parameters": {"split_uuid": split_uuid, "splitter": "random", "split_strategy": "train_valid_test"},
        "training_metrics": [],
    }
    model_path = _make_model_tarball(tmp_path, cfg)

    def _safe_extract_passthrough(tar, path):
        tar.extractall(path=path)

    monkeypatch.setattr(perf_plots.futils, "safe_extract", _safe_extract_passthrough)

    # We must ensure merge_response_cols_from_original is exercised: rewritten df lacks y, orig has y.
    def _fake_predict_from_model_file(model_path_in, input_df, **kwargs):
        assert "y" in input_df.columns  # confirms merge happened
        out = input_df.copy()
        out["y_actual"] = out["y"].astype(float)
        out["y_pred"] = out["y_actual"]
        return out

    monkeypatch.setattr(perf_plots.pfm, "predict_from_model_file", _fake_predict_from_model_file)

    fig = perf_plots.plot_pred_vs_actual_from_file(model_path, plot_size=3)
    assert fig is not None
    plt.close(fig)