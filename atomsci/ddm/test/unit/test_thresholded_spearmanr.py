# test_thresholded_spearmanr.py
#
# Pytest unit tests targeting full coverage of:
#   atomsci.ddm.utils.thresholded_spearmanr
#
# Key design points:
# - We stub atomsci submodules that the module under test imports:
#     atomsci.ddm.utils.file_utils.safe_extract
#     atomsci.ddm.pipeline.perf_plots.merge_response_cols_from_original
#     atomsci.ddm.pipeline.predict_from_model.predict_from_model_file
# - We install these stubs into sys.modules BEFORE importing the module under test,
#   so the import succeeds even in minimal CI environments.
# - We then test:
#     - thresholded_spearmanr() branches and edge cases
#     - spearmenr_from_file() and thresholded_spearmenr_from_file() wrappers
#     - _metric_from_file() end-to-end with filesystem and IO monkeypatched
#
# Run:
#   pytest -q

import io
import json
import os
import sys
import types
import importlib

import numpy as np
import pandas as pd
import pytest
import tarfile


MODULE_UNDER_TEST = "atomsci.ddm.utils.thresholded_spearmanr"


def _install_stub_atomsci_deps(monkeypatch):
    """
    Install stub modules required by atomsci.ddm.utils.thresholded_spearmanr imports.

    We do NOT stub the module under test, we only stub its dependencies that may
    be unavailable or undesired in unit tests.
    """
    # Ensure package hierarchy exists
    atomsci = sys.modules.get("atomsci") or types.ModuleType("atomsci")
    ddm = sys.modules.get("atomsci.ddm") or types.ModuleType("atomsci.ddm")
    utils = sys.modules.get("atomsci.ddm.utils") or types.ModuleType("atomsci.ddm.utils")
    pipeline = sys.modules.get("atomsci.ddm.pipeline") or types.ModuleType("atomsci.ddm.pipeline")

    file_utils = types.ModuleType("atomsci.ddm.utils.file_utils")
    perf_plots = types.ModuleType("atomsci.ddm.pipeline.perf_plots")
    predict_from_model = types.ModuleType("atomsci.ddm.pipeline.predict_from_model")

    def safe_extract(_tar, path):
        # no-op, tests control file placement/reads via monkeypatching open/read_csv/listdir
        return None

    def merge_response_cols_from_original(df, orig_df, id_col, response_cols, **kwargs):
        # Ensure response cols exist on df by merging from orig_df
        if all(c in df.columns for c in response_cols):
            return df
        return df.merge(orig_df[[id_col] + response_cols], on=id_col, how="left")

    def predict_from_model_file(
        model_path,
        df,
        id_col,
        smiles_col,
        response_col,
        is_featurized,
        AD_method,
        dont_standardize,
    ):
        # Minimal pred_df contract expected by _metric_from_file:
        pred_df = df.copy()
        if "subset" not in pred_df.columns:
            pred_df["subset"] = "train"
        for resp in response_col:
            pred_df[f"{resp}_actual"] = pred_df[resp]
            pred_df[f"{resp}_pred"] = pred_df[resp].astype(float) + 1.0
        return pred_df

    file_utils.safe_extract = safe_extract
    perf_plots.merge_response_cols_from_original = merge_response_cols_from_original
    predict_from_model.predict_from_model_file = predict_from_model_file

    monkeypatch.setitem(sys.modules, "atomsci", atomsci)
    monkeypatch.setitem(sys.modules, "atomsci.ddm", ddm)
    monkeypatch.setitem(sys.modules, "atomsci.ddm.utils", utils)
    monkeypatch.setitem(sys.modules, "atomsci.ddm.pipeline", pipeline)

    monkeypatch.setitem(sys.modules, "atomsci.ddm.utils.file_utils", file_utils)
    monkeypatch.setitem(sys.modules, "atomsci.ddm.pipeline.perf_plots", perf_plots)
    monkeypatch.setitem(sys.modules, "atomsci.ddm.pipeline.predict_from_model", predict_from_model)


@pytest.fixture()
def mod(monkeypatch):
    _install_stub_atomsci_deps(monkeypatch)
    if MODULE_UNDER_TEST in sys.modules:
        del sys.modules[MODULE_UNDER_TEST]
    return importlib.import_module(MODULE_UNDER_TEST)


# -----------------------
# thresholded_spearmanr()
# -----------------------

def test_thresholded_spearmanr_shape_mismatch_raises(mod):
    with pytest.raises(ValueError, match="Shapes must match"):
        mod.thresholded_spearmanr([1, 2], [1], threshold=0.0)


def test_thresholded_spearmanr_bad_nan_policy_raises(mod):
    with pytest.raises(ValueError, match="nan_policy must be"):
        mod.thresholded_spearmanr([1, 2], [1, 2], threshold=0.0, nan_policy="drop")


def test_thresholded_spearmanr_nan_policy_raise_raises_on_nans(mod):
    with pytest.raises(ValueError, match="NaNs present in inputs"):
        mod.thresholded_spearmanr(
            [np.nan, 1.0], [0.0, 1.0], threshold=0.0, nan_policy="raise"
        )


def test_thresholded_spearmanr_nan_policy_omit_all_nan_returns_nan(mod):
    out = mod.thresholded_spearmanr([np.nan], [np.nan], threshold=0.0, nan_policy="omit")
    assert np.isnan(out)


def test_thresholded_spearmanr_no_predicted_actives_returns_empty_score(mod):
    out = mod.thresholded_spearmanr(
        y_pred=[-1.0, -2.0],
        y_true=[10.0, 11.0],
        threshold=0.0,
        empty_pred_active_score=0.123,
    )
    assert out == pytest.approx(0.123)


def test_thresholded_spearmanr_fp_only_rank_insufficient(mod):
    # Predicted actives: both, but both are false positives since y_true < threshold.
    out = mod.thresholded_spearmanr(
        y_pred=[1.0, 2.0],
        y_true=[-1.0, -2.0],
        threshold=0.0,
        fp_weight=1.0,
        min_ranked=3,
    )
    # fp_rate=1, fp_score=0, no correct actives => rank_score=0, score=0
    assert out == pytest.approx(0.0)


def test_thresholded_spearmanr_rank_term_min_ranked_sets_zero(mod):
    # Only 2 correctly predicted actives, min_ranked=3 => rank_score=0
    out = mod.thresholded_spearmanr(
        y_pred=[1.0, 2.0, -1.0],
        y_true=[1.1, 2.2, -5.0],
        threshold=0.0,
        fp_weight=0.0,   # isolate rank_score contribution
        min_ranked=3,
    )
    assert out == pytest.approx(0.0)


def test_thresholded_spearmanr_rank_term_nan_rho_maps_to_half(monkeypatch, mod):
    # Force spearmanr() to return NaN to hit rank_score=0.5 path.
    def fake_spearmanr(a, b, nan_policy):
        return (np.nan, np.nan)

    monkeypatch.setattr(mod, "spearmanr", fake_spearmanr)

    out = mod.thresholded_spearmanr(
        y_pred=[1.0, 2.0, 3.0],
        y_true=[1.0, 2.0, 3.0],
        threshold=0.0,
        fp_weight=0.0,  # isolate rank_score
        min_ranked=3,
    )
    assert out == pytest.approx(0.5)


def test_thresholded_spearmanr_happy_path_scores_in_0_1(mod):
    # 3 predicted actives, 1 false positive, and perfect rank among correct actives
    out = mod.thresholded_spearmanr(
        y_pred=[10.0, 20.0, 30.0],
        y_true=[10.0, 20.0, -1.0],  # last is FP
        threshold=0.0,
        fp_weight=1.0,
        min_ranked=2,
    )
    # fp_rate=1/3 => fp_score=2/3
    # correct actives are first two => rho=1 => rank_score=1
    expected = ((2 / 3) + 1.0) / 2.0
    assert out == pytest.approx(expected, rel=1e-6)
    assert 0.0 <= out <= 1.0


# -------------------------
# Wrapper function behavior
# -------------------------

def test_spearmenr_from_file_len_lt_3_returns_zero(monkeypatch, mod):
    # Patch _metric_from_file to call metric on a tiny array (<3) to hit len<3 path
    def fake_metric_from_file(model_path, metric):
        return {"x_train": metric(np.array([1.0, 2.0]), np.array([2.0, 1.0]))}

    monkeypatch.setattr(mod, "_metric_from_file", fake_metric_from_file)

    out = mod.spearmenr_from_file("dummy.tar.gz")
    assert out["x_train"] == 0


def test_thresholded_spearmenr_from_file_passes_params(monkeypatch, mod):
    captured = {}

    def fake_metric_from_file(model_path, metric):
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])
        captured["val"] = metric(y_pred, y_true)
        return {"task_train": captured["val"]}

    monkeypatch.setattr(mod, "_metric_from_file", fake_metric_from_file)

    out = mod.thresholded_spearmenr_from_file(
        "dummy.tar.gz",
        threshold=0.0,
        fp_weight=0.0,
        min_ranked=3,
        empty_pred_active_score=0.9,
        nan_policy="omit",
    )
    # fp_weight=0 isolates rank_score; perfect rank => 1
    assert out["task_train"] == pytest.approx(1.0)


# -------------------------
# _metric_from_file end-to-end
# -------------------------

def _fake_environment(
    monkeypatch,
    *,
    prediction_type: str = "regression",
    featurizer: str = "ecfp",
    multiple_split: bool = False,
    no_split: bool = False,
):
    """
    Monkeypatch IO and filesystem behavior used by _metric_from_file:
      - tarfile.open context manager
      - open(model_metadata.json)
      - pd.read_csv for dataset and split
      - os.listdir and path helpers for split file detection
    """
    # Fake tarfile.open
    class FakeTar:
        def __enter__(self):  # pragma: no cover
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover
            return False

    monkeypatch.setattr(tarfile, "open", lambda *args, **kwargs: FakeTar())

    config = {
        "model_parameters": {"prediction_type": prediction_type, "featurizer": featurizer},
        "training_dataset": {
            "dataset_key": "/data/dataset.csv",
            "response_cols": ["r1", "r2"],
            "id_col": "id",
            "smiles_col": "smiles",
        },
        "splitting_parameters": {"split_uuid": "UUID123"},
    }
    if featurizer in ["descriptors", "computed_descriptors"]:
        config["descriptor_specific"] = {"descriptor_type": "rdkit"}

    meta_text = json.dumps(config)

    def fake_open(path, *args, **kwargs):
        if path.endswith("model_metadata.json"):
            return io.StringIO(meta_text)
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", fake_open)

    # Path helpers for split file discovery
    monkeypatch.setattr(os.path, "realpath", lambda p: p)
    monkeypatch.setattr(os.path, "dirname", lambda p: "/data")

    if no_split:
        matches = []
    elif multiple_split:
        matches = ["split_UUID123_a.csv", "split_UUID123_b.csv"]
    else:
        matches = ["split_UUID123.csv"]
    monkeypatch.setattr(os, "listdir", lambda _p: matches)

    dataset_df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5"],
            "smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "r1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "r2": [5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    split_df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5"],
            "subset": ["train", "train", "valid", "test", "test"],
        }
    )

    def fake_read_csv(path, *args, **kwargs):
        if path.endswith("dataset.csv"):
            return dataset_df.copy()
        if "scaled_descriptors" in path:
            # For descriptor featurizer branch: return a dataset that lacks responses
            # so merge_response_cols_from_original must add them.
            return dataset_df[["id", "smiles"]].copy()
        if "split_UUID123" in path:
            return split_df.copy()
        return dataset_df.copy()

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    return config


def test__metric_from_file_classification_raises(monkeypatch, mod):
    _fake_environment(monkeypatch, prediction_type="classification")
    with pytest.raises(ValueError, match="regression models"):
        mod._metric_from_file("dummy.tar.gz", metric_function=lambda y_pred, y_true: 0.0)


def test__metric_from_file_no_split_file_raises(monkeypatch, mod):
    _fake_environment(monkeypatch, no_split=True)
    # Current implementation selects split[0] without checking, so IndexError is expected.
    with pytest.raises(IndexError):
        mod._metric_from_file("dummy.tar.gz", metric_function=lambda y_pred, y_true: 0.0)


def test__metric_from_file_multiple_split_files_uses_first(monkeypatch, mod):
    _fake_environment(monkeypatch, multiple_split=True)

    def metric_fn(y_pred, y_true):
        return float(np.mean(np.abs(y_pred - y_true)))

    out = mod._metric_from_file("dummy.tar.gz", metric_function=metric_fn)
    assert set(out.keys()) == {"r1_train", "r1_valid", "r1_test", "r2_train", "r2_valid", "r2_test"}
    # Stub predictor sets pred = actual + 1 => MAE = 1
    assert out["r1_train"] == pytest.approx(1.0)
    assert out["r2_test"] == pytest.approx(1.0)


def test__metric_from_file_descriptor_featurizer_branch(monkeypatch, mod):
    # Cover featurizer in ['descriptors','computed_descriptors'] path and response merge.
    _fake_environment(monkeypatch, featurizer="descriptors")

    def metric_fn(y_pred, y_true):
        return float(np.mean(y_pred - y_true))  # should be 1 with our stub predictor

    out = mod._metric_from_file("dummy.tar.gz", metric_function=metric_fn)
    assert out["r1_train"] == pytest.approx(1.0)
    assert out["r2_valid"] == pytest.approx(1.0)


def test__metric_from_file_metric_called_with_empty_subset(monkeypatch, mod):
    _fake_environment(monkeypatch)

    def fake_read_csv(path, *args, **kwargs):
        if path.endswith("dataset.csv"):
            return pd.DataFrame(
                {
                    "id": ["1", "2", "3"],
                    "smiles": ["C", "CC", "CCC"],
                    "r1": [0.0, 1.0, 2.0],
                    "r2": [5.0, 6.0, 7.0],
                }
            )
        if "split_UUID123" in path:
            return pd.DataFrame({"id": ["1", "2", "3"], "subset": ["train", "train", "train"]})
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    def metric_fn(y_pred, y_true):
        if len(y_pred) == 0:
            return 42.0
        return 1.0

    out = mod._metric_from_file("dummy.tar.gz", metric_function=metric_fn)
    assert out["r1_valid"] == pytest.approx(42.0)
    assert out["r2_test"] == pytest.approx(42.0)
    assert out["r1_train"] == pytest.approx(1.0)