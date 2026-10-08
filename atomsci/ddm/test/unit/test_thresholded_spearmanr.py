"""Tests for atomsci.ddm.utils.thresholded_spearmanr.

ATOM dependencies are stubbed before loading the module under test.
Filesystem tests use temporary datasets, split files, and model archives.
"""

import importlib.util
import io
import json
import sys
import tarfile
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


MODULE_UNDER_TEST = "atomsci.ddm.utils.thresholded_spearmanr"
SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "utils"
    / "thresholded_spearmanr.py"
)


@pytest.fixture
def mod(monkeypatch):
    """Load the real source module with isolated ATOM dependency stubs."""
    package_names = (
        "atomsci",
        "atomsci.ddm",
        "atomsci.ddm.utils",
        "atomsci.ddm.pipeline",
    )
    dependency_names = (
        "atomsci.ddm.utils.file_utils",
        "atomsci.ddm.pipeline.perf_plots",
        "atomsci.ddm.pipeline.predict_from_model",
    )

    modules = {}
    for name in package_names + dependency_names:
        module = types.ModuleType(name)
        if name in package_names:
            module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    # Populate parent attributes as well as sys.modules entries.
    for name, module in modules.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)

    def safe_extract(archive, path):
        # Only metadata is needed from the test-generated archive.
        member = archive.extractfile("model_metadata.json")
        assert member is not None
        with member:
            (Path(path) / "model_metadata.json").write_bytes(member.read())

    def merge_response_cols_from_original(
        df, orig_df, *, id_col, response_cols, **kwargs
    ):
        missing = [col for col in response_cols if col not in df.columns]
        if not missing:
            return df.copy()
        return df.merge(
            orig_df[[id_col] + missing],
            on=id_col,
            how="left",
        )

    def predict_from_model_file(model_path, df, *, response_col, **kwargs):
        pred_df = df.copy()
        for response in response_col:
            pred_df[f"{response}_actual"] = pred_df[response]
            pred_df[f"{response}_pred"] = pred_df[response].astype(float) + 1.0
        return pred_df

    modules["atomsci.ddm.utils.file_utils"].safe_extract = Mock(
        side_effect=safe_extract
    )
    modules[
        "atomsci.ddm.pipeline.perf_plots"
    ].merge_response_cols_from_original = Mock(
        side_effect=merge_response_cols_from_original
    )
    modules[
        "atomsci.ddm.pipeline.predict_from_model"
    ].predict_from_model_file = Mock(side_effect=predict_from_model_file)

    spec = importlib.util.spec_from_file_location(
        MODULE_UNDER_TEST, SOURCE_PATH
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_UNDER_TEST, module)
    modules["atomsci.ddm.utils"].thresholded_spearmanr = module
    spec.loader.exec_module(module)
    return module


def _make_model_environment(
    monkeypatch,
    tmp_path,
    mod,
    *,
    prediction_type="regression",
    featurizer="ecfp",
    split_count=1,
    subsets=("train", "train", "valid", "test", "test"),
):
    """Create real temporary input files for the stubbed prediction pipeline."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset_path = data_dir / "dataset.csv"

    dataset = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "r1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "r2": [5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    dataset.to_csv(dataset_path, index=False)

    config = {
        "model_parameters": {
            "prediction_type": prediction_type,
            "featurizer": featurizer,
        },
        "training_dataset": {
            "dataset_key": str(dataset_path),
            "response_cols": ["r1", "r2"],
            "id_col": "id",
            "smiles_col": "smiles",
        },
        "splitting_parameters": {"split_uuid": "UUID123"},
    }

    if featurizer in {"descriptors", "computed_descriptors"}:
        config["descriptor_specific"] = {"descriptor_type": "rdkit"}
        descriptor_dir = data_dir / "scaled_descriptors"
        descriptor_dir.mkdir()
        descriptors = dataset[["id", "smiles"]].copy()
        descriptors["descriptor_0"] = np.arange(len(dataset), dtype=float)
        descriptors.to_csv(
            descriptor_dir / "dataset_with_rdkit_descriptors.csv",
            index=False,
        )

    # Use cmpd_id to exercise the loader's split-column rename.
    split = pd.DataFrame(
        {
            "cmpd_id": dataset["id"],
            "subset": list(subsets),
        }
    )
    for index in range(split_count):
        split.to_csv(
            data_dir / f"split_UUID123_{index}.csv",
            index=False,
        )

    model_path = tmp_path / "model.tar.gz"
    metadata = json.dumps(config).encode("utf-8")
    with tarfile.open(model_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("model_metadata.json")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))

    reload_dir = tmp_path / "reload"
    reload_dir.mkdir()

    # Replace only this module's tempfile reference, not global tempfile behavior.
    monkeypatch.setattr(
        mod,
        "tempfile",
        types.SimpleNamespace(
            mkdtemp=Mock(return_value=str(reload_dir))
        ),
    )

    return types.SimpleNamespace(
        model_path=str(model_path),
        config=config,
        dataset=dataset,
        reload_dir=reload_dir,
        subsets=list(subsets),
    )


# ---------------------------------------------------------------------------
# thresholded_spearmanr
# ---------------------------------------------------------------------------


def test_thresholded_spearmanr_shape_mismatch_raises(mod):
    with pytest.raises(ValueError, match="Shapes must match"):
        mod.thresholded_spearmanr([1, 2], [1], threshold=0.0)


def test_thresholded_spearmanr_bad_nan_policy_raises(mod):
    with pytest.raises(ValueError, match="nan_policy must be"):
        mod.thresholded_spearmanr(
            [1, 2], [1, 2], threshold=0.0, nan_policy="drop"
        )


@pytest.mark.parametrize(
    "y_pred,y_true",
    [
        ([np.nan, 1.0], [0.0, 1.0]),
        ([0.0, 1.0], [np.nan, 1.0]),
    ],
)
def test_thresholded_spearmanr_nan_policy_raise(mod, y_pred, y_true):
    with pytest.raises(ValueError, match="NaNs present in inputs"):
        mod.thresholded_spearmanr(
            y_pred, y_true, threshold=0.0, nan_policy="raise"
        )


@pytest.mark.parametrize(
    "y_pred,y_true",
    [
        ([], []),
        ([np.nan], [np.nan]),
    ],
)
def test_thresholded_spearmanr_no_usable_pairs_returns_nan(
    mod, y_pred, y_true
):
    result = mod.thresholded_spearmanr(
        y_pred, y_true, threshold=0.0, nan_policy="omit"
    )
    assert np.isnan(result)


def test_thresholded_spearmanr_omits_nan_pairs(mod):
    result = mod.thresholded_spearmanr(
        y_pred=[1.0, np.nan, 2.0, 3.0, 100.0],
        y_true=[1.0, 99.0, 2.0, 3.0, np.nan],
        threshold=0.0,
    )
    assert result == pytest.approx(1.0)


def test_thresholded_spearmanr_no_predicted_actives(mod):
    result = mod.thresholded_spearmanr(
        y_pred=[-1.0, -2.0],
        y_true=[10.0, 11.0],
        threshold=0.0,
        empty_pred_active_score=0.123,
    )
    assert result == pytest.approx(0.123)


def test_thresholded_spearmanr_all_false_positives(mod):
    result = mod.thresholded_spearmanr(
        y_pred=[1.0, 2.0],
        y_true=[-1.0, -2.0],
        threshold=0.0,
    )
    assert result == pytest.approx(0.0)


def test_thresholded_spearmanr_insufficient_correct_actives(mod):
    result = mod.thresholded_spearmanr(
        y_pred=[1.0, 2.0, -1.0],
        y_true=[1.1, 2.2, -5.0],
        threshold=0.0,
        fp_weight=0.0,
        min_ranked=3,
    )
    assert result == pytest.approx(0.0)


@pytest.mark.parametrize("nan_policy", ["omit", "raise"])
def test_thresholded_spearmanr_threshold_is_inclusive(mod, nan_policy):
    result = mod.thresholded_spearmanr(
        y_pred=[0.0, 1.0, 2.0],
        y_true=[0.0, 1.0, 2.0],
        threshold=0.0,
        min_ranked=3,
        nan_policy=nan_policy,
    )
    assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    "y_true,expected",
    [
        ([1.0, 2.0, 3.0], 1.0),
        ([3.0, 2.0, 1.0], 0.0),
    ],
)
def test_thresholded_spearmanr_rank_mapping(mod, y_true, expected):
    result = mod.thresholded_spearmanr(
        y_pred=[1.0, 2.0, 3.0],
        y_true=y_true,
        threshold=0.0,
        fp_weight=0.0,
    )
    assert result == pytest.approx(expected)


def test_thresholded_spearmanr_nan_rho_maps_to_half(monkeypatch, mod):
    monkeypatch.setattr(
        mod, "spearmanr", Mock(return_value=(np.nan, np.nan))
    )
    result = mod.thresholded_spearmanr(
        y_pred=[1.0, 2.0, 3.0],
        y_true=[1.0, 2.0, 3.0],
        threshold=0.0,
        fp_weight=0.0,
    )
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize("fp_weight", [0.0, 1.0, 3.0])
def test_thresholded_spearmanr_weighted_false_positive_score(mod, fp_weight):
    result = mod.thresholded_spearmanr(
        y_pred=[10.0, 20.0, 30.0],
        y_true=[10.0, 20.0, -1.0],
        threshold=0.0,
        fp_weight=fp_weight,
        min_ranked=2,
    )
    # Precision is 2/3; ranking of the two correct actives is perfect.
    expected = (fp_weight * (2.0 / 3.0) + 1.0) / (fp_weight + 1.0)
    assert result == pytest.approx(expected)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# evaluate_metric_per_subset_df
# ---------------------------------------------------------------------------


@pytest.fixture
def pred_df():
    return pd.DataFrame(
        {
            "subset": ["train", "train", "valid", "test", "test"],
            "r1_actual": [1.0, 2.0, 3.0, np.nan, 5.0],
            "r1_pred": [2.0, 4.0, 6.0, 7.0, np.nan],
            "r2_actual": [10.0, 20.0, 30.0, 40.0, 50.0],
            "r2_pred": [9.0, 18.0, 27.0, 36.0, 45.0],
        }
    )


def test_evaluate_metric_per_subset_df_values_and_argument_order(mod, pred_df):
    def signed_error(y_pred, y_true):
        assert y_pred.dtype == np.dtype(float)
        assert y_true.dtype == np.dtype(float)
        if y_pred.size == 0:
            return np.nan
        return float(np.mean(y_pred - y_true))

    result = mod.evaluate_metric_per_subset_df(
        pred_df,
        ["r1", "r2"],
        metric_fn=signed_error,
    )
    expected = pd.DataFrame(
        [
            [1.5, 3.0, np.nan],
            [-1.5, -3.0, -4.5],
        ],
        index=pd.Index(["r1", "r2"], name="response"),
        columns=["train", "valid", "test"],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_evaluate_metric_per_subset_df_missing_subset_raises(mod, pred_df):
    with pytest.raises(KeyError, match="missing required column 'subset'"):
        mod.evaluate_metric_per_subset_df(
            pred_df.drop(columns="subset"),
            ["r1"],
            metric_fn=lambda y_pred, y_true: 0.0,
        )


@pytest.mark.parametrize("missing_column", ["r1_actual", "r1_pred"])
def test_evaluate_metric_per_subset_df_missing_response_column_raises(
    mod, pred_df, missing_column
):
    with pytest.raises(KeyError, match="Missing expected columns"):
        mod.evaluate_metric_per_subset_df(
            pred_df.drop(columns=missing_column),
            ["r1"],
            metric_fn=lambda y_pred, y_true: 0.0,
        )


@pytest.mark.parametrize("dropna", [True, False])
def test_evaluate_metric_per_subset_df_dropna(mod, pred_df, dropna):
    metric = Mock(return_value=42.0)

    result = mod.evaluate_metric_per_subset_df(
        pred_df,
        ["r1"],
        metric_fn=metric,
        subsets=("test",),
        dropna=dropna,
    )

    metric.assert_called_once()
    y_pred, y_true = metric.call_args.args
    if dropna:
        assert y_pred.size == y_true.size == 0
    else:
        np.testing.assert_allclose(
            y_pred, [7.0, np.nan], equal_nan=True
        )
        np.testing.assert_allclose(
            y_true, [np.nan, 5.0], equal_nan=True
        )

    assert result.loc["r1", "test"] == pytest.approx(42.0)


def test_evaluate_metric_per_subset_df_custom_order_and_empty_subset(
    mod, pred_df
):
    result = mod.evaluate_metric_per_subset_df(
        pred_df,
        ["r2", "r1"],
        metric_fn=lambda y_pred, y_true: len(y_pred),
        subsets=("test", "holdout", "train"),
    )
    expected = pd.DataFrame(
        [
            [2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        index=pd.Index(["r2", "r1"], name="response"),
        columns=["test", "holdout", "train"],
    )
    pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def test_metric_from_file_df_loads_predictions(monkeypatch, mod, pred_df):
    loader = Mock(return_value=(pred_df, ["r1", "r2"], {}))
    monkeypatch.setattr(mod, "predictions_from_model_file", loader)

    result = mod.metric_from_file_df(
        "dummy.tar.gz",
        metric_fn=lambda y_pred, y_true: 42.0,
    )

    loader.assert_called_once_with("dummy.tar.gz")
    expected = pd.DataFrame(
        42.0,
        index=pd.Index(["r1", "r2"], name="response"),
        columns=["train", "valid", "test"],
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.fixture
def metric_delegate(monkeypatch, mod):
    output = pd.DataFrame(
        {"train": [123.0]},
        index=pd.Index(["task"], name="response"),
    )
    delegate = Mock(return_value=output)
    monkeypatch.setattr(mod, "metric_from_file_df", delegate)
    return delegate


@pytest.mark.parametrize(
    "y_pred,y_true,expected",
    [
        ([], [], 0.0),
        ([1.0, 2.0], [2.0, 1.0], 0.0),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
        ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], -1.0),
    ],
)
def test_spearmanr_from_file_df(
    monkeypatch, mod, metric_delegate, y_pred, y_true, expected
):
    spearman = Mock(wraps=mod.spearmanr)
    monkeypatch.setattr(mod, "spearmanr", spearman)

    result = mod.spearmanr_from_file_df(
        "dummy.tar.gz", nan_policy="raise"
    )

    assert result is metric_delegate.return_value
    metric_delegate.assert_called_once()
    model_path, metric_fn = metric_delegate.call_args.args
    assert model_path == "dummy.tar.gz"

    score = metric_fn(
        np.asarray(y_pred, dtype=float),
        np.asarray(y_true, dtype=float),
    )
    assert score == pytest.approx(expected)

    if len(y_pred) < 3:
        spearman.assert_not_called()
    else:
        spearman.assert_called_once()
        assert spearman.call_args.kwargs == {"nan_policy": "raise"}


def test_spearmanr_from_file_df_nan_rho_returns_zero(
    monkeypatch, mod, metric_delegate
):
    spearman = Mock(return_value=(np.nan, np.nan))
    monkeypatch.setattr(mod, "spearmanr", spearman)

    mod.spearmanr_from_file_df("dummy.tar.gz")
    metric_fn = metric_delegate.call_args.args[1]

    result = metric_fn(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert result == pytest.approx(0.0)
    spearman.assert_called_once()
    assert spearman.call_args.kwargs == {"nan_policy": "omit"}


def test_thresholded_spearmanr_from_file_df_passes_all_parameters(
    monkeypatch, mod, metric_delegate
):
    metric = Mock(return_value=0.625)
    monkeypatch.setattr(mod, "thresholded_spearmanr", metric)

    parameters = {
        "threshold": 2.5,
        "fp_weight": 3.0,
        "min_ranked": 4,
        "empty_pred_active_score": 0.2,
        "nan_policy": "raise",
    }
    result = mod.thresholded_spearmanr_from_file_df(
        "dummy.tar.gz", **parameters
    )

    assert result is metric_delegate.return_value
    metric_delegate.assert_called_once()
    model_path, metric_fn = metric_delegate.call_args.args
    assert model_path == "dummy.tar.gz"

    y_pred = np.array([3.0, 4.0, 5.0, 6.0])
    y_true = np.array([3.1, 4.1, 5.1, 6.1])
    assert metric_fn(y_pred, y_true) == pytest.approx(0.625)

    metric.assert_called_once()
    kwargs = metric.call_args.kwargs.copy()
    assert kwargs.pop("y_pred") is y_pred
    assert kwargs.pop("y_true") is y_true
    assert kwargs == parameters


# ---------------------------------------------------------------------------
# predictions_from_model_file and filesystem-backed integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "featurizer",
    ["ecfp", "descriptors", "computed_descriptors"],
)
def test_predictions_from_model_file(
    monkeypatch, tmp_path, mod, featurizer
):
    env = _make_model_environment(
        monkeypatch, tmp_path, mod, featurizer=featurizer
    )

    result, response_cols, config = mod.predictions_from_model_file(
        env.model_path
    )

    assert response_cols == ["r1", "r2"]
    assert config == env.config
    assert result["subset"].tolist() == env.subsets
    assert result["id"].tolist() == ["1", "2", "3", "4", "5"]

    for response in response_cols:
        np.testing.assert_allclose(
            result[f"{response}_actual"],
            env.dataset[response],
        )
        np.testing.assert_allclose(
            result[f"{response}_pred"],
            env.dataset[response] + 1.0,
        )

    merge = mod.pp.merge_response_cols_from_original
    merge.assert_called_once()
    assert merge.call_args.kwargs == {
        "id_col": "id",
        "response_cols": ["r1", "r2"],
        "max_missing_frac": 0.01,
        "error_on_extra_feat_ids": False,
        "coerce_id_to_str": True,
        "sample_n": 20,
    }

    is_featurized = featurizer in {"descriptors", "computed_descriptors"}
    input_df = merge.call_args.args[0]
    if is_featurized:
        assert "descriptor_0" in input_df.columns
        assert not {"r1", "r2"}.intersection(input_df.columns)
    else:
        assert {"r1", "r2"}.issubset(input_df.columns)

    predictor = mod.pfm.predict_from_model_file
    predictor.assert_called_once()
    assert predictor.call_args.args[0] == env.model_path
    assert predictor.call_args.kwargs == {
        "id_col": "id",
        "smiles_col": "smiles",
        "response_col": ["r1", "r2"],
        "is_featurized": is_featurized,
        "AD_method": None,
        "dont_standardize": True,
    }

    mod.futils.safe_extract.assert_called_once()
    assert mod.futils.safe_extract.call_args.kwargs == {
        "path": str(env.reload_dir)
    }
    mod.tempfile.mkdtemp.assert_called_once_with()


def test_predictions_from_model_file_classification_raises(
    monkeypatch, tmp_path, mod
):
    env = _make_model_environment(
        monkeypatch, tmp_path, mod, prediction_type="classification"
    )

    with pytest.raises(ValueError, match="only supports regression models"):
        mod.predictions_from_model_file(env.model_path)

    mod.pfm.predict_from_model_file.assert_not_called()


@pytest.mark.parametrize(
    "split_count,expected_exception",
    [
        (0, FileNotFoundError),
        (2, FileExistsError),
    ],
)
def test_predictions_from_model_file_invalid_split_count_raises(
    monkeypatch, tmp_path, mod, split_count, expected_exception
):
    env = _make_model_environment(
        monkeypatch, tmp_path, mod, split_count=split_count
    )

    with pytest.raises(expected_exception, match="UUID123"):
        mod.predictions_from_model_file(env.model_path)

    mod.pfm.predict_from_model_file.assert_not_called()


def test_metric_from_file_df_end_to_end(monkeypatch, tmp_path, mod):
    env = _make_model_environment(monkeypatch, tmp_path, mod)

    result = mod.metric_from_file_df(
        env.model_path,
        metric_fn=lambda y_pred, y_true: float(
            np.mean(np.abs(y_pred - y_true))
        ),
    )
    expected = pd.DataFrame(
        1.0,
        index=pd.Index(["r1", "r2"], name="response"),
        columns=["train", "valid", "test"],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_metric_from_file_df_calls_metric_for_empty_subsets(
    monkeypatch, tmp_path, mod
):
    env = _make_model_environment(
        monkeypatch,
        tmp_path,
        mod,
        subsets=("train",) * 5,
    )

    def metric_fn(y_pred, y_true):
        if y_pred.size == 0:
            assert y_true.size == 0
            return 42.0
        return float(np.mean(np.abs(y_pred - y_true)))

    result = mod.metric_from_file_df(
        env.model_path,
        metric_fn=metric_fn,
    )
    expected = pd.DataFrame(
        [
            [1.0, 42.0, 42.0],
            [1.0, 42.0, 42.0],
        ],
        index=pd.Index(["r1", "r2"], name="response"),
        columns=["train", "valid", "test"],
    )
    pd.testing.assert_frame_equal(result, expected)