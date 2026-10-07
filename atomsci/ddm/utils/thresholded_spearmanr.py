"""
thresholded_spearmanr_metrics.py

Utilities to:
  1) Load an ATOM/atomsci DDM regression model tarball
  2) Locate and load the training dataset and split file
  3) Merge response columns onto featurized data if needed
  4) Run predictions on train/valid/test
  5) Compute arbitrary per-task, per-subset metrics via a simple metric_fn(y_pred, y_true) interface

Output format:
  - Metrics are returned as a small pandas DataFrame:
        index: response (task) name
        columns: train, valid, test (configurable)
"""

import json
import os
import tarfile
import tempfile
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import atomsci.ddm.pipeline.perf_plots as pp
from atomsci.ddm.pipeline import predict_from_model as pfm
from atomsci.ddm.utils import file_utils as futils

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def thresholded_spearmanr(
    y_pred,
    y_true,
    *,
    threshold: float,
    fp_weight: float = 1.0,
    min_ranked: int = 3,
    empty_pred_active_score: float = 0.0,
    nan_policy: str = "omit",
) -> float:
    """
    Metric in [0, 1], higher is better.

    Definitions
    ----------
    predicted active: y_pred >= threshold
    false positive: predicted active AND y_true < threshold

    Score components
    ----------------
    fp_score: 1 - FP_rate among predicted actives
    rank_score: Spearman rank correlation between y_pred and y_true,
                computed only on correctly predicted actives
                (predicted active AND y_true >= threshold),
                mapped from [-1,1] to [0,1].

    Final score
    -----------
    Weighted average: (fp_weight * fp_score + rank_score) / (fp_weight + 1)

    Parameters
    ----------
    fp_weight:
        Higher values penalize false positives more strongly.
    min_ranked:
        Minimum number of correctly predicted actives required to compute Spearman.
        If fewer, rank_score is set to 0.0 (strong penalty, matches implementation).
    empty_pred_active_score:
        Returned when no compounds are predicted active.
    nan_policy:
        "omit" drops NaN pairs before computing, "raise" errors if any NaNs.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shapes must match, got {y_pred.shape} vs {y_true.shape}")

    if nan_policy not in {"omit", "raise"}:
        raise ValueError("nan_policy must be 'omit' or 'raise'")

    if nan_policy == "raise":
        if np.isnan(y_pred).any() or np.isnan(y_true).any():
            raise ValueError("NaNs present in inputs")
        yp, yt = y_pred, y_true
    else:
        m = ~(np.isnan(y_pred) | np.isnan(y_true))
        yp, yt = y_pred[m], y_true[m]

    if yp.size == 0:
        return float("nan")

    pred_active = yp >= threshold
    n_pred_active = int(np.sum(pred_active))
    if n_pred_active == 0:
        return float(empty_pred_active_score)

    fp = pred_active & (yt < threshold)
    fp_rate = float(np.sum(fp)) / float(n_pred_active)
    fp_score = 1.0 - fp_rate

    correct_active = pred_active & (yt >= threshold)
    yp_ca = yp[correct_active]
    yt_ca = yt[correct_active]

    if yp_ca.size < min_ranked:
        rank_score = 0.0
    else:
        res = spearmanr(yp_ca, yt_ca, nan_policy=nan_policy)
        rho = getattr(res, "statistic", res[0])
        rank_score = 0.5 if np.isnan(rho) else 0.5 * (float(rho) + 1.0)

    score = (fp_weight * fp_score + rank_score) / (fp_weight + 1.0)
    return float(np.clip(score, 0.0, 1.0))


def predictions_from_model_file(model_path: str) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Load a regression model tarball and reproduce the training split assignment,
    then run predict_from_model_file() to produce predictions for all rows.

    Returns
    -------
    pred_df:
        Predictions dataframe from atomsci, expected to contain:
          - 'subset' column with values in {'train','valid','test'} (depending on split)
          - for each response 'resp': 'resp_actual' and 'resp_pred'
    response_cols:
        List of response/task names
    config:
        Parsed model_metadata.json dictionary
    """
    reload_dir = tempfile.mkdtemp()
    try:
        with tarfile.open(model_path, mode="r:gz") as tar:
            futils.safe_extract(tar, path=reload_dir)

        with open(os.path.join(reload_dir, "model_metadata.json")) as f:
            config = json.loads(f.read())

        if config["model_parameters"]["prediction_type"] == "classification":
            raise ValueError("predictions_from_model_file() only supports regression models.")

        dataset_dict = config["training_dataset"]
        original_dataset_key = dataset_dict["dataset_key"]
        response_cols = dataset_dict["response_cols"]
        id_col = dataset_dict["id_col"]
        smiles_col = dataset_dict["smiles_col"]

        featurizer = config["model_parameters"]["featurizer"]
        is_featurized = False
        dataset_key_to_read = original_dataset_key

        if featurizer in ["descriptors", "computed_descriptors"]:
            desc = config["descriptor_specific"]["descriptor_type"]
            parts = original_dataset_key.rsplit("/", maxsplit=1)
            dataset_key_to_read = os.path.join(
                parts[0],
                "scaled_descriptors",
                parts[1].replace(".csv", f"_with_{desc}_descriptors.csv"),
            )
            is_featurized = True

        df = pd.read_csv(dataset_key_to_read)
        orig_df = pd.read_csv(original_dataset_key)

        df = pp.merge_response_cols_from_original(
            df,
            orig_df,
            id_col=id_col,
            response_cols=response_cols,
            max_missing_frac=0.01,
            error_on_extra_feat_ids=False,
            coerce_id_to_str=True,
            sample_n=20,
        )

        split_uuid = config["splitting_parameters"]["split_uuid"]
        data_dir = os.path.dirname(os.path.realpath(original_dataset_key))
        matches = [fn for fn in os.listdir(data_dir) if split_uuid in fn]

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No split file found containing split_uuid={split_uuid} in {data_dir}"
            )
        if len(matches) > 1:
            raise FileExistsError(
                f"More than one split file found containing split_uuid={split_uuid} in {data_dir}: {matches}"
            )

        split_file = os.path.join(data_dir, matches[0])
        split = pd.read_csv(split_file).rename(columns={"cmpd_id": id_col})

        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str)
        if id_col in split.columns:
            split[id_col] = split[id_col].astype(str)

        df = df.merge(split, how="left", on=id_col)

        pred_df = pfm.predict_from_model_file(
            model_path,
            df,
            id_col=id_col,
            smiles_col=smiles_col,
            response_col=response_cols,
            is_featurized=is_featurized,
            AD_method=None,
            dont_standardize=True,
        )

        return pred_df, response_cols, config

    finally:
        # Optional cleanup for shared systems:
        # import shutil
        # shutil.rmtree(reload_dir, ignore_errors=True)
        pass


def evaluate_metric_per_subset_df(
    pred_df: pd.DataFrame,
    response_cols: list[str],
    metric_fn: MetricFn,
    subsets: tuple[str, ...] = ("train", "valid", "test"),
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute per-task, per-subset metrics.

    Returns
    -------
    metrics_df:
        DataFrame indexed by response/task, columns=subsets.
    """
    if "subset" not in pred_df.columns:
        raise KeyError("pred_df is missing required column 'subset'")

    rows = []
    for resp in response_cols:
        actual_col = f"{resp}_actual"
        pred_col = f"{resp}_pred"

        if actual_col not in pred_df.columns or pred_col not in pred_df.columns:
            raise KeyError(f"Missing expected columns: {actual_col} and/or {pred_col}")

        task_df = pred_df
        if dropna:
            task_df = task_df[task_df[actual_col].notna() & task_df[pred_col].notna()]

        row = {"response": resp}
        for subset in subsets:
            tmp = task_df[task_df["subset"] == subset]
            y_pred = tmp[pred_col].to_numpy(dtype=float)
            y_true = tmp[actual_col].to_numpy(dtype=float)
            row[subset] = float(metric_fn(y_pred, y_true))

        rows.append(row)

    metrics_df = pd.DataFrame(rows).set_index("response")
    metrics_df = metrics_df.reindex(columns=list(subsets))
    return metrics_df


def metric_from_file_df(model_path: str, metric_fn: MetricFn) -> pd.DataFrame:
    """
    Convenience wrapper: load predictions from a model tarball, then evaluate metric per subset.
    """
    pred_df, response_cols, _config = predictions_from_model_file(model_path)
    return evaluate_metric_per_subset_df(pred_df, response_cols, metric_fn)


def thresholded_spearmanr_from_file_df(
    model_path: str,
    *,
    threshold: float,
    fp_weight: float = 1.0,
    min_ranked: int = 3,
    empty_pred_active_score: float = 0.0,
    nan_policy: str = "omit",
) -> pd.DataFrame:
    """
    Compute thresholded_spearmanr per task and per subset for a model tarball.
    """
    def metric_fn(y_pred, y_true):
        return thresholded_spearmanr(
            y_pred=y_pred,
            y_true=y_true,
            threshold=threshold,
            fp_weight=fp_weight,
            min_ranked=min_ranked,
            empty_pred_active_score=empty_pred_active_score,
            nan_policy=nan_policy,
        )

    return metric_from_file_df(model_path, metric_fn)


def spearmanr_from_file_df(model_path: str, *, nan_policy: str = "omit") -> pd.DataFrame:
    """
    Compute Spearman's rho per task and per subset for a model tarball.
    """
    def metric_fn(y_pred, y_true):
        if len(y_pred) < 3:
            return 0.0
        res = spearmanr(y_pred, y_true, nan_policy=nan_policy)
        rho = getattr(res, "statistic", res[0])
        return 0.0 if np.isnan(rho) else float(rho)

    return metric_from_file_df(model_path, metric_fn)