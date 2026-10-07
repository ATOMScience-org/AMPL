from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from functools import reduce
from typing import Callable, Literal, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import atomsci.ddm.pipeline.predict_from_model as pfm
from atomsci.ddm.utils.model_file_reader import ModelFileReader

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def _sanitize_model_key(model_path: Union[str, Path]) -> str:
    base_name = Path(model_path).name
    base_name = base_name[base_name.rfind("_") + 1 : base_name.rfind(".tar.gz")]
    match = re.search(r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", base_name)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract model UUID from path: {model_path}. Ensure the filename contains a UUID.")


def _infer_response_columns(
    pred_df: pd.DataFrame,
    response_col: Optional[str],
) -> Tuple[str, str, Optional[str], Optional[str]]:
    if response_col is not None:
        resp = response_col
        pred_col = f"{resp}_pred"
        std_col = f"{resp}_std" if f"{resp}_std" in pred_df.columns else None
        actual_col = f"{resp}_actual" if f"{resp}_actual" in pred_df.columns else None
        if pred_col not in pred_df.columns:
            raise ValueError(
                f"Expected prediction column '{pred_col}' not found in model output. "
                f"Available columns: {list(pred_df.columns)}"
            )
        return resp, pred_col, std_col, actual_col

    pred_cols = [c for c in pred_df.columns if c.endswith("_pred")]
    if len(pred_cols) != 1:
        raise ValueError(
            "response_col was not provided, and model output did not have exactly one '*_pred' column. "
            f"Found: {pred_cols}"
        )
    pred_col = pred_cols[0]
    resp = pred_col[: -len("_pred")]
    std_col = f"{resp}_std" if f"{resp}_std" in pred_df.columns else None
    actual_col = f"{resp}_actual" if f"{resp}_actual" in pred_df.columns else None
    return resp, pred_col, std_col, actual_col


def _dedupe(df: pd.DataFrame, id_col: str, how: str) -> pd.DataFrame:
    if how == "first":
        return df.drop_duplicates(subset=[id_col], keep="first")
    if how == "mean":
        num_cols = [c for c in df.columns if c != id_col]
        return df.groupby(id_col, as_index=False)[num_cols].mean(numeric_only=True)
    raise ValueError("dedupe must be one of: 'first', 'mean'")


def _choose_input_df_for_model(
    model_path: Union[str, Path],
    input_dfs: Dict[str, pd.DataFrame],
    raw_df_key: str,
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Returns:
        selected_df, is_featurized, source_key
    """
    reader = ModelFileReader(str(model_path))
    featurizer = reader.get_featurizer()
    descriptor_type = reader.get_descriptor_type()

    if featurizer == "computed_descriptors" and descriptor_type in input_dfs:
        return input_dfs[descriptor_type], True, descriptor_type

    # if the descriptor_type is not found, any input_df will do
    return input_dfs[raw_df_key], False, raw_df_key

def query_by_committee_regression(
    model_paths: List[Union[str, Path]],
    input_dfs: Dict[str, pd.DataFrame],
    id_col: str = "compound_id",
    smiles_col: str = "rdkit_smiles",
    predicted_response: Optional[str] = None,
    predict_fn: Optional[Callable[..., pd.DataFrame]] = None,
    base: str = "intersection",
    dedupe: str = "first",
    disagreement_metric: str = "committee_std",
    sort_desc: bool = True,
    return_long: bool = False,
    raw_df_key: str = "raw",
    **predict_kwargs,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Query-by-committee for regression using a list of saved models and multiple possible input dataframes.

    base:
        - "intersection": only keep compounds present in raw input and all model-relevant inputs
        - "input": keep all compounds from raw input
        - "union": keep any compound present in any model output
    """
    if predict_fn is None:
        predict_fn = pfm.predict_from_model_file

    if not isinstance(input_dfs, dict) or len(input_dfs) == 0:
        raise ValueError("input_dfs must be a non-empty dict mapping featurization name to DataFrame.")

    for key, df in input_dfs.items():
        if id_col not in df.columns:
            raise ValueError(f"input_dfs['{key}'] must contain '{id_col}'.")
        if key == raw_df_key and smiles_col not in df.columns:
            raise ValueError(f"Raw dataframe input_dfs['{key}'] must contain '{smiles_col}'.")

    if base not in {"intersection", "input", "union"}:
        raise ValueError("base must be one of: 'intersection', 'input', 'union'")

    if raw_df_key not in input_dfs:
        raise ValueError(f"input_dfs must include raw dataframe key '{raw_df_key}'.")

    base_df = input_dfs[raw_df_key][[id_col, smiles_col]].copy()
    base_df = base_df.drop_duplicates(subset=[id_col], keep="first")

    per_model_wide_parts: List[pd.DataFrame] = []
    long_parts: List[pd.DataFrame] = []
    resp_name_global: Optional[str] = None
    actual_col_global: Optional[str] = None

    # Compute intersection IDs across all model-relevant input sources
    intersection_ids = set(base_df[id_col].dropna().tolist())

    model_inputs = []
    for mpath in model_paths:
        selected_df, is_featurized, source_key = _choose_input_df_for_model(
            model_path=mpath,
            input_dfs=input_dfs,
            raw_df_key=raw_df_key,
        )

        selected_ids = set(selected_df[id_col].dropna().tolist())
        if base == "intersection":
            intersection_ids &= selected_ids

        model_inputs.append((mpath, selected_df, is_featurized, source_key))

    if base == "intersection":
        base_df = base_df[base_df[id_col].isin(intersection_ids)].copy()
        base_df = base_df.drop_duplicates(subset=[id_col], keep="first")

    for mpath, selected_df, is_featurized, source_key in model_inputs:
        model_key = _sanitize_model_key(mpath)

        if base == "intersection":
            selected_df = selected_df[selected_df[id_col].isin(intersection_ids)].copy()

        pred_df = predict_fn(
            model_path=str(mpath),
            input_df=selected_df,
            id_col=id_col,
            smiles_col=smiles_col,
            response_col=None,
            is_featurized=is_featurized,
            **predict_kwargs,
        ).copy()

        if id_col not in pred_df.columns:
            raise ValueError(f"Model output for {mpath} does not contain '{id_col}'.")

        if base == "intersection":
            pred_df = pred_df[pred_df[id_col].isin(intersection_ids)].copy()

        resp_name, pred_col, std_col, actual_col = _infer_response_columns(pred_df, predicted_response)

        if resp_name_global is None:
            resp_name_global = resp_name
        elif resp_name_global != resp_name:
            raise ValueError(
                f"Different response names inferred across models: '{resp_name_global}' vs '{resp_name}'. "
                "Pass predicted_response explicitly to avoid ambiguity."
            )

        if actual_col_global is None and actual_col is not None:
            actual_col_global = actual_col

        keep_cols = [id_col, pred_col]
        if std_col is not None:
            keep_cols.append(std_col)
        if actual_col is not None:
            keep_cols.append(actual_col)

        model_part = pred_df[keep_cols].copy()
        model_part = _dedupe(model_part, id_col=id_col, how=dedupe)

        pred_out_col = f"pred_{model_key}"
        std_out_col = f"std_{model_key}" if std_col is not None else None

        rename_map = {pred_col: pred_out_col}
        if std_col is not None:
            rename_map[std_col] = std_out_col

        model_part = model_part.rename(columns=rename_map)
        per_model_wide_parts.append(model_part)

        long_tmp = model_part[[id_col, pred_out_col]].copy()
        long_tmp = long_tmp.rename(columns={pred_out_col: "pred"})
        long_tmp["model_key"] = model_key
        long_tmp["input_source"] = source_key
        long_tmp["is_featurized"] = is_featurized
        if std_out_col is not None and std_out_col in model_part.columns:
            long_tmp["std"] = model_part[std_out_col]
        long_parts.append(long_tmp)

    if base == "input":
        wide = base_df.copy()
        for part in per_model_wide_parts:
            wide = wide.merge(part, on=id_col, how="left")
    elif base == "union":
        wide = reduce(lambda left, right: left.merge(right, on=id_col, how="outer"), [base_df] + per_model_wide_parts)
    else:  # intersection
        wide = base_df.copy()
        for part in per_model_wide_parts:
            wide = wide.merge(part, on=id_col, how="inner")

    pred_cols = [c for c in wide.columns if c.startswith("pred_")]
    std_cols = [c for c in wide.columns if c.startswith("std_")]

    pred_matrix = wide[pred_cols].to_numpy(dtype=float)
    wide["committee_n_models"] = np.sum(~np.isnan(pred_matrix), axis=1)
    wide["committee_mean_pred"] = np.nanmean(pred_matrix, axis=1)
    wide["committee_std"] = np.nanstd(pred_matrix, axis=1, ddof=0)
    wide["committee_range"] = np.nanmax(pred_matrix, axis=1) - np.nanmin(pred_matrix, axis=1)

    if std_cols:
        aleatoric = wide[std_cols].to_numpy(dtype=float)
        wide["committee_mean_model_std"] = np.nanmean(aleatoric, axis=1)
    else:
        wide["committee_mean_model_std"] = np.nan

    if disagreement_metric not in wide.columns:
        raise ValueError(
            f"disagreement_metric '{disagreement_metric}' not found. "
            f"Available committee columns include: committee_std, committee_range, committee_n_models."
        )

    wide = wide.sort_values(by=disagreement_metric, ascending=not sort_desc).reset_index(drop=True)

    if return_long:
        long_df = pd.concat(long_parts, ignore_index=True)
        if base == "intersection":
            long_df = long_df[long_df[id_col].isin(set(wide[id_col]))].copy()
        return wide, long_df

    return wide

@dataclass(frozen=True)
class QBCColumns:
    pred: str = "committee_mean_pred"
    committee_std: str = "committee_std"
    mean_model_std: str = "committee_mean_model_std"


UncertaintyMode = Literal["committee", "combined", "model"]


def _robust_z(x: pd.Series, med: float, iqr: float) -> pd.Series:
    # Robust z-score using IQR, guard for iqr ~ 0
    denom = iqr if (iqr is not None and np.isfinite(iqr) and iqr > 0) else 1.0
    return (x - med) / denom


def _compute_uncertainty(
    df: pd.DataFrame,
    *,
    cols: QBCColumns,
    mode: UncertaintyMode,
    out_col: str = "uncertainty",
) -> pd.DataFrame:
    out = df.copy()

    if mode in ("committee", "combined"):
        if cols.committee_std not in out.columns:
            raise KeyError(f"Missing required column: {cols.committee_std}")
        committee = pd.to_numeric(out[cols.committee_std], errors="coerce")
    else:
        committee = None

    if mode in ("model", "combined"):
        if cols.mean_model_std not in out.columns:
            if mode == "model":
                raise KeyError(f"Missing required column: {cols.mean_model_std}")
            model = None
        else:
            model = pd.to_numeric(out[cols.mean_model_std], errors="coerce")
    else:
        model = None

    if mode == "committee":
        out[out_col] = committee
    elif mode == "model":
        out[out_col] = model
    elif mode == "combined":
        out[out_col] = committee if model is None else np.sqrt(np.square(committee) + np.square(model))
    else:
        raise ValueError(f"Unknown uncertainty mode: {mode}")

    return out


def select_high_pred_high_certainty(
    df: pd.DataFrame,
    *,
    n: int = 48,
    cols: QBCColumns = QBCColumns(),
    id_col: str = "compound_id",
    uncertainty_mode: UncertaintyMode = "combined",
    uncertainty_col: str = "uncertainty",
    # “High prediction” and “high certainty” definitions (global quantiles)
    pred_thr: float = 6,
    unc_max_quantile: float = 0.20,
    # Candidate pool control (keeps it fast on 400k rows)
    pool_mult: int = 300,
    max_pool: int = 200_000,
    # Ranking tradeoff (on robust z-scales)
    alpha: float = 1.5,
) -> pd.DataFrame:
    """
    Function 1: pick ~top n compounds with high pIC50 and high certainty.

    Steps:
      1) Pull top pool by prediction using nlargest (fast).
      2) Filter to pred >= global Q(pred_min_quantile)
      3) Filter to uncertainty <= global Q(unc_max_quantile)
      4) Rank by robust score: z_pred - alpha*z_unc
      5) Return top n

    If filters are too strict to yield n, it will expand the pool up to max_pool.
    """
    if cols.pred not in df.columns:
        raise KeyError(f"Missing required column: {cols.pred}")
    if id_col not in df.columns:
        raise KeyError(f"Missing required column: {id_col}")
    if n <= 0:
        raise ValueError("n must be positive")

    base = df.copy()
    base[cols.pred] = pd.to_numeric(base[cols.pred], errors="coerce")

    # Global thresholds (computed once)
    pred_all = base[cols.pred]

    base_u = _compute_uncertainty(base, cols=cols, mode=uncertainty_mode, out_col=uncertainty_col)
    unc_all = pd.to_numeric(base_u[uncertainty_col], errors="coerce")
    unc_thr = unc_all.quantile(unc_max_quantile)

    # Global robust scaling stats (for stable scoring)
    pred_med = float(pred_all.quantile(0.50))
    pred_iqr = float(pred_all.quantile(0.75) - pred_all.quantile(0.25))
    unc_med = float(unc_all.quantile(0.50))
    unc_iqr = float(unc_all.quantile(0.75) - unc_all.quantile(0.25))

    pool_size = min(len(base_u), max(n * pool_mult, n))
    pool_size = min(pool_size, max_pool)

    while True:
        pool = base_u.nlargest(pool_size, cols.pred).copy()

        pred = pd.to_numeric(pool[cols.pred], errors="coerce")
        unc = pd.to_numeric(pool[uncertainty_col], errors="coerce")

        mask = (pred >= pred_thr) & (unc <= unc_thr) & np.isfinite(pred) & np.isfinite(unc)
        cand = pool.loc[mask].copy()

        if len(cand) >= n or pool_size >= min(max_pool, len(base_u)):
            if len(cand) == 0:
                return cand  # empty, constraints too strict or data missing

            z_pred = _robust_z(pd.to_numeric(cand[cols.pred], errors="coerce"), pred_med, pred_iqr)
            z_unc = _robust_z(pd.to_numeric(cand[uncertainty_col], errors="coerce"), unc_med, unc_iqr)
            cand["selection_score"] = z_pred - alpha * z_unc

            cand = cand.sort_values(
                by=["selection_score", cols.pred, uncertainty_col],
                ascending=[False, False, True],
                kind="mergesort",
            )
            return cand.head(n)

        pool_size = min(pool_size * 2, max_pool)


def select_high_pred_medium_certainty(
    df: pd.DataFrame,
    *,
    n: int = 48,
    cols: QBCColumns = QBCColumns(),
    id_col: str = "compound_id",
    uncertainty_mode: UncertaintyMode = "combined",
    uncertainty_col: str = "uncertainty",
    # “High prediction” and “medium certainty” definitions (global quantiles)
    pred_min_quantile: float = 0.95,
    unc_band: Tuple[float, float] = (0.20, 0.60),
    pool_mult: int = 500,
    max_pool: int = 200_000,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """
    Function 2: pick ~top n compounds with high pIC50 and medium certainty.

    Medium certainty is enforced as a global uncertainty quantile band:
      Q(unc_band[0]) <= uncertainty <= Q(unc_band[1])

    Ranking still favors higher prediction, but penalizes uncertainty less than strict mode.
    """
    lo_q, hi_q = unc_band
    if not (0.0 <= lo_q < hi_q <= 1.0):
        raise ValueError("unc_band must satisfy 0 <= lo < hi <= 1")
    if cols.pred not in df.columns:
        raise KeyError(f"Missing required column: {cols.pred}")
    if id_col not in df.columns:
        raise KeyError(f"Missing required column: {id_col}")
    if n <= 0:
        raise ValueError("n must be positive")

    base = df.copy()
    base[cols.pred] = pd.to_numeric(base[cols.pred], errors="coerce")

    pred_all = base[cols.pred]
    pred_thr = pred_all.quantile(pred_min_quantile)

    base_u = _compute_uncertainty(base, cols=cols, mode=uncertainty_mode, out_col=uncertainty_col)
    unc_all = pd.to_numeric(base_u[uncertainty_col], errors="coerce")
    unc_lo = unc_all.quantile(lo_q)
    unc_hi = unc_all.quantile(hi_q)

    pred_med = float(pred_all.quantile(0.50))
    pred_iqr = float(pred_all.quantile(0.75) - pred_all.quantile(0.25))
    unc_med = float(unc_all.quantile(0.50))
    unc_iqr = float(unc_all.quantile(0.75) - unc_all.quantile(0.25))

    pool_size = min(len(base_u), max(n * pool_mult, n))
    pool_size = min(pool_size, max_pool)

    while True:
        pool = base_u.nlargest(pool_size, cols.pred).copy()

        pred = pd.to_numeric(pool[cols.pred], errors="coerce")
        unc = pd.to_numeric(pool[uncertainty_col], errors="coerce")

        mask = (pred >= pred_thr) & (unc >= unc_lo) & (unc <= unc_hi) & np.isfinite(pred) & np.isfinite(unc)
        cand = pool.loc[mask].copy()

        if len(cand) >= n or pool_size >= min(max_pool, len(base_u)):
            if len(cand) == 0:
                return cand

            z_pred = _robust_z(pd.to_numeric(cand[cols.pred], errors="coerce"), pred_med, pred_iqr)
            z_unc = _robust_z(pd.to_numeric(cand[uncertainty_col], errors="coerce"), unc_med, unc_iqr)
            cand["selection_score"] = z_pred - alpha * z_unc

            cand = cand.sort_values(
                by=["selection_score", cols.pred, uncertainty_col],
                ascending=[False, False, True],
                kind="mergesort",
            )
            return cand.head(n)

        pool_size = min(pool_size * 2, max_pool)


def plot_qbc_overview(
    df: pd.DataFrame,
    *,
    cols: QBCColumns = QBCColumns(),
    uncertainty_mode: UncertaintyMode = "combined",
    uncertainty_col: str = "uncertainty",
    bins: int = 250,
    clip_quantiles: Tuple[float, float] = (0.001, 0.999),
    log_counts: bool = True,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Function 3: scalable visualization of prediction vs uncertainty.
    Uses a 2D histogram (no scatter), suitable for 400k rows.
    """
    if cols.pred not in df.columns:
        raise KeyError(f"Missing required column: {cols.pred}")

    work = df.copy()
    work[cols.pred] = pd.to_numeric(work[cols.pred], errors="coerce")
    work = _compute_uncertainty(work, cols=cols, mode=uncertainty_mode, out_col=uncertainty_col)

    x = pd.to_numeric(work[cols.pred], errors="coerce").to_numpy()
    y = pd.to_numeric(work[uncertainty_col], errors="coerce").to_numpy()

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        raise ValueError("No finite data points to plot.")

    x_lo, x_hi = np.quantile(x, clip_quantiles)
    y_lo, y_hi = np.quantile(y, clip_quantiles)

    x = np.clip(x, x_lo, x_hi)
    y = np.clip(y, y_lo, y_hi)

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_lo, x_hi], [y_lo, y_hi]])
    H = H.T

    H_plot = np.log10(H + 1.0) if log_counts else H
    cbar_label = "log10(count + 1)" if log_counts else "count"

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    im = ax_main.imshow(
        H_plot,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
    )

    ax_main.set_xlabel(cols.pred)
    ax_main.set_ylabel(f"{uncertainty_col} ({uncertainty_mode})")
    ax_main.set_title("QBC overview: prediction vs uncertainty (binned)")

    ax_top.hist(x, bins=bins, range=(x_lo, x_hi), color="gray")
    ax_right.hist(y, bins=bins, range=(y_lo, y_hi), orientation="horizontal", color="gray")

    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    cbar = fig.colorbar(im, ax=[ax_main, ax_right], fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label)

    return fig


def select_diverse_subset(
    ranked_df: pd.DataFrame,
    *,
    n: int = 48,
    smiles_col: str = "base_rdkit_smiles",
    score_col: str = "selection_score",
    id_col: str = "compound_id",
    max_candidates: int = 20000,
    max_tanimoto: float = 0.35,
    fp_radius: int = 2,
    fp_nbits: int = 2048,
    invalid_smiles: Literal["drop", "keep_no_fp"] = "drop",
    fallback: Literal["relax", "fill"] = "relax",
    relax_step: float = 0.05,
    relax_max: float = 0.85,
) -> pd.DataFrame:
    """
    Greedy diversity filter on a ranked list.

    Parameters
    ----------
    ranked_df:
      Must include smiles_col and score_col. Should already be sorted high-to-low by score_col,
      or at least contain score_col for sorting.
    n:
      Number of diverse compounds to return.
    max_candidates:
      Only consider the top max_candidates rows by score_col (keeps it fast).
    max_tanimoto:
      Diversity constraint, any selected pair must have Tanimoto similarity <= max_tanimoto.
      Smaller is more diverse.
    invalid_smiles:
      - "drop": remove rows with invalid SMILES (recommended)
      - "keep_no_fp": keep them, but they will not be considered for selection (no fingerprint)
    fallback:
      If the constraint is too strict to select n:
      - "relax": gradually increase max_tanimoto until n is reached or relax_max is hit
      - "fill": keep max_tanimoto fixed, then fill remaining slots with best-scoring leftovers
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if score_col not in ranked_df.columns:
        raise KeyError(f"Missing required column: {score_col}")
    if smiles_col not in ranked_df.columns:
        raise KeyError(f"Missing required column: {smiles_col}")
    if id_col not in ranked_df.columns:
        raise KeyError(f"Missing required column: {id_col}")

    # Work on a top-ranked pool for speed
    work = ranked_df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.sort_values(score_col, ascending=False, kind="mergesort")
    work = work.head(max_candidates).copy()

    # Build molecules and fingerprints
    mols = []
    fps = []
    valid_mask = np.ones(len(work), dtype=bool)

    for i, smi in enumerate(work[smiles_col].astype(str).tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask[i] = False
            mols.append(None)
            fps.append(None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_nbits)
        mols.append(mol)
        fps.append(fp)

    work["_fp"] = fps

    if invalid_smiles == "drop":
        work = work.loc[valid_mask].copy()

    # If everything got dropped, return empty
    if len(work) == 0:
        return work.drop(columns=["_fp"], errors="ignore")

    def _greedy_pick(curr_max_tanimoto: float) -> pd.Index:
        selected_idx = []
        selected_fps = []

        # iterate in rank order
        for idx, fp in zip(work.index, work["_fp"].tolist()):
            if fp is None:
                continue

            if not selected_fps:
                selected_idx.append(idx)
                selected_fps.append(fp)
                if len(selected_idx) >= n:
                    break
                continue

            sims = DataStructs.BulkTanimotoSimilarity(fp, selected_fps)
            if max(sims) <= curr_max_tanimoto:
                selected_idx.append(idx)
                selected_fps.append(fp)
                if len(selected_idx) >= n:
                    break

        return pd.Index(selected_idx)

    curr = max_tanimoto
    selected = _greedy_pick(curr)

    if len(selected) < n:
        if fallback == "relax":
            while len(selected) < n and curr < relax_max:
                curr = min(curr + relax_step, relax_max)
                selected = _greedy_pick(curr)

        if fallback == "fill" and len(selected) < n:
            # Fill remaining slots with best-scoring not-yet-selected, ignoring diversity.
            remaining = work.index.difference(selected)
            fill_needed = n - len(selected)
            selected = selected.append(remaining[:fill_needed])

    out = work.loc[selected].copy()
    out["diversity_max_tanimoto_used"] = curr
    out = out.drop(columns=["_fp"], errors="ignore")

    # Return in final score order (highest first)
    out = out.sort_values(score_col, ascending=False, kind="mergesort").head(n)
    return out