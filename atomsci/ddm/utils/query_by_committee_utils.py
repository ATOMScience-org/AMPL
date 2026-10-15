from __future__ import annotations

import hashlib
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def add_murcko_scaffolds(
    df: pd.DataFrame,
    *,
    smiles_col: str = "rdkit_smiles",
    scaffold_col: str = "murcko_scaffold",
    generic: bool = False,
    drop_invalid: bool = False,
) -> pd.DataFrame:
    """
    Add a Murcko scaffold column to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a SMILES column.
    smiles_col : str
        Column containing SMILES.
    scaffold_col : str
        Name of output scaffold column.
    generic : bool
        If True, convert scaffolds to generic Murcko scaffolds.
    drop_invalid : bool
        If True, drop rows with invalid SMILES.

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with scaffold column added.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"Missing required column: {smiles_col}")

    out = df.copy()
    scaffolds = []
    valid_mask = []

    for smi in out[smiles_col].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds.append(np.nan)
            valid_mask.append(False)
            continue

        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)

        scaffold_smi = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else np.nan
        if scaffold_smi == "":
            scaffold_smi = np.nan

        scaffolds.append(scaffold_smi)
        valid_mask.append(True)

    out[scaffold_col] = scaffolds

    if drop_invalid:
        out = out.loc[valid_mask].copy()

    return out


def cluster_by_scaffold(
    df: pd.DataFrame,
    *,
    scaffold_col: str = "murcko_scaffold",
    id_col: str = "compound_id",
    sort_scaffolds_by_size: bool = True,
    drop_na_scaffold: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Return a dict mapping scaffold -> dataframe of compounds in that scaffold.
    """
    if scaffold_col not in df.columns:
        raise KeyError(f"Missing required column: {scaffold_col}")
    if id_col not in df.columns:
        raise KeyError(f"Missing required column: {id_col}")

    work = df.copy()
    if drop_na_scaffold:
        work = work.loc[work[scaffold_col].notna()].copy()

    grouped = []
    for scaffold, group in work.groupby(scaffold_col, dropna=False, sort=False):
        grouped.append((scaffold, group.copy()))

    if sort_scaffolds_by_size:
        grouped.sort(key=lambda x: len(x[1]), reverse=True)

    return {scaffold: group.reset_index(drop=True) for scaffold, group in grouped}


def scaffold_summary(
    df: pd.DataFrame,
    *,
    scaffold_col: str = "murcko_scaffold",
    pred_col: str = "committee_mean_pred",
    std_col: str = "committee_std",
    id_col: str = "compound_id",
    smiles_col: str = "rdkit_smiles",
    drop_na_scaffold: bool = True,
) -> pd.DataFrame:
    """
    Create a summary table with one row per scaffold.
    """
    required = [scaffold_col, id_col]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    work = df.copy()
    if drop_na_scaffold:
        work = work.loc[work[scaffold_col].notna()].copy()

    agg = {
        id_col: "count",
    }
    if pred_col in work.columns:
        agg[pred_col] = ["max", "mean", "median"]
    if std_col in work.columns:
        agg[std_col] = ["mean", "median"]
    if smiles_col in work.columns:
        agg[smiles_col] = "first"

    summary = work.groupby(scaffold_col).agg(agg)
    summary.columns = [
        "_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary = summary.reset_index()

    rename_map = {
        f"{id_col}_count": "n_compounds",
        f"{smiles_col}_first": "example_smiles",
    }
    summary = summary.rename(columns=rename_map)

    if "n_compounds" in summary.columns:
        summary = summary.sort_values("n_compounds", ascending=False, kind="mergesort")

    return summary.reset_index(drop=True)


def pick_one_per_scaffold(
    df: pd.DataFrame,
    scorer: Callable[[pd.DataFrame], pd.Series],
    *,
    scaffold_col: str = "murcko_scaffold",
    id_col: str = "compound_id",
    drop_na_scaffold: bool = True,
    score_col: str = "scaffold_selection_score",
) -> pd.DataFrame:
    """
    Pick one compound per scaffold using a scoring function.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, should already contain scaffold assignments.
    scorer : Callable[[pd.DataFrame], pd.Series]
        Function that takes the dataframe and returns a numeric score per row.
        Higher score is better.
    """
    if scaffold_col not in df.columns:
        raise KeyError(f"Missing required column: {scaffold_col}")
    if id_col not in df.columns:
        raise KeyError(f"Missing required column: {id_col}")

    work = df.copy()
    if drop_na_scaffold:
        work = work.loc[work[scaffold_col].notna()].copy()

    scores = pd.to_numeric(scorer(work), errors="coerce")
    if len(scores) != len(work):
        raise ValueError("scorer(df) must return a Series with the same length as df")

    work[score_col] = scores

    # Stable sort, then take best row per scaffold
    work = work.sort_values(
        by=[scaffold_col, score_col],
        ascending=[True, False],
        kind="mergesort",
    )

    picked = work.drop_duplicates(subset=[scaffold_col], keep="first").copy()
    picked = picked.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    return picked


def scorer_most_active(
    df: pd.DataFrame,
    *,
    pred_col: str = "committee_mean_pred",
) -> pd.Series:
    """
    Higher prediction is better.
    """
    if pred_col not in df.columns:
        raise KeyError(f"Missing required column: {pred_col}")
    return pd.to_numeric(df[pred_col], errors="coerce")


def scorer_pred_minus_k_std(
    df: pd.DataFrame,
    *,
    pred_col: str = "committee_mean_pred",
    std_col: str = "committee_std",
    k: float = 1.0,
) -> pd.Series:
    """
    Score = prediction - k * std, higher is better.
    """
    if pred_col not in df.columns:
        raise KeyError(f"Missing required column: {pred_col}")
    if std_col not in df.columns:
        raise KeyError(f"Missing required column: {std_col}")

    pred = pd.to_numeric(df[pred_col], errors="coerce")
    std = pd.to_numeric(df[std_col], errors="coerce")
    return pred - k * std


def _scaffold_to_color(scaffold: str, cmap_name: str = "tab20") -> tuple[float, float, float, float]:
    """
    Deterministic scaffold -> color mapping.
    """
    cmap = plt.get_cmap(cmap_name)
    digest = hashlib.md5(str(scaffold).encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % cmap.N
    return cmap(idx)


def plot_scaffold_prediction_uncertainty(
    df: pd.DataFrame,
    *,
    pred_col: str = "committee_mean_pred",
    std_col: str = "committee_std",
    scaffold_col: str = "murcko_scaffold",
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 0.8,
    s: float = 20,
    max_scaffolds_in_legend: int = 25,
    clip_quantiles: tuple[float, float] | None = (0.001, 0.999),
) -> plt.Figure:
    """
    Scatter plot of prediction vs uncertainty, colored by Murcko scaffold.

    For many scaffolds, the legend can become large, so only the top
    max_scaffolds_in_legend most frequent scaffolds are shown.
    """
    required = [pred_col, std_col, scaffold_col]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    work = df.copy()
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
    work[std_col] = pd.to_numeric(work[std_col], errors="coerce")
    work = work.loc[
        work[pred_col].notna() & work[std_col].notna() & work[scaffold_col].notna()
    ].copy()

    if len(work) == 0:
        raise ValueError("No valid rows to plot.")

    if clip_quantiles is not None:
        lo, hi = clip_quantiles
        x_lo, x_hi = work[pred_col].quantile([lo, hi])
        y_lo, y_hi = work[std_col].quantile([lo, hi])
        work[pred_col] = work[pred_col].clip(x_lo, x_hi)
        work[std_col] = work[std_col].clip(y_lo, y_hi)

    fig, ax = plt.subplots(figsize=figsize)

    scaffold_counts = work[scaffold_col].value_counts()
    legend_scaffolds = set(scaffold_counts.head(max_scaffolds_in_legend).index)

    for scaffold, group in work.groupby(scaffold_col, sort=False):
        color = _scaffold_to_color(scaffold)
        label = scaffold if scaffold in legend_scaffolds else None
        ax.scatter(
            group[pred_col],
            group[std_col],
            color=color,
            alpha=alpha,
            s=s,
            label=label,
        )

    ax.set_xlabel(pred_col)
    ax.set_ylabel(std_col)
    ax.set_title("Prediction vs uncertainty colored by Murcko scaffold")

    if max_scaffolds_in_legend > 0 and len(legend_scaffolds) > 0:
        ax.legend(
            title="Scaffold",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=False,
        )

    fig.tight_layout()
    return fig