"""Dataset loaders for benchmark experiments.

Loads four standard survival analysis datasets and returns them
as SurvivalData objects with a unified interface.

Datasets
--------
- GBSG2 (n=686, p=8): breast cancer, clean low-dimensional
- METABRIC (n=1904, p=9): breast cancer, popular DL benchmark
- DLBCL (n=240, p=7399): gene expression, high-dimensional (p >> n)
- SUPPORT (n=8873, p=14): critically ill patients, large-n
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import SurvivalData


# ── GBSG2 ────────────────────────────────────────────────────────────

def load_gbsg2() -> SurvivalData:
    """German Breast Cancer Study Group 2.

    Low-dimensional clinical dataset (n=686, p=9 after one-hot).
    Source: scikit-survival built-in.
    """
    from sksurv.datasets import load_gbsg2 as _load

    X_df, y = _load()

    # One-hot (dummy) encoding for categorical features: each level gets its
    # own indicator column, with the first level dropped to avoid collinearity.
    # No assumption is made about ordering or spacing between levels.
    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns.tolist()
    X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dtype=np.float64)

    X = X_df.values.astype(np.float64)
    feature_names = list(X_df.columns)
    T = y["time"].astype(np.float64)
    E = y["cens"].astype(np.float64)

    return SurvivalData(
        X=X, T=T, E=E,
        feature_names=feature_names,
        dataset_name="gbsg2",
    )


# ── METABRIC ─────────────────────────────────────────────────────────

def load_metabric() -> SurvivalData:
    """METABRIC breast cancer dataset.

    Medium-sized clinical+genomic dataset (n=1904, p=9).
    Popular benchmark in deep learning survival papers.
    Source: pycox built-in.
    """
    from pycox.datasets import metabric

    df = metabric.read_df()

    T = df["duration"].values.astype(np.float64)
    E = df["event"].values.astype(np.float64)
    feature_cols = [c for c in df.columns if c not in ("duration", "event")]
    X = df[feature_cols].values.astype(np.float64)

    return SurvivalData(
        X=X, T=T, E=E,
        feature_names=feature_cols,
        dataset_name="metabric",
    )


# ── DLBCL ────────────────────────────────────────────────────────────

def load_dlbcl() -> SurvivalData:
    """Diffuse Large B-Cell Lymphoma gene expression dataset.

    High-dimensional omics dataset (n=240, p=7399).
    Classic benchmark for regularized / high-dimensional survival models.
    Source: SurvSet package.
    """
    from SurvSet.data import SurvLoader

    loader = SurvLoader()
    result = loader.load_dataset("DLBCL")
    df = result["df"]

    T = df["time"].values.astype(np.float64)
    E = df["event"].values.astype(np.float64)
    feature_cols = [c for c in df.columns if c not in ("pid", "time", "event")]
    X = df[feature_cols].values.astype(np.float64)

    # Handle NaNs: fill with column median
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]

    return SurvivalData(
        X=X, T=T, E=E,
        feature_names=feature_cols,
        dataset_name="dlbcl",
    )


# ── SUPPORT ──────────────────────────────────────────────────────────

def load_support() -> SurvivalData:
    """Study to Understand Prognoses Preferences Outcomes and Risks.

    Large clinical dataset (n=8873, p=14).
    Tests scalability and performance under heavier censoring.
    Source: pycox built-in.
    """
    from pycox.datasets import support

    df = support.read_df()

    T = df["duration"].values.astype(np.float64)
    E = df["event"].values.astype(np.float64)
    feature_cols = [c for c in df.columns if c not in ("duration", "event")]
    X = df[feature_cols].values.astype(np.float64)

    # Handle NaNs
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]

    return SurvivalData(
        X=X, T=T, E=E,
        feature_names=feature_cols,
        dataset_name="support",
    )


# ── Registry ─────────────────────────────────────────────────────────

LOADERS = {
    "gbsg2": load_gbsg2,
    "metabric": load_metabric,
    "dlbcl": load_dlbcl,
    "support": load_support,
}

# Which (dataset, model) pairs to skip
SKIP = {
    ("dlbcl", "CoxPH"),  # p=7399 >> n=240, unpenalised Cox fails
}
