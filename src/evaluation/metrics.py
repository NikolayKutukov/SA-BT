"""Survival analysis evaluation metrics.

Thin wrappers around scikit-survival metric functions that accept
plain numpy arrays (matching the SurvivalModel interface).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw as _concordance_index_ipcw,
    integrated_brier_score as _integrated_brier_score,
    cumulative_dynamic_auc as _cumulative_dynamic_auc,
)
from sksurv.util import Surv


# ── helpers ──────────────────────────────────────────────────────────

def _make_structured(T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Convert (T, E) to scikit-survival structured array."""
    return Surv.from_arrays(E.astype(bool), T)


def _default_times(T: np.ndarray, E: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Equidistant evaluation times between first and last observed event."""
    event_times = T[E.astype(bool)]
    if len(event_times) == 0:
        return np.linspace(T.min(), T.max(), n_points)
    lo, hi = event_times.min(), event_times.max()
    # Shrink slightly to stay inside observed range (required by IBS/AUC)
    return np.linspace(lo + 1e-6, hi - 1e-6, n_points)


# ── metrics ──────────────────────────────────────────────────────────

def concordance_index(
    T: np.ndarray,
    E: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Harrell's concordance index."""
    c, *_ = concordance_index_censored(E.astype(bool), T, risk_scores)
    return float(c)


def concordance_index_ipcw(
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    risk_scores: np.ndarray,
    tau: float | None = None,
) -> float:
    """Inverse-probability-of-censoring-weighted C-index.

    Especially important under heavy censoring (Setup 5).
    """
    y_train = _make_structured(T_train, E_train)
    y_test = _make_structured(T_test, E_test)
    if tau is None:
        tau = T_test[E_test.astype(bool)].max()
    c, *_ = _concordance_index_ipcw(y_train, y_test, risk_scores, tau=tau)
    return float(c)


def integrated_brier_score(
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    surv_probs: np.ndarray,
    times: np.ndarray | None = None,
) -> float:
    """Integrated Brier Score (IBS).

    Parameters
    ----------
    surv_probs : (n_test, len(times)) predicted S(t|X).
    times      : evaluation time grid.  Auto-computed if None.
    """
    y_train = _make_structured(T_train, E_train)
    y_test = _make_structured(T_test, E_test)
    if times is None:
        times = _default_times(T_test, E_test)
    return float(_integrated_brier_score(y_train, y_test, surv_probs, times))


def time_dependent_auc(
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    risk_scores: np.ndarray,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Cumulative/dynamic time-dependent AUC.

    Returns
    -------
    auc_values : (len(times),) AUC at each time point.
    mean_auc   : scalar, mean AUC across the time grid.
    """
    y_train = _make_structured(T_train, E_train)
    y_test = _make_structured(T_test, E_test)
    if times is None:
        times = _default_times(T_test, E_test)
    auc_values, mean_auc = _cumulative_dynamic_auc(
        y_train, y_test, risk_scores, times
    )
    return auc_values, float(mean_auc)


def calibration_curve(
    T: np.ndarray,
    E: np.ndarray,
    surv_probs_at_t: np.ndarray,
    time_point: float,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration: observed vs predicted survival at a fixed time.

    Uses a simple binning approach (Kaplan-Meier within predicted-risk groups).

    Parameters
    ----------
    surv_probs_at_t : (n,) predicted S(time_point | X) for each subject.

    Returns
    -------
    predicted_means : (n_bins,) mean predicted survival per bin.
    observed_surv   : (n_bins,) KM-estimated observed survival per bin.
    """
    from lifelines import KaplanMeierFitter

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    predicted_means = np.zeros(n_bins)
    observed_surv = np.zeros(n_bins)
    kmf = KaplanMeierFitter()

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (surv_probs_at_t >= lo) & (surv_probs_at_t <= hi)
        else:
            mask = (surv_probs_at_t >= lo) & (surv_probs_at_t < hi)

        if mask.sum() == 0:
            predicted_means[i] = (lo + hi) / 2
            observed_surv[i] = np.nan
            continue

        predicted_means[i] = surv_probs_at_t[mask].mean()
        kmf.fit(T[mask], E[mask].astype(bool))
        # Evaluate KM at the time point
        surv_at_t = kmf.predict(time_point)
        observed_surv[i] = float(surv_at_t.iloc[0]) if len(surv_at_t) > 0 else np.nan

    return predicted_means, observed_surv


# ── convenience ──────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    times: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all standard metrics for a fitted single-event model.

    Returns a dict with keys: c_index, c_index_ipcw, ibs, mean_auc.
    """
    if times is None:
        times = _default_times(T_test, E_test)

    risk = model.predict_risk(X_test)
    surv = model.predict_survival_function(X_test, times)

    results: dict[str, float] = {}
    results["c_index"] = concordance_index(T_test, E_test, risk)

    try:
        results["c_index_ipcw"] = concordance_index_ipcw(
            T_train, E_train, T_test, E_test, risk
        )
    except Exception:
        results["c_index_ipcw"] = np.nan

    try:
        results["ibs"] = integrated_brier_score(
            T_train, E_train, T_test, E_test, surv, times
        )
    except Exception:
        results["ibs"] = np.nan

    try:
        _, mean_auc = time_dependent_auc(
            T_train, E_train, T_test, E_test, risk, times
        )
        results["mean_auc"] = mean_auc
    except Exception:
        results["mean_auc"] = np.nan

    return results
