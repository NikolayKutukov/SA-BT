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
        return np.linspace(T.min() + 1e-6, T.max() - 1e-6, n_points)
    # Use percentiles to avoid extreme outliers
    lo = np.percentile(event_times, 5)
    hi = np.percentile(event_times, 95)
    return np.linspace(max(lo, 1e-6), hi, n_points)


def _ipcw_tau(
    T_train: np.ndarray, E_train: np.ndarray,
    T_test: np.ndarray, E_test: np.ndarray,
) -> float:
    """Upper time bound τ for IPCW-based metrics (Uno et al., 2011).

    IPCW estimands require G(τ) > 0, where G is the KM estimator of the
    censoring distribution fitted on the training data. The largest
    training event time is the natural τ: by construction G is strictly
    positive at any event time. We additionally cap by the largest test
    event time so the test set covers the evaluation interval (0, τ].
    """
    ev_tr = T_train[E_train.astype(bool)]
    ev_te = T_test[E_test.astype(bool)]
    if len(ev_tr) == 0 or len(ev_te) == 0:
        return float(min(T_train.max(), T_test.max()))
    return float(min(ev_tr.max(), ev_te.max()))


def _safe_times(
    T_train: np.ndarray, E_train: np.ndarray,
    T_test: np.ndarray, E_test: np.ndarray,
    n_points: int = 100,
) -> np.ndarray:
    """Evaluation time grid for time-resolved IPCW metrics (IBS, AUC).

    Upper bound is τ from ``_ipcw_tau``. Lower bound is the 5th
    percentile of training events to avoid near-zero hazard regions
    where Brier/AUC are numerically unstable.
    """
    ev_tr = T_train[E_train.astype(bool)]
    ev_te = T_test[E_test.astype(bool)]
    if len(ev_tr) == 0 or len(ev_te) == 0:
        return _default_times(T_test, E_test, n_points)
    lo = max(np.percentile(ev_tr, 5), np.percentile(ev_te, 5), 1e-6)
    hi = _ipcw_tau(T_train, E_train, T_test, E_test)
    if lo >= hi:
        return _default_times(T_test, E_test, n_points)
    return np.linspace(lo, hi, n_points)


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
        tau = _ipcw_tau(T_train, E_train, T_test, E_test)
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
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Calibration: observed vs predicted survival at a fixed time.

    Uses quantile-based binning (equal-sized groups by predicted probability)
    so that every bin is populated regardless of how the predictions cluster.

    Parameters
    ----------
    surv_probs_at_t : (n,) predicted S(time_point | X) for each subject.

    Returns
    -------
    predicted_means : (n_bins,) mean predicted survival per bin.
    observed_surv   : (n_bins,) KM-estimated observed survival per bin.
    slope           : calibration slope (ideal = 1.0).
    intercept       : calibration intercept (ideal = 0.0).
    """
    from lifelines import KaplanMeierFitter

    n = len(surv_probs_at_t)
    # Reduce bins if too few subjects to fill them
    n_bins = min(n_bins, n // 5)
    if n_bins < 2:
        return np.array([np.nan]), np.array([np.nan]), np.nan, np.nan

    # Quantile-based bin assignment: equal number of subjects per bin
    quantile_edges = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(surv_probs_at_t, quantile_edges)
    bin_indices = np.digitize(surv_probs_at_t, bin_boundaries[1:-1])

    predicted_means = np.full(n_bins, np.nan)
    observed_surv = np.full(n_bins, np.nan)
    kmf = KaplanMeierFitter()

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() < 3:
            continue
        predicted_means[i] = surv_probs_at_t[mask].mean()
        kmf.fit(T[mask], E[mask].astype(bool))
        observed_surv[i] = float(kmf.predict(time_point))

    # Calibration slope & intercept via linear regression on valid bins
    valid = ~(np.isnan(predicted_means) | np.isnan(observed_surv))
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(predicted_means[valid], observed_surv[valid], 1)
    else:
        slope, intercept = np.nan, np.nan

    return predicted_means, observed_surv, float(slope), float(intercept)


# ── single-metric (inner CV) ──────────────────────────────────────────

def compute_single_metric(
    model: Any,
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    metric: str = "c_index_ipcw",
    times: np.ndarray | None = None,
) -> float:
    """Compute a single metric for HP selection in inner CV.

    Lighter than ``evaluate_model`` — avoids computing metrics that
    are not needed for the selection criterion.
    """
    if metric == "c_index":
        risk = model.predict_risk(X_test)
        return concordance_index(T_test, E_test, risk)

    # All IPCW-based metrics need filtered test data
    mask = _ipcw_mask(T_test, T_train)
    X_f, T_f, E_f = X_test[mask], T_test[mask], E_test[mask]

    if metric == "c_index_ipcw":
        risk = model.predict_risk(X_f)
        return concordance_index_ipcw(T_train, E_train, T_f, E_f, risk)

    if metric == "ibs":
        if times is None:
            times = _safe_times(T_train, E_train, T_f, E_f)
        surv = model.predict_survival_function(X_f, times)
        return integrated_brier_score(T_train, E_train, T_f, E_f, surv, times)

    if metric == "mean_auc":
        if times is None:
            times = _safe_times(T_train, E_train, T_f, E_f)
        risk = model.predict_risk(X_f)
        _, mean_auc = time_dependent_auc(
            T_train, E_train, T_f, E_f, risk, times
        )
        return mean_auc

    raise ValueError(
        f"Unknown metric '{metric}'. Choose from: c_index, c_index_ipcw, ibs, mean_auc"
    )


# Higher is better for these metrics; lower is better for IBS
METRIC_DIRECTION: dict[str, bool] = {
    "c_index": True,
    "c_index_ipcw": True,
    "ibs": False,
    "mean_auc": True,
}


# ── convenience ──────────────────────────────────────────────────────

def _ipcw_mask(T_test: np.ndarray, T_train: np.ndarray) -> np.ndarray:
    """Boolean mask keeping test subjects with T <= max(T_train).

    sksurv's IPCW estimator cannot handle test observations that exceed
    the training time range.
    """
    return T_test <= T_train.max()


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
    risk = model.predict_risk(X_test)

    # C-index (Harrell) works on all data — no IPCW restriction
    results: dict[str, float] = {}
    results["c_index"] = concordance_index(T_test, E_test, risk)

    # IPCW-based metrics require T_test <= max(T_train)
    mask = _ipcw_mask(T_test, T_train)
    X_f, T_f, E_f, risk_f = X_test[mask], T_test[mask], E_test[mask], risk[mask]

    if times is None:
        times = _safe_times(T_train, E_train, T_f, E_f)

    surv_f = model.predict_survival_function(X_f, times)

    try:
        results["c_index_ipcw"] = concordance_index_ipcw(
            T_train, E_train, T_f, E_f, risk_f
        )
    except Exception:
        results["c_index_ipcw"] = np.nan

    try:
        results["ibs"] = integrated_brier_score(
            T_train, E_train, T_f, E_f, surv_f, times
        )
    except Exception:
        results["ibs"] = np.nan

    try:
        _, mean_auc = time_dependent_auc(
            T_train, E_train, T_f, E_f, risk_f, times
        )
        results["mean_auc"] = mean_auc
    except Exception:
        results["mean_auc"] = np.nan

    return results
