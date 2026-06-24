"""Synthetic data generators for interpretability validation.

Adapted from Krzyzinski et al., "SurvSHAP(t): Time-dependent explanations
of machine learning survival models" (Knowledge-Based Systems 262, 2023;
arXiv:2208.11080), file `paper/other_codes/data_generation.R` in the
MI2DataLab/survshap repository.

Two scenarios — both share the **same** feature distribution so that
methods can be compared apples-to-apples:

    x0, x1 ~ Bernoulli(0.5)        (binary, important)
    x2     ~ N(10, 2)              (continuous, important)
    x3     ~ N(20, 4)              (continuous, important)
    x4..x7 ~ N(0, 1)               (noise — true effect 1e-6, not 0,
                                    to avoid degenerate gradients in
                                    SurvSHAP renormalization)

1. generate_interpretability_data()   — Cox PH, constant betas.
2. generate_tv_interpretability_data() — Non-PH, x1 and x3 are time-varying.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .types import SurvivalData


# Shared feature distribution (paper Exp.1 layout, p=8) ───────────────
def _generate_features(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw n samples of the shared 8-feature design matrix."""
    X = np.empty((n, 8))
    X[:, 0] = rng.binomial(1, 0.5, size=n).astype(float)   # Bernoulli
    X[:, 1] = rng.binomial(1, 0.5, size=n).astype(float)   # Bernoulli
    X[:, 2] = rng.normal(10.0, 2.0, size=n)                # N(10, 2)
    X[:, 3] = rng.normal(20.0, 4.0, size=n)                # N(20, 4)
    X[:, 4:] = rng.standard_normal((n, 4))                 # noise
    return X


# Censoring scheme (paper, common across both DGMs) ──────────────────
def _apply_censoring(
    rng: np.random.Generator,
    T_event: np.ndarray,
    admin_lo: float,
    admin_hi: float,
    random_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Administrative censoring U(admin_lo, admin_hi) ∪ random U(0, random_max)."""
    n = len(T_event)
    C_admin = rng.uniform(admin_lo, admin_hi, size=n)
    C_rand = rng.uniform(0.0, random_max, size=n)
    C = np.minimum(C_admin, C_rand)
    T_obs = np.minimum(T_event, C)
    E = (T_event <= C).astype(np.float64)
    return T_obs, E


def generate_interpretability_data(
    n: int = 1000,
    seed: int = 42,
    h0: float = 0.08,
    admin_lo: float = 11.0,
    admin_hi: float = 16.0,
    random_max: float = 24.0,
) -> SurvivalData:
    """Cox PH dataset with constant log-hazard ratios — sanity check for SurvLIME.

    Linear predictor (paper Exp.1, non-time-dependent variant)
    ----------------------------------------------------------
        η = 1.0·x0 + 0.5·x1 − 0.2·x2 + 0.1·x3 + 1e-6·(x4 + x5 + x6 + x7)

    Coefficients are scaled per feature distribution so |β·E[x]| stays O(1):
    Bernoulli features get O(1) betas, N(10,2) gets β=−0.2, N(20,4) gets β=0.1.

    Baseline hazard: h0(t) = 0.08 (paper "exponential" variant).
    Sampling: closed-form T = −log(U) / (h0 · exp(η)).

    Censoring (paper recipe): administrative U(11, 16) ∩ random U(0, 24).
    """
    rng = np.random.default_rng(seed)

    feature_names = [f"x{i}" for i in range(8)]
    true_betas = np.array([1.0, 0.5, -0.2, 0.1, 1e-6, 1e-6, 1e-6, 1e-6])

    X = _generate_features(rng, n)
    eta = X @ true_betas

    U = rng.uniform(size=n)
    T_event = -np.log(U) / (h0 * np.exp(eta))
    T_obs, E = _apply_censoring(rng, T_event, admin_lo, admin_hi, random_max)

    return SurvivalData(
        X=X,
        T=T_obs,
        E=E,
        feature_names=feature_names,
        dataset_name="synthetic_interpretability",
        true_betas=true_betas,
    )


def generate_tv_interpretability_data(
    n: int = 1000,
    seed: int = 42,
    h0: float = 0.08,
    admin_lo: float = 11.0,
    admin_hi: float = 16.0,
    random_max: float = 24.0,
) -> SurvivalData:
    """Non-PH dataset with one time-varying coefficient — stress test for SurvSHAP(t).

    Hazard
    ------
        h(t|x) = h0 · exp(A(x) + B(x)·t)

    Constant part:
        A(x) =  1.0·x0 − 1.0·x1 − 0.2·x2 + 0.1·x3 + 1e-6·(x4+x5+x6+x7)

    Time-slope part (encodes β1(t)):
        B(x) =  2.0·x1                          ∈ {0, 2}  (always ≥ 0 — no cure fraction)

    Equivalently, the per-feature coefficients are:
        β0    =  1.0                        (constant)
        β1(t) = −1.0 + 2.0·t                (TV; sign change at t = 0.5; on Bernoulli x1)
        β2    = −0.2                        (constant)
        β3    =  0.1                        (constant; β3 kept non-TV because making it
                                             time-varying with N(20,4)-scale x3 would push
                                             B(x) strongly negative → cure fraction → 95%+
                                             censoring.  Single TV feature is enough to
                                             showcase SurvSHAP's time dimension.)
        β4..7 =  1e-6                       (noise)

    Closed-form cumulative hazard:
        H(t|x) = h0 · exp(A) · (exp(B·t) − 1) / B    [B > 0  → x1 = 1]
                 h0 · exp(A) · t                       [B = 0  → x1 = 0]

    Event times: Brent root-finding on H(T) = −log(U), bracket [1e-8, 20]
    (paper convention).  Censoring: same recipe as the PH dataset.
    """
    rng = np.random.default_rng(seed)

    feature_names = [f"x{i}" for i in range(8)]
    # true_betas stores t=0 values; time slope of β1 is in the docstring.
    true_betas = np.array([1.0, -1.0, -0.2, 0.1, 1e-6, 1e-6, 1e-6, 1e-6])

    X = _generate_features(rng, n)
    noise_sum = X[:, 4:].sum(axis=1)

    A = 1.0 * X[:, 0] - 1.0 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * X[:, 3] + 1e-6 * noise_sum
    B = 2.0 * X[:, 1]

    def cumhaz(t: float, i: int) -> float:
        if abs(B[i]) < 1e-10:
            return h0 * np.exp(A[i]) * t
        return h0 * np.exp(A[i]) * (np.exp(B[i] * t) - 1.0) / B[i]

    U = rng.uniform(size=n)
    targets = -np.log(U)
    T_event = np.empty(n)
    T_MAX = 20.0  # paper convention
    for i in range(n):
        tgt = targets[i]
        if cumhaz(T_MAX, i) < tgt:
            # Cure fraction or improper survival within bracket — censor at T_MAX.
            T_event[i] = T_MAX
        else:
            T_event[i] = brentq(lambda t: cumhaz(t, i) - tgt, 1e-8, T_MAX)

    T_obs, E = _apply_censoring(rng, T_event, admin_lo, admin_hi, random_max)

    return SurvivalData(
        X=X,
        T=T_obs,
        E=E,
        feature_names=feature_names,
        dataset_name="synthetic_tv_interpretability",
        true_betas=true_betas,
    )
