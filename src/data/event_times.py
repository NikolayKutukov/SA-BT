"""Event-time sampling via inverse-CDF (Bender et al., 2005).

* Closed-form for Weibull PH and Gompertz PH.
* Numerical root-finding (brentq) for non-PH / time-varying effects.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


# ── Closed-form Weibull PH ──────────────────────────────────────────
# H(t|x) = scale * t^shape * exp(x @ beta)
# T = ( -log(U) / (scale * exp(x @ beta)) )^(1/shape)

def sample_event_times_ph_weibull(
    X: np.ndarray,
    betas: np.ndarray,
    shape: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Closed-form inverse CDF for Weibull PH model."""
    n = X.shape[0]
    U = rng.uniform(0.0, 1.0, size=n)
    lp = X @ betas  # linear predictor
    T = np.power(-np.log(U) / (scale * np.exp(lp)), 1.0 / shape)
    return T


# ── Closed-form Gompertz PH ─────────────────────────────────────────
# H(t|x) = (b/c)(exp(c*t) - 1) * exp(g(x))
# T = (1/c) * log(1 + c*(-log(U)) / (b * exp(g(x))))

def sample_event_times_ph_gompertz(
    X: np.ndarray,
    betas: np.ndarray,
    b: float,
    c: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Closed-form inverse CDF for Gompertz PH model."""
    n = X.shape[0]
    U = rng.uniform(0.0, 1.0, size=n)
    lp = X @ betas
    arg = 1.0 + c * (-np.log(U)) / (b * np.exp(lp))
    # Guard against non-positive argument (would mean T → ∞)
    T = np.where(arg > 0, np.log(arg) / c, 1e6)
    return T


def sample_event_times_risk_score(
    risk_scores: np.ndarray,
    shape: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weibull PH with pre-computed risk scores g(x_i).

    T_i = ( -log(U_i) / (scale * exp(g_i)) )^(1/shape)
    """
    n = risk_scores.shape[0]
    U = rng.uniform(0.0, 1.0, size=n)
    T = np.power(-np.log(U) / (scale * np.exp(risk_scores)), 1.0 / shape)
    return T


def sample_event_times_risk_score_gompertz(
    risk_scores: np.ndarray,
    b: float,
    c: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gompertz PH with pre-computed risk scores g(x_i).

    T_i = (1/c) * log(1 + c * (-log(U_i)) / (b * exp(g_i)))
    """
    n = risk_scores.shape[0]
    U = rng.uniform(0.0, 1.0, size=n)
    arg = 1.0 + c * (-np.log(U)) / (b * np.exp(risk_scores))
    T = np.where(arg > 0, np.log(arg) / c, 1e6)
    return T


# ── Numerical root-finding for non-PH (Setup 2) ─────────────────────
#
# h(t|x) = h_0(t) * exp( eta_static + beta(t) * x_tv )
# where beta(t) = tv_intercept + tv_slope * t
#
# H(t|x) = exp(eta_static) * integral_0^t h_0(s) * exp((b0 + b1*s) * x_tv) ds
#
# Solve H(t|x) = -log(U) via brentq.

def sample_event_times_non_ph(
    X: np.ndarray,
    static_betas: np.ndarray,
    tv_idx: int,
    tv_intercept: float,
    tv_slope: float,
    shape: float,
    scale: float,
    rng: np.random.Generator,
    t_max: float = 50.0,
) -> np.ndarray:
    """Numerical inverse-CDF for non-PH model with one time-varying coefficient.

    Uses scipy.integrate.quad for the cumulative hazard integral and
    scipy.optimize.brentq for root-finding.  Loops over samples — the
    main computational bottleneck.
    """
    n = X.shape[0]
    U = rng.uniform(0.0, 1.0, size=n)
    targets = -np.log(U)  # H(T_i | x_i) = target_i

    # Pre-compute static linear predictor (excluding tv covariate)
    betas_no_tv = static_betas.copy()
    betas_no_tv[tv_idx] = 0.0  # zero-out; handled via time-varying part
    eta_static = X @ betas_no_tv  # (n,)

    T = np.empty(n)
    for i in range(n):
        x_tv = X[i, tv_idx]
        exp_eta = np.exp(eta_static[i])

        def cumhaz(t: float) -> float:
            """H(t | x_i)."""
            if t <= 0:
                return 0.0

            def integrand(s: float) -> float:
                h0 = shape * scale * s ** (shape - 1)
                return h0 * np.exp((tv_intercept + tv_slope * s) * x_tv)

            val, _ = quad(integrand, 0, t, limit=100)
            return exp_eta * val

        def equation(t: float) -> float:
            return cumhaz(t) - targets[i]

        # Check if event can occur within t_max
        if cumhaz(t_max) < targets[i]:
            T[i] = t_max  # effectively censored by horizon
        else:
            try:
                T[i] = brentq(equation, 1e-10, t_max, xtol=1e-8)
            except ValueError:
                T[i] = t_max

    return T
