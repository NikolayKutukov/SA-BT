"""Censoring mechanisms: uniform, exponential, administrative, covariate-dependent."""

from __future__ import annotations

import numpy as np


def apply_censoring(
    event_times: np.ndarray,
    censor_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-censor: T_obs = min(T_event, C), E = I(T_event <= C).

    Returns (observed_times, event_indicators).
    """
    T = np.minimum(event_times, censor_times)
    E = (event_times <= censor_times).astype(np.int32)
    return T, E


# ── censoring-time generators ────────────────────────────────────────

def generate_uniform_censoring(
    n: int, censor_max: float, rng: np.random.Generator,
) -> np.ndarray:
    """C_i ~ Uniform(0, censor_max).  Setups 1, 3."""
    return rng.uniform(0.0, censor_max, size=n)


def generate_exponential_censoring(
    n: int, rate: float, rng: np.random.Generator,
) -> np.ndarray:
    """C_i ~ Exp(rate), mean = 1/rate.  Setup 2."""
    return rng.exponential(1.0 / rate, size=n)


def generate_administrative_censoring(n: int, admin_time: float) -> np.ndarray:
    """C_i = tau (fixed).  Used as a component in Setups 4, 5."""
    return np.full(n, admin_time)


def generate_combined_censoring(
    n: int,
    admin_time: float,
    additional_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """C_i = min(tau, C_random) where C_random ~ Exp(additional_rate).  Setup 4."""
    C_admin = generate_administrative_censoring(n, admin_time)
    C_rand = generate_exponential_censoring(n, additional_rate, rng)
    return np.minimum(C_admin, C_rand)


def generate_covariate_dependent_censoring(
    n: int,
    X: np.ndarray,
    covariate_idx: int,
    base_rate: float,
    covariate_effect: float,
    admin_time: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Covariate-dependent censoring for Setup 5.

    C_i ~ Exp(rate_i) where rate_i = base_rate * exp(covariate_effect * X[i, idx])
    Then C_i = min(C_i, admin_time).
    """
    rates = base_rate * np.exp(covariate_effect * X[:, covariate_idx])
    C = rng.exponential(1.0 / rates)
    return np.minimum(C, admin_time)
