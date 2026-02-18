"""Parametric baseline hazard / cumulative-hazard / inverse-cumulative-hazard.

All functions are vectorised over *t* (or *H*) via numpy broadcasting.
"""

from __future__ import annotations

import numpy as np


# ── Weibull PH parametrisation ───────────────────────────────────────
#  h_0(t) = a * lam * t^(a-1)
#  H_0(t) = lam * t^a
#  S_0(t) = exp(-lam * t^a)

def weibull_hazard(t: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """h_0(t) = shape * scale * t^{shape-1}."""
    return shape * scale * np.power(t, shape - 1)


def weibull_cumulative_hazard(t: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """H_0(t) = scale * t^{shape}."""
    return scale * np.power(t, shape)


def weibull_inverse_cumulative_hazard(H: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """H_0^{-1}(H) = (H / scale)^{1/shape}."""
    return np.power(H / scale, 1.0 / shape)


# ── Gompertz ─────────────────────────────────────────────────────────
#  h_0(t) = b * exp(c*t)
#  H_0(t) = (b/c) * (exp(c*t) - 1)
#  S_0(t) = exp(-(b/c)(exp(c*t) - 1))

def gompertz_hazard(t: np.ndarray, b: float, c: float) -> np.ndarray:
    """h_0(t) = b * exp(c * t)."""
    return b * np.exp(c * t)


def gompertz_cumulative_hazard(t: np.ndarray, b: float, c: float) -> np.ndarray:
    """H_0(t) = (b / c) * (exp(c*t) - 1)."""
    return (b / c) * (np.exp(c * t) - 1.0)


def gompertz_inverse_cumulative_hazard(H: np.ndarray, b: float, c: float) -> np.ndarray:
    """H_0^{-1}(H) = (1/c) * log(1 + (c/b) * H).

    Returns np.inf when the argument of log is non-positive (should not
    happen for valid H >= 0, b > 0, c > 0).
    """
    arg = 1.0 + (c / b) * H
    out = np.full_like(H, np.inf, dtype=float)
    mask = arg > 0
    out[mask] = np.log(arg[mask]) / c
    return out


# ── Exponential (Weibull with shape=1) ───────────────────────────────

def exponential_cumulative_hazard(t: np.ndarray, rate: float) -> np.ndarray:
    """H_0(t) = rate * t."""
    return rate * t


def exponential_inverse_cumulative_hazard(H: np.ndarray, rate: float) -> np.ndarray:
    """H_0^{-1}(H) = H / rate."""
    return H / rate
