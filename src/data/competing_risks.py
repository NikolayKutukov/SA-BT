"""Multi-state simulation for Setup 4: Healthy → Relapse → Death + Healthy → Death.

Uses the latent failure-time framework: generate independent cause-specific
Weibull PH event times, take the minimum.  For subjects who relapse, a second
Weibull PH draw gives the Relapse → Death transition time (clock-reset /
semi-Markov).
"""

from __future__ import annotations

import numpy as np

from .event_times import sample_event_times_ph_weibull


def simulate_multistate(
    X: np.ndarray,
    betas_01: np.ndarray,
    betas_02: np.ndarray,
    betas_12: np.ndarray,
    shape_01: float, scale_01: float,
    shape_02: float, scale_02: float,
    shape_12: float, scale_12: float,
    rng: np.random.Generator,
) -> dict:
    """Simulate multi-state trajectories.

    Returns a dict with:
        T_first      : (n,) time of first event from Healthy
        cause_first  : (n,) 1 = relapse, 2 = direct death
        T_death      : (n,) time of death (via either pathway)
        relapse_mask : (n,) bool, True if subject relapsed
        state_history: list of per-subject dicts with trajectory info
    """
    n = X.shape[0]

    # Latent times from Healthy
    T_01 = sample_event_times_ph_weibull(X, betas_01, shape_01, scale_01, rng)
    T_02 = sample_event_times_ph_weibull(X, betas_02, shape_02, scale_02, rng)

    T_first = np.minimum(T_01, T_02)
    cause_first = np.where(T_01 < T_02, 1, 2)  # 1=relapse, 2=direct death

    # For relapsers: Relapse → Death (clock resets at relapse)
    relapse_mask = cause_first == 1
    n_relapse = relapse_mask.sum()

    T_death = np.copy(T_first)  # default: death at first event (direct deaths)

    if n_relapse > 0:
        T_12 = sample_event_times_ph_weibull(
            X[relapse_mask], betas_12, shape_12, scale_12, rng,
        )
        T_death[relapse_mask] = T_first[relapse_mask] + T_12

    # Per-subject trajectory records
    state_history = []
    for i in range(n):
        if cause_first[i] == 1:
            state_history.append({
                "path": "Healthy → Relapse → Death",
                "t_relapse": float(T_first[i]),
                "t_death": float(T_death[i]),
            })
        else:
            state_history.append({
                "path": "Healthy → Death",
                "t_death": float(T_death[i]),
            })

    return {
        "T_first": T_first,
        "cause_first": cause_first,
        "T_death": T_death,
        "relapse_mask": relapse_mask,
        "state_history": state_history,
    }
