"""Per-model hyperparameter search spaces and random sampling.

Search budgets:
  - CoxPH=3, CoxNet=5, RSF=5, DeepSurv=3
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ── Search budgets (max configs per model) ───────────────────────────

SEARCH_BUDGETS: dict[str, int] = {
    "cox_ph": 1,
    "cox_net": 30,
    "rsf": 15,
    "deep_surv": 50,
}


# ── Search spaces ────────────────────────────────────────────────────

SEARCH_SPACES: dict[str, dict[str, Any]] = {
    # Unpenalised Cox PH has no hyperparameters: it is the methodological
    # baseline. The ridge-penalty role is covered by CoxNet (l1_ratio<1).
    "cox_ph": {},
    "cox_net": {
        "l1_ratio": [0.25, 0.5, 0.75, 1.0],
        "alpha_min_ratio": ("loguniform", 0.0001, 0.1),
        "n_alphas": [50, 100],
    },
    "rsf": {
        "n_estimators": [100, 200, 300, 500],
        "max_features": ["sqrt", 0.33, 0.5],
        "min_samples_leaf": [10, 15, 20],
        # "max_depth": [10, 15, 20, None],
        # "min_samples_split": [6, 10, 20],
    },
    "deep_surv": {
        "hidden_layers": [
            [8], [16], [32], [64], [128],
            [32, 32], [64, 64], [128, 128],
            [64, 64, 64], [128, 128, 128],
        ],
        "dropout": [0.0, 0.1, 0.2, 0.4, 0.6],
        "lr": ("loguniform", 1e-4, 1e-1),
        "weight_decay": ("loguniform", 1e-5, 1e-1),
        "epochs": [200],
        "batch_size": [64, 128, 256],
    },
}


def _sample_param(spec: Any, rng: np.random.Generator) -> Any:
    if isinstance(spec, tuple) and spec[0] == "loguniform":
        _, lo, hi = spec
        log_val = rng.uniform(np.log(lo), np.log(hi))
        return float(np.exp(log_val))
    if isinstance(spec, list):
        idx = rng.integers(len(spec))
        return spec[idx]
    raise ValueError(f"Unknown param spec: {spec}")


def sample_configs(
    model_name: str,
    n_configs: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Sample random HP configs for a given model."""
    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(SEARCH_SPACES)}"
        )

    if n_configs is None:
        n_configs = SEARCH_BUDGETS[model_name]
    if rng is None:
        rng = np.random.default_rng()

    space = SEARCH_SPACES[model_name]

    configs: list[dict[str, Any]] = []
    max_attempts = n_configs * 10
    attempts = 0

    while len(configs) < n_configs and attempts < max_attempts:
        attempts += 1
        config = {k: _sample_param(v, rng) for k, v in space.items()}

        if config not in configs:
            configs.append(config)

    return configs
