"""Per-model hyperparameter search spaces and random sampling.

Search budgets follow the HP tuning guideline:
  - Classical models (CoxPH, CoxNet, RSF): up to 30 configs
  - Deep models (DeepSurv, DeepHit): 15 configs
  - Transformer (SurvTRACE): 10 configs
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ── Search budgets (max configs per model) ───────────────────────────

SEARCH_BUDGETS: dict[str, int] = {
    "cox_ph": 5,
    "cox_net": 30,
    "rsf": 30,
    "deep_surv": 15,
    "deep_hit": 15,
    "survtrace": 10,
}


# ── Search spaces ────────────────────────────────────────────────────
# Each value is a list of discrete choices or a tuple ("loguniform", lo, hi)
# for continuous log-uniform sampling.

SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "cox_ph": {
        "alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
        "ties": ["breslow", "efron"],
    },
    "cox_net": {
        "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha_min_ratio": ("loguniform", 0.001, 0.1),
        "n_alphas": [50, 100],
    },
    "rsf": {
        "n_estimators": [200, 500, 1000],
        "max_features": ["sqrt", 0.33, 0.5],
        "min_samples_leaf": [5, 10, 20],
        "max_depth": [None, 10, 20],
        "min_samples_split": [6, 10, 20],
    },
    "deep_surv": {
        "hidden_layers": [[64, 64], [128, 128], [64, 64, 64], [128, 128, 128], [256, 256]],
        "dropout": [0.0, 0.1, 0.2, 0.5],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [0, 1e-4, 1e-3],
        "batch_size": [128, 256],
    },
    "deep_hit": {
        "hidden_layers": [[64, 64], [128, 128], [64, 64, 64], [128, 128, 128]],
        "dropout": [0.0, 0.1, 0.2, 0.5],
        "lr": [1e-4, 3e-4, 1e-3],
        "alpha": [0.1, 0.2, 0.5],
        "sigma": [0.1, 0.5, 1.0],
        "batch_size": [128, 256],
    },
    "survtrace": {
        "num_hidden_layers": [2, 3, 4],
        "num_attention_heads": [2, 4, 8],
        "hidden_size": [32, 64, 128, 256],
        "intermediate_size": [64, 128, 256],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [5e-5, 1e-4, 3e-4],
    },
}


def _sample_param(spec: Any, rng: np.random.Generator) -> Any:
    """Sample a single parameter value from its specification."""
    if isinstance(spec, tuple) and spec[0] == "loguniform":
        _, lo, hi = spec
        log_val = rng.uniform(np.log(lo), np.log(hi))
        return float(np.exp(log_val))
    if isinstance(spec, list):
        idx = rng.integers(len(spec))
        return spec[idx]
    raise ValueError(f"Unknown param spec: {spec}")


def _is_valid_survtrace_config(config: dict) -> bool:
    """Check SurvTRACE constraint: hidden_size % num_attention_heads == 0."""
    hs = config.get("hidden_size")
    heads = config.get("num_attention_heads")
    if hs is not None and heads is not None:
        return hs % heads == 0
    return True


def sample_configs(
    model_name: str,
    n_configs: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Sample random HP configs for a given model.

    Parameters
    ----------
    model_name : key in SEARCH_SPACES (e.g. "cox_ph", "rsf").
    n_configs  : number of configs to sample (default: SEARCH_BUDGETS[model_name]).
    rng        : numpy Generator for reproducibility.

    Returns
    -------
    List of dicts, each ready to be passed as ``**kwargs`` to the model constructor.
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(SEARCH_SPACES)}"
        )

    if n_configs is None:
        n_configs = SEARCH_BUDGETS[model_name]
    if rng is None:
        rng = np.random.default_rng()

    space = SEARCH_SPACES[model_name]
    is_survtrace = model_name == "survtrace"

    configs: list[dict[str, Any]] = []
    max_attempts = n_configs * 10  # avoid infinite loop on constraint violations
    attempts = 0

    while len(configs) < n_configs and attempts < max_attempts:
        attempts += 1
        config = {k: _sample_param(v, rng) for k, v in space.items()}

        # Enforce SurvTRACE constraint
        if is_survtrace and not _is_valid_survtrace_config(config):
            continue

        # Avoid exact duplicates
        if config not in configs:
            configs.append(config)

    return configs
