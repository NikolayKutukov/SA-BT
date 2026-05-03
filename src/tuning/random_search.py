"""Inner cross-validation search for hyperparameter tuning."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import compute_single_metric, METRIC_DIRECTION


class InnerCVSearch:
    """Random search over HP configs using inner K-fold CV.

    For each candidate config, trains the model on each inner fold and
    computes a single metric on the held-out fold.  Selects the config
    with the best average score.
    """

    def __init__(
        self,
        model_class: type,
        model_name: str,
        n_inner_folds: int = 3,
        metric: str = "c_index_ipcw",
        seed: int = 42,
    ) -> None:
        self.model_class = model_class
        self.model_name = model_name
        self.n_inner_folds = n_inner_folds
        self.metric = metric
        self.higher_is_better = METRIC_DIRECTION.get(metric, True)
        self.seed = seed

    def search(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        param_configs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], float]:
        """Run inner CV search and return the best config."""
        stratify_col = E.astype(int)

        n_folds = min(self.n_inner_folds, _min_class_count(stratify_col))
        if n_folds < 2:
            n_folds = 2

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.seed,
        )
        fold_indices = list(skf.split(X, stratify_col))

        scores = []
        for config in param_configs:
            fold_scores = []
            for train_idx, val_idx in fold_indices:
                score = self._evaluate_config(
                    config, X, T, E, train_idx, val_idx,
                )
                fold_scores.append(score)

            valid = [s for s in fold_scores if not np.isnan(s)]
            mean_score = np.mean(valid) if valid else np.nan
            scores.append(mean_score)

        scores_arr = np.array(scores)
        valid_mask = ~np.isnan(scores_arr)

        if not valid_mask.any():
            return {}, np.nan

        if self.higher_is_better:
            best_idx = int(np.nanargmax(scores_arr))
        else:
            best_idx = int(np.nanargmin(scores_arr))

        return param_configs[best_idx], float(scores_arr[best_idx])

    def _evaluate_config(
        self,
        config: dict[str, Any],
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> float:
        X_tr, X_val = X[train_idx], X[val_idx]
        T_tr, T_val = T[train_idx], T[val_idx]
        E_tr, E_val = E[train_idx], E[val_idx]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = self.model_class(**config)
                model.fit(X_tr, T_tr, E_tr)
                return compute_single_metric(
                    model, X_tr, T_tr, E_tr,
                    X_val, T_val, E_val,
                    metric=self.metric,
                )
        except Exception:
            return np.nan


def _min_class_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())
