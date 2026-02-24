"""Inner cross-validation search for hyperparameter tuning."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import compute_single_metric, METRIC_DIRECTION
from src.models.base import SurvivalModel, CauseSpecificWrapper


class InnerCVSearch:
    """Random search over HP configs using inner K-fold CV.

    For each candidate config, trains the model on each inner fold and
    computes a single metric on the held-out fold.  Selects the config
    with the best average score.

    Parameters
    ----------
    model_class : SurvivalModel subclass (e.g. ``CoxPHModel``).
    model_name  : key in SEARCH_SPACES (used for logging only).
    n_inner_folds : number of inner CV folds.
    metric      : selection criterion (see ``METRIC_DIRECTION``).
    seed        : random seed for fold splitting.
    """

    def __init__(
        self,
        model_class: type,
        model_name: str,
        n_inner_folds: int = 5,
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
        cause: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Run inner CV search and return the best config.

        Parameters
        ----------
        X, T, E : training data (outer-train fold).
        param_configs : list of HP dicts to evaluate.
        cause : cause indicators for competing risks (Setup 4).

        Returns
        -------
        best_config : dict of best HP values.
        best_score  : mean inner-CV score for the best config.
        """
        is_competing = cause is not None
        stratify_col = cause.astype(int) if is_competing else E.astype(int)

        # Ensure enough samples per class for stratified splitting
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
                    config, X, T, E, cause,
                    train_idx, val_idx, is_competing,
                )
                fold_scores.append(score)

            # Average non-NaN scores
            valid = [s for s in fold_scores if not np.isnan(s)]
            mean_score = np.mean(valid) if valid else np.nan
            scores.append(mean_score)

        # Select best
        scores_arr = np.array(scores)
        valid_mask = ~np.isnan(scores_arr)

        if not valid_mask.any():
            # All configs failed — fall back to default (empty config)
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
        cause: np.ndarray | None,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        is_competing: bool,
    ) -> float:
        """Fit model with config on inner-train, score on inner-val."""
        X_tr, X_val = X[train_idx], X[val_idx]
        T_tr, T_val = T[train_idx], T[val_idx]
        E_tr, E_val = E[train_idx], E[val_idx]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if is_competing and not self.model_class.supports_competing_risks:
                    # Cause-specific approach
                    cause_tr = cause[train_idx]
                    cause_val = cause[val_idx]
                    return self._eval_cause_specific(
                        config, X_tr, T_tr, cause_tr, X_val, T_val, cause_val,
                    )

                if is_competing and self.model_class.supports_competing_risks:
                    # Native competing risks: pass cause as E
                    cause_tr = cause[train_idx]
                    cause_val = cause[val_idx]
                    model = self.model_class(**config)
                    model.fit(X_tr, T_tr, cause_tr)
                    overall_E_val = (cause_val > 0).astype(float)
                    return compute_single_metric(
                        model, X_tr, T_tr, (cause_tr > 0).astype(float),
                        X_val, T_val, overall_E_val,
                        metric=self.metric,
                    )

                # Standard single-event
                model = self.model_class(**config)
                model.fit(X_tr, T_tr, E_tr)
                return compute_single_metric(
                    model, X_tr, T_tr, E_tr,
                    X_val, T_val, E_val,
                    metric=self.metric,
                )

        except Exception:
            return np.nan

    def _eval_cause_specific(
        self,
        config: dict[str, Any],
        X_tr: np.ndarray,
        T_tr: np.ndarray,
        cause_tr: np.ndarray,
        X_val: np.ndarray,
        T_val: np.ndarray,
        cause_val: np.ndarray,
    ) -> float:
        """Evaluate a cause-specific model on inner fold."""
        from src.evaluation.metrics import concordance_index

        factory = lambda: self.model_class(**config)  # noqa: E731
        wrapper = CauseSpecificWrapper(factory)
        wrapper.fit(X_tr, T_tr, cause_tr, cause=cause_tr)

        # Average C-index across causes
        c_values = []
        for cause_k in wrapper._causes:
            risk_k = wrapper.predict_cause_risk(X_val, cause_k)
            E_k_val = (cause_val == cause_k).astype(float)
            if E_k_val.sum() > 0:
                c = concordance_index(T_val, E_k_val, risk_k)
                c_values.append(c)

        return float(np.mean(c_values)) if c_values else np.nan


def _min_class_count(y: np.ndarray) -> int:
    """Minimum number of samples in any class."""
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())
