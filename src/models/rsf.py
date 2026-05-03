"""Random Survival Forest wrapper around scikit-survival."""

from __future__ import annotations

import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from .base import SurvivalModel


class RSFModel(SurvivalModel):
    """Random Survival Forest.

    Wraps ``sksurv.ensemble.RandomSurvivalForest``.
    """

    name = "RSF"

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_split: int = 10,
        min_samples_leaf: int = 6,
        max_features: str | int | float = "sqrt",
        max_depth: int | None = 15,
        max_samples: int | float | None = None,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_depth=max_depth,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._model = RandomSurvivalForest(**self._params)

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "RSFModel":
        # For large datasets, cap tree depth and subsample to avoid OOM
        n = X.shape[0]
        if n > 10_000:
            params = self._params.copy()
            if params["max_depth"] is None:
                params["max_depth"] = 15
            if params["max_samples"] is None:
                params["max_samples"] = 10_000
            self._model = RandomSurvivalForest(**params)
        y = Surv.from_arrays(E.astype(bool), T)
        self._model.fit(X, y)
        return self

    _BATCH = 1000  # prediction batch size to avoid OOM on large datasets

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if n <= self._BATCH:
            return self._model.predict(X)
        parts = []
        for s in range(0, n, self._BATCH):
            parts.append(self._model.predict(X[s : s + self._BATCH]))
        return np.concatenate(parts)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        n = len(X)
        if n <= self._BATCH:
            step_fns = self._model.predict_survival_function(X)
            domain = step_fns[0].domain
            clipped = np.clip(times, domain[0], domain[1])
            out = np.column_stack([fn(clipped) for fn in step_fns]).T
            return np.clip(out, 0.0, 1.0)
        parts = []
        for s in range(0, n, self._BATCH):
            step_fns = self._model.predict_survival_function(X[s : s + self._BATCH])
            domain = step_fns[0].domain
            clipped = np.clip(times, domain[0], domain[1])
            batch_out = np.column_stack([fn(clipped) for fn in step_fns]).T
            parts.append(batch_out)
        return np.clip(np.vstack(parts), 0.0, 1.0)
