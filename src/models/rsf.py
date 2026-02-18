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
        max_samples: int | float | None = None,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self._model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "RSFModel":
        y = Surv.from_arrays(E.astype(bool), T)
        self._model.fit(X, y)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        step_fns = self._model.predict_survival_function(X)
        out = np.column_stack([fn(times) for fn in step_fns]).T
        return np.clip(out, 0.0, 1.0)
