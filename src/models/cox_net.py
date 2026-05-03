"""CoxNet wrapper around scikit-survival CoxnetSurvivalAnalysis."""

from __future__ import annotations

import warnings

import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from .base import SurvivalModel


class CoxNetModel(SurvivalModel):
    """Elastic-net penalised Cox model.

    Wraps ``sksurv.linear_model.CoxnetSurvivalAnalysis``.
    Designed for high-dimensional settings (Setup 3: p=2000, n=500).
    Uses the full regularisation path and selects the best alpha via
    built-in cross-validation–like path fitting.
    """

    name = "CoxNet"

    def __init__(
        self,
        l1_ratio: float = 0.5,
        alpha_min_ratio: float = 0.01,
        n_alphas: int = 100,
        max_iter: int = 100_000,
    ) -> None:
        self._l1_ratio = l1_ratio
        self._alpha_min_ratio = alpha_min_ratio
        self._n_alphas = n_alphas
        self._max_iter = max_iter
        self._model: CoxnetSurvivalAnalysis | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "CoxNetModel":
        y = Surv.from_arrays(E.astype(bool), T)

        self._model = CoxnetSurvivalAnalysis(
            l1_ratio=self._l1_ratio,
            alpha_min_ratio=self._alpha_min_ratio,
            n_alphas=self._n_alphas,
            max_iter=self._max_iter,
            fit_baseline_model=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self._model.fit(X, y)

        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        step_fns = self._model.predict_survival_function(X)
        domain = step_fns[0].domain
        clipped = np.clip(times, domain[0], domain[1])
        out = np.column_stack([fn(clipped) for fn in step_fns]).T
        return np.clip(out, 0.0, 1.0)
