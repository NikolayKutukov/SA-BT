"""CoxPH wrapper around scikit-survival CoxPHSurvivalAnalysis."""

from __future__ import annotations

import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from .base import SurvivalModel


class CoxPHModel(SurvivalModel):
    """Unpenalised Cox proportional hazards model.

    Wraps ``sksurv.linear_model.CoxPHSurvivalAnalysis``.
    Not suitable for high-dimensional data (p >> n); use CoxNetModel instead.
    """

    name = "CoxPH"

    def __init__(self, alpha: float = 0.0, ties: str = "efron") -> None:
        self._model = CoxPHSurvivalAnalysis(alpha=alpha, ties=ties)

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "CoxPHModel":
        y = Surv.from_arrays(E.astype(bool), T)
        self._model.fit(X, y)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        step_fns = self._model.predict_survival_function(X)
        # Clip times to the domain of the step functions (training time range)
        domain = step_fns[0].domain
        clipped = np.clip(times, domain[0], domain[1])
        out = np.column_stack([fn(clipped) for fn in step_fns]).T  # (n, len(times))
        return np.clip(out, 0.0, 1.0)
