"""Abstract base class for all survival models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SurvivalModel(ABC):
    """Unified interface for survival analysis models.

    All concrete wrappers implement this ABC so the benchmark runner
    can treat every model identically.
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        **kwargs,
    ) -> "SurvivalModel":
        """Train the model.

        Parameters
        ----------
        X : (n, p) covariate matrix.
        T : (n,) observed times.
        E : (n,) event indicators (1 = event, 0 = censored).
        """

    @abstractmethod
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Return (n,) risk scores — higher value = worse prognosis."""

    @abstractmethod
    def predict_survival_function(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Return (n, len(times)) matrix of S(t | X) values."""
