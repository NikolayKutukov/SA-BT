"""Abstract base class for all survival models + cause-specific wrapper."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np


class SurvivalModel(ABC):
    """Unified interface for survival analysis models.

    All concrete wrappers implement this ABC so the benchmark runner
    can treat every model identically.
    """

    name: str = "base"
    supports_competing_risks: bool = False

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk.

        Subclasses may override for custom serialisation (e.g. torch
        state_dict).  The default implementation uses pickle.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "SurvivalModel":
        """Load a previously saved model from disk.

        Returns the full model object regardless of the concrete
        subclass that was saved.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

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
             For competing risks: integer cause code (0 = censored, 1, 2, … = causes).
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

    def predict_cumulative_incidence(
        self,
        X: np.ndarray,
        times: np.ndarray,
        cause: int,
    ) -> np.ndarray:
        """Return (n, len(times)) CIF for a given cause (competing risks).

        Only implemented by models with ``supports_competing_risks = True``.
        """
        raise NotImplementedError(
            f"{self.name} does not support competing risks prediction."
        )


class CauseSpecificWrapper:
    """Adapt any single-event SurvivalModel for competing risks via the
    cause-specific hazard approach.

    For Setup 4 (multi-state): fits one independent model *per cause*,
    treating all other event types as censored.

    Parameters
    ----------
    model_factory : callable returning a fresh SurvivalModel instance.
    """

    def __init__(self, model_factory: Callable[[], SurvivalModel]) -> None:
        self._factory = model_factory
        self._models: dict[int, SurvivalModel] = {}
        self._causes: list[int] = []

    @property
    def name(self) -> str:
        return f"{self._factory().name}_cause_specific"

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        cause: np.ndarray,
    ) -> "CauseSpecificWrapper":
        """Fit one model per cause.

        Parameters
        ----------
        cause : (n,) integer cause indicator.
                0 = censored, k > 0 = cause k.
        """
        self._causes = sorted(set(int(c) for c in cause if c > 0))

        for k in self._causes:
            # Binary indicator: 1 if this subject had cause k, else 0 (censored)
            E_k = (cause == k).astype(np.float64)
            model_k = self._factory()
            model_k.fit(X, T, E_k)
            self._models[k] = model_k

        return self

    def predict_cause_risk(self, X: np.ndarray, cause: int) -> np.ndarray:
        """Risk scores from the cause-specific model for *cause*."""
        return self._models[cause].predict_risk(X)

    def predict_cause_survival(
        self,
        X: np.ndarray,
        times: np.ndarray,
        cause: int,
    ) -> np.ndarray:
        """Cause-specific survival S_k(t|X) from the model for *cause*."""
        return self._models[cause].predict_survival_function(X, times)

    def predict_cause_cif(
        self,
        X: np.ndarray,
        times: np.ndarray,
        cause: int,
    ) -> np.ndarray:
        """Crude CIF approximation: 1 - S_k(t|X).

        Note: this is a simplification. The true CIF depends on the
        overall survival, but this is the standard cause-specific approach.
        """
        return 1.0 - self.predict_cause_survival(X, times, cause)
