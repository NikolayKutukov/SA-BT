from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SurvivalData:
    """Standardized output for all synthetic data setups."""

    # Core arrays
    X: np.ndarray  # (n, p) covariate matrix
    T: np.ndarray  # (n,) observed times: min(event_time, censor_time)
    E: np.ndarray  # (n,) event indicator: 1=event, 0=censored
    feature_names: list[str]

    # Ground-truth metadata
    true_event_times: np.ndarray  # (n,) uncensored event times
    true_betas: np.ndarray | dict  # (p,) or dict for non-PH / multi-transition
    setup_name: str
    config: object  # the config dataclass used

    # Optional: competing risks (Setup 4)
    cause: Optional[np.ndarray] = None  # (n,) cause indicator
    state_history: Optional[list] = None  # per-subject trajectory dicts

    # ── convenience helpers ──────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single DataFrame with covariates, T, E (and cause)."""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["T"] = self.T
        df["E"] = self.E
        if self.cause is not None:
            df["cause"] = self.cause
        return df

    @property
    def censoring_rate(self) -> float:
        return 1.0 - float(self.E.mean())

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    def summary(self) -> str:
        lines = [
            f"Setup: {self.setup_name}",
            f"  n={self.n}, p={self.p}",
            f"  censoring rate: {self.censoring_rate:.1%}",
            f"  median observed time: {np.median(self.T):.3f}",
            f"  event count: {int(self.E.sum())}",
        ]
        if self.cause is not None:
            unique, counts = np.unique(self.cause[self.E == 1], return_counts=True)
            for u, c in zip(unique, counts):
                lines.append(f"  cause {int(u)}: {c} events")
        return "\n".join(lines)
