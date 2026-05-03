from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SurvivalData:
    """Standardized container for survival analysis datasets."""

    # Core arrays
    X: np.ndarray  # (n, p) covariate matrix
    T: np.ndarray  # (n,) observed times: min(event_time, censor_time)
    E: np.ndarray  # (n,) event indicator: 1=event, 0=censored
    feature_names: list[str]

    # Dataset identifier
    dataset_name: str = ""

    # Ground-truth metadata (only for synthetic data)
    true_betas: Optional[np.ndarray | dict] = None

    # ── convenience helpers ──────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single DataFrame with covariates, T, E."""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["T"] = self.T
        df["E"] = self.E
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
            f"Dataset: {self.dataset_name}",
            f"  n={self.n}, p={self.p}",
            f"  censoring rate: {self.censoring_rate:.1%}",
            f"  median observed time: {np.median(self.T):.3f}",
            f"  event count: {int(self.E.sum())}",
        ]
        return "\n".join(lines)
