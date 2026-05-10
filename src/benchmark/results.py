"""Results storage and formatting for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ResultsTable:
    """Accumulates per-(setup, model) metric dictionaries and exports them."""

    # {(setup_name, model_name): {metric_name: value}}
    _data: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    # Optional timing info
    _timings: dict[tuple[str, str], float] = field(default_factory=dict)

    def add(
        self,
        setup: str,
        model: str,
        metrics: dict[str, float],
        fit_time: float | None = None,
    ) -> None:
        self._data[(setup, model)] = metrics
        if fit_time is not None:
            self._timings[(setup, model)] = fit_time

    def to_dataframe(self) -> pd.DataFrame:
        """Wide-format DataFrame: rows = (setup, model), columns = metrics."""
        records = []
        for (setup, model), metrics in self._data.items():
            row = {"setup": setup, "model": model, **metrics}
            if (setup, model) in self._timings:
                row["fit_time_s"] = self._timings[(setup, model)]
            records.append(row)
        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values(["setup", "model"]).reset_index(drop=True)
        return df

