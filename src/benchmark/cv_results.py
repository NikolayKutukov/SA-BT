"""Results storage and aggregation for nested cross-validation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CVResultsTable:
    """Fold-level results from nested CV with aggregation methods.

    Each row stores metrics for one (repeat, setup, model, fold) combination,
    along with the best HP config selected by inner CV and timing info.
    """

    _rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        repeat: int,
        setup: str,
        model: str,
        fold: int,
        metrics: dict[str, float],
        best_config: dict[str, Any] | None = None,
        inner_cv_score: float | None = None,
        fit_time: float | None = None,
        n_samples: int | None = None,
    ) -> None:
        """Append one fold result."""
        row: dict[str, Any] = {
            "repeat": repeat,
            "setup": setup,
            "model": model,
            "fold": fold,
            **metrics,
        }
        if best_config is not None:
            row["best_config"] = str(best_config)
        if inner_cv_score is not None:
            row["inner_cv_score"] = inner_cv_score
        if fit_time is not None:
            row["fit_time_s"] = fit_time
        if n_samples is not None:
            row["n_samples"] = n_samples
        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        """Raw fold-level DataFrame."""
        if not self._rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._rows)
        sort_cols = [c for c in ["setup", "model", "repeat", "fold"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CVResultsTable":
        """Restore from a fold-level DataFrame (e.g. loaded from CSV)."""
        table = cls()
        table._rows = df.to_dict(orient="records")
        return table

    def aggregate(
        self,
        metric_cols: list[str] | None = None,
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute mean and std per (setup, model) for each metric.

        Parameters
        ----------
        metric_cols : which metric columns to aggregate.
                      Default: c_index, c_index_ipcw, ibs, mean_auc.
        group_cols  : grouping columns.
                      Default: ["setup", "model"].

        Returns
        -------
        DataFrame with columns like ``c_index_mean``, ``c_index_std``, etc.
        """
        df = self.to_dataframe()
        if df.empty:
            return df

        if metric_cols is None:
            metric_cols = [
                c for c in ["c_index", "c_index_ipcw", "ibs", "mean_auc", "fit_time_s"]
                if c in df.columns
            ]
        if group_cols is None:
            group_cols = ["setup", "model"]
            # Include n_samples if it varies
            if "n_samples" in df.columns and df["n_samples"].nunique() > 1:
                group_cols.append("n_samples")

        agg_funcs = {col: ["mean", "std"] for col in metric_cols if col in df.columns}
        if not agg_funcs:
            return df[group_cols].drop_duplicates().reset_index(drop=True)

        agg = df.groupby(group_cols, sort=True).agg(agg_funcs)
        # Flatten multi-level columns: ("c_index", "mean") -> "c_index_mean"
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        return agg.reset_index()

    def format_table(
        self,
        metric_cols: list[str] | None = None,
        fmt: str = "{:.3f}",
    ) -> pd.DataFrame:
        """Aggregated table with 'mean +/- SD' strings.

        Ready for LaTeX or display.
        """
        agg = self.aggregate(metric_cols=metric_cols)
        if agg.empty:
            return agg

        # Identify metric base names from columns ending in _mean
        mean_cols = [c for c in agg.columns if c.endswith("_mean")]
        group_cols = [c for c in agg.columns if not c.endswith(("_mean", "_std"))]

        formatted = agg[group_cols].copy()
        for mc in mean_cols:
            base = mc.removesuffix("_mean")
            std_col = f"{base}_std"
            if std_col in agg.columns:
                formatted[base] = agg.apply(
                    lambda row, m=mc, s=std_col: (
                        f"{fmt.format(row[m])} +/- {fmt.format(row[s])}"
                        if not np.isnan(row[m]) else "N/A"
                    ),
                    axis=1,
                )
            else:
                formatted[base] = agg[mc].apply(
                    lambda v: fmt.format(v) if not np.isnan(v) else "N/A"
                )

        return formatted

    def rank_table(
        self,
        metric: str = "c_index_ipcw",
        higher_is_better: bool = True,
    ) -> pd.DataFrame:
        """Per-setup ranking + average rank across setups.

        Parameters
        ----------
        metric : which metric to rank on (uses the mean across folds).
        higher_is_better : rank direction.

        Returns
        -------
        DataFrame with columns: model, setup_1_rank, ..., avg_rank.
        """
        agg = self.aggregate(metric_cols=[metric])
        if agg.empty:
            return agg

        mean_col = f"{metric}_mean"
        if mean_col not in agg.columns:
            return pd.DataFrame()

        setups = sorted(agg["setup"].unique())
        models = sorted(agg["model"].unique())

        rank_data: dict[str, dict[str, float]] = {m: {} for m in models}

        for setup in setups:
            subset = agg[agg["setup"] == setup].set_index("model")[mean_col]
            if higher_is_better:
                ranks = subset.rank(ascending=False, method="min")
            else:
                ranks = subset.rank(ascending=True, method="min")
            for model in models:
                rank_data[model][setup] = ranks.get(model, np.nan)

        rows = []
        for model in models:
            row = {"model": model}
            rank_vals = []
            for setup in setups:
                r = rank_data[model].get(setup, np.nan)
                row[f"{setup}_rank"] = r
                if not np.isnan(r):
                    rank_vals.append(r)
            row["avg_rank"] = np.mean(rank_vals) if rank_vals else np.nan
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("avg_rank").reset_index(drop=True)
        return df

    def to_csv(self, path: str) -> None:
        """Save raw fold-level results to CSV."""
        self.to_dataframe().to_csv(path, index=False)

    def to_latex(self, **kwargs) -> str:
        """LaTeX table with mean +/- SD."""
        return self.format_table().to_latex(index=False, **kwargs)

    def summary(self) -> str:
        """Human-readable aggregated summary."""
        fmt = self.format_table()
        if fmt.empty:
            return "No results yet."
        return fmt.to_string(index=False)
