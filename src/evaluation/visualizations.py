"""Visualization functions for benchmark results.

All functions return ``matplotlib.figure.Figure`` for flexible saving/display.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric_heatmap(
    agg_df: pd.DataFrame,
    metric: str = "c_index_ipcw",
    pivot: str = "setups",
    ax: plt.Axes | None = None,
    cmap: str = "YlGnBu",
    fmt: str = ".3f",
) -> plt.Figure:
    """Heatmap of a metric across models and setups or sample sizes.

    Parameters
    ----------
    agg_df : aggregated DataFrame from ``CVResultsTable.aggregate()``.
    metric : base metric name (e.g. "c_index_ipcw").
    pivot  : "setups" for models x setups, or "sample_sizes" for models x n.
    cmap   : colormap.
    fmt    : annotation format.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in agg_df.columns:
        raise ValueError(f"Column '{mean_col}' not found in DataFrame")

    if pivot == "sample_sizes" and "n_samples" in agg_df.columns:
        x_col = "n_samples"
        x_label = "Sample size"
    else:
        x_col = "setup"
        x_label = "Setup"

    pivot_df = agg_df.pivot(index="model", columns=x_col, values=mean_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(pivot_df.columns) * 1.5), len(pivot_df) * 0.8 + 2))
    else:
        fig = ax.get_figure()

    # Annotation with mean +/- SD
    annot = pivot_df.copy().astype(str)
    if std_col in agg_df.columns:
        std_pivot = agg_df.pivot(index="model", columns=x_col, values=std_col)
        for row in pivot_df.index:
            for col in pivot_df.columns:
                m = pivot_df.loc[row, col]
                s = std_pivot.loc[row, col]
                if np.isnan(m):
                    annot.loc[row, col] = "N/A"
                else:
                    annot.loc[row, col] = f"{m:{fmt}}\n+/-{s:{fmt}}"

    sns.heatmap(
        pivot_df, annot=annot, fmt="", cmap=cmap, ax=ax,
        linewidths=0.5, cbar_kws={"label": metric},
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Model")
    ax.set_title(f"{metric} — Models vs {x_label}")
    fig.tight_layout()
    return fig


def plot_sample_size_curves(
    agg_df: pd.DataFrame,
    setup: str,
    metric: str = "c_index_ipcw",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Line plot: metric vs sample size, one line per model with error bars.

    Requires ``n_samples`` column in ``agg_df``.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    subset = agg_df[agg_df["setup"] == setup].copy()
    if subset.empty or mean_col not in subset.columns:
        fig, ax_ = plt.subplots()
        ax_.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    models = sorted(subset["model"].unique())
    for model in models:
        m_data = subset[subset["model"] == model].sort_values("n_samples")
        x = m_data["n_samples"].values
        y = m_data[mean_col].values
        yerr = m_data[std_col].values if std_col in m_data.columns else None
        ax.errorbar(x, y, yerr=yerr, marker="o", label=model, capsize=3)

    ax.set_xlabel("Sample size")
    ax.set_ylabel(metric)
    ax.set_title(f"{setup} — {metric} vs sample size")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_calibration_comparison(
    calibration_data: list[tuple[str, np.ndarray, np.ndarray]],
    setup_name: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Overlay calibration curves for multiple models.

    Parameters
    ----------
    calibration_data : list of (model_name, predicted_means, observed_surv).
    setup_name : for the title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    for name, pred, obs in calibration_data:
        valid = ~np.isnan(obs)
        if valid.any():
            ax.scatter(pred[valid], obs[valid], label=name, s=30, alpha=0.8)
            ax.plot(pred[valid], obs[valid], alpha=0.5)

    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Observed survival (KM)")
    ax.set_title(f"Calibration — {setup_name}")
    ax.legend(loc="best", fontsize="small")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rank_chart(
    rank_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Horizontal bar chart of average ranks across setups.

    Parameters
    ----------
    rank_df : from ``CVResultsTable.rank_table()``.
    """
    if rank_df.empty or "avg_rank" not in rank_df.columns:
        fig, ax_ = plt.subplots()
        ax_.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, len(rank_df) * 0.5 + 1)))
    else:
        fig = ax.get_figure()

    df = rank_df.sort_values("avg_rank", ascending=True)
    colors = sns.color_palette("viridis", len(df))

    ax.barh(df["model"], df["avg_rank"], color=colors)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title("Model ranking across setups")
    ax.invert_yaxis()

    # Annotate bars
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["avg_rank"] + 0.05, i, f"{row['avg_rank']:.2f}",
            va="center", fontsize=9,
        )

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_brier_score_over_time(
    time_points: np.ndarray,
    brier_scores: dict[str, np.ndarray],
    setup_name: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Line plot of time-dependent Brier score for each model.

    Parameters
    ----------
    time_points  : (n_times,) array of evaluation times.
    brier_scores : {model_name: (n_times,) Brier score array}.
    setup_name   : for the title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    for name, bs in brier_scores.items():
        ax.plot(time_points, bs, label=name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Brier score")
    ax.set_title(f"Time-dependent Brier score — {setup_name}")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
