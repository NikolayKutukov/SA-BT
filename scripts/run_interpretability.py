#!/usr/bin/env python
"""Run interpretability experiment: SurvLIME + SurvSHAP(t) on synthetic data.

Usage:
    python scripts/run_interpretability.py
    python scripts/run_interpretability.py --models cox_ph rsf
    python scripts/run_interpretability.py --n-individuals 3 --n-seeds 1
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import generate_interpretability_data
from src.models import ALL_MODELS
from src.interpretability.explainers import SurvLIMEExplainer, SurvSHAPExplainer

MAX_INDIVIDUALS = 10

# Ground truth: features x0-x3 are important, x4-x7 are noise
IMPORTANT_FEATURES = ["x0", "x1", "x2", "x3"]
TRUE_SIGNS = {"x0": 1.0, "x1": -1.0, "x2": 1.0, "x3": 1.0}


def select_individuals(
    X_test: np.ndarray,
    risk_scores: np.ndarray,
    n_individuals: int,
) -> tuple[np.ndarray, list[str]]:
    """Select representative test individuals by risk percentiles."""
    indices = []
    labels = []

    percentiles = [10, 50, 90]
    for pct in percentiles[:n_individuals]:
        target = np.percentile(risk_scores, pct)
        diffs = np.abs(risk_scores - target)
        diffs[indices] = np.inf
        idx = int(np.argmin(diffs))
        indices.append(idx)
        labels.append(f"P{pct} (risk={risk_scores[idx]:.3f})")

    # Fill remaining with evenly spaced percentiles
    while len(indices) < n_individuals and len(indices) < len(risk_scores):
        frac = len(indices) / (n_individuals + 1)
        target = np.percentile(risk_scores, frac * 100)
        diffs = np.abs(risk_scores - target)
        diffs[indices] = np.inf
        idx = int(np.argmin(diffs))
        indices.append(idx)
        labels.append(f"P{int(frac*100)} (risk={risk_scores[idx]:.3f})")

    return np.array(indices), labels


def compute_precision_at_k(
    feature_ranking: list[str],
    true_important: list[str],
    k: int,
) -> float:
    """Precision@k: fraction of top-k ranked features that are truly important."""
    top_k = set(feature_ranking[:k])
    return len(top_k & set(true_important)) / k


def check_sign_consistency(
    coefficients: np.ndarray,
    feature_names: list[str],
    true_signs: dict[str, float],
) -> dict[str, bool]:
    """Check if coefficient signs match ground-truth effect directions."""
    results = {}
    for i, name in enumerate(feature_names):
        if name in true_signs:
            pred_sign = np.sign(coefficients[i])
            results[name] = bool(pred_sign == true_signs[name]) if abs(coefficients[i]) > 1e-6 else False
    return results


def run_single_seed(
    seed: int,
    model_names: list[str],
    n_individuals: int,
    outdir: Path,
    save_plots: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the interpretability experiment for one seed."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

    data = generate_interpretability_data(seed=seed)
    feature_names = data.feature_names

    if verbose:
        print(data.summary())

    # Train/test split
    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        data.X, data.T, data.E,
        test_size=0.2, stratify=data.E.astype(int), random_state=seed,
    )

    if verbose:
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Censoring rate: {data.censoring_rate:.1%}")

    # Time grid
    event_times = T_test[E_test.astype(bool)]
    if len(event_times) > 0:
        t_max = np.percentile(event_times, 95)
        t_min = event_times.min() + 1e-6
    else:
        t_max = np.percentile(T_test, 95)
        t_min = T_test.min() + 1e-6
    times = np.linspace(t_min, t_max, 50)

    seed_results = {}

    for model_name in model_names:
        model_class = ALL_MODELS[model_name]
        display_name = model_class.name if hasattr(model_class, "name") else model_name

        if verbose:
            print(f"\n--- Model: {display_name} ---")

        model = model_class()
        model.fit(X_train, T_train, E_train)
        risk_scores = model.predict_risk(X_test)

        if verbose:
            print(f"  Risk scores: min={risk_scores.min():.3f}, "
                  f"max={risk_scores.max():.3f}, "
                  f"mean={risk_scores.mean():.3f}")

        ind_indices, ind_labels = select_individuals(
            X_test, risk_scores, n_individuals,
        )

        if verbose:
            print(f"  Selected {len(ind_indices)} individuals:")
            for idx, label in zip(ind_indices, ind_labels):
                print(f"    [{idx}] {label}")

        # SurvLIME
        if verbose:
            print(f"\n  Running SurvLIME...")

        lime_results = {}
        try:
            lime_explainer = SurvLIMEExplainer(model, X_train, T_train, E_train)

            all_lime_coefs = []
            for j, (idx, label) in enumerate(zip(ind_indices, ind_labels)):
                try:
                    coefficients = lime_explainer.explain(X_test[idx], num_samples=1000)
                    all_lime_coefs.append(coefficients)

                    if verbose:
                        order = np.argsort(np.abs(coefficients))[::-1]
                        top_names = [feature_names[k] for k in order[:4]]
                        print(f"    Individual {j+1} ({label}): top-4 = {top_names}")

                    if save_plots:
                        fig = lime_explainer.plot_weights(
                            coefficients, feature_names,
                            true_important=IMPORTANT_FEATURES,
                        )
                        fig.suptitle(f"SurvLIME — {display_name} — {label}", fontsize=10)
                        fig.savefig(
                            outdir / f"survlime_{model_name}_ind{j+1}_seed{seed}.png",
                            dpi=150, bbox_inches="tight",
                        )
                        plt.close(fig)

                except Exception as exc:
                    if verbose:
                        print(f"    SurvLIME failed for individual {j+1}: {exc}")
                    all_lime_coefs.append(np.zeros(len(feature_names)))

            if all_lime_coefs:
                mean_abs = np.mean(np.abs(np.stack(all_lime_coefs)), axis=0)
                lime_ranking = [feature_names[i] for i in np.argsort(mean_abs)[::-1]]
                lime_p_at_k = compute_precision_at_k(
                    lime_ranking, IMPORTANT_FEATURES, k=4,
                )
                mean_coefs = np.mean(np.stack(all_lime_coefs), axis=0)
                sign_check = check_sign_consistency(mean_coefs, feature_names, TRUE_SIGNS)

                lime_results = {
                    "ranking": lime_ranking,
                    "precision_at_4": lime_p_at_k,
                    "sign_consistency": sign_check,
                    "mean_abs_coefficients": dict(zip(feature_names, mean_abs)),
                }

                if verbose:
                    print(f"  SurvLIME precision@4: {lime_p_at_k:.2f}")
                    print(f"  SurvLIME ranking: {lime_ranking[:6]}")
                    print(f"  Sign consistency: {sign_check}")

        except Exception as exc:
            if verbose:
                print(f"  SurvLIME setup failed: {exc}")

        # SurvSHAP(t)
        if verbose:
            print(f"\n  Running SurvSHAP(t)...")

        shap_results = {}
        try:
            shap_explainer = SurvSHAPExplainer(
                model, X_train, T_train, E_train, feature_names=feature_names,
            )

            for j, (idx, label) in enumerate(zip(ind_indices, ind_labels)):
                try:
                    result_df = shap_explainer.explain(X_test[idx], times)

                    if save_plots and result_df is not None and not result_df.empty:
                        fig = shap_explainer.plot_shap_panel(
                            result_df, feature_names,
                            highlight_features=IMPORTANT_FEATURES,
                            title=f"SurvSHAP(t) — {display_name} — {label}",
                        )
                        fig.savefig(
                            outdir / f"survshap_time_{model_name}_ind{j+1}_seed{seed}.png",
                            dpi=150, bbox_inches="tight",
                        )
                        plt.close(fig)

                    if verbose:
                        print(f"    Individual {j+1} ({label}): SurvSHAP computed")

                except Exception as exc:
                    if verbose:
                        print(f"    SurvSHAP failed for individual {j+1}: {exc}")

            # Global importance
            try:
                importance_df = shap_explainer.explain_multiple(
                    X_test[ind_indices], times, feature_names,
                )

                if not importance_df.empty:
                    shap_ranking = importance_df["feature"].tolist()
                    shap_p_at_k = compute_precision_at_k(
                        shap_ranking, IMPORTANT_FEATURES, k=4,
                    )

                    shap_results = {
                        "ranking": shap_ranking,
                        "precision_at_4": shap_p_at_k,
                        "importance": importance_df,
                    }

                    if verbose:
                        print(f"  SurvSHAP precision@4: {shap_p_at_k:.2f}")
                        print(f"  SurvSHAP ranking: {shap_ranking[:6]}")

                    if save_plots:
                        fig = shap_explainer.plot_global_importance(
                            importance_df,
                            true_important=IMPORTANT_FEATURES,
                        )
                        fig.suptitle(
                            f"Global importance — {display_name}", fontsize=10,
                        )
                        fig.savefig(
                            outdir / f"survshap_global_{model_name}_seed{seed}.png",
                            dpi=150, bbox_inches="tight",
                        )
                        plt.close(fig)

            except Exception as exc:
                if verbose:
                    print(f"  SurvSHAP global importance failed: {exc}")

        except Exception as exc:
            if verbose:
                print(f"  SurvSHAP setup failed: {exc}")

        seed_results[model_name] = {
            "survlime": lime_results,
            "survshap": shap_results,
        }

    return seed_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run interpretability experiment (SurvLIME + SurvSHAP(t)).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--models", nargs="*",
        default=["cox_ph", "rsf"],
        choices=list(ALL_MODELS.keys()),
        help="Which models to explain (default: cox_ph rsf)",
    )
    parser.add_argument(
        "--n-individuals", type=int, default=5,
        help="Number of test individuals to explain (default: 5, max: 10)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=3,
        help="Number of independent seeds for stability (default: 3)",
    )
    parser.add_argument(
        "--outdir", type=str, default="results/interpretability",
        help="Output directory for plots and tables (default: results/interpretability/)",
    )
    args = parser.parse_args()

    args.n_individuals = min(args.n_individuals, MAX_INDIVIDUALS)

    print(f"Models       : {args.models}")
    print(f"Individuals  : {args.n_individuals}")
    print(f"Seeds        : {args.n_seeds}")
    print(f"Base seed    : {args.seed}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for s in range(args.n_seeds):
        seed = args.seed + s * 100
        save_plots = (s == 0)
        result = run_single_seed(
            seed=seed,
            model_names=args.models,
            n_individuals=args.n_individuals,
            outdir=outdir,
            save_plots=save_plots,
            verbose=True,
        )
        all_results.append(result)

    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")

    summary_rows = []
    for model_name in args.models:
        lime_p4_vals = []
        shap_p4_vals = []
        lime_sign_vals = {}

        for result in all_results:
            if model_name in result:
                mr = result[model_name]
                if "survlime" in mr and "precision_at_4" in mr["survlime"]:
                    lime_p4_vals.append(mr["survlime"]["precision_at_4"])
                if "survlime" in mr and "sign_consistency" in mr["survlime"]:
                    for feat, ok in mr["survlime"]["sign_consistency"].items():
                        lime_sign_vals.setdefault(feat, []).append(ok)
                if "survshap" in mr and "precision_at_4" in mr["survshap"]:
                    shap_p4_vals.append(mr["survshap"]["precision_at_4"])

        row = {"model": model_name}
        if lime_p4_vals:
            row["survlime_p@4_mean"] = np.mean(lime_p4_vals)
            row["survlime_p@4_std"] = np.std(lime_p4_vals)
        if shap_p4_vals:
            row["survshap_p@4_mean"] = np.mean(shap_p4_vals)
            row["survshap_p@4_std"] = np.std(shap_p4_vals)
        for feat, vals in lime_sign_vals.items():
            row[f"sign_{feat}"] = np.mean(vals)

        summary_rows.append(row)
        print(f"\n  {model_name}:")
        for k, v in row.items():
            if k != "model":
                if isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                else:
                    print(f"    {k}: {v}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = outdir / "interpretability_summary.csv"
        summary_df.to_csv(str(summary_path), index=False)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
