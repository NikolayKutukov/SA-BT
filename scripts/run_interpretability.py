#!/usr/bin/env python
"""Run interpretability experiment: SurvLIME + SurvSHAP(t) on setup_interp.

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

# Suppress noisy sklearn/sksurv warnings and survlimepy logging
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate import generate_setup_interp
from src.models import ALL_MODELS
from src.interpretability.explainers import SurvLIMEExplainer, SurvSHAPExplainer

MAX_INDIVIDUALS = 10


def select_individuals(
    X_test: np.ndarray,
    risk_scores: np.ndarray,
    feature_names: list[str],
    tv_idx: int,
    n_individuals: int,
) -> tuple[np.ndarray, list[str]]:
    """Select representative test individuals by risk percentiles.

    Returns indices and description labels.
    """
    indices = []
    labels = []

    # Low risk (10th percentile)
    p10 = np.percentile(risk_scores, 10)
    idx = np.argmin(np.abs(risk_scores - p10))
    indices.append(idx)
    labels.append(f"low_risk (P10, risk={risk_scores[idx]:.3f})")

    # Borderline (50th percentile)
    if n_individuals >= 2:
        p50 = np.percentile(risk_scores, 50)
        idx = np.argmin(np.abs(risk_scores - p50))
        if idx in indices:
            # Find next closest that's not already selected
            diffs = np.abs(risk_scores - p50)
            diffs[indices] = np.inf
            idx = np.argmin(diffs)
        indices.append(idx)
        labels.append(f"borderline (P50, risk={risk_scores[idx]:.3f})")

    # High risk (90th percentile)
    if n_individuals >= 3:
        p90 = np.percentile(risk_scores, 90)
        idx = np.argmin(np.abs(risk_scores - p90))
        if idx in indices:
            diffs = np.abs(risk_scores - p90)
            diffs[indices] = np.inf
            idx = np.argmin(diffs)
        indices.append(idx)
        labels.append(f"high_risk (P90, risk={risk_scores[idx]:.3f})")

    # Extreme x3 (time-varying feature)
    if n_individuals >= 4:
        x3 = X_test[:, tv_idx]
        # High x3
        remaining = [i for i in range(len(x3)) if i not in indices]
        if remaining:
            best = max(remaining, key=lambda i: abs(x3[i]))
            indices.append(best)
            labels.append(f"extreme_x3 (x3={x3[best]:.3f})")

    # Fill remaining with spaced percentiles
    while len(indices) < n_individuals and len(indices) < len(risk_scores):
        frac = len(indices) / (n_individuals + 1)
        pct = np.percentile(risk_scores, frac * 100)
        idx = np.argmin(np.abs(risk_scores - pct))
        if idx not in indices:
            indices.append(idx)
            labels.append(f"P{int(frac*100)} (risk={risk_scores[idx]:.3f})")
        else:
            # Pick any unused index
            unused = [i for i in range(len(risk_scores)) if i not in indices]
            if unused:
                indices.append(unused[0])
                labels.append(f"extra_{len(indices)}")
            else:
                break

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
    true_betas: dict,
) -> dict[str, bool]:
    """Check if coefficient signs match ground-truth effect directions."""
    effects = true_betas.get("effects", {})
    results = {}
    for i, name in enumerate(feature_names):
        if name in effects:
            true_sign = effects[name].get("sign", 0)
            pred_sign = np.sign(coefficients[i])
            results[name] = bool(pred_sign == true_sign) if abs(coefficients[i]) > 1e-6 else False
    return results


def run_single_seed(
    seed: int,
    model_names: list[str],
    n_individuals: int,
    outdir: Path,
    save_plots: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the interpretability experiment for one seed.

    Returns a dict with quantitative results.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

    # 1. Generate data
    data = generate_setup_interp(seed=seed)
    if verbose:
        print(data.summary())

    true_betas = data.true_betas
    important_features = true_betas["important_features"]
    feature_names = data.feature_names
    tv_idx = data.config.tv_idx

    # 2. Train/test split
    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        data.X, data.T, data.E,
        test_size=0.2, stratify=data.E.astype(int), random_state=seed,
    )

    if verbose:
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Censoring rate: {1 - data.E.mean():.1%}")

    # 3. Time grid: clipped to P95 of event times
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

        # 4. Fit model
        model = model_class()
        model.fit(X_train, T_train, E_train)
        risk_scores = model.predict_risk(X_test)

        if verbose:
            print(f"  Risk scores: min={risk_scores.min():.3f}, "
                  f"max={risk_scores.max():.3f}, "
                  f"mean={risk_scores.mean():.3f}")

        # 5. Select individuals
        ind_indices, ind_labels = select_individuals(
            X_test, risk_scores, feature_names, tv_idx, n_individuals,
        )

        if verbose:
            print(f"  Selected {len(ind_indices)} individuals:")
            for idx, label in zip(ind_indices, ind_labels):
                print(f"    [{idx}] {label}")

        # 6. SurvLIME
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
                        # Top features
                        order = np.argsort(np.abs(coefficients))[::-1]
                        top_names = [feature_names[k] for k in order[:4]]
                        print(f"    Individual {j+1} ({label}): top-4 = {top_names}")

                    # Plot
                    if save_plots:
                        fig = lime_explainer.plot_weights(
                            coefficients, feature_names,
                            true_important=important_features,
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

            # Aggregate SurvLIME: mean |coefficient| ranking
            if all_lime_coefs:
                mean_abs = np.mean(np.abs(np.stack(all_lime_coefs)), axis=0)
                lime_ranking = [feature_names[i] for i in np.argsort(mean_abs)[::-1]]
                lime_p_at_k = compute_precision_at_k(
                    lime_ranking, important_features, k=4,
                )
                # Sign consistency (use mean coefficients)
                mean_coefs = np.mean(np.stack(all_lime_coefs), axis=0)
                sign_check = check_sign_consistency(mean_coefs, feature_names, true_betas)

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

        # 7. SurvSHAP(t)
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
                        fig = shap_explainer.plot_shap_over_time(
                            result_df, feature_names,
                            highlight_features=important_features,
                        )
                        fig.suptitle(
                            f"SurvSHAP(t) — {display_name} — {label}",
                            fontsize=10,
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
                        shap_ranking, important_features, k=4,
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
                            true_important=important_features,
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

    # Guard
    args.n_individuals = min(args.n_individuals, MAX_INDIVIDUALS)

    print(f"Models       : {args.models}")
    print(f"Individuals  : {args.n_individuals}")
    print(f"Seeds        : {args.n_seeds}")
    print(f"Base seed    : {args.seed}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Run across seeds
    all_results = []
    for s in range(args.n_seeds):
        seed = args.seed + s * 100
        save_plots = (s == 0)  # Save plots only from first seed
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

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = outdir / "interpretability_summary.csv"
        summary_df.to_csv(str(summary_path), index=False)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
