#!/usr/bin/env python
"""Run nested cross-validation benchmark with hyperparameter tuning.

Usage:
    python scripts/run_nested_cv.py
    python scripts/run_nested_cv.py --datasets gbsg2 metabric --models cox_ph rsf
    python scripts/run_nested_cv.py --outer-folds 5 --inner-folds 5 --repeats 3
    python scripts/run_nested_cv.py --inner-metric ibs --seed 123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ALL_MODELS
from src.data.loaders import LOADERS
from src.benchmark.nested_cv_runner import NestedCVRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nested CV benchmark with HP tuning.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--outdir", type=str, default="results",
        help="Output directory for results CSV (default: results/)",
    )
    parser.add_argument(
        "--datasets", nargs="*",
        default=list(LOADERS.keys()),
        choices=list(LOADERS.keys()),
        help="Which datasets to run (default: all)",
    )
    parser.add_argument(
        "--models", nargs="*",
        default=list(ALL_MODELS.keys()),
        choices=list(ALL_MODELS.keys()),
        help="Which models to run (default: all)",
    )
    parser.add_argument(
        "--outer-folds", type=int, default=5,
        help="Number of outer CV folds (default: 5)",
    )
    parser.add_argument(
        "--inner-folds", type=int, default=3,
        help="Number of inner CV folds for HP tuning (default: 3)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of independent repeats (default: 1)",
    )
    parser.add_argument(
        "--inner-metric", type=str, default="c_index_ipcw",
        choices=["c_index", "c_index_ipcw", "ibs", "mean_auc"],
        help="Metric for HP selection in inner CV (default: c_index_ipcw)",
    )
    args = parser.parse_args()

    print(f"Datasets    : {args.datasets}")
    print(f"Models      : {args.models}")
    print(f"Outer folds : {args.outer_folds}")
    print(f"Inner folds : {args.inner_folds}")
    print(f"Repeats     : {args.repeats}")
    print(f"Inner metric: {args.inner_metric}")
    print(f"Seed        : {args.seed}")

    runner = NestedCVRunner(
        model_names=args.models,
        datasets=args.datasets,
        n_outer_folds=args.outer_folds,
        n_inner_folds=args.inner_folds,
        n_repeats=args.repeats,
        inner_metric=args.inner_metric,
        seed=args.seed,
    )

    results = runner.run(verbose=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_path = outdir / "nested_cv_raw.csv"
    results.to_csv(str(raw_path))
    print(f"\nRaw fold-level results saved to {raw_path}")

    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS (mean +/- SD)")
    print(f"{'='*60}")
    print(results.summary())

    print(f"\n{'='*60}")
    print("MODEL RANKINGS (by IPCW C-index)")
    print(f"{'='*60}")
    rank_df = results.rank_table(metric="c_index_ipcw")
    if not rank_df.empty:
        print(rank_df.to_string(index=False))

    agg_path = outdir / "nested_cv_aggregated.csv"
    results.aggregate().to_csv(str(agg_path), index=False)
    print(f"\nAggregated results saved to {agg_path}")


if __name__ == "__main__":
    main()
