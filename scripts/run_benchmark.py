#!/usr/bin/env python
"""Run survival model benchmark across all datasets.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --datasets gbsg2 metabric
    python scripts/run_benchmark.py --models cox_ph rsf
    python scripts/run_benchmark.py --seed 123 --outdir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ALL_MODELS
from src.data.loaders import LOADERS
from src.benchmark.runner import BenchmarkRunner


def build_model(name: str):
    """Instantiate a model by name with default hyperparameters."""
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(ALL_MODELS)}")
    return ALL_MODELS[name]()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run survival model benchmark.",
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
        "--test-fraction", type=float, default=0.2,
        help="Held-out test set fraction (default: 0.2)",
    )
    args = parser.parse_args()

    models = [build_model(name) for name in args.models]

    print(f"Datasets: {args.datasets}")
    print(f"Models  : {[m.name for m in models]}")
    print(f"Seed    : {args.seed}")
    print(f"Test%   : {args.test_fraction}")

    runner = BenchmarkRunner(
        models=models,
        datasets=args.datasets,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )

    results = runner.run(verbose=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "benchmark_results.csv"
    results.to_csv(str(csv_path))

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(results.summary())
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
