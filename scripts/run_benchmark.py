#!/usr/bin/env python
"""Run survival model benchmark across all setups.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --setups setup_1 setup_2
    python scripts/run_benchmark.py --models cox_ph rsf
    python scripts/run_benchmark.py --seed 123 --outdir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ALL_MODELS
from src.benchmark.runner import BenchmarkRunner, GENERATORS


def build_model(name: str):
    """Instantiate a model by name with default hyperparameters."""
    from src.models import (
        CoxPHModel, CoxNetModel, RSFModel,
        DeepSurvModel, DeepHitModel, SurvTRACEModel,
    )

    factories = {
        "cox_ph": CoxPHModel,
        "cox_net": CoxNetModel,
        "rsf": RSFModel,
        "deep_surv": DeepSurvModel,
        "deep_hit": DeepHitModel,
        "survtrace": SurvTRACEModel,
    }
    if name not in factories:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(factories)}")
    return factories[name]()


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
        "--setups", nargs="*",
        default=list(GENERATORS.keys()),
        choices=list(GENERATORS.keys()),
        help="Which setups to run (default: all)",
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

    # Build model instances
    models = [build_model(name) for name in args.models]

    print(f"Setups : {args.setups}")
    print(f"Models : {[m.name for m in models]}")
    print(f"Seed   : {args.seed}")
    print(f"Test%  : {args.test_fraction}")

    runner = BenchmarkRunner(
        models=models,
        setups=args.setups,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )

    results = runner.run(verbose=True)

    # Save results
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
