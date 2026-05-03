"""Benchmark runner: fit models across datasets and collect evaluation metrics."""

from __future__ import annotations

import time
import traceback

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loaders import LOADERS, SKIP
from src.evaluation.metrics import evaluate_model
from src.models.base import SurvivalModel
from .results import ResultsTable


class BenchmarkRunner:
    """Run all model x dataset combinations and collect metrics.

    Parameters
    ----------
    models : list of SurvivalModel instances (already configured).
    datasets : which datasets to run (default: all four).
    seed : random seed for train/test split.
    test_fraction : held-out test set proportion.
    """

    def __init__(
        self,
        models: list[SurvivalModel],
        datasets: list[str] | None = None,
        seed: int = 42,
        test_fraction: float = 0.2,
    ) -> None:
        self.models = models
        self.datasets = datasets or list(LOADERS.keys())
        self.seed = seed
        self.test_fraction = test_fraction

    def run(self, verbose: bool = True) -> ResultsTable:
        results = ResultsTable()

        for dataset_name in self.datasets:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Dataset: {dataset_name}")
                print(f"{'='*60}")

            data = LOADERS[dataset_name]()

            if verbose:
                print(data.summary())

            # Train / test split (stratified by event indicator)
            X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
                data.X, data.T, data.E,
                test_size=self.test_fraction,
                random_state=self.seed,
                stratify=data.E.astype(int),
            )

            # Evaluation time grid
            event_times = T_test[E_test.astype(bool)]
            if len(event_times) > 0:
                times = np.linspace(
                    event_times.min() + 1e-6,
                    event_times.max() - 1e-6,
                    100,
                )
            else:
                times = np.linspace(T_test.min(), T_test.max(), 100)

            for model in self.models:
                key = (dataset_name, model.name)

                if key in SKIP:
                    if verbose:
                        print(f"  SKIP {model.name} (incompatible with {dataset_name})")
                    results.add(dataset_name, model.name, {"status": "skipped"})
                    continue

                if verbose:
                    print(f"\n  Fitting {model.name} ...")

                try:
                    t0 = time.perf_counter()
                    model.fit(X_train, T_train, E_train)
                    t_fit = time.perf_counter() - t0

                    metrics = evaluate_model(
                        model, X_train, T_train, E_train,
                        X_test, T_test, E_test, times,
                    )
                    results.add(dataset_name, model.name, metrics, fit_time=t_fit)

                    if verbose:
                        self._print_metrics(model.name, metrics, t_fit)

                except Exception as exc:
                    if verbose:
                        print(f"  ERROR {model.name}: {exc}")
                        traceback.print_exc()
                    results.add(dataset_name, model.name, {"status": "error", "error": str(exc)})

        return results

    @staticmethod
    def _print_metrics(name: str, metrics: dict, t_fit: float | None = None) -> None:
        print(f"  {name}:")
        if t_fit is not None:
            print(f"    fit time: {t_fit:.2f}s")
        for k, v in metrics.items():
            if k.startswith("_"):
                continue
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
