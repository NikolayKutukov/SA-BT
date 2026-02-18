"""Benchmark runner: fit models across setups and collect evaluation metrics."""

from __future__ import annotations

import time
import traceback

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.generate import (
    generate_setup_1,
    generate_setup_2,
    generate_setup_3,
    generate_setup_4,
    generate_setup_5,
)
from src.evaluation.metrics import evaluate_model, concordance_index
from src.models.base import SurvivalModel, CauseSpecificWrapper
from .results import ResultsTable


GENERATORS = {
    "setup_1": generate_setup_1,
    "setup_2": generate_setup_2,
    "setup_3": generate_setup_3,
    "setup_4": generate_setup_4,
    "setup_5": generate_setup_5,
}

# Which (setup, model) pairs to skip — model can't handle the setup
SKIP = {
    ("setup_3", "CoxPH"),  # p=2000 >> n=500, unpenalised Cox fails
}


class BenchmarkRunner:
    """Run all model × setup combinations and collect metrics.

    Parameters
    ----------
    models : list of SurvivalModel instances (already configured).
    setups : which setups to run (default: all five).
    seed   : base random seed for data generation.
    test_fraction : held-out test set proportion.
    """

    def __init__(
        self,
        models: list[SurvivalModel],
        setups: list[str] | None = None,
        seed: int = 42,
        test_fraction: float = 0.2,
    ) -> None:
        self.models = models
        self.setups = setups or list(GENERATORS.keys())
        self.seed = seed
        self.test_fraction = test_fraction

    def run(self, verbose: bool = True) -> ResultsTable:
        results = ResultsTable()

        for i, setup_name in enumerate(self.setups):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Setup: {setup_name}")
                print(f"{'='*60}")

            data = GENERATORS[setup_name](seed=self.seed + i)

            if verbose:
                print(data.summary())

            is_competing = data.cause is not None

            # Train / test split (stratified by event indicator)
            stratify_col = data.E.astype(int)
            X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
                data.X, data.T, data.E,
                test_size=self.test_fraction,
                random_state=self.seed,
                stratify=stratify_col,
            )

            # For competing risks, also split cause
            cause_train = cause_test = None
            if is_competing:
                _, _, cause_train, cause_test = train_test_split(
                    data.X, data.cause,
                    test_size=self.test_fraction,
                    random_state=self.seed,
                    stratify=stratify_col,
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
                key = (setup_name, model.name)

                if key in SKIP:
                    if verbose:
                        print(f"  SKIP {model.name} (incompatible with {setup_name})")
                    results.add(setup_name, model.name, {"status": "skipped"})
                    continue

                if verbose:
                    print(f"\n  Fitting {model.name} ...")

                try:
                    # ── Competing risks: cause-specific approach ──
                    if is_competing and not model.supports_competing_risks:
                        metrics = self._run_cause_specific(
                            model, X_train, T_train, cause_train,
                            X_test, T_test, cause_test, times, verbose,
                        )
                        t_fit = metrics.pop("_fit_time", None)
                        results.add(
                            setup_name,
                            f"{model.name}_cause_specific",
                            metrics,
                            fit_time=t_fit,
                        )
                        continue

                    # ── Competing risks: native approach ──
                    if is_competing and model.supports_competing_risks:
                        metrics = self._run_native_competing(
                            model, X_train, T_train, E_train, cause_train,
                            X_test, T_test, E_test, cause_test, times, verbose,
                        )
                        t_fit = metrics.pop("_fit_time", None)
                        results.add(setup_name, model.name, metrics, fit_time=t_fit)
                        continue

                    # ── Standard single-event ──
                    t0 = time.perf_counter()
                    model.fit(X_train, T_train, E_train)
                    t_fit = time.perf_counter() - t0

                    metrics = evaluate_model(
                        model, X_train, T_train, E_train,
                        X_test, T_test, E_test, times,
                    )
                    results.add(setup_name, model.name, metrics, fit_time=t_fit)

                    if verbose:
                        self._print_metrics(model.name, metrics, t_fit)

                except Exception as exc:
                    if verbose:
                        print(f"  ERROR {model.name}: {exc}")
                        traceback.print_exc()
                    results.add(setup_name, model.name, {"status": "error", "error": str(exc)})

        return results

    def _run_cause_specific(
        self, model, X_train, T_train, cause_train,
        X_test, T_test, cause_test, times, verbose,
    ) -> dict:
        """Fit cause-specific models and compute per-cause metrics."""
        factory = type(model)
        wrapper = CauseSpecificWrapper(factory)

        t0 = time.perf_counter()
        wrapper.fit(X_train, T_train, cause_train, cause=cause_train)
        t_fit = time.perf_counter() - t0

        metrics: dict = {"_fit_time": t_fit}

        for cause_k in wrapper._causes:
            risk_k = wrapper.predict_cause_risk(X_test, cause_k)
            E_k_test = (cause_test == cause_k).astype(float)

            if E_k_test.sum() > 0:
                c = concordance_index(T_test, E_k_test, risk_k)
                metrics[f"c_index_cause_{cause_k}"] = c

        if verbose:
            self._print_metrics(f"{model.name}_cause_specific", metrics, t_fit)

        return metrics

    def _run_native_competing(
        self, model, X_train, T_train, E_train, cause_train,
        X_test, T_test, E_test, cause_test, times, verbose,
    ) -> dict:
        """Fit native competing risks model and compute per-cause metrics."""
        t0 = time.perf_counter()
        # For native competing risks, pass cause codes as E
        model.fit(X_train, T_train, cause_train)
        t_fit = time.perf_counter() - t0

        metrics: dict = {"_fit_time": t_fit}

        # Overall survival metrics
        risk = model.predict_risk(X_test)
        overall_E = (cause_test > 0).astype(float)
        metrics["c_index_overall"] = concordance_index(T_test, overall_E, risk)

        # Per-cause CIF metrics
        causes = sorted(set(int(c) for c in cause_test if c > 0))
        for cause_k in causes:
            try:
                cif = model.predict_cumulative_incidence(X_test, times, cause_k)
                metrics[f"has_cif_cause_{cause_k}"] = True
            except NotImplementedError:
                pass

        if verbose:
            self._print_metrics(model.name, metrics, t_fit)

        return metrics

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
