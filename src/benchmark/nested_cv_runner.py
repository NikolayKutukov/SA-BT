"""Nested cross-validation benchmark runner with hyperparameter tuning.

Implements the evaluation protocol from the guideline:
  - Outer 5-fold CV for unbiased performance estimation
  - Inner 5-fold CV for hyperparameter selection via random search
  - Aggregation of metrics across folds and repeats (mean +/- SD)
"""

from __future__ import annotations

import time
import traceback
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.generate import (
    generate_setup_1,
    generate_setup_2,
    generate_setup_3,
    generate_setup_4,
    generate_setup_5,
)
from src.evaluation.metrics import evaluate_model, concordance_index
from src.models import ALL_MODELS
from src.models.base import CauseSpecificWrapper
from src.tuning.search_spaces import sample_configs, SEARCH_BUDGETS
from src.tuning.random_search import InnerCVSearch

from .cv_results import CVResultsTable


GENERATORS = {
    "setup_1": generate_setup_1,
    "setup_2": generate_setup_2,
    "setup_3": generate_setup_3,
    "setup_4": generate_setup_4,
    "setup_5": generate_setup_5,
}

# Which (setup, model_name) pairs to skip
SKIP = {
    ("setup_3", "CoxPH"),  # p=2000 >> n=500, unpenalised Cox fails
}


class NestedCVRunner:
    """Nested cross-validation with HP tuning for all model x setup combinations.

    Parameters
    ----------
    model_names   : which models to run (keys in ALL_MODELS). Default: all.
    setups        : which setups to run. Default: all five.
    n_outer_folds : outer CV folds for performance estimation.
    n_inner_folds : inner CV folds for HP selection.
    n_repeats     : number of independent repeats (different data seeds).
    inner_metric  : metric used for HP selection in inner CV.
    seed          : base random seed.
    config_timeout : optional per-config time cap in seconds (not enforced
                     via signal — just skips configs that exceed it).
    """

    def __init__(
        self,
        model_names: list[str] | None = None,
        setups: list[str] | None = None,
        n_outer_folds: int = 5,
        n_inner_folds: int = 5,
        n_repeats: int = 1,
        inner_metric: str = "c_index_ipcw",
        seed: int = 42,
        config_timeout: float | None = None,
    ) -> None:
        self.model_names = model_names or list(ALL_MODELS.keys())
        self.setups = setups or list(GENERATORS.keys())
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_repeats = n_repeats
        self.inner_metric = inner_metric
        self.seed = seed
        self.config_timeout = config_timeout

    def run(self, verbose: bool = True) -> CVResultsTable:
        """Execute the full nested CV benchmark.

        Returns
        -------
        CVResultsTable with fold-level results for all (repeat, setup, model, fold).
        """
        results = CVResultsTable()

        for repeat in range(self.n_repeats):
            data_seed = self.seed + repeat * 100

            for setup_idx, setup_name in enumerate(self.setups):
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Repeat {repeat + 1}/{self.n_repeats} | Setup: {setup_name}")
                    print(f"{'='*60}")

                # Generate data
                data = GENERATORS[setup_name](seed=data_seed + setup_idx)
                n_samples = len(data.T)

                if verbose:
                    print(data.summary())

                is_competing = data.cause is not None

                # Stratification column
                stratify_col = (
                    data.cause.astype(int) if is_competing
                    else data.E.astype(int)
                )

                # Outer CV
                n_outer = min(self.n_outer_folds, _min_class_count(stratify_col))
                if n_outer < 2:
                    n_outer = 2

                outer_cv = StratifiedKFold(
                    n_splits=n_outer,
                    shuffle=True,
                    random_state=self.seed + repeat,
                )

                for fold_idx, (train_idx, test_idx) in enumerate(
                    outer_cv.split(data.X, stratify_col)
                ):
                    X_train = data.X[train_idx]
                    X_test = data.X[test_idx]
                    T_train = data.T[train_idx]
                    T_test = data.T[test_idx]
                    E_train = data.E[train_idx]
                    E_test = data.E[test_idx]

                    cause_train = cause_test = None
                    if is_competing:
                        cause_train = data.cause[train_idx]
                        cause_test = data.cause[test_idx]

                    # Evaluation time grid
                    times = _eval_times(T_test, E_test)

                    for model_name in self.model_names:
                        model_class = ALL_MODELS[model_name]
                        display_name = model_class.name if hasattr(model_class, 'name') else model_name

                        if (setup_name, display_name) in SKIP:
                            if verbose:
                                print(f"  SKIP {display_name} (incompatible)")
                            results.add(
                                repeat, setup_name, display_name, fold_idx,
                                {"status": "skipped"}, n_samples=n_samples,
                            )
                            continue

                        if verbose:
                            print(
                                f"\n  Fold {fold_idx + 1}/{n_outer} | "
                                f"{display_name} | tuning ..."
                            )

                        try:
                            # ── Inner CV: HP search ──
                            rng = np.random.default_rng(
                                self.seed + repeat * 1000 + fold_idx * 100
                                + hash(model_name) % 1000
                            )
                            configs = sample_configs(
                                model_name,
                                SEARCH_BUDGETS.get(model_name, 10),
                                rng,
                            )

                            searcher = InnerCVSearch(
                                model_class=model_class,
                                model_name=model_name,
                                n_inner_folds=self.n_inner_folds,
                                metric=self.inner_metric,
                                seed=self.seed + repeat * 10 + fold_idx,
                            )

                            best_config, inner_score = searcher.search(
                                X_train, T_train, E_train, configs,
                                cause=cause_train,
                            )

                            if verbose:
                                print(f"    Best config: {best_config}")
                                direction = "(higher=better)" if searcher.higher_is_better else "(lower=better)"
                                score_str = f"{inner_score:.4f}" if not np.isnan(inner_score) else "N/A"
                                print(
                                    f"    Inner {self.inner_metric}: "
                                    f"{score_str} {direction}"
                                )

                            # ── Refit on full outer-train ──
                            t0 = time.perf_counter()

                            if is_competing and not model_class.supports_competing_risks:
                                metrics = self._fit_eval_cause_specific(
                                    model_class, best_config,
                                    X_train, T_train, cause_train,
                                    X_test, T_test, cause_test,
                                    times,
                                )
                            elif is_competing and model_class.supports_competing_risks:
                                metrics = self._fit_eval_native_cr(
                                    model_class, best_config,
                                    X_train, T_train, E_train, cause_train,
                                    X_test, T_test, E_test, cause_test,
                                    times,
                                )
                            else:
                                model = model_class(**best_config)
                                model.fit(X_train, T_train, E_train)
                                metrics = evaluate_model(
                                    model, X_train, T_train, E_train,
                                    X_test, T_test, E_test, times,
                                )

                            t_fit = time.perf_counter() - t0

                            results.add(
                                repeat, setup_name, display_name, fold_idx,
                                metrics, best_config=best_config,
                                inner_cv_score=inner_score,
                                fit_time=t_fit, n_samples=n_samples,
                            )

                            if verbose:
                                _print_metrics(display_name, metrics, t_fit)

                        except Exception as exc:
                            if verbose:
                                print(f"  ERROR {display_name}: {exc}")
                                traceback.print_exc()
                            results.add(
                                repeat, setup_name, display_name, fold_idx,
                                {"status": "error", "error": str(exc)},
                                n_samples=n_samples,
                            )

        return results

    @staticmethod
    def _fit_eval_cause_specific(
        model_class, config,
        X_train, T_train, cause_train,
        X_test, T_test, cause_test,
        times,
    ) -> dict:
        """Fit cause-specific models and evaluate per cause."""
        factory = lambda: model_class(**config)  # noqa: E731
        wrapper = CauseSpecificWrapper(factory)
        wrapper.fit(X_train, T_train, cause_train, cause=cause_train)

        metrics: dict = {}
        for cause_k in wrapper._causes:
            risk_k = wrapper.predict_cause_risk(X_test, cause_k)
            E_k_test = (cause_test == cause_k).astype(float)
            if E_k_test.sum() > 0:
                c = concordance_index(T_test, E_k_test, risk_k)
                metrics[f"c_index_cause_{cause_k}"] = c

        return metrics

    @staticmethod
    def _fit_eval_native_cr(
        model_class, config,
        X_train, T_train, E_train, cause_train,
        X_test, T_test, E_test, cause_test,
        times,
    ) -> dict:
        """Fit native competing risks model and evaluate."""
        model = model_class(**config)
        model.fit(X_train, T_train, cause_train)

        metrics: dict = {}
        risk = model.predict_risk(X_test)
        overall_E = (cause_test > 0).astype(float)
        metrics["c_index_overall"] = concordance_index(T_test, overall_E, risk)

        causes = sorted(set(int(c) for c in cause_test if c > 0))
        for cause_k in causes:
            try:
                cif = model.predict_cumulative_incidence(X_test, times, cause_k)
                metrics[f"has_cif_cause_{cause_k}"] = True
            except NotImplementedError:
                pass

        return metrics


def _eval_times(T: np.ndarray, E: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Compute evaluation time grid from test data."""
    event_times = T[E.astype(bool)]
    if len(event_times) > 0:
        return np.linspace(
            event_times.min() + 1e-6,
            event_times.max() - 1e-6,
            n_points,
        )
    return np.linspace(T.min(), T.max(), n_points)


def _min_class_count(y: np.ndarray) -> int:
    """Minimum number of samples in any class."""
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def _print_metrics(name: str, metrics: dict, t_fit: float | None = None) -> None:
    """Pretty-print metrics for a single model."""
    print(f"    {name}:")
    if t_fit is not None:
        print(f"      fit time: {t_fit:.2f}s")
    for k, v in metrics.items():
        if k.startswith("_") or k == "status":
            continue
        if isinstance(v, float):
            print(f"      {k}: {v:.4f}")
        else:
            print(f"      {k}: {v}")
