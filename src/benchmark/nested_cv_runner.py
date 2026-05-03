"""Nested cross-validation benchmark runner with hyperparameter tuning.

Implements the evaluation protocol:
  - Outer K-fold CV for unbiased performance estimation
  - Inner K-fold CV for hyperparameter selection via random search
  - Aggregation of metrics across folds and repeats (mean +/- SD)
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.loaders import LOADERS, SKIP
from src.evaluation.metrics import evaluate_model
from src.models import ALL_MODELS
from src.tuning.search_spaces import sample_configs, SEARCH_BUDGETS
from src.tuning.random_search import InnerCVSearch

from .cv_results import CVResultsTable


class NestedCVRunner:
    """Nested cross-validation with HP tuning for all model x dataset combinations.

    Parameters
    ----------
    model_names   : which models to run (keys in ALL_MODELS). Default: all.
    datasets      : which datasets to run. Default: all four.
    n_outer_folds : outer CV folds for performance estimation.
    n_inner_folds : inner CV folds for HP selection.
    n_repeats     : number of independent repeats.
    inner_metric  : metric used for HP selection in inner CV.
    seed          : base random seed.
    """

    def __init__(
        self,
        model_names: list[str] | None = None,
        datasets: list[str] | None = None,
        n_outer_folds: int = 5,
        n_inner_folds: int = 3,
        n_repeats: int = 1,
        inner_metric: str = "c_index_ipcw",
        seed: int = 42,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_names = model_names or list(ALL_MODELS.keys())
        self.datasets = datasets or list(LOADERS.keys())
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_repeats = n_repeats
        self.inner_metric = inner_metric
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_key(self, repeat: int, dataset: str, model: str, fold: int) -> str:
        return f"r{repeat}_{dataset}_{model}_f{fold}"

    def _load_cache(self, key: str) -> dict | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def run(self, verbose: bool = True) -> CVResultsTable:
        results = CVResultsTable()

        for repeat in range(self.n_repeats):
            for dataset_name in self.datasets:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Repeat {repeat + 1}/{self.n_repeats} | Dataset: {dataset_name}")
                    print(f"{'='*60}")

                data = LOADERS[dataset_name]()
                n_samples = len(data.T)

                if verbose:
                    print(data.summary())

                stratify_col = data.E.astype(int)

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

                    for model_name in self.model_names:
                        model_class = ALL_MODELS[model_name]
                        display_name = model_class.name if hasattr(model_class, 'name') else model_name

                        if (dataset_name, display_name) in SKIP:
                            if verbose:
                                print(f"  SKIP {display_name} (incompatible)")
                            results.add(
                                repeat, dataset_name, display_name, fold_idx,
                                {"status": "skipped"}, n_samples=n_samples,
                            )
                            continue

                        if verbose:
                            print(
                                f"\n  Fold {fold_idx + 1}/{n_outer} | "
                                f"{display_name} | tuning ..."
                            )

                        try:
                            cache_key = self._cache_key(repeat, dataset_name, model_name, fold_idx)
                            cached = self._load_cache(cache_key)

                            if cached is not None:
                                # Restore from cache — skip tuning
                                best_config = cached["best_config"]
                                inner_score = cached["inner_score"]
                                if verbose:
                                    print(f"    [cached] Best config: {best_config}")
                            else:
                                # Inner CV: HP search
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
                                )

                                self._save_cache(cache_key, {
                                    "best_config": best_config,
                                    "inner_score": float(inner_score) if not np.isnan(inner_score) else None,
                                })

                            if verbose:
                                if cached is None:
                                    print(f"    Best config: {best_config}")
                                direction = "(higher=better)" if self.inner_metric in ("c_index", "c_index_ipcw", "mean_auc") else "(lower=better)"
                                score_str = f"{inner_score:.4f}" if inner_score is not None and not np.isnan(float(inner_score if inner_score is not None else float('nan'))) else "N/A"
                                print(
                                    f"    Inner {self.inner_metric}: "
                                    f"{score_str} {direction}"
                                )

                            # Refit on full outer-train
                            t0 = time.perf_counter()
                            model = model_class(**best_config)
                            model.fit(X_train, T_train, E_train)
                            metrics = evaluate_model(
                                model, X_train, T_train, E_train,
                                X_test, T_test, E_test,
                            )
                            t_fit = time.perf_counter() - t0

                            results.add(
                                repeat, dataset_name, display_name, fold_idx,
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
                                repeat, dataset_name, display_name, fold_idx,
                                {"status": "error", "error": str(exc)},
                                n_samples=n_samples,
                            )

        return results



def _min_class_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def _print_metrics(name: str, metrics: dict, t_fit: float | None = None) -> None:
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
