"""Wrapper classes adapting SurvivalModel to survlimepy / survshap interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.util import Surv

from src.models.base import SurvivalModel


# ── numpy 2.x compat patch for survlimepy ────────────────────────────
# survlimepy uses np.reshape(x, newshape=...) which was renamed to
# shape= in numpy 2.x.  Store the true original on np itself so that
# repeated reload() calls never chain multiple patched versions.
if not hasattr(np, "_orig_reshape_survlime_compat"):
    np._orig_reshape_survlime_compat = np.reshape  # type: ignore[attr-defined]

_orig_reshape = np._orig_reshape_survlime_compat  # always the real original


def _patched_reshape(a, *args, **kwargs):
    if "newshape" in kwargs:
        kwargs["shape"] = kwargs.pop("newshape")
    return _orig_reshape(a, *args, **kwargs)


np.reshape = _patched_reshape  # type: ignore[assignment]


# Metadata columns in survshap result DataFrames (not SHAP values).
_SURVSHAP_META_COLS = frozenset({
    "variable_str", "variable_name", "variable_value",
    "B", "aggregated_change",
})


def _shap_time_cols(result: pd.DataFrame) -> list[str]:
    """Return only the SHAP time-point columns from a survshap result."""
    return [c for c in result.columns if c not in _SURVSHAP_META_COLS]


# ── Helpers ──────────────────────────────────────────────────────────


def _is_sksurv_model(model: SurvivalModel) -> bool:
    """Check if the model wraps a scikit-survival estimator."""
    return hasattr(model, "_model") and hasattr(model._model, "predict_cumulative_hazard_function")


def _make_cumhaz_predict_fn(
    model: SurvivalModel,
    model_output_times: np.ndarray,
) -> callable:
    """Create a predict_fn suitable for survlimepy.

    survlimepy calls predict_fn(data) where data is (n_samples, n_features).
    Must return (n_samples, n_times) array of cumulative hazard values.
    """
    if _is_sksurv_model(model):
        def predict_fn(x):
            x_2d = np.atleast_2d(x)
            chf_fns = model._model.predict_cumulative_hazard_function(x_2d)
            return np.array([fn(model_output_times) for fn in chf_fns])
        return predict_fn

    # Generic fallback: H(t) = -log(S(t))
    def predict_fn(x):
        x_2d = np.atleast_2d(x)
        sf = model.predict_survival_function(x_2d, model_output_times)
        return -np.log(np.clip(sf, 1e-10, 1.0))

    return predict_fn


def _make_sksurv_adapter(model: SurvivalModel, X_train: np.ndarray, T_train: np.ndarray, E_train: np.ndarray):
    """Create an adapter object compatible with survshap for non-sksurv models."""

    class _SurvModelAdapter:
        """Lightweight adapter exposing sksurv-like prediction methods for survshap."""

        def __init__(self, model, X_train, T_train, E_train):
            self._model = model
            self._unique_times = np.sort(np.unique(T_train[E_train.astype(bool)]))
            if len(self._unique_times) == 0:
                self._unique_times = np.sort(np.unique(T_train))

        def predict_survival_function(self, X):
            times = self._unique_times
            sf = self._model.predict_survival_function(np.atleast_2d(X), times)

            # survshap expects list of StepFunction-like objects
            from sksurv.functions import StepFunction
            result = []
            for i in range(sf.shape[0]):
                result.append(StepFunction(times, sf[i]))
            return result

        def predict_cumulative_hazard_function(self, X):
            times = self._unique_times
            sf = self._model.predict_survival_function(np.atleast_2d(X), times)
            chf = -np.log(np.clip(sf, 1e-10, 1.0))

            # survshap expects list of StepFunction-like objects
            from sksurv.functions import StepFunction
            result = []
            for i in range(chf.shape[0]):
                result.append(StepFunction(times, chf[i]))
            return result

        def predict(self, X):
            return self._model.predict_risk(np.atleast_2d(X))

    return _SurvModelAdapter(model, X_train, T_train, E_train)


# ── SurvLIME Explainer ──────────────────────────────────────────────


class SurvLIMEExplainer:
    """Wrapper around survlimepy for local survival explanations.

    Parameters
    ----------
    model      : fitted SurvivalModel instance.
    X_train    : training feature matrix.
    T_train    : training times.
    E_train    : training event indicators.
    """

    def __init__(
        self,
        model: SurvivalModel,
        X_train: np.ndarray,
        T_train: np.ndarray,
        E_train: np.ndarray,
    ) -> None:
        from survlimepy import SurvLimeExplainer

        self._model = model
        self._X_train = X_train
        self._T_train = T_train
        self._E_train = E_train

        # Model output times: unique event times from training data
        event_mask = E_train.astype(bool)
        self._model_output_times = np.sort(np.unique(T_train[event_mask]))

        self._predict_fn = _make_cumhaz_predict_fn(model, self._model_output_times)

        # survlimepy has a bug: validate_events_times checks len(vector) > 1
        # instead of vector.ndim > 1, rejecting any numpy array with >1 element.
        # Workaround: pass as Python lists to bypass the numpy branch.
        self._explainer = SurvLimeExplainer(
            training_features=X_train,
            training_events=E_train.astype(bool).tolist(),
            training_times=T_train.tolist(),
            model_output_times=self._model_output_times,
        )

    def explain(
        self,
        x_instance: np.ndarray,
        num_samples: int = 1000,
    ) -> np.ndarray:
        """Compute local feature importance coefficients for one instance.

        Returns
        -------
        coefficients : (p,) array of Cox surrogate coefficients.
        """
        coefficients = self._explainer.explain_instance(
            data_row=x_instance,
            predict_fn=self._predict_fn,
            num_samples=num_samples,
        )
        return np.asarray(coefficients)

    def plot_weights(
        self,
        coefficients: np.ndarray,
        feature_names: list[str],
        true_important: list[str] | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Bar plot of SurvLIME coefficients, sorted by absolute value.

        Parameters
        ----------
        coefficients    : from explain().
        feature_names   : feature labels.
        true_important  : names of truly important features (highlighted).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names) * 0.4)))
        else:
            fig = ax.get_figure()

        order = np.argsort(feature_names)[::-1]  # alphabetical, reversed for barh (a at top)
        sorted_names = [feature_names[i] for i in order]
        sorted_coefs = coefficients[order]

        colors = []
        for name in sorted_names:
            if true_important and name in true_important:
                colors.append("steelblue")
            else:
                colors.append("lightgray")

        ax.barh(sorted_names, sorted_coefs, color=colors)
        ax.set_xlabel("SurvLIME coefficient")
        ax.set_title("Local feature importance (SurvLIME)")
        ax.axvline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        return fig

    def compare_survival_curves(
        self,
        x_instance: np.ndarray,
        times: np.ndarray,
        coefficients: np.ndarray | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Overlay black-box vs SurvLIME surrogate survival curves.

        Parameters
        ----------
        x_instance   : feature vector to explain.
        times        : time grid for the survival curves.
        coefficients : SurvLIME coefficients (from explain()). If None,
                       computed automatically.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        if coefficients is None:
            coefficients = self.explain(x_instance)

        # Black-box prediction
        sf_bb = self._model.predict_survival_function(
            np.atleast_2d(x_instance), times
        )[0]
        ax.plot(times, sf_bb, label="Black-box model", linewidth=2)

        # SurvLIME surrogate: Cox model S(t|x) = S_0(t)^exp(beta @ x)
        # Estimate baseline cumulative hazard from training data using KM
        from lifelines import CoxPHFitter
        df_train = pd.DataFrame(self._X_train, columns=[f"x{i}" for i in range(self._X_train.shape[1])])
        df_train["T"] = self._T_train
        df_train["E"] = self._E_train.astype(int)
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df_train, duration_col="T", event_col="E")
            # Predict baseline survival and adjust with SurvLIME coefficients
            baseline_surv = cph.predict_survival_function(
                pd.DataFrame(np.zeros((1, self._X_train.shape[1])),
                             columns=[f"x{i}" for i in range(self._X_train.shape[1])]),
                times=times,
            ).values.flatten()
            log_hr = np.dot(coefficients, x_instance)
            sf_surr = np.clip(baseline_surv ** np.exp(log_hr), 0.0, 1.0)
            ax.plot(times, sf_surr, "--", label="SurvLIME surrogate", linewidth=2)
        except Exception:
            # Fallback: use Nelson-Aalen baseline from training data
            from lifelines import NelsonAalenFitter
            naf = NelsonAalenFitter()
            naf.fit(self._T_train, event_observed=self._E_train.astype(bool))
            H0 = np.interp(times, naf.cumulative_hazard_.index.values,
                           naf.cumulative_hazard_.values.flatten(), left=0.0)
            log_hr = np.dot(coefficients, x_instance)
            sf_surr = np.clip(np.exp(-H0 * np.exp(log_hr)), 0.0, 1.0)
            ax.plot(times, sf_surr, "--", label="SurvLIME surrogate", linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("S(t)")
        ax.set_title("Black-box vs SurvLIME surrogate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


# ── SurvSHAP(t) Explainer ───────────────────────────────────────────


class SurvSHAPExplainer:
    """Wrapper around survshap for time-dependent Shapley values.

    Parameters
    ----------
    model      : fitted SurvivalModel instance.
    X_train    : training feature matrix.
    T_train    : training times.
    E_train    : training event indicators.
    """

    def __init__(
        self,
        model: SurvivalModel,
        X_train: np.ndarray,
        T_train: np.ndarray,
        E_train: np.ndarray,
        feature_names: list[str] | None = None,
        function_type: str = "sf",
    ) -> None:
        from survshap import SurvivalModelExplainer

        self._model = model
        self._X_train = X_train
        self._T_train = T_train
        self._E_train = E_train

        # Feature names for DataFrame conversion (survshap requires DataFrames)
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X_train.shape[1])]
        self._feature_names = feature_names
        self._function_type = function_type

        # Structured array for survshap
        self._y_train = Surv.from_arrays(E_train.astype(bool), T_train)

        # Get or create sksurv-compatible model
        if _is_sksurv_model(model):
            self._sksurv_model = model._model
        else:
            self._sksurv_model = _make_sksurv_adapter(
                model, X_train, T_train, E_train,
            )

        # Subsample background data for performance (SurvSHAP is O(n_background))
        max_background = 50
        if X_train.shape[0] > max_background:
            rng = np.random.default_rng(42)
            bg_idx = rng.choice(X_train.shape[0], max_background, replace=False)
            X_bg = X_train[bg_idx]
            y_bg = self._y_train[bg_idx]
        else:
            X_bg = X_train
            y_bg = self._y_train

        # survshap expects DataFrames with column names
        X_bg_df = pd.DataFrame(X_bg, columns=self._feature_names)

        self._explainer = SurvivalModelExplainer(
            model=self._sksurv_model,
            data=X_bg_df,
            y=y_bg,
        )

    def explain(
        self,
        x_instance: np.ndarray,
        times: np.ndarray,
    ) -> pd.DataFrame:
        """Compute SurvSHAP(t) values for one instance.

        Returns
        -------
        DataFrame with SHAP values per feature and time point.
        """
        from survshap import PredictSurvSHAP

        # survshap expects a DataFrame for new_observation
        obs_df = pd.DataFrame(
            np.atleast_2d(x_instance), columns=self._feature_names,
        )

        shap_obj = PredictSurvSHAP(function_type=self._function_type)
        shap_obj.fit(
            explainer=self._explainer,
            new_observation=obs_df,
            timestamps=times,
        )

        # Extract SHAP values — result is a DataFrame
        result = shap_obj.result
        if isinstance(result, np.ndarray):
            # Fallback: wrap raw array into a DataFrame
            result = pd.DataFrame(result, columns=[f"t = {t}" for t in times])
            result.insert(0, "variable_name", self._feature_names[:result.shape[0]])
        return result

    def explain_multiple(
        self,
        X_instances: np.ndarray,
        times: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute mean absolute SurvSHAP(t) across multiple instances.

        Returns
        -------
        DataFrame with (n_features,) mean |SHAP| values, averaged over
        time and individuals.
        """
        all_abs_shap = []

        for i in range(X_instances.shape[0]):
            result = self.explain(X_instances[i], times)
            # result is a DataFrame; extract the SHAP value columns
            shap_cols = _shap_time_cols(result)
            if shap_cols:
                shap_vals = result[shap_cols].values.astype(float)
                abs_mean = np.nanmean(np.abs(shap_vals), axis=1)
                all_abs_shap.append(abs_mean)

        if not all_abs_shap:
            return pd.DataFrame()

        # Average across individuals
        stacked = np.stack(all_abs_shap, axis=0)  # (n_individuals, n_features)
        global_importance = np.nanmean(stacked, axis=0)

        names = feature_names if feature_names is not None else self._feature_names
        return pd.DataFrame({
            "feature": names[:len(global_importance)],
            "mean_abs_shap": global_importance,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    @staticmethod
    def _extract_shap_matrix(result: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (times, var_names, shap_matrix) sorted alphabetically by feature name."""
        shap_cols = _shap_time_cols(result)

        def _parse_time(col: str) -> float:
            return float(col.split("=")[-1].strip()) if "=" in col else float(col)

        times = np.array([_parse_time(c) for c in shap_cols])
        shap_matrix = result[shap_cols].values.astype(float)
        var_names = result["variable_name"].values if "variable_name" in result.columns else np.arange(shap_matrix.shape[0]).astype(str)

        order = np.argsort(var_names)
        return times, var_names[order], shap_matrix[order]

    @staticmethod
    def _normalize_shap(shap_matrix: np.ndarray) -> np.ndarray:
        """Normalize SHAP values at each time point by sum of absolute values (eq. 5 in paper)."""
        denom = np.abs(shap_matrix).sum(axis=0, keepdims=True)
        denom = np.where(denom < 1e-12, 1.0, denom)
        return shap_matrix / denom

    def plot_shap_over_time(
        self,
        result: pd.DataFrame,
        feature_names: list[str],
        highlight_features: list[str] | None = None,
        normalized: bool = False,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Line plot of SurvSHAP(t) values vs time for key features.

        Parameters
        ----------
        result             : from explain() — one individual.
        feature_names      : all feature names.
        highlight_features : features to plot (default: all).
        normalized         : if True, normalize by sum of |SHAP| at each time point.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        times, var_names, shap_matrix = self._extract_shap_matrix(result)
        if normalized:
            shap_matrix = self._normalize_shap(shap_matrix)

        for i, name in enumerate(var_names):
            if i >= shap_matrix.shape[0]:
                break
            if highlight_features and name not in highlight_features:
                ax.plot(times, shap_matrix[i], alpha=0.15, color="gray", linewidth=0.8)
            else:
                ax.plot(times, shap_matrix[i], label=name, linewidth=2)

        ax.set_xlabel("Time")
        ylabel = "Normalized SurvSHAP(t)" if normalized else "SurvSHAP(t)"
        ax.set_ylabel(ylabel)
        ax.set_title("Time-dependent SHAP values" + (" (normalized)" if normalized else ""))
        ax.legend(loc="best", fontsize="small")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_shap_panel(
        self,
        result: pd.DataFrame,
        feature_names: list[str],
        highlight_features: list[str] | None = None,
        title: str = "",
    ) -> plt.Figure:
        """Two-row panel: raw SurvSHAP(t) on top, normalized on bottom (Fig 3 style).

        Parameters
        ----------
        result             : from explain() — one individual.
        feature_names      : all feature names.
        highlight_features : features to draw in colour (rest drawn faint).
        title              : overall figure title.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        times, var_names, shap_matrix = self._extract_shap_matrix(result)
        norm_matrix = self._normalize_shap(shap_matrix)

        for ax, matrix, ylabel in zip(
            axes,
            [shap_matrix, norm_matrix],
            ["SurvSHAP(t)", "Normalized SurvSHAP(t)"],
        ):
            for i, name in enumerate(var_names):
                if i >= matrix.shape[0]:
                    break
                if highlight_features and name not in highlight_features:
                    ax.plot(times, matrix[i], alpha=0.15, color="gray", linewidth=0.8)
                else:
                    ax.plot(times, matrix[i], label=name, linewidth=2)

            ax.set_ylabel(ylabel)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize="small")

        axes[-1].set_xlabel("Time")
        if title:
            fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        return fig

    def plot_global_importance(
        self,
        importance_df: pd.DataFrame,
        feature_names: list[str] | None = None,
        highlight_features: list[str] | None = None,
        true_important: list[str] | None = None,
        top_k: int | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Bar chart of global mean |SurvSHAP(t)| importance.

        Parameters
        ----------
        importance_df  : from explain_multiple().
        true_important : names of truly important features (highlighted).
        top_k          : show only top k features.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(3, len(importance_df) * 0.4)))
        else:
            fig = ax.get_figure()

        df = importance_df.copy()
        if top_k:
            df = df.head(top_k)

        # Sort alphabetically by feature name (reversed for barh so a appears at top)
        df = df.sort_values("feature", ascending=True).reset_index(drop=True)

        highlighted = highlight_features or true_important
        colors = []
        for name in df["feature"]:
            if highlighted and name in highlighted:
                colors.append("steelblue")
            else:
                colors.append("lightgray")

        ax.barh(df["feature"], df["mean_abs_shap"], color=colors)
        ax.set_xlabel("Mean |SurvSHAP(t)|")
        ax.set_title("Global feature importance (SurvSHAP(t))")
        fig.tight_layout()
        return fig
