"""Top-level generators: one function per setup, all returning SurvivalData."""

from __future__ import annotations

import numpy as np

from .censoring import (
    apply_censoring,
    generate_combined_censoring,
    generate_covariate_dependent_censoring,
    generate_exponential_censoring,
    generate_uniform_censoring,
)
from .competing_risks import simulate_multistate
from scipy.integrate import quad
from scipy.optimize import brentq

from .configs import (
    Setup1Config,
    Setup2Config,
    Setup3Config,
    Setup4Config,
    Setup5Config,
    SetupInterpConfig,
)
from .covariates import generate_covariates_block, generate_covariates_standard
from .event_times import (
    sample_event_times_non_ph,
    sample_event_times_ph_weibull,
    sample_event_times_risk_score,
    sample_event_times_risk_score_gompertz,
)
from .types import SurvivalData


# ── Setup 1 – Clean PH, low-dimensional ─────────────────────────────

def generate_setup_1(config: Setup1Config | None = None, seed: int = 42) -> SurvivalData:
    """Weibull PH with linear effects, ~30-40 % censoring."""
    if config is None:
        config = Setup1Config(seed=seed)
    rng = np.random.default_rng(config.seed)

    X, names = generate_covariates_standard(
        n=config.n,
        n_continuous=config.n_continuous,
        n_binary=config.n_binary,
        rng=rng,
        ar1_rho=config.ar1_rho,
        n_categorical=config.n_categorical,
        categorical_levels=config.categorical_levels,
    )

    betas = np.array(config.betas, dtype=np.float64)
    T_event = sample_event_times_ph_weibull(X, betas, config.weibull_shape, config.weibull_scale, rng)

    C = generate_uniform_censoring(config.n, config.censor_max, rng)
    T_obs, E = apply_censoring(T_event, C)

    return SurvivalData(
        X=X, T=T_obs, E=E, feature_names=names,
        true_event_times=T_event, true_betas=betas,
        setup_name=config.setup_name, config=config,
    )


# ── Setup 2 – Non-PH, time-varying effects ──────────────────────────

def generate_setup_2(config: Setup2Config | None = None, seed: int = 42) -> SurvivalData:
    """Non-PH with time-varying coefficient on one covariate."""
    if config is None:
        config = Setup2Config(seed=seed)
    rng = np.random.default_rng(config.seed)

    X, names = generate_covariates_standard(
        n=config.n,
        n_continuous=config.n_continuous,
        n_binary=config.n_binary,
        rng=rng,
        ar1_rho=config.ar1_rho,
    )

    static_betas = np.array(config.static_betas, dtype=np.float64)

    T_event = sample_event_times_non_ph(
        X, static_betas,
        tv_idx=config.tv_covariate_idx,
        tv_intercept=config.tv_beta_intercept,
        tv_slope=config.tv_beta_slope,
        shape=config.weibull_shape,
        scale=config.weibull_scale,
        rng=rng,
    )

    C = generate_exponential_censoring(config.n, config.censor_rate, rng)
    T_obs, E = apply_censoring(T_event, C)

    true_betas = {
        "static": static_betas,
        "tv_idx": config.tv_covariate_idx,
        "tv_intercept": config.tv_beta_intercept,
        "tv_slope": config.tv_beta_slope,
    }

    return SurvivalData(
        X=X, T=T_obs, E=E, feature_names=names,
        true_event_times=T_event, true_betas=true_betas,
        setup_name=config.setup_name, config=config,
    )


# ── Setup 3 – High-dimensional, sparse, nonlinear ───────────────────

def _risk_score_setup3(X: np.ndarray, active_idx: np.ndarray) -> np.ndarray:
    """Nonlinear risk score for Setup 3.

    g(x) = 0.8*x[0] + 0.6*x[1] - 0.5*x[2]
           + 0.4*x[3]^2
           + 0.7*x[4]*x[5]
           + 0.9*I(x[6] > 0)
           - 0.3*x[7]
           + 0.5*x[8]^2
           + 0.6*x[9]*x[10]
           + sum_{j=11..19} 0.2*x[j]

    where x[k] refers to X[:, active_idx[k]].
    """
    a = X[:, active_idx]  # (n, 20)
    g = (
        0.8 * a[:, 0]
        + 0.6 * a[:, 1]
        - 0.5 * a[:, 2]
        + 0.4 * a[:, 3] ** 2
        + 0.7 * a[:, 4] * a[:, 5]
        + 0.9 * (a[:, 6] > 0).astype(float)
        - 0.3 * a[:, 7]
        + 0.5 * a[:, 8] ** 2
        + 0.6 * a[:, 9] * a[:, 10]
        + 0.2 * a[:, 11:20].sum(axis=1)
    )
    return g


def generate_setup_3(config: Setup3Config | None = None, seed: int = 42) -> SurvivalData:
    """High-dimensional sparse with nonlinear effects."""
    if config is None:
        config = Setup3Config(seed=seed)
    rng = np.random.default_rng(config.seed)

    X, names = generate_covariates_block(
        n=config.n,
        n_blocks=config.n_blocks,
        block_size=config.block_size,
        rho=config.within_block_rho,
        rng=rng,
    )

    # Active feature indices: first feature in each block
    active_idx = np.arange(config.n_active) * config.block_size  # [0, 100, 200, …, 1900]
    risk_scores = _risk_score_setup3(X, active_idx)

    T_event = sample_event_times_risk_score(
        risk_scores, config.weibull_shape, config.weibull_scale, rng,
    )

    C = generate_uniform_censoring(config.n, config.censor_max, rng)
    T_obs, E = apply_censoring(T_event, C)

    true_betas = {
        "type": "nonlinear",
        "active_indices": active_idx.tolist(),
        "description": (
            "g(x) = 0.8*x0 + 0.6*x1 - 0.5*x2 + 0.4*x3^2 + 0.7*x4*x5 "
            "+ 0.9*I(x6>0) - 0.3*x7 + 0.5*x8^2 + 0.6*x9*x10 "
            "+ 0.2*sum(x11..x19)"
        ),
    }

    return SurvivalData(
        X=X, T=T_obs, E=E, feature_names=names,
        true_event_times=T_event, true_betas=true_betas,
        setup_name=config.setup_name, config=config,
    )


# ── Setup 4 – Competing risks / multi-state ─────────────────────────

def generate_setup_4(config: Setup4Config | None = None, seed: int = 42) -> SurvivalData:
    """Multi-state: Healthy→Relapse→Death + Healthy→Death."""
    if config is None:
        config = Setup4Config(seed=seed)
    rng = np.random.default_rng(config.seed)

    X, names = generate_covariates_standard(
        n=config.n,
        n_continuous=config.n_continuous,
        n_binary=config.n_binary,
        rng=rng,
        ar1_rho=config.ar1_rho,
    )

    betas_01 = np.array(config.betas_01, dtype=np.float64)
    betas_02 = np.array(config.betas_02, dtype=np.float64)
    betas_12 = np.array(config.betas_12, dtype=np.float64)

    ms = simulate_multistate(
        X,
        betas_01, betas_02, betas_12,
        config.trans_01_shape, config.trans_01_scale,
        config.trans_02_shape, config.trans_02_scale,
        config.trans_12_shape, config.trans_12_scale,
        rng,
    )

    # For competing-risks analysis: first event from Healthy
    T_event = ms["T_first"]
    cause = ms["cause_first"]

    C = generate_combined_censoring(
        config.n, config.admin_censor_time, config.additional_censor_rate, rng,
    )
    T_obs, E = apply_censoring(T_event, C)

    # When censored, cause is 0
    cause_obs = np.where(E == 1, cause, 0)

    true_betas = {
        "betas_01": betas_01,
        "betas_02": betas_02,
        "betas_12": betas_12,
    }

    return SurvivalData(
        X=X, T=T_obs, E=E, feature_names=names,
        true_event_times=T_event, true_betas=true_betas,
        setup_name=config.setup_name, config=config,
        cause=cause_obs,
        state_history=ms["state_history"],
    )


# ── Setup 5 – Large-n, heavy censoring ──────────────────────────────

def _risk_score_setup5(X: np.ndarray, config: Setup5Config) -> np.ndarray:
    """Linear + quadratic age + age×treatment interaction."""
    betas = np.array(config.linear_betas, dtype=np.float64)
    g = X @ betas
    g += config.age_quadratic_coeff * X[:, 0] ** 2
    g += config.age_treatment_interaction * X[:, 0] * X[:, config.treatment_idx]
    return g


def generate_setup_5(config: Setup5Config | None = None, seed: int = 42) -> SurvivalData:
    """Large-n with Gompertz baseline and heavy covariate-dependent censoring."""
    if config is None:
        config = Setup5Config(seed=seed)
    rng = np.random.default_rng(config.seed)

    X, names = generate_covariates_standard(
        n=config.n,
        n_continuous=config.n_continuous,
        n_binary=config.n_binary,
        rng=rng,
        ar1_rho=config.ar1_rho,
        n_categorical=config.n_categorical,
        categorical_levels=tuple(3 for _ in range(config.n_categorical)),
    )

    risk_scores = _risk_score_setup5(X, config)

    T_event = sample_event_times_risk_score_gompertz(
        risk_scores, config.gompertz_b, config.gompertz_c, rng,
    )

    C = generate_covariate_dependent_censoring(
        config.n, X,
        covariate_idx=config.censor_covariate_idx,
        base_rate=config.base_censor_rate,
        covariate_effect=config.censor_covariate_effect,
        admin_time=config.admin_censor_time,
        rng=rng,
    )
    T_obs, E = apply_censoring(T_event, C)

    true_betas = {
        "linear": np.array(config.linear_betas),
        "age_quadratic": config.age_quadratic_coeff,
        "age_treatment_interaction": config.age_treatment_interaction,
    }

    return SurvivalData(
        X=X, T=T_obs, E=E, feature_names=names,
        true_event_times=T_event, true_betas=true_betas,
        setup_name=config.setup_name, config=config,
    )


# ── Setup Interp – Interpretability experiment ───────────────────────

def _risk_score_interp(X: np.ndarray, config: SetupInterpConfig) -> np.ndarray:
    """Static (time-constant) part of the risk score for the interpretability setup.

    g(x) = 0.8*x0 + 1.0*I(x1 > 0) + 0.5*x2^2
    """
    g = (
        config.beta_linear * X[:, config.linear_idx]
        + config.beta_threshold * (X[:, config.threshold_idx] > config.threshold_value).astype(float)
        + config.beta_quadratic * X[:, config.quadratic_idx] ** 2
    )
    return g


def _sample_interp_event_times(
    X: np.ndarray,
    g_static: np.ndarray,
    config: SetupInterpConfig,
    rng: np.random.Generator,
    t_max: float = 50.0,
) -> np.ndarray:
    """Event times for interpretability setup via numerical root-finding.

    h(t|x) = h0(t) * exp(g_static + beta(t) * x_tv)
    where beta(t) = tv_intercept + tv_slope * t

    Uses the same brentq approach as sample_event_times_non_ph but with
    a pre-computed nonlinear static risk score.
    """
    n = X.shape[0]
    U = rng.uniform(0.0, 1.0, size=n)
    targets = -np.log(U)

    shape = config.weibull_shape
    scale = config.weibull_scale
    tv_intercept = config.tv_intercept
    tv_slope = config.tv_slope
    tv_idx = config.tv_idx

    T = np.empty(n)
    for i in range(n):
        x_tv = X[i, tv_idx]
        exp_g = np.exp(g_static[i])

        def cumhaz(t: float) -> float:
            if t <= 0:
                return 0.0

            def integrand(s: float) -> float:
                h0 = shape * scale * s ** (shape - 1)
                return h0 * np.exp((tv_intercept + tv_slope * s) * x_tv)

            val, _ = quad(integrand, 0, t, limit=100)
            return exp_g * val

        def equation(t: float) -> float:
            return cumhaz(t) - targets[i]

        if cumhaz(t_max) < targets[i]:
            T[i] = t_max
        else:
            try:
                T[i] = brentq(equation, 1e-10, t_max, xtol=1e-8)
            except ValueError:
                T[i] = t_max

    return T


def generate_setup_interp(
    config: SetupInterpConfig | None = None, seed: int = 42,
) -> SurvivalData:
    """Interpretability scenario: 4 known effects (linear, threshold, nonlinear,
    time-varying) + 8 noise features.  Designed for SurvLIME / SurvSHAP(t) validation.
    """
    if config is None:
        config = SetupInterpConfig(seed=seed)
    rng = np.random.default_rng(config.seed)

    X, names = generate_covariates_standard(
        n=config.n,
        n_continuous=config.n_continuous,
        n_binary=config.n_binary,
        rng=rng,
        ar1_rho=config.ar1_rho,
    )

    g_static = _risk_score_interp(X, config)

    T_event = _sample_interp_event_times(X, g_static, config, rng)

    C = generate_uniform_censoring(config.n, config.censor_max, rng)
    T_obs, E = apply_censoring(T_event, C)

    # Rich true_betas with effect type labels for automated validation
    true_betas = {
        "effects": {
            names[config.linear_idx]: {
                "type": "linear", "beta": config.beta_linear, "sign": +1,
            },
            names[config.threshold_idx]: {
                "type": "threshold", "beta": config.beta_threshold,
                "sign": +1, "threshold": config.threshold_value,
            },
            names[config.quadratic_idx]: {
                "type": "nonlinear", "beta": config.beta_quadratic,
                "sign": +1, "form": "quadratic",
            },
            names[config.tv_idx]: {
                "type": "time_varying",
                "tv_intercept": config.tv_intercept,
                "tv_slope": config.tv_slope, "sign": +1,
            },
        },
        "important_features": [
            names[config.linear_idx],
            names[config.threshold_idx],
            names[config.quadratic_idx],
            names[config.tv_idx],
        ],
        "noise_features": [
            n for i, n in enumerate(names)
            if i not in {config.linear_idx, config.threshold_idx,
                         config.quadratic_idx, config.tv_idx}
        ],
        "n_important": 4,
    }

    return SurvivalData(
        X=X, T=T_obs, E=E, feature_names=names,
        true_event_times=T_event, true_betas=true_betas,
        setup_name=config.setup_name, config=config,
    )


# ── convenience ──────────────────────────────────────────────────────

def generate_all(seed: int = 42) -> dict[str, SurvivalData]:
    """Generate all 5 setups (each with a distinct seed)."""
    return {
        "setup_1": generate_setup_1(seed=seed),
        "setup_2": generate_setup_2(seed=seed + 1),
        "setup_3": generate_setup_3(seed=seed + 2),
        "setup_4": generate_setup_4(seed=seed + 3),
        "setup_5": generate_setup_5(seed=seed + 4),
    }
