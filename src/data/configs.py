from __future__ import annotations

from dataclasses import dataclass


# ── Base ─────────────────────────────────────────────────────────────

@dataclass
class BaseConfig:
    n: int = 0
    p: int = 0
    seed: int = 42
    setup_name: str = ""


# ── Setup 1 – Clean PH, low-dimensional ─────────────────────────────

@dataclass
class Setup1Config(BaseConfig):
    n: int = 1000
    p: int = 10
    setup_name: str = "setup_1_clean_ph"

    # Covariates: 5 continuous, 3 binary, 2 categorical (ordinal-encoded)
    n_continuous: int = 5
    n_binary: int = 3
    n_categorical: int = 2
    categorical_levels: tuple = (3, 4)
    ar1_rho: float = 0.3  # AR(1) for continuous vars

    # Baseline hazard – Weibull PH:  h_0(t) = a * lam * t^(a-1)
    weibull_shape: float = 1.5   # a
    weibull_scale: float = 0.1   # lambda  (median T ≈ 3.6 at X=0)

    # True coefficients (5 nonzero out of 10)
    betas: tuple = (0.8, -0.6, 0.5, 0.0, 0.0,  # continuous
                    1.0, 0.0, -0.7,               # binary
                    0.0, 0.0)                      # categorical

    # Censoring: Uniform(0, censor_max) → target ~30-40 %
    censor_max: float = 12.0


# ── Setup 2 – Non-PH, time-varying effects ──────────────────────────

@dataclass
class Setup2Config(BaseConfig):
    n: int = 1500
    p: int = 20
    setup_name: str = "setup_2_non_ph"

    n_continuous: int = 10
    n_binary: int = 10
    ar1_rho: float = 0.5

    # Baseline – Weibull (mild increasing hazard)
    weibull_shape: float = 1.2
    weibull_scale: float = 0.08

    # Static betas (p=20; index 1 is handled by time-varying part)
    static_betas: tuple = (
        0.7, 0.0, -0.5, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0,   # continuous
        0.8, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # binary
    )
    # Time-varying covariate X_1: beta(t) = tv_intercept + tv_slope * t
    tv_covariate_idx: int = 1
    tv_beta_intercept: float = 1.5   # strong early
    tv_beta_slope: float = -0.3       # weakens over time; crosses 0 at t=5

    # Censoring: Exp(rate) → target ~40-50 %
    censor_rate: float = 0.15


# ── Setup 3 – High-dimensional, sparse, nonlinear ───────────────────

@dataclass
class Setup3Config(BaseConfig):
    n: int = 500
    p: int = 2000
    setup_name: str = "setup_3_high_dim"

    # Block-correlated covariates
    n_blocks: int = 20
    block_size: int = 100  # 20 * 100 = 2000
    within_block_rho: float = 0.6

    # Baseline – Weibull
    weibull_shape: float = 1.3
    weibull_scale: float = 0.05

    # 20 active features (first in each block: 0, 100, 200, …, 1900)
    n_active: int = 20

    # Censoring: Uniform(0, censor_max) → target ~30-50 %
    censor_max: float = 12.0


# ── Setup 4 – Competing risks / multi-state ─────────────────────────

@dataclass
class Setup4Config(BaseConfig):
    n: int = 2000
    p: int = 15
    setup_name: str = "setup_4_competing_risks"

    n_continuous: int = 8
    n_binary: int = 7
    ar1_rho: float = 0.3

    # Transition 0→1 (Healthy → Relapse)
    trans_01_shape: float = 1.3
    trans_01_scale: float = 0.08
    betas_01: tuple = (0.6, -0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.8, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Transition 0→2 (Healthy → Death, direct)
    trans_02_shape: float = 1.1
    trans_02_scale: float = 0.03
    betas_02: tuple = (0.3, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.7, -0.4, 0.0, 0.0, 0.0)

    # Transition 1→2 (Relapse → Death)
    trans_12_shape: float = 1.8
    trans_12_scale: float = 0.15
    betas_12: tuple = (0.4, 0.0, -0.3, 0.0, 0.6, 0.0, 0.0, 0.0,
                       0.5, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0)

    # Censoring: min(admin_time, Exp(1/additional_rate))
    admin_censor_time: float = 15.0
    additional_censor_rate: float = 0.10


# ── Setup 5 – Large-n, heavy censoring ──────────────────────────────

@dataclass
class Setup5Config(BaseConfig):
    n: int = 50_000
    p: int = 30
    setup_name: str = "setup_5_large_n"

    n_continuous: int = 10
    n_binary: int = 12
    n_categorical: int = 8
    ar1_rho: float = 0.2

    # Baseline – Gompertz: h_0(t) = b * exp(c*t)
    gompertz_b: float = 0.02
    gompertz_c: float = 0.08

    # Linear betas (30 features)
    linear_betas: tuple = (
        0.6, -0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # continuous
        0.5, 0.0, -0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0,    # binary
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     # binary/cat
    )
    # Nonlinear additions to risk score
    age_quadratic_coeff: float = 0.1       # 0.1 * X_0^2
    age_treatment_interaction: float = -0.3  # -0.3 * X_0 * X_15
    treatment_idx: int = 15

    # Censoring: heavy, covariate-dependent
    admin_censor_time: float = 20.0
    base_censor_rate: float = 0.08
    censor_covariate_idx: int = 0       # "age"
    censor_covariate_effect: float = -0.02  # younger → more censoring


# ── Setup Interp – Interpretability experiment ────────────────────────

@dataclass
class SetupInterpConfig(BaseConfig):
    n: int = 1000
    p: int = 12
    setup_name: str = "setup_interp"

    # Covariates: 8 continuous (AR(1)), 4 binary
    n_continuous: int = 8
    n_binary: int = 4
    ar1_rho: float = 0.3

    # Baseline hazard – Weibull PH (same as Setup 1)
    weibull_shape: float = 1.5
    weibull_scale: float = 0.1

    # Important feature effects
    # x0: linear
    beta_linear: float = 0.8
    linear_idx: int = 0
    # x1: threshold at 0
    beta_threshold: float = 1.0
    threshold_idx: int = 1
    threshold_value: float = 0.0
    # x2: quadratic (nonlinear)
    beta_quadratic: float = 0.5
    quadratic_idx: int = 2
    # x3: time-varying β(t) = tv_intercept + tv_slope * t
    tv_idx: int = 3
    tv_intercept: float = 0.5
    tv_slope: float = 0.3

    # Censoring: Uniform(0, censor_max) → target ~30-40 %
    censor_max: float = 12.0
