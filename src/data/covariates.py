"""Covariate generation: continuous (correlated), binary, categorical."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, cholesky


# ── correlation matrices ─────────────────────────────────────────────

def make_ar1_correlation_matrix(p: int, rho: float) -> np.ndarray:
    """AR(1): Sigma[i,j] = rho^|i-j|."""
    idx = np.arange(p)
    return np.power(rho, np.abs(idx[:, None] - idx[None, :]))


def make_block_correlation_matrix(
    n_blocks: int, block_size: int, rho: float,
) -> np.ndarray:
    """Block-diagonal compound-symmetry correlation.

    Each block: (1-rho)*I + rho*J  (ones matrix).
    Blocks are independent of each other.
    """
    single_block = (1 - rho) * np.eye(block_size) + rho * np.ones((block_size, block_size))
    blocks = [single_block for _ in range(n_blocks)]
    return block_diag(*blocks)


# ── sampling ─────────────────────────────────────────────────────────

def generate_continuous(
    n: int,
    p: int,
    rng: np.random.Generator,
    correlation: np.ndarray | None = None,
) -> np.ndarray:
    """N(0, Sigma) via Cholesky: X = Z @ L^T where Z ~ N(0, I)."""
    Z = rng.standard_normal((n, p))
    if correlation is not None:
        L = cholesky(correlation, lower=True)
        Z = Z @ L.T
    return Z


def generate_binary(
    n: int, p: int, rng: np.random.Generator, prob: float = 0.5,
) -> np.ndarray:
    """Bernoulli(prob), returns 0/1 array of shape (n, p)."""
    return rng.binomial(1, prob, size=(n, p)).astype(np.float64)


def generate_categorical(
    n: int, n_levels: int, rng: np.random.Generator,
) -> np.ndarray:
    """Uniform-discrete {0, 1, …, n_levels-1}, shape (n,)."""
    return rng.integers(0, n_levels, size=n).astype(np.float64)


# ── master dispatcher ────────────────────────────────────────────────

def generate_covariates_standard(
    n: int,
    n_continuous: int,
    n_binary: int,
    rng: np.random.Generator,
    ar1_rho: float = 0.0,
    n_categorical: int = 0,
    categorical_levels: tuple = (),
) -> tuple[np.ndarray, list[str]]:
    """Generate mixed covariates for Setups 1/2/4/5.

    Returns
    -------
    X : (n, p) array
    feature_names : list of length p
    """
    parts: list[np.ndarray] = []
    names: list[str] = []

    # continuous (possibly correlated)
    if n_continuous > 0:
        corr = make_ar1_correlation_matrix(n_continuous, ar1_rho) if ar1_rho > 0 else None
        parts.append(generate_continuous(n, n_continuous, rng, corr))
        names.extend([f"X_cont_{i}" for i in range(n_continuous)])

    # binary
    if n_binary > 0:
        parts.append(generate_binary(n, n_binary, rng))
        names.extend([f"X_bin_{i}" for i in range(n_binary)])

    # categorical (ordinal-encoded as integers)
    for k, lev in enumerate(categorical_levels[:n_categorical]):
        col = generate_categorical(n, lev, rng).reshape(-1, 1)
        parts.append(col)
        names.append(f"X_cat_{k}")

    X = np.hstack(parts)
    return X, names


def generate_covariates_block(
    n: int,
    n_blocks: int,
    block_size: int,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Block-correlated continuous covariates for Setup 3.

    Returns
    -------
    X : (n, n_blocks * block_size)
    feature_names : list of length p
    """
    corr = make_block_correlation_matrix(n_blocks, block_size, rho)
    X = generate_continuous(n, n_blocks * block_size, rng, corr)
    names = [f"X_{i}" for i in range(X.shape[1])]
    return X, names
