"""Simulation and diagnostics for geometric Brownian motion."""

from __future__ import annotations

import numpy as np


def simulate_gbm(
    spot: float, rate: float, volatility: float, maturity: float, steps: int,
    paths: int, dividend_yield: float = 0.0, seed: int | None = None,
) -> np.ndarray:
    """Simulate risk-neutral GBM paths with exact lognormal increments."""
    if min(spot, volatility, maturity) <= 0 or steps < 1 or paths < 1:
        raise ValueError("invalid GBM parameters")
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    shocks = rng.standard_normal((paths, steps))
    log_increments = (rate - dividend_yield - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    log_paths = np.c_[np.zeros(paths), np.cumsum(log_increments, axis=1)]
    return spot * np.exp(log_paths)


def quadratic_variation(paths: np.ndarray) -> np.ndarray:
    """Realized quadratic variation of each path over its sampled grid."""
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_times >= 2)")
    return np.sum(np.diff(paths, axis=1) ** 2, axis=1)


def discounted_martingale(paths: np.ndarray, rate: float, maturity: float) -> np.ndarray:
    """Discount a GBM path at every observation time for martingale checks."""
    times = np.linspace(0.0, maturity, paths.shape[1])
    return paths * np.exp(-rate * times)
