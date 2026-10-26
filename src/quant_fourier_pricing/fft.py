"""Carr--Madan fast Fourier transform pricing for European calls."""

from __future__ import annotations

import numpy as np


def black_scholes_log_cf(
    u: np.ndarray | complex,
    maturity: float,
    spot: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> np.ndarray | complex:
    """Characteristic function of log(S_T) under the risk-neutral measure."""
    mean = np.log(spot) + (rate - dividend_yield - 0.5 * volatility**2) * maturity
    return np.exp(1j * u * mean - 0.5 * volatility**2 * maturity * u**2)


def carr_madan_call_prices(
    spot: float,
    rate: float,
    maturity: float,
    volatility: float,
    strikes: np.ndarray,
    dividend_yield: float = 0.0,
    alpha: float = 1.5,
    n: int = 2**14,
    eta: float = 0.08,
) -> np.ndarray:
    """Price calls at arbitrary strikes with the damped Carr--Madan FFT.

    The method evaluates the characteristic function only O(N) times and
    interpolates the FFT log-strike grid. `alpha > 0` damps the call payoff.
    """
    strikes = np.asarray(strikes, dtype=float)
    if np.any(strikes <= 0) or maturity <= 0 or spot <= 0 or volatility <= 0:
        raise ValueError("spot, strikes, maturity, and volatility must be positive")
    if alpha <= 0 or n & (n - 1):
        raise ValueError("alpha must be positive and n must be a power of two")

    v = np.arange(n, dtype=float) * eta
    lambd = 2.0 * np.pi / (n * eta)
    b = 0.5 * n * lambd
    k_grid = -b + np.arange(n) * lambd
    shifted_u = v - 1j * (alpha + 1.0)
    cf = black_scholes_log_cf(
        shifted_u, maturity, spot, rate, volatility, dividend_yield
    )
    denominator = alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-rate * maturity) * cf / denominator

    # Simpson weights improve integration accuracy without changing complexity.
    weights = np.ones(n)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights /= 3.0
    transform = np.fft.fft(np.exp(1j * b * v) * psi * eta * weights)
    call_grid = np.exp(-alpha * k_grid) * transform.real / np.pi
    return np.interp(np.log(strikes), k_grid, call_grid)
