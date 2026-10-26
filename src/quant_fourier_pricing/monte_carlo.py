"""Monte Carlo estimators and delta-hedging experiments."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .black_scholes import BlackScholesModel
from .stochastic import simulate_gbm


@dataclass(frozen=True)
class MonteCarloEstimate:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]


def european_call_mc(
    spot: float, strike: float, rate: float, volatility: float, maturity: float,
    paths: int = 100_000, seed: int | None = 7,
) -> MonteCarloEstimate:
    """Estimate an arbitrage-free call value under the risk-neutral measure."""
    terminal = simulate_gbm(spot, rate, volatility, maturity, 1, paths, seed=seed)[:, -1]
    discounted_payoffs = np.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
    price = float(discounted_payoffs.mean())
    se = float(discounted_payoffs.std(ddof=1) / np.sqrt(paths))
    return MonteCarloEstimate(price, se, (price - 1.96 * se, price + 1.96 * se))


def delta_hedge_pnl(
    paths: np.ndarray, strike: float, rate: float, volatility: float,
    maturity: float,
) -> np.ndarray:
    """Terminal P&L of a discretely rebalanced BSM delta hedge for a call."""
    steps = paths.shape[1] - 1
    if steps < 1:
        raise ValueError("at least one rebalancing interval is required")
    dt = maturity / steps
    model = BlackScholesModel(float(paths[0, 0]), rate, volatility)
    cash = np.full(paths.shape[0], model.price(strike, maturity))
    delta = np.full(paths.shape[0], model.delta(strike, maturity))
    cash -= delta * paths[:, 0]
    for j in range(1, steps):
        cash *= np.exp(rate * dt)
        tau = maturity - j * dt
        next_delta = np.array([BlackScholesModel(s, rate, volatility).delta(strike, tau) for s in paths[:, j]])
        cash -= (next_delta - delta) * paths[:, j]
        delta = next_delta
    portfolio = cash * np.exp(rate * dt) + delta * paths[:, -1]
    return portfolio - np.maximum(paths[:, -1] - strike, 0.0)
