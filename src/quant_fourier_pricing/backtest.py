"""Historical delta-hedging studies using real underlying-price observations.

This module deliberately distinguishes the observed underlying from the option:
the option is priced by BSM with a documented volatility proxy. Historical
option-chain prices are not inferred or presented as observed market quotes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from io import StringIO
from urllib.request import urlopen
import csv
import json

import numpy as np

from .black_scholes import BlackScholesModel


TRADING_DAYS = 252


@dataclass(frozen=True)
class HedgeTrade:
    opened: str
    closed: str
    spot_open: float
    spot_close: float
    strike: float
    volatility_proxy: float
    option_premium: float
    payoff: float
    hedging_pnl: float


def download_spy_daily(start: str, end: str) -> list[tuple[str, float]]:
    """Download SPY daily closes from a public, versioned historical CSV.

    The source is a static public GitHub dataset containing Yahoo Finance
    history through 2025-08-29. Dates are ISO strings. The fixed source URL
    and end date make this particular research run reproducible.
    """
    url = (
        "https://raw.githubusercontent.com/gsl0001/"
        "daily-spy-data-from-2000-2025/refs/heads/main/spy_data.csv"
    )
    with urlopen(url, timeout=30) as response:  # nosec B310: fixed public HTTPS source
        text = response.read().decode("utf-8")
    # The source has two metadata lines after its header; data begins at line 4.
    rows = csv.DictReader(StringIO("\n".join(text.splitlines()[2:])))
    prices = [
        (row["Price"], float(row["Close"]))
        for row in rows
        if start <= row.get("Price", "") <= end and row.get("Close") not in {None, "N/D"}
    ]
    if len(prices) < 50:
        raise ValueError(f"received insufficient SPY data: {len(prices)} observations")
    return prices


def realized_volatility(closes: np.ndarray, window: int = 21) -> float:
    """Annualized sample standard deviation of log returns."""
    if len(closes) <= window:
        raise ValueError("not enough observations for realized-volatility window")
    returns = np.diff(np.log(closes[-(window + 1) :]))
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def run_delta_hedge_backtest(
    observations: list[tuple[str, float]],
    maturity_days: int = 21,
    rebalance_every: int = 1,
    vol_window: int = 21,
    rate: float = 0.04,
) -> list[HedgeTrade]:
    """Backtest successive ATM calls on observed closes with daily delta hedging.

    A trade begins every ``maturity_days`` observations after the volatility
    lookback. Calls are model-valued using trailing realized volatility, and
    hedge error is marked against the observed terminal underlying payoff.
    This isolates discrete-hedging risk on real market paths.
    """
    if maturity_days < 2 or rebalance_every < 1:
        raise ValueError("maturity_days must be >= 2 and rebalance_every must be >= 1")
    dates = [date for date, _ in observations]
    closes = np.array([close for _, close in observations], dtype=float)
    trades: list[HedgeTrade] = []
    for start in range(vol_window, len(closes) - maturity_days, maturity_days):
        stop = start + maturity_days
        sigma = max(realized_volatility(closes[: start + 1], vol_window), 0.05)
        spot = float(closes[start])
        strike = spot  # at-the-money contract
        model = BlackScholesModel(spot, rate, sigma)
        premium = model.price(strike, maturity_days / TRADING_DAYS)
        delta = model.delta(strike, maturity_days / TRADING_DAYS)
        cash = premium - delta * spot
        for day in range(1, maturity_days):
            cash *= np.exp(rate / TRADING_DAYS)
            if day % rebalance_every == 0:
                tau = (maturity_days - day) / TRADING_DAYS
                next_delta = BlackScholesModel(float(closes[start + day]), rate, sigma).delta(strike, tau)
                cash -= (next_delta - delta) * closes[start + day]
                delta = next_delta
        cash *= np.exp(rate / TRADING_DAYS)
        terminal = float(closes[stop])
        payoff = max(terminal - strike, 0.0)
        hedge_value = cash + delta * terminal
        trades.append(HedgeTrade(
            opened=dates[start], closed=dates[stop], spot_open=spot, spot_close=terminal,
            strike=strike, volatility_proxy=sigma, option_premium=premium, payoff=payoff,
            hedging_pnl=hedge_value - payoff,
        ))
    return trades


def summarize_backtest(trades: list[HedgeTrade]) -> dict[str, float | int]:
    """Return transparent, contract-level hedging-error metrics."""
    if not trades:
        raise ValueError("backtest produced no trades")
    pnl = np.array([trade.hedging_pnl for trade in trades])
    return {
        "trades": len(trades),
        "mean_hedging_pnl": float(pnl.mean()),
        "median_hedging_pnl": float(np.median(pnl)),
        "std_hedging_pnl": float(pnl.std(ddof=1)),
        "mean_absolute_hedging_error": float(np.abs(pnl).mean()),
        "root_mean_squared_hedging_error": float(np.sqrt(np.mean(pnl**2))),
        "worst_hedging_pnl": float(pnl.min()),
        "best_hedging_pnl": float(pnl.max()),
        "positive_hedge_rate": float((pnl > 0).mean()),
    }


def save_results(path: str, trades: list[HedgeTrade], summary: dict[str, float | int]) -> None:
    """Write machine-readable, reproducible results to JSON."""
    with open(path, "w", encoding="utf-8") as output:
        json.dump({"summary": summary, "trades": [asdict(trade) for trade in trades]}, output, indent=2)
