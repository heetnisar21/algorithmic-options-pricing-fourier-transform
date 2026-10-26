import numpy as np

from quant_fourier_pricing.backtest import run_delta_hedge_backtest, summarize_backtest


def test_backtest_produces_finite_trade_metrics() -> None:
    dates = [f"2024-01-{day:02d}" for day in range(1, 91)]
    prices = 100 * np.exp(np.linspace(0, 0.12, 90))
    trades = run_delta_hedge_backtest(list(zip(dates, prices)), maturity_days=10, vol_window=21)
    summary = summarize_backtest(trades)
    assert summary["trades"] > 0
    assert np.isfinite(summary["root_mean_squared_hedging_error"])
