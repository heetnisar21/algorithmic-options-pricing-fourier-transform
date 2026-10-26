"""Run a real-SPY-price delta-hedging study and save the results."""

from pathlib import Path

from quant_fourier_pricing.backtest import (
    download_spy_daily,
    run_delta_hedge_backtest,
    save_results,
    summarize_backtest,
)


def main() -> None:
    observations = download_spy_daily("2023-01-01", "2025-08-29")
    trades = run_delta_hedge_backtest(observations)
    summary = summarize_backtest(trades)
    destination = Path("outputs/spy_delta_hedge_backtest_2023_2025.json")
    destination.parent.mkdir(exist_ok=True)
    save_results(str(destination), trades, summary)
    print("SPY delta-hedging backtest: 2023-01-01 to 2025-08-29")
    for metric, value in summary.items():
        print(f"{metric}: {value:.6f}" if isinstance(value, float) else f"{metric}: {value}")
    print(f"Saved detailed trade ledger to {destination}")


if __name__ == "__main__":
    main()
