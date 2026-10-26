# Algorithmic Options Pricing with Fourier Transforms

Research-oriented Python implementation of arbitrage-free European option pricing. The project bridges stochastic calculus, the Black--Scholes--Merton (BSM) PDE, risk-neutral valuation, and fast Fourier transforms (FFT).

Built as a **Stamatics, IIT Kanpur Summer Project (May--July 2026)**.

## What is implemented

- Exact simulation of risk-neutral geometric Brownian motion, discounted-martingale diagnostics, and realized quadratic variation.
- Black--Scholes--Merton European call/put prices and first/second-order hedging sensitivities.
- Risk-neutral Monte Carlo estimator with a 95% confidence interval.
- Discretely rebalanced delta-hedging P&L experiment.
- Reproducible historical SPY delta-hedging study using public daily closes and a trailing realized-volatility proxy.
- Carr--Madan damped Fourier pricing using an `O(N log N)` FFT, with vectorized strike interpolation.
- Automated tests that validate pricing identities and compare every numerical engine to the BSM solution.

## Mathematical setup

Under the risk-neutral measure \(\mathbb{Q}\), the asset follows

\[
dS_t = (r-q)S_tdt + \sigma S_tdW_t^{\mathbb{Q}}.
\]

No-arbitrage implies that a derivative value \(V(S,t)\) solves

\[
V_t + \tfrac{1}{2}\sigma^2S^2V_{SS} + (r-q)SV_S-rV=0.
\]

The FFT module prices a damped call transform of the terminal log-price characteristic function, then inverts it over an entire log-strike grid at once.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python examples/compare_methods.py
python examples/backtest_spy.py
```

Example output should show FFT prices within a few cents of BSM across strikes, plus a Monte Carlo confidence interval containing the analytical value.

## Repository layout

```text
src/quant_fourier_pricing/
  black_scholes.py  # BSM valuation and Greeks
  fft.py            # Carr--Madan characteristic-function pricing
  stochastic.py     # GBM, martingales, quadratic variation
  monte_carlo.py    # MC valuation and discrete delta hedging
tests/               # analytical and numerical validation
examples/            # runnable comparison
```

## Research extensions

The architecture is designed to support characteristic functions beyond BSM (Heston, Variance Gamma, Merton jump diffusion), calibration to implied-volatility surfaces, and transaction-cost-aware super/sub-hedging.

## Historical backtest methodology

`examples/backtest_spy.py` downloads daily SPY closes (2023-01-01 through 2025-08-29) from a public, versioned historical CSV and records the exact source in code. It opens a new 21-trading-day at-the-money call every 21 trading days, uses the preceding 21 days' realized volatility as the BSM volatility input, and rebalances a delta hedge daily. The output is the hedge error versus terminal payoff for each real SPY path.

The underlying prices are observed market data. The options are **model-priced**, rather than historical option-chain quotes; this is a controlled discrete-hedging experiment, not a claim of historical option-trading profits. For a market-price backtest, replace the volatility proxy with synchronized historical implied-volatility/option-trade data from an appropriately licensed provider.

### Latest reproducible run

The included result ledger covers 667 SPY daily closes from 2023-01-03 through 2025-08-29, producing 30 non-overlapping 21-trading-day trades. Its mean absolute hedging error was **$2.52 per share** and root mean squared error was **$4.30 per share**. See `outputs/spy_delta_hedge_backtest_2023_2025.json` for the complete trade-by-trade ledger and methodology inputs.

## Disclaimer

Educational research code only. It is not investment advice or production trading infrastructure.
