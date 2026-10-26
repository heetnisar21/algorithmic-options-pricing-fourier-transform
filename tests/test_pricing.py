import numpy as np

from quant_fourier_pricing import BlackScholesModel, carr_madan_call_prices
from quant_fourier_pricing.monte_carlo import european_call_mc
from quant_fourier_pricing.stochastic import discounted_martingale, simulate_gbm


def test_bsm_reference_value() -> None:
    model = BlackScholesModel(100, 0.05, 0.2)
    assert abs(model.price(100, 1) - 10.4506) < 1e-3


def test_put_call_parity() -> None:
    model = BlackScholesModel(100, 0.05, 0.2, 0.01)
    lhs = model.price(105, 1.25, "call") - model.price(105, 1.25, "put")
    rhs = 100 * np.exp(-0.01 * 1.25) - 105 * np.exp(-0.05 * 1.25)
    assert abs(lhs - rhs) < 1e-10


def test_fft_agrees_with_closed_form() -> None:
    strikes = np.array([85.0, 100.0, 115.0])
    fft = carr_madan_call_prices(100, 0.03, 1.0, 0.25, strikes)
    model = BlackScholesModel(100, 0.03, 0.25)
    expected = np.array([model.price(k, 1.0) for k in strikes])
    assert np.max(np.abs(fft - expected)) < 0.03


def test_monte_carlo_interval_contains_bsm() -> None:
    estimate = european_call_mc(100, 100, 0.05, 0.2, 1, paths=200_000, seed=4)
    exact = BlackScholesModel(100, 0.05, 0.2).price(100, 1)
    assert estimate.confidence_interval[0] < exact < estimate.confidence_interval[1]


def test_discounted_stock_is_empirically_a_martingale() -> None:
    paths = simulate_gbm(100, 0.04, 0.2, 1.0, 12, 100_000, seed=42)
    discounted = discounted_martingale(paths, 0.04, 1.0)
    assert abs(discounted[:, -1].mean() - discounted[:, 0].mean()) < 0.2
