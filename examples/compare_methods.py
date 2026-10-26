"""Compare analytical, FFT, and Monte Carlo call prices."""

import numpy as np

from quant_fourier_pricing import BlackScholesModel, carr_madan_call_prices
from quant_fourier_pricing.monte_carlo import european_call_mc


def main() -> None:
    spot, rate, sigma, maturity = 100.0, 0.05, 0.20, 1.0
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    model = BlackScholesModel(spot, rate, sigma)
    fft_prices = carr_madan_call_prices(spot, rate, maturity, sigma, strikes)

    print(" strike | BSM price | FFT price | abs. error")
    print("--------|-----------|-----------|-----------")
    for strike, fft_price in zip(strikes, fft_prices):
        bsm = model.price(float(strike), maturity)
        print(f" {strike:6.1f} | {bsm:9.4f} | {fft_price:9.4f} | {abs(bsm - fft_price):9.5f}")
    estimate = european_call_mc(spot, 100.0, rate, sigma, maturity)
    print(f"\nMonte Carlo (K=100): {estimate.price:.4f} ± {1.96 * estimate.standard_error:.4f}")


if __name__ == "__main__":
    main()
