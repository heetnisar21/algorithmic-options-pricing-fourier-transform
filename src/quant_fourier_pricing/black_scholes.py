"""Black--Scholes--Merton formulas and Greeks under a risk-neutral measure."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Literal


OptionType = Literal["call", "put"]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class BlackScholesModel:
    """Constant-coefficient diffusion model dS = (r-q)Sdt + sigma SdW."""

    spot: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0 or self.volatility <= 0:
            raise ValueError("spot and volatility must be strictly positive")

    def d1_d2(self, strike: float, maturity: float) -> tuple[float, float]:
        if strike <= 0 or maturity <= 0:
            raise ValueError("strike and maturity must be strictly positive")
        root_t = sqrt(maturity)
        d1 = (
            log(self.spot / strike)
            + (self.rate - self.dividend_yield + 0.5 * self.volatility**2) * maturity
        ) / (self.volatility * root_t)
        return d1, d1 - self.volatility * root_t

    def price(self, strike: float, maturity: float, option_type: OptionType = "call") -> float:
        d1, d2 = self.d1_d2(strike, maturity)
        discounted_spot = self.spot * exp(-self.dividend_yield * maturity)
        discounted_strike = strike * exp(-self.rate * maturity)
        call = discounted_spot * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)
        if option_type == "call":
            return call
        if option_type == "put":
            return call - discounted_spot + discounted_strike
        raise ValueError("option_type must be 'call' or 'put'")

    def delta(self, strike: float, maturity: float, option_type: OptionType = "call") -> float:
        d1, _ = self.d1_d2(strike, maturity)
        scale = exp(-self.dividend_yield * maturity)
        if option_type == "call":
            return scale * _norm_cdf(d1)
        if option_type == "put":
            return scale * (_norm_cdf(d1) - 1.0)
        raise ValueError("option_type must be 'call' or 'put'")

    def gamma(self, strike: float, maturity: float) -> float:
        d1, _ = self.d1_d2(strike, maturity)
        return exp(-self.dividend_yield * maturity) * _norm_pdf(d1) / (
            self.spot * self.volatility * sqrt(maturity)
        )


def black_scholes_price(
    spot: float, strike: float, maturity: float, rate: float, volatility: float,
    option_type: OptionType = "call", dividend_yield: float = 0.0,
) -> float:
    """Price a European option via the BSM PDE's closed-form solution."""
    return BlackScholesModel(spot, rate, volatility, dividend_yield).price(
        strike, maturity, option_type
    )
