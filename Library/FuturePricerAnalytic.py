"""Closed-form pricing of equity (index) forwards / futures under full cost-of-carry.

Fair (par) future price:   F = S0 * exp((r - q - b) * T)
Long contract value:       V = exp(-r T) * (F - K) = S0 * exp(-(q + b) T) - K * exp(-r T)

where
    S0 = spot index level,        K = delivery (strike) price,
    r  = risk-free rate,          q = continuous dividend yield,
    b  = borrow / repo rate,      T = time to delivery (years).

A future/forward has no optionality, so it is volatility-independent and
gamma = vega = 0. Style mirrors Library/OptionPricerBSM1973.py: vectorized numpy
functions with inputs reshaped to column vectors. Drop-in for the repo's Library/.
"""
import numpy as np
from numpy.typing import NDArray


def _col(x) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64).reshape((-1, 1))


def future_fair_price(und_price, risk_free_rate, dividend_yield, borrow_rate, time_to_expiry):
    """Fair / par future (forward) price  F = S0 * exp((r - q - b) T)."""
    S0 = _col(und_price); r = _col(risk_free_rate); q = _col(dividend_yield)
    b = _col(borrow_rate); T = _col(time_to_expiry)
    return S0 * np.exp((r - q - b) * T)


def forward_value(und_price, delivery_price, risk_free_rate, dividend_yield, borrow_rate, time_to_expiry):
    """PV of a LONG forward/future struck at delivery_price K:
        V = S0 * exp(-(q + b) T) - K * exp(-r T).
    For a brand-new contract K = F gives V = 0."""
    S0 = _col(und_price); K = _col(delivery_price); r = _col(risk_free_rate)
    q = _col(dividend_yield); b = _col(borrow_rate); T = _col(time_to_expiry)
    return S0 * np.exp(-(q + b) * T) - K * np.exp(-r * T)


def forward_delta(risk_free_rate, dividend_yield, borrow_rate, time_to_expiry):
    """dV/dS0 = exp(-(q + b) T)  (spot delta of the contract value)."""
    q = _col(dividend_yield); b = _col(borrow_rate); T = _col(time_to_expiry)
    return np.exp(-(q + b) * T)


def forward_rho(delivery_price, risk_free_rate, time_to_expiry):
    """dV/dr = K * T * exp(-r T)."""
    K = _col(delivery_price); r = _col(risk_free_rate); T = _col(time_to_expiry)
    return K * T * np.exp(-r * T)


def forward_theta(und_price, delivery_price, risk_free_rate, dividend_yield, borrow_rate, time_to_expiry):
    """dV/dT (sensitivity of contract value to time-to-delivery)."""
    S0 = _col(und_price); K = _col(delivery_price); r = _col(risk_free_rate)
    q = _col(dividend_yield); b = _col(borrow_rate); T = _col(time_to_expiry)
    return -(q + b) * S0 * np.exp(-(q + b) * T) + r * K * np.exp(-r * T)


def implied_carry_from_price(market_future, und_price, time_to_expiry):
    """Closed-form inverse of F = S0 exp(cT) for total carry c = r - q - b:
        c = ln(F / S0) / T.
    (The notebook also shows the same inversion via Library.RootFinder.bisection.)"""
    F = _col(market_future); S0 = _col(und_price); T = _col(time_to_expiry)
    return np.log(F / S0) / T



def forward_ir01(delivery_price, risk_free_rate, time_to_expiry, bump=1e-4):
    """Value change for a +`bump` (default 1bp) parallel rise in r:
        IR01 = dV/dr * bump = K * T * exp(-r T) * bump.
    For a forward, rate risk enters only through discounting the fixed delivery price K."""
    return forward_rho(delivery_price, risk_free_rate, time_to_expiry) * bump


def forward_dividend_rho(und_price, dividend_yield, borrow_rate, time_to_expiry):
    """dV/dq = -S0 * T * exp(-(q + b) T). Equals dV/db (the borrow sensitivity)."""
    S0 = _col(und_price); q = _col(dividend_yield); b = _col(borrow_rate); T = _col(time_to_expiry)
    return -S0 * T * np.exp(-(q + b) * T)


def forward_div01(und_price, dividend_yield, borrow_rate, time_to_expiry, bump=1e-4):
    """Value change for a +`bump` (default 1bp) rise in the dividend yield:
        Div01 = dV/dq * bump. (Borrow01 is identical.)"""
    return forward_dividend_rho(und_price, dividend_yield, borrow_rate, time_to_expiry) * bump



def future_price_delta(risk_free_rate, dividend_yield, borrow_rate, time_to_expiry):
    """Sensitivity of the futures/forward PRICE to spot:
        dF/dS0 = exp((r - q - b) T).
    This is > 1 whenever the net carry r - q - b > 0 (the usual case for index futures,
    where r > q). It is NOT discounted, unlike the contract-value delta forward_delta()
    = exp(-(q + b) T) < 1; the two are linked by dF/dS0 = exp(r T) * dV/dS0."""
    r = _col(risk_free_rate); q = _col(dividend_yield); b = _col(borrow_rate); T = _col(time_to_expiry)
    return np.exp((r - q - b) * T)



def futures_convexity_logadj(equity_vol, rate_vol, correlation, time_to_expiry, mean_reversion=0.0):
    """Log futures/forward convexity adjustment under a Gaussian (Hull-White) short-rate model
    with constant equity vol. For lognormal spot and Gaussian rates,
        Fut0 = Fwd0 * exp(c),   c = Cov(ln S_T, integral_0^T r_s ds),
    and with Hull-White mean reversion a and instantaneous correlation rho between equity and
    the short rate,
        c = rho * sigma_S * sigma_r * (1/a) * (T - (1 - e^{-aT}) / a).
    As a -> 0 (Ho-Lee / Gaussian random-walk rates) this collapses to the textbook
        c -> rho * sigma_S * sigma_r * T^2 / 2,
    showing the adjustment grows ~ T^2. Sign follows rho: positive equity-rate correlation
    makes the future richer than the forward."""
    sS = _col(equity_vol); sr = _col(rate_vol); rho = _col(correlation)
    T = _col(time_to_expiry); a = _col(mean_reversion)
    small = np.abs(a) < 1e-10
    a_safe = np.where(small, 1.0, a)
    B = (1.0 - np.exp(-a_safe * T)) / a_safe
    term = np.where(small, 0.5 * T ** 2, (T - B) / a_safe)
    return rho * sS * sr * term


def futures_price_from_forward(forward_price, equity_vol, rate_vol, correlation,
                               time_to_expiry, mean_reversion=0.0):
    """Convexity-adjusted futures price  Fut0 = Fwd0 * exp(c)  (see futures_convexity_logadj)."""
    c = futures_convexity_logadj(equity_vol, rate_vol, correlation, time_to_expiry, mean_reversion)
    return _col(forward_price) * np.exp(c)


if __name__ == "__main__":
    S0, r, q, b, T, K = 5500., 0.043, 0.013, 0.0025, 0.25, 5400.
    F = future_fair_price(np.array(S0), np.array(r), np.array(q), np.array(b), np.array(T))
    V = forward_value(np.array(S0), np.array(K), np.array(r), np.array(q), np.array(b), np.array(T))
    print("fair price F :", float(F.reshape(-1)[0]))
    print("value (K=%.0f):" % K, float(V.reshape(-1)[0]))
    print("delta        :", float(forward_delta(np.array(r), np.array(q), np.array(b), np.array(T)).reshape(-1)[0]))
    c = implied_carry_from_price(F, np.array(S0), np.array(T))
    print("implied carry:", float(c.reshape(-1)[0]), "(should equal r-q-b =", r - q - b, ")")
