import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
# from Library.ImportLibs import *


def heston_call(
        und_strike: NDArray[np.float64],
        und_price: NDArray[np.float64],
        initial_variance: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    und_strike = und_strike.reshape((-1, 1))
    und_price = und_price.reshape((-1, 1))
    initial_variance = initial_variance.reshape((-1, 1))
    kappa = kappa.reshape((-1, 1))
    theta = theta.reshape((-1, 1))
    sigma = sigma.reshape((-1, 1))
    rhosv = rhosv.reshape((-1, 1))
    lamb = lamb.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    # args = (und_strike, und_price, initial_variance, kappa, theta, sigma, rhosv, lamb, time_to_expiry, risk_free_rate, dividend_yield)
    # P1 = _P1(*args)
    # P2 = _P2(*args)
    # price = und_price * P1 - (und_strike * np.exp(-(risk_free_rate - dividend_yield) * time_to_expiry) * P2)
    price = []
    for i in range(time_to_expiry.shape[0]):
        strike = und_strike[i, 0]
        spot = und_price[i, 0]
        tau = time_to_expiry[i, 0]
        r = risk_free_rate[i, 0]
        d = dividend_yield[i, 0]
        args = (
            strike, spot, initial_variance[0, 0], kappa[0, 0], theta[0, 0],
            sigma[0, 0], rhosv[0, 0], lamb[0, 0], tau, r, d
        )
        P1 = _P1(*args)
        P2 = _P2(*args)
        price.append(spot * P1 - (strike * np.exp(-(r - d) * tau) * P2))
    return np.array(price).reshape((-1, 1))


def heston_put(
        und_strike: NDArray[np.float64],
        und_price: NDArray[np.float64],
        initial_variance: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    und_strike = und_strike.reshape((-1, 1))
    und_price = und_price.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))

    args = (
        und_strike, und_price, initial_variance, kappa, theta, sigma, rhosv,
        lamb, time_to_expiry, risk_free_rate, dividend_yield
    )
    put  = heston_call(*args)
    put += und_strike * np.exp(-risk_free_rate * time_to_expiry) - und_price * np.exp(-dividend_yield * time_to_expiry)
    return put


def _P1(
        und_strike: NDArray[np.float64],
        und_price: NDArray[np.float64],
        initial_variance: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    # P, umax, N = 0, 100, 10_000
    # dphi = umax / N
    #
    # for i in range(1, N):
    #     phi = dphi * (2 * i + 1) * .5
    #     characteristic_func_args = (
    #         phi, und_strike, und_price, initial_variance, kappa, theta, sigma, rhosv,
    #         lamb, time_to_expiry, risk_free_rate, dividend_yield
    #     )
    #     P += _integrand1(*characteristic_func_args) * dphi

    characteristic_func_args = (
        und_strike, und_price, initial_variance, kappa, theta, sigma, rhosv,
        lamb, time_to_expiry, risk_free_rate, dividend_yield
    )

    P, _  = np.real(quad(_integrand1, 0., 100., args=characteristic_func_args))
    P /= np.pi
    P += np.array(.5)
    return P


def _P2(
        und_strike: NDArray[np.float64],
        und_price: NDArray[np.float64],
        initial_variance: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    # P, umax, N = 0, 100, 10_000
    # dphi = umax / N
    #
    # for i in range(1, N):
    #     phi = dphi * (2 * i + 1) * .5
    #     characteristic_func_args = (
    #         phi, und_strike, und_price, initial_variance, kappa, theta, sigma, rhosv,
    #         lamb, time_to_expiry, risk_free_rate, dividend_yield
    #     )
    #     P += _integrand2(*characteristic_func_args) * dphi

    characteristic_func_args = (
        und_strike, und_price, initial_variance, kappa, theta, sigma, rhosv,
        lamb, time_to_expiry, risk_free_rate, dividend_yield
    )

    P, _  = np.real(quad(_integrand2, 0., 100., args=characteristic_func_args))
    P /= np.pi
    P += np.array(.5)
    return P


def _integrand1(
        phi: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        und_price: NDArray[np.float64],
        initial_variance: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    characteristic_func_args = (
        phi, und_price, initial_variance, kappa, theta, sigma, rhosv,
        lamb, time_to_expiry, risk_free_rate, dividend_yield
    )
    integrand  = np.exp(-1j * phi * np.log(und_strike))
    integrand *= _characteristic_func1(*characteristic_func_args)
    integrand /= 1j * phi
    return integrand


def _integrand2(
        phi: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        und_price: NDArray[np.float64],
        initial_variance: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    characteristic_func_args = (
        phi, und_price, initial_variance, kappa, theta, sigma, rhosv,
        lamb, time_to_expiry, risk_free_rate, dividend_yield
    )
    integrand  = np.exp(-1j * phi * np.log(und_strike))
    integrand *= _characteristic_func2(*characteristic_func_args)
    integrand /= 1j * phi
    return integrand


def _characteristic_func1(
        phi: NDArray[np.float64],
        und_price: NDArray[np.float64],
        initial_variance: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    C_args = (phi, kappa, theta, sigma, rhosv, lamb, time_to_expiry, risk_free_rate, dividend_yield)
    D_args = (phi, kappa, sigma, rhosv, lamb, time_to_expiry)
    C1 = _C1(*C_args)
    D1 = _D1(*D_args)
    return np.exp(C1 + (D1 * initial_variance) + (1j * phi * np.log(und_price)))


def _characteristic_func2(
        phi: NDArray[np.float64],
        und_price: NDArray[np.float64],
        initial_variance: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    C_args = (phi, kappa, theta, sigma, rhosv, lamb, time_to_expiry, risk_free_rate, dividend_yield)
    D_args = (phi, kappa, sigma, rhosv, lamb, time_to_expiry)
    C2 = _C2(*C_args)
    D2 = _D2(*D_args)
    return np.exp(C2 + (D2 * initial_variance) + (1j * phi * np.log(und_price)))


def _C1(
        phi: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    a_args = (kappa, theta)
    b_args = (kappa, sigma, rhosv, lamb)
    d_args = (phi, kappa, sigma, rhosv, lamb)
    g_args = (phi, kappa, sigma, rhosv, lamb)
    C1  = (_b1(*b_args) - (rhosv * sigma * phi * 1j) + _d1(*d_args)) * time_to_expiry
    C1 -= 2. * np.log((1. - _g1(*g_args) * np.exp(_d1(*d_args) * time_to_expiry)) / (1. - _g1(*g_args)))
    C1 *= _a(*a_args) / sigma**2
    C1 += (risk_free_rate - dividend_yield) * phi * 1j * time_to_expiry
    return C1


def _C2(
        phi: NDArray[np.float64],
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64]
) -> NDArray[np.float64]:
    a_args = (kappa, theta)
    b_args = (kappa, lamb)
    d_args = (phi, kappa, sigma, rhosv, lamb)
    g_args = (phi, kappa, sigma, rhosv, lamb)
    C2  = (_b2(*b_args) - (rhosv * sigma * phi * 1j) + _d2(*d_args)) * time_to_expiry
    C2 -= 2. * np.log((1. - _g2(*g_args) * np.exp(_d2(*d_args) * time_to_expiry)) / (1. - _g2(*g_args)))
    C2 *= _a(*a_args) / sigma**2
    C2 += (risk_free_rate - dividend_yield) * phi * 1j * time_to_expiry
    return C2


def _D1(
        phi: NDArray[np.float64],
        kappa: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:
    b_args = (kappa, sigma, rhosv, lamb)
    d_args = (phi, kappa, sigma, rhosv, lamb)
    g_args = (phi, kappa, sigma, rhosv, lamb)
    D1  = (1. - np.exp(_d1(*d_args) * time_to_expiry)) / (1. - _g1(*g_args) * np.exp(_d1(*d_args) * time_to_expiry))
    D1 *= (_b1(*b_args) - (rhosv * sigma * phi * 1j) + _d1(*d_args)) / sigma**2
    return D1


def _D2(
        phi: NDArray[np.float64],
        kappa: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:
    b_args = (kappa, lamb)
    d_args = (phi, kappa, sigma, rhosv, lamb)
    g_args = (phi, kappa, sigma, rhosv, lamb)
    D2  = (1. - np.exp(_d2(*d_args) * time_to_expiry)) / (1. - _g2(*g_args) * np.exp(_d2(*d_args) * time_to_expiry))
    D2 *= (_b2(*b_args) - (rhosv * sigma * phi * 1j) + _d2(*d_args)) / sigma**2
    return D2


def _g1(
        phi: NDArray[np.float64],
        kappa: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64]
) -> NDArray[np.float64]:
    b_args = (kappa, sigma, rhosv, lamb)
    d_args = (phi, kappa, sigma, rhosv, lamb)
    g1  = _b1(*b_args) - (rhosv * sigma * phi * 1j) + _d1(*d_args)
    g1 /= _b1(*b_args) - (rhosv * sigma * phi * 1j) - _d1(*d_args)
    return g1


def _g2(
        phi: NDArray[np.float64],
        kappa: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64]
) -> NDArray[np.float64]:
    b_args = (kappa, lamb)
    d_args = (phi, kappa, sigma, rhosv, lamb)
    g2  = _b2(*b_args) - (rhosv * sigma * phi * 1j) + _d2(*d_args)
    g2 /= _b2(*b_args) - (rhosv * sigma * phi * 1j) - _d2(*d_args)
    return g2


def _d1(
        phi: NDArray[np.float64],
        kappa: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64]
) -> NDArray[np.float64]:
    b_args = (kappa, sigma, rhosv, lamb)
    d1  = (rhosv * sigma * phi * 1j - _b1(*b_args))**2
    d1 -= sigma**2 * (2. * _u1() * phi * 1j - phi**2)
    return np.sqrt(d1)


def _d2(
        phi: NDArray[np.float64],
        kappa: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64]
) -> NDArray[np.float64]:
    b_args = (kappa, lamb)
    d2  = (rhosv * sigma * phi * 1j - _b2(*b_args))**2
    d2 -= sigma**2 * (2. * _u2() * phi * 1j - phi**2)
    return np.sqrt(d2)


def _u1() -> NDArray[np.float64]:
    return np.array(.5)


def _u2() -> NDArray[np.float64]:
    return np.array(-.5)


def _b1(
        kappa: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rhosv: NDArray[np.float64],
        lamb: NDArray[np.float64],
) -> NDArray[np.float64]:
    return _b2(kappa, lamb) - (rhosv * sigma)


def _b2(
        kappa: NDArray[np.float64],
        lamb: NDArray[np.float64]
) -> NDArray[np.float64]:
    return kappa + lamb


def _a(
        kappa: NDArray[np.float64],
        theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    return kappa * theta


if __name__=='__main__':
    import time

    beg = time.perf_counter()

    S0 = [100., 100.]  # initial asset price
    K = [100., 120.]  # strike
    v0 = 0.1  # initial variance
    r = [.03, .03]  # risk free rate
    d = [.0, .0]
    kappa = 1.5768  # rate of mean reversion of variance process
    theta = 0.0398  # long-term mean variance
    sigma = 0.3  # volatility of volatility
    lambd = 0.575  # risk premium of variance
    rho = -0.5711  # correlation between variance and stock process
    tau = [7/365, 7/365]  # time to maturity

    call_price = heston_call(
        und_strike=np.array(K).reshape((-1, 1)),
        und_price=np.array(S0).reshape((-1, 1)),
        initial_variance=np.array(v0).reshape((-1, 1)),
        kappa=np.array(kappa).reshape((-1, 1)),
        theta=np.array(theta).reshape((-1, 1)),
        sigma=np.array(sigma).reshape((-1, 1)),
        rhosv=np.array(rho).reshape((-1, 1)),
        lamb=np.array(lambd).reshape((-1, 1)),
        time_to_expiry=np.array(tau).reshape((-1, 1)),
        risk_free_rate=np.array(r).reshape((-1, 1)),
        dividend_yield=np.array(d).reshape((-1, 1))
    )
    print(call_price)

    put_price = heston_put(
        und_strike=np.array(K).reshape((-1, 1)),
        und_price=np.array(S0).reshape((-1, 1)),
        initial_variance=np.array(v0).reshape((-1, 1)),
        kappa=np.array(kappa).reshape((-1, 1)),
        theta=np.array(theta).reshape((-1, 1)),
        sigma=np.array(sigma).reshape((-1, 1)),
        rhosv=np.array(rho).reshape((-1, 1)),
        lamb=np.array(lambd).reshape((-1, 1)),
        time_to_expiry=np.array(tau).reshape((-1, 1)),
        risk_free_rate=np.array(r).reshape((-1, 1)),
        dividend_yield=np.array(d).reshape((-1, 1))
    )
    print(put_price)

    end = time.perf_counter()

    print(f"Time took {end - beg:.4f} s")
