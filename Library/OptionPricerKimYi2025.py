import numpy as np
from numpy.typing import NDArray
from Library.OptionPricerKou2002 import kou_call, kou_put

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)


def _col(x) -> NDArray[np.float64]:
    """As a column vector, accepting a scalar as readily as an array."""
    return np.asarray(x, dtype=float).reshape((-1, 1))


def psi_vol(
        betai: NDArray[np.float64],
        kappai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64]
) -> NDArray[np.float64]:
    psi  = (sigma * betai)**2
    psi += 2. * sigma * betai * kappai * rhoix
    psi += kappai**2
    return np.sqrt(psi)


def kimyi_call(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    # _col, not .reshape: the calibrators hand these in as PYTHON FLOATS
    # whenever a parameter is pinned rather than fitted - sigma on the
    # systematic side when FIT_SIGMA is off, phi_i on the idiosyncratic side
    # when FIT_PHI is off - and a float has no .reshape. Those are the
    # DEFAULT paths, so the Q calibration raised AttributeError before this.
    und_price = _col(und_price)
    und_strike = _col(und_strike)
    r = _col(risk_free_rate)
    d = _col(dividend_yield)
    kappai = _col(kappai)
    gammai = _col(gammai)
    betai = _col(betai)
    rhoix = _col(rhoix)
    sigma = _col(sigma)
    pprob = _col(pprob)
    lamb = _col(lamb)
    eta1 = _col(eta1)
    eta2 = _col(eta2)
    time_to_expiry = _col(time_to_expiry)

    psi = psi_vol(betai=betai, kappai=kappai, rhoix=rhoix, sigma=sigma)

    value = kou_call(
        r=r,
        d=d,
        sigma=psi,
        lam=lamb,
        p=pprob,
        eta1=eta1/gammai,
        eta2=eta2/gammai,
        S0=und_price,
        K=und_strike,
        expiry=time_to_expiry
    )

    return value.reshape((-1, 1))


def kimyi_put(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    # _col, not .reshape: the calibrators hand these in as PYTHON FLOATS
    # whenever a parameter is pinned rather than fitted - sigma on the
    # systematic side when FIT_SIGMA is off, phi_i on the idiosyncratic side
    # when FIT_PHI is off - and a float has no .reshape. Those are the
    # DEFAULT paths, so the Q calibration raised AttributeError before this.
    und_price = _col(und_price)
    und_strike = _col(und_strike)
    r = _col(risk_free_rate)
    d = _col(dividend_yield)
    kappai = _col(kappai)
    gammai = _col(gammai)
    betai = _col(betai)
    rhoix = _col(rhoix)
    sigma = _col(sigma)
    pprob = _col(pprob)
    lamb = _col(lamb)
    eta1 = _col(eta1)
    eta2 = _col(eta2)
    time_to_expiry = _col(time_to_expiry)

    psi = psi_vol(betai=betai, kappai=kappai, rhoix=rhoix, sigma=sigma)

    value = kou_put(
        r=r,
        d=d,
        sigma=psi,
        lam=lamb,
        p=pprob,
        eta1=eta1 / gammai,
        eta2=eta2 / gammai,
        S0=und_price,
        K=und_strike,
        expiry=time_to_expiry
    )

    return value.reshape((-1, 1))


if __name__=='__main__':
    und_price = np.array([100., 110., 120.])
    und_strike = np.array(98.)
    r = np.array(0.05)
    d = np.array(0.0)
    kappai = np.array(0.16)
    gammai = np.array(1.0)
    betai = np.array(0.)
    rhoix = np.array(0.)
    sigma = np.array(0.1)
    pprob = np.array(0.4)
    lamb = np.array(1.)
    eta1 = np.array(10.)
    eta2 = np.array(5.)

    time_to_expiry = np.array(0.5)

    call = kimyi_call(und_price=und_price, und_strike=und_strike, risk_free_rate=r, dividend_yield=d, kappai=kappai, gammai=gammai,
                    betai=betai, rhoix=rhoix, sigma=sigma, pprob=pprob, lamb=lamb, eta1=eta1, eta2=eta2, time_to_expiry=time_to_expiry)
    put = kimyi_put(und_price=und_price, und_strike=und_strike, risk_free_rate=r, dividend_yield=d, kappai=kappai, gammai=gammai,
                  betai=betai, rhoix=rhoix, sigma=sigma, pprob=pprob, lamb=lamb, eta1=eta1, eta2=eta2, time_to_expiry=time_to_expiry)

    _LOG.info(f"Call prices: {call.squeeze()}")
    _LOG.info(f"Put prices: {put.squeeze()}")
