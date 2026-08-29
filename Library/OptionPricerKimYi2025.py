import numpy as np
from numpy.typing import NDArray
from Library.OptionPricerKou2002 import kou_call, kou_put


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

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    r = risk_free_rate.reshape((-1, 1))
    d = dividend_yield.reshape((-1, 1))
    kappai = kappai.reshape((-1, 1))
    gammai = gammai.reshape((-1, 1))
    betai = betai.reshape((-1, 1))
    rhoix = rhoix.reshape((-1, 1))
    sigma = sigma.reshape((-1, 1))
    pprob = pprob.reshape((-1, 1))
    lamb = lamb.reshape((-1, 1))
    eta1 = eta1.reshape((-1, 1))
    eta2 = eta2.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

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

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    r = risk_free_rate.reshape((-1, 1))
    d = dividend_yield.reshape((-1, 1))
    kappai = kappai.reshape((-1, 1))
    gammai = gammai.reshape((-1, 1))
    betai = betai.reshape((-1, 1))
    rhoix = rhoix.reshape((-1, 1))
    sigma = sigma.reshape((-1, 1))
    pprob = pprob.reshape((-1, 1))
    lamb = lamb.reshape((-1, 1))
    eta1 = eta1.reshape((-1, 1))
    eta2 = eta2.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

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

    print(f"Call prices: {call.squeeze()}")
    print(f"Put prices: {put.squeeze()}")
