import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
# from Library.RootFinder import newton_raphson
from Library.SkewCalibrationBase import SkewCalibrationBase
from Library.OptionPricerBSM1973 import BlackScholesMertonCall, BlackScholesMertonPut
from Library.OptionPricerKimYi2025 import kimyi_call, kimyi_put


class KimYiSkewCalibrationSystematic(SkewCalibrationBase):

    def __init__(
            self,
            mkt_imp_vol: NDArray[np.float64],
            und_price: NDArray[np.float64],
            und_strike: NDArray[np.float64],
            risk_free_rate: NDArray[np.float64],
            dividend_yield: NDArray[np.float64],
            time_to_expiry: NDArray[np.float64],
            is_call_option: NDArray[np.bool_],
            option_weights: NDArray[np.float64]
    ):
        self.mkt_imp_vol = mkt_imp_vol
        self.und_price = und_price
        self.und_strike = und_strike
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.time_to_expiry = time_to_expiry
        self.is_call_option = is_call_option
        self.option_weights = option_weights
        self.penalty = np.array(0.)

    def target(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        mod_imp_vol = self.model_vol(x=x)

        # If Newton IV-inversion diverged for any strike inside model_vol(),
        # the resulting NaN(s) would poison the sum below. Return a large
        # finite penalty so SLSQP sees "very high objective" and moves away
        # from this parameter region, rather than crashing outright on a
        # pathological (sigma, pprob, lamb, eta1, eta2) trial.
        if not np.all(np.isfinite(mod_imp_vol)):
            return np.array(1e6)

        return 0.5 * np.sum((self.mkt_imp_vol - mod_imp_vol) ** 2 * self.option_weights) + self.penalty

    def model_vol(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        sigma, pprob, lamb, eta1, eta2 = x

        mod_imp_vol_put = _kimyi_imp_vol_put(
            kappai=np.array(0.),
            gammai=np.array(1.),
            betai=np.array(1.),
            rhoix=np.array(0.),
            sigma=sigma,
            pprob=pprob,
            lamb=lamb,
            eta1=eta1,
            eta2=eta2,
            und_price=self.und_price[~self.is_call_option],
            und_strike=self.und_strike[~self.is_call_option],
            risk_free_rate=self.risk_free_rate[~self.is_call_option],
            dividend_yield=self.dividend_yield[~self.is_call_option],
            time_to_expiry=self.time_to_expiry[~self.is_call_option]
        )

        mod_imp_vol_call = _kimyi_imp_vol_call(
            kappai=np.array(0.),
            gammai=np.array(1.),
            betai=np.array(1.),
            rhoix=np.array(0.),
            sigma=sigma,
            pprob=pprob,
            lamb=lamb,
            eta1=eta1,
            eta2=eta2,
            und_price=self.und_price[self.is_call_option],
            und_strike=self.und_strike[self.is_call_option],
            risk_free_rate=self.risk_free_rate[self.is_call_option],
            dividend_yield=self.dividend_yield[self.is_call_option],
            time_to_expiry=self.time_to_expiry[self.is_call_option]
        )

        return np.vstack((mod_imp_vol_put, mod_imp_vol_call))

    @property
    def mkt_imp_vol(self) -> NDArray[np.float64]:
        return self._mkt_imp_vol

    @mkt_imp_vol.setter
    def mkt_imp_vol(self, value: NDArray[np.float64]):
        self._mkt_imp_vol = value.reshape((-1, 1))

    @property
    def und_price(self) -> NDArray[np.float64]:
        return self._und_price

    @und_price.setter
    def und_price(self, value: NDArray[np.float64]):
        self._und_price = value.reshape((-1, 1))

    @property
    def risk_free_rate(self) -> NDArray[np.float64]:
        return self._risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, value: NDArray[np.float64]):
        self._risk_free_rate = value.reshape((-1, 1))

    @property
    def dividend_yield(self) -> NDArray[np.float64]:
        return self._dividend_yield

    @dividend_yield.setter
    def dividend_yield(self, value: NDArray[np.float64]):
        self._dividend_yield = value.reshape((-1, 1))

    @property
    def time_to_expiry(self) -> NDArray[np.float64]:
        return self._time_to_expiry

    @time_to_expiry.setter
    def time_to_expiry(self, value: NDArray[np.float64]):
        self._time_to_expiry = value.reshape((-1, 1))

    @property
    def option_weights(self) -> NDArray[np.float64]:
        return self._option_weights

    @option_weights.setter
    def option_weights(self, value: NDArray[np.float64]):
        self._option_weights = value.reshape((-1, 1))


class KimYiSkewCalibrationIdiosyncratic(SkewCalibrationBase):

    def __init__(
            self,
            sigma: NDArray[np.float64],
            pprob: NDArray[np.float64],
            lamb: NDArray[np.float64],
            eta1: NDArray[np.float64],
            eta2: NDArray[np.float64],
            mkt_imp_vol: NDArray[np.float64],
            und_price: NDArray[np.float64],
            und_strike: NDArray[np.float64],
            risk_free_rate: NDArray[np.float64],
            dividend_yield: NDArray[np.float64],
            time_to_expiry: NDArray[np.float64],
            is_call_option: NDArray[np.bool_],
            option_weights: NDArray[np.float64]
    ):
        self.sigma = sigma
        self.pprob = pprob
        self.lamb = lamb
        self.eta1 = eta1
        self.eta2 = eta2
        self.mkt_imp_vol = mkt_imp_vol
        self.und_price = und_price
        self.und_strike = und_strike
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.time_to_expiry = time_to_expiry
        self.is_call_option = is_call_option
        self.option_weights = option_weights
        self.penalty = np.array(0.)

    def target(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        mod_imp_vol = self.model_vol(x=x)

        # If Newton IV-inversion diverged for any strike inside model_vol(),
        # the resulting NaN(s) would poison the sum below. Return a large
        # finite penalty so SLSQP sees "very high objective" and moves away
        # from this parameter region, rather than crashing outright on a
        # pathological (sigma, pprob, lamb, eta1, eta2) trial.
        if not np.all(np.isfinite(mod_imp_vol)):
            return np.array(1e6)

        return 0.5 * np.sum((self.mkt_imp_vol - mod_imp_vol) ** 2 * self.option_weights) + self.penalty

    def model_vol(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        kappai, gammai, betai, rhoix = x

        mod_imp_vol_put = _kimyi_imp_vol_put(
            kappai=kappai,
            gammai=gammai,
            betai=betai,
            rhoix=rhoix,
            sigma=self.sigma,
            pprob=self.pprob,
            lamb=self.lamb,
            eta1=self.eta1,
            eta2=self.eta2,
            und_price=self.und_price[~self.is_call_option],
            und_strike=self.und_strike[~self.is_call_option],
            risk_free_rate=self.risk_free_rate[~self.is_call_option],
            dividend_yield=self.dividend_yield[~self.is_call_option],
            time_to_expiry=self.time_to_expiry[~self.is_call_option]
        )

        mod_imp_vol_call = _kimyi_imp_vol_call(
            kappai=kappai,
            gammai=gammai,
            betai=betai,
            rhoix=rhoix,
            sigma=self.sigma,
            pprob=self.pprob,
            lamb=self.lamb,
            eta1=self.eta1,
            eta2=self.eta2,
            und_price=self.und_price[self.is_call_option],
            und_strike=self.und_strike[self.is_call_option],
            risk_free_rate=self.risk_free_rate[self.is_call_option],
            dividend_yield=self.dividend_yield[self.is_call_option],
            time_to_expiry=self.time_to_expiry[self.is_call_option]
        )

        return np.vstack((mod_imp_vol_put, mod_imp_vol_call))

    @property
    def sigma(self) -> NDArray[np.float64]:
        return self._sigma

    @sigma.setter
    def sigma(self, value: NDArray[np.float64]):
        self._sigma = np.array(value).reshape((-1, 1))

    @property
    def pprob(self) -> NDArray[np.float64]:
        return self._pprob

    @pprob.setter
    def pprob(self, value: NDArray[np.float64]):
        self._pprob = np.array(value).reshape((-1, 1))

    @property
    def lamb(self) -> NDArray[np.float64]:
        return self._lamb

    @lamb.setter
    def lamb(self, value: NDArray[np.float64]):
        self._lamb = np.array(value).reshape((-1, 1))

    @property
    def eta1(self) -> NDArray[np.float64]:
        return self._eta1

    @eta1.setter
    def eta1(self, value: NDArray[np.float64]):
        self._eta1 = np.array(value).reshape((-1, 1))

    @property
    def eta2(self) -> NDArray[np.float64]:
        return self._eta2

    @eta2.setter
    def eta2(self, value: NDArray[np.float64]):
        self._eta2 = np.array(value).reshape((-1, 1))

    @property
    def mkt_imp_vol(self) -> NDArray[np.float64]:
        return self._mkt_imp_vol

    @mkt_imp_vol.setter
    def mkt_imp_vol(self, value: NDArray[np.float64]):
        self._mkt_imp_vol = value.reshape((-1, 1))

    @property
    def und_price(self) -> NDArray[np.float64]:
        return self._und_price

    @und_price.setter
    def und_price(self, value: NDArray[np.float64]):
        self._und_price = value.reshape((-1, 1))

    @property
    def risk_free_rate(self) -> NDArray[np.float64]:
        return self._risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, value: NDArray[np.float64]):
        self._risk_free_rate = value.reshape((-1, 1))

    @property
    def dividend_yield(self) -> NDArray[np.float64]:
        return self._dividend_yield

    @dividend_yield.setter
    def dividend_yield(self, value: NDArray[np.float64]):
        self._dividend_yield = value.reshape((-1, 1))

    @property
    def time_to_expiry(self) -> NDArray[np.float64]:
        return self._time_to_expiry

    @time_to_expiry.setter
    def time_to_expiry(self, value: NDArray[np.float64]):
        self._time_to_expiry = value.reshape((-1, 1))

    @property
    def option_weights(self) -> NDArray[np.float64]:
        return self._option_weights

    @option_weights.setter
    def option_weights(self, value: NDArray[np.float64]):
        self._option_weights = value.reshape((-1, 1))


def _kimyi_imp_vol_call(
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    prices = kimyi_call(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        kappai=kappai,
        gammai=gammai,
        betai=betai,
        rhoix=rhoix,
        sigma=sigma,
        pprob=pprob,
        lamb=lamb,
        eta1=eta1,
        eta2=eta2,
        time_to_expiry=time_to_expiry
    )

    bsm_call_obj = BlackScholesMertonCall(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_expiry=time_to_expiry
    )

    obj_func = lambda x: bsm_call_obj.price(volatility=x) - prices

    # Initial guess x0=1.5 (150% vol) matches the paper's original code.
    # Starting high gives Newton non-negligible BSM vega even at deep OTM
    # strikes, whereas a strike-adaptive low x0 (e.g. Brenner-Subrahmanyam)
    # can leave Newton stuck at the floor for deep-OTM options where BSM
    # vega at low vol is ~zero. Wrapped in try/except because scipy.newton
    # raises RuntimeError when its 200 iterations don't converge -- happens
    # when the outer SLSQP tries a pathological (sigma, pprob, lamb, eta1,
    # eta2) combination that pushes model prices near the arbitrage bounds.
    x0 = np.full(prices.shape, 1.5).reshape((-1, 1))
    try:
        mod_iv_call = newton(
            func=obj_func,
            x0=x0,
            fprime=bsm_call_obj.vega,
            fprime2=bsm_call_obj.vomma,
            maxiter=200,
            tol=1e-8,
        )
    except RuntimeError:
        mod_iv_call = np.full(prices.shape, np.nan).reshape((-1, 1))

    # Newton can also silently return inf/NaN (via overflow warnings, not
    # exceptions) for individual strikes when the outer SLSQP tries a
    # pathological (sigma, pprob, lamb, eta1, eta2) combination. Mark only
    # those bad entries as NaN -- keep the good ones intact so plotting +
    # per-strike diagnostics still work. target() below checks isfinite on
    # the whole array and returns a 1e6 penalty if any NaN is present, so
    # SLSQP's avoidance behavior is preserved.
    mod_iv_call = np.where(np.isfinite(mod_iv_call), mod_iv_call, np.nan)

    # mod_iv_call = newton_raphson(
    #     func=bsm_call_obj.price,
    #     func_deriv=bsm_call_obj.vega,
    #     target_value=prices,
    #     initial_value=np.array([1.5] * prices.shape[0]).reshape((-1, 1))
    # )

    return mod_iv_call


def _kimyi_imp_vol_put(
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
    ) -> NDArray[np.float64]:

    prices = kimyi_put(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        kappai=kappai,
        gammai=gammai,
        betai=betai,
        rhoix=rhoix,
        sigma=sigma,
        pprob=pprob,
        lamb=lamb,
        eta1=eta1,
        eta2=eta2,
        time_to_expiry=time_to_expiry
    )

    bsm_put_obj = BlackScholesMertonPut(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_expiry=time_to_expiry
    )

    obj_func = lambda x: bsm_put_obj.price(volatility=x) - prices

    # See _kimyi_imp_vol_call above for the rationale on x0=1.5, maxiter=200,
    # try/except, and per-strike NaN handling -- same treatment for symmetry.
    x0 = np.full(prices.shape, 1.5).reshape((-1, 1))
    try:
        mod_iv_put = newton(
            func=obj_func,
            x0=x0,
            fprime=bsm_put_obj.vega,
            fprime2=bsm_put_obj.vomma,
            maxiter=200,
            tol=1e-8,
        )
    except RuntimeError:
        mod_iv_put = np.full(prices.shape, np.nan).reshape((-1, 1))

    # See _kimyi_imp_vol_call above for the per-strike NaN rationale.
    mod_iv_put = np.where(np.isfinite(mod_iv_put), mod_iv_put, np.nan)

    # mod_iv_put = newton_raphson(
    #     func=bsm_put_obj.price,
    #     func_deriv=bsm_put_obj.vega,
    #     target_value=prices,
    #     initial_value=np.array([1.5] * prices.shape[0]).reshape((-1, 1))
    # )

    return mod_iv_put


def kimyi_vol_surface(
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    mask_call = und_strike > 100.
    mask_put = und_strike <= 100.

    mod_iv_call = _kimyi_imp_vol_call(
        kappai=np.array(kappai).reshape((-1, 1)),
        gammai=np.array(gammai).reshape((-1, 1)),
        betai=np.array(betai).reshape((-1, 1)),
        rhoix=np.array(rhoix).reshape((-1, 1)),
        sigma=np.array(sigma).reshape((-1, 1)),
        pprob=np.array(pprob).reshape((-1, 1)),
        lamb=np.array(lamb).reshape((-1, 1)),
        eta1=np.array(eta1).reshape((-1, 1)),
        eta2=np.array(eta2).reshape((-1, 1)),
        und_price=np.array(und_price).reshape((-1, 1)),
        und_strike=np.array(und_strike[mask_call]).reshape((-1, 1)),
        risk_free_rate=np.array(risk_free_rate).reshape((-1, 1)),
        dividend_yield=np.array(dividend_yield).reshape((-1, 1)),
        time_to_expiry=np.array(time_to_expiry).reshape((-1, 1))
    )

    mod_iv_put = _kimyi_imp_vol_call(
        kappai=np.array(kappai).reshape((-1, 1)),
        gammai=np.array(gammai).reshape((-1, 1)),
        betai=np.array(betai).reshape((-1, 1)),
        rhoix=np.array(rhoix).reshape((-1, 1)),
        sigma=np.array(sigma).reshape((-1, 1)),
        pprob=np.array(pprob).reshape((-1, 1)),
        lamb=np.array(lamb).reshape((-1, 1)),
        eta1=np.array(eta1).reshape((-1, 1)),
        eta2=np.array(eta2).reshape((-1, 1)),
        und_price=np.array(und_price).reshape((-1, 1)),
        und_strike=np.array(und_strike[mask_put]).reshape((-1, 1)),
        risk_free_rate=np.array(risk_free_rate).reshape((-1, 1)),
        dividend_yield=np.array(dividend_yield).reshape((-1, 1)),
        time_to_expiry=np.array(time_to_expiry).reshape((-1, 1))
    )

    # results = {}
    # results['dMONEYNESS'] = np.array(und_strike) / np.array(und_price) * 100.
    # results['iEXPIRY'] = np.array(expiry_in_days, dtype=np.int64)
    # results['dVOL'] = np.vstack((mod_iv_put, mod_iv_call)) * 100.

    return np.vstack((mod_iv_put, mod_iv_call)) * 100.
