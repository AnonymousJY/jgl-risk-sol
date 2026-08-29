import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
# from Library.RootFinder import newton_raphson
from Library.SkewCalibrationBase import SkewCalibrationBase
from Library.OptionPricerBSM1973 import BlackScholesMertonCall, BlackScholesMertonPut
from Library.OptionPricerHeston1993 import heston_call, heston_put


def feller_condition(x: NDArray[np.float64]) -> NDArray[np.float64]:
    _, kappa, theta, sigma, _, _ = x
    return 2. * kappa * theta - sigma**2


class HestonSkewCalibration(SkewCalibrationBase):

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

        return .5 * np.sum((self.mkt_imp_vol - mod_imp_vol) ** 2 * self.option_weights) + self.penalty

    def model_vol(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        v0, kappa, theta, sigma, rhosv, lambd = x

        mod_imp_vol_put = _heston_imp_vol_put(
            und_strike=self.und_strike[~self.is_call_option],
            und_price=self.und_price[~self.is_call_option],
            initial_variance=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rhosv=rhosv,
            lamb=lambd,
            time_to_expiry=self.time_to_expiry[~self.is_call_option],
            risk_free_rate=self.risk_free_rate[~self.is_call_option],
            dividend_yield=self.dividend_yield[~self.is_call_option]
        )

        mod_imp_vol_call = _heston_imp_vol_call(
            und_strike=self.und_strike[self.is_call_option],
            und_price=self.und_price[self.is_call_option],
            initial_variance=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rhosv=rhosv,
            lamb=lambd,
            time_to_expiry=self.time_to_expiry[self.is_call_option],
            risk_free_rate=self.risk_free_rate[self.is_call_option],
            dividend_yield=self.dividend_yield[self.is_call_option]
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


def _heston_imp_vol_call(
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

    prices = heston_call(
        und_strike=und_strike,
        und_price=und_price,
        initial_variance=initial_variance,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rhosv=rhosv,
        lamb=lamb,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield
    )

    bsm_call_obj = BlackScholesMertonCall(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_expiry=time_to_expiry
    )

    obj_func = lambda x: bsm_call_obj.price(volatility=x) - prices

    mod_iv_call = newton(
        func=obj_func,
        x0=np.array([1.] * prices.shape[0]).reshape((-1, 1)),
        fprime=bsm_call_obj.vega,
        fprime2=bsm_call_obj.vomma
    )

    # mod_iv_call = newton_raphson(
    #     func=bsm_call_obj.price,
    #     func_deriv=bsm_call_obj.vega,
    #     target_value=prices,
    #     initial_value=np.array([1.5] * prices.shape[0]).reshape((-1, 1))
    # )

    return mod_iv_call


def _heston_imp_vol_put(
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

    prices = heston_put(
        und_strike=und_strike,
        und_price=und_price,
        initial_variance=initial_variance,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rhosv=rhosv,
        lamb=lamb,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield
    )

    bsm_put_obj = BlackScholesMertonPut(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_expiry=time_to_expiry
    )

    obj_func = lambda x: bsm_put_obj.price(volatility=x) - prices

    mod_iv_put = newton(
        func=obj_func,
        x0=np.array([1.] * prices.shape[0]).reshape((-1, 1)),
        fprime=bsm_put_obj.vega,
        fprime2=bsm_put_obj.vomma
    )

    # mod_iv_put = newton_raphson(
    #     func=bsm_put_obj.price,
    #     func_deriv=bsm_put_obj.vega,
    #     target_value=prices,
    #     initial_value=np.array([1.5] * prices.shape[0]).reshape((-1, 1))
    # )

    return mod_iv_put
