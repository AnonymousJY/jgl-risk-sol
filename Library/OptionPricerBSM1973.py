import numpy as np
from scipy.stats import norm
from numpy.typing import NDArray


def bsm_call_price(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    moneyness = np.log(und_price / und_strike)
    volatility_scaled = volatility * np.sqrt(time_to_expiry)

    d1 = (moneyness + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / volatility_scaled
    d2 = d1 - volatility_scaled
    value = und_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
    value = value - und_strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    return value


def bsm_put_price(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    moneyness = np.log(und_price / und_strike)
    volatility_scaled = volatility * np.sqrt(time_to_expiry)

    d1 = (moneyness + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / volatility_scaled
    d2 = d1 - volatility_scaled

    value = und_strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
    value -= und_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
    return value


def bsm_vega_call(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    d1 = (np.log(und_price / und_strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

    return und_price * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * np.sqrt(time_to_expiry)


def bsm_vega_put(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    d1 = (np.log(und_price / und_strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - (volatility * np.sqrt(time_to_expiry))

    return und_strike * np.exp(-risk_free_rate * time_to_expiry) * norm.pdf(d2) * np.sqrt(time_to_expiry)


def bsm_delta_call(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    d1 = (np.log(und_price / und_strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

    return np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)


def bsm_delta_put(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    d1 = (np.log(und_price / und_strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

    return -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)


def bsm_gamma_call(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    d1 = (np.log(und_price / und_strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

    return np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) / (und_price * volatility * np.sqrt(time_to_expiry))


def bsm_gamma_put(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    d1 = (np.log(und_price / und_strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    return und_strike * np.exp(-risk_free_rate * time_to_expiry) * norm.pdf(d2) / (und_price**2 * volatility * np.sqrt(time_to_expiry))


def bsm_vanna_call(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    d1 = (np.log(und_price / und_strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    return -np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * d2 / volatility


def bsm_vanna_put(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    vega = bsm_vega_put(und_price=und_price, und_strike=und_strike, volatility=volatility, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield, time_to_expiry=time_to_expiry)

    d1 = (np.log(und_price / und_strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

    return vega / und_price * (1. - d1 / (volatility * np.sqrt(time_to_expiry)))


def bsm_vomma_call(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:
    und_price = und_price.reshape((-1, 1))
    und_strike = und_strike.reshape((-1, 1))
    volatility = volatility.reshape((-1, 1))
    risk_free_rate = risk_free_rate.reshape((-1, 1))
    dividend_yield = dividend_yield.reshape((-1, 1))
    time_to_expiry = time_to_expiry.reshape((-1, 1))

    d1 = (np.log(und_price / und_strike) + (
                risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (
                     volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    vega = bsm_vega_call(und_price=und_price, und_strike=und_strike, volatility=volatility, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield, time_to_expiry=time_to_expiry)

    return vega * d1 * d2 / volatility


def bsm_vomma_put(
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        volatility: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    vomma = bsm_vomma_call(und_price=und_price, und_strike=und_strike, volatility=volatility, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield, time_to_expiry=time_to_expiry)

    return vomma


class BlackScholesMertonCall:

    def __init__(
            self,
            und_price: NDArray[np.float64],
            und_strike: NDArray[np.float64],
            risk_free_rate: NDArray[np.float64],
            dividend_yield: NDArray[np.float64],
            time_to_expiry: NDArray[np.float64]
    ):
        self.und_price = und_price.reshape((-1, 1))
        self.und_strike = und_strike.reshape((-1, 1))
        self.risk_free_rate = risk_free_rate.reshape((-1, 1))
        self.dividend_yield = dividend_yield.reshape((-1, 1))
        self.time_to_expiry = time_to_expiry.reshape((-1, 1))

    def price(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_call_price(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                              risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield,
                              time_to_expiry=self.time_to_expiry)

    def vega(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_vega_call(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield,
                             time_to_expiry=self.time_to_expiry)

    def delta(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_delta_call(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield, time_to_expiry=self.time_to_expiry)

    def gamma(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_gamma_call(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield, time_to_expiry=self.time_to_expiry)

    def vanna(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_vanna_call(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield, time_to_expiry=self.time_to_expiry)

    def vomma(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_vomma_call(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield, time_to_expiry=self.time_to_expiry)


class BlackScholesMertonPut:

    def __init__(
            self,
            und_price: NDArray[np.float64],
            und_strike: NDArray[np.float64],
            risk_free_rate: NDArray[np.float64],
            dividend_yield: NDArray[np.float64],
            time_to_expiry: NDArray[np.float64]
    ):
        self.und_price = und_price.reshape((-1, 1))
        self.und_strike = und_strike.reshape((-1, 1))
        self.risk_free_rate = risk_free_rate.reshape((-1, 1))
        self.dividend_yield = dividend_yield.reshape((-1, 1))
        self.time_to_expiry = time_to_expiry.reshape((-1, 1))

    def price(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_put_price(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                              risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield,
                              time_to_expiry=self.time_to_expiry)

    def vega(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_vega_put(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield,
                            time_to_expiry=self.time_to_expiry)

    def delta(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_delta_put(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield, time_to_expiry=self.time_to_expiry)

    def gamma(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_gamma_put(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield, time_to_expiry=self.time_to_expiry)

    def vanna(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_vanna_put(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield, time_to_expiry=self.time_to_expiry)

    def vomma(self, volatility: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_vomma_put(und_price=self.und_price, und_strike=self.und_strike, volatility=volatility,
                             risk_free_rate=self.risk_free_rate, dividend_yield=self.dividend_yield, time_to_expiry=self.time_to_expiry)
