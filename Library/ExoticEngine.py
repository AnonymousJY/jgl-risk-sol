import numpy as np
from numpy.typing import NDArray
from typing import Union
from typing import List
from abc import abstractmethod
from Library.Wrapper import Wrapper
from Library.Random import RandomBase
from Library.Parameters import ParametersBase
from Library.OptionPricerKimYi2025 import psi_vol
from Library.PathDependent import PathDependentBase, CashFlow
from Library.StatisticsMC import StatisticsMCBase, get_corr_mat

class ExoticEngine:

    def __init__(
            self,
            the_product: PathDependentBase,
            risk_free_rate: ParametersBase,
            number_of_paths: np.uint64
    ) -> None:
        self.product_ptr = the_product
        self.discount = np.exp(-risk_free_rate.integral(np.array(0), self.product_ptr.possible_cash_flow_times()))
        self.these_cash_flows = [CashFlow()] * self.product_ptr.max_number_of_cash_flows()
        self.number_of_paths = number_of_paths
        self.number_of_assets = self.product_ptr.number_of_assets

    def do_simulation(self, gatherer: StatisticsMCBase) -> None:
        spot_values = np.zeros(
            shape=(
                self.number_of_assets,
                self.number_of_paths,
                self.product_ptr.fixings.shape[0]
            )
        )
        self.get_paths(spot_values=spot_values)
        this_value = self.do_paths(spot_values=spot_values)
        gatherer.dump_result(result=this_value)

    def do_paths(self, spot_values: NDArray[np.float64]) -> NDArray[np.float64]:
        number_flows = self.product_ptr.cash_flows(spot_values=spot_values, generated_flows=self.these_cash_flows)
        value = np.zeros(shape=(spot_values.shape[1], number_flows))
        for i in range(number_flows):
            value += self.these_cash_flows[i].amount * self.discount[self.these_cash_flows[i].time_idx]

        return value

    @abstractmethod
    def get_paths(self, spot_values: NDArray[np.float64]) -> None:
        pass

    @property
    def product_ptr(self) -> Wrapper[PathDependentBase]:
        return self._product_ptr

    @product_ptr.setter
    def product_ptr(self, the_product: PathDependentBase) -> None:
        self._product_ptr = Wrapper(the_product)

    @property
    def number_of_paths(self) -> np.uint64:
        return self._number_of_paths

    @number_of_paths.setter
    def number_of_paths(self, number_of_paths: np.uint64) -> None:
        self._number_of_paths = np.uint64(number_of_paths)


class ExoticEngineBlackScholesMerton(ExoticEngine):

    def __init__(
            self,
            the_product: PathDependentBase,
            risk_free_rate: ParametersBase,
            dividend_yield: List[ParametersBase],
            imp_volatility: List[ParametersBase],
            rand_generator: RandomBase,
            spot_price: NDArray[np.float64],
            number_of_paths: np.uint64 = np.uint64(10_000),
            correlations: List[ParametersBase]=None
    ) -> None:
        super().__init__(the_product, risk_free_rate, number_of_paths)

        self.generator_ptr = rand_generator
        self.spot_price = spot_price

        self.times = self.product_ptr.fixings
        self.number_of_assets = self.product_ptr.number_of_assets
        self.number_of_fixings = np.uint64(self.times.shape[0])

        self.drifts = np.zeros(shape=(self.number_of_assets, self.number_of_paths, self.number_of_fixings))
        self.stddev = np.zeros(shape=(self.number_of_assets, self.number_of_paths, self.number_of_fixings))
        self.variates = np.zeros(shape=(self.number_of_assets, self.number_of_paths, self.number_of_fixings))
        self.log_spot = np.zeros(shape=(self.number_of_assets, self.number_of_paths, self.number_of_fixings))

        if correlations is None:
            self.L = np.array(1.).reshape((-1, 1))
        else:
            correl = np.array([x.integral(np.array(0.), np.array(1.)) for x in correlations]).reshape((-1, 1))
            self.L = np.linalg.cholesky(get_corr_mat(correl, self.number_of_assets))

        for i in range(self.number_of_assets):
            volatility_sq = imp_volatility[i].integral_square(np.array(0.), self.times[0])
            self.drifts[i, :, 0] = (
                    risk_free_rate.integral(np.array(0.), self.times[0]) -
                    dividend_yield[i].integral(np.array(0.), self.times[0]) -
                    volatility_sq * .5
            )
            self.stddev[i, :, 0] = np.sqrt(volatility_sq)

        for i in range(self.number_of_assets):
            for j in range(1, self.number_of_fixings):
                volatility_sq = imp_volatility[i].integral_square(self.times[j-1], self.times[j])
                self.drifts[:, :, j] = (
                    risk_free_rate.integral(self.times[j-1], self.times[j]) -
                    dividend_yield[i].integral(self.times[j-1], self.times[j]) -
                    volatility_sq * .5
                )
                self.stddev[:, :, j] = np.sqrt(volatility_sq)

        self.log_spot[:, :, 0] = np.log(self.spot_price)

    def get_paths(self, spot_values: NDArray[np.float64]) -> None:
        self.generator_ptr.get_gaussian(self.variates)

        current_log_spot = self.log_spot[:, :, 0]

        for j in range(self.number_of_fixings):
            current_log_spot += self.drifts[:, :, j]
            current_log_spot += self.stddev[:, :, j] * np.dot(self.L, self.variates[:, :, j])
            spot_values[:, :, j] = current_log_spot

        spot_values[:] = np.exp(spot_values)

    @property
    def spot_price(self) -> NDArray[np.float64]:
        return self._spot_price

    @spot_price.setter
    def spot_price(self, spot_price: NDArray[np.float64]) -> None:
        self._spot_price = np.array(spot_price).reshape((-1, 1))

    @property
    def generator_ptr(self) -> RandomBase:
        return self._generator_ptr

    @generator_ptr.setter
    def generator_ptr(self, value: RandomBase) -> None:
        self._generator_ptr = Wrapper(value)


class ExoticEngineKimYi(ExoticEngine):

    def __init__(
            self,
            the_product: PathDependentBase,
            risk_free_rate: ParametersBase,
            dividend_yield: list[ParametersBase],
            sigma: ParametersBase,
            kappai: list[ParametersBase],
            rhoij: list[ParametersBase],
            betai: list[ParametersBase],
            rhoix: list[ParametersBase],
            gammai: list[ParametersBase],
            p_prob: ParametersBase,
            lamb: ParametersBase,
            eta1: ParametersBase,
            eta2: ParametersBase,
            generator: RandomBase,
            spot: Union[np.float64, NDArray[np.float64]],
            number_of_paths: np.uint64 = np.uint64(100_000)
    ) -> None:
        super().__init__(the_product=the_product, risk_free_rate=risk_free_rate, number_of_paths=number_of_paths)

        self.generator_ptr = generator
        self.spot = spot

        self.times = the_product.fixings
        self.number_of_fixings = np.uint64(self.times.shape[0])

        # self.generator_ptr.reset_dimensionality(new_dimensionality=self.number_of_fixings) # not used now
        self.scaled_lamb = np.zeros(shape=(self.number_of_fixings, 1))

        t1, t2 = np.array(0), np.array(1)
        self.eta1 = eta1.integral(time1=t1, time2=t2)
        self.eta2 = eta2.integral(time1=t1, time2=t2)
        self.p_prob = p_prob.integral(time1=t1, time2=t2)
        q = np.float64(1 - self.p_prob)

        number_of_underlying = np.int64(self.spot.shape[0])
        self.drifts = np.zeros(shape=(number_of_underlying, self.number_of_paths, self.number_of_fixings))
        self.stddev = np.zeros(shape=(number_of_underlying, self.number_of_paths, self.number_of_fixings))
        self.variates_cont = np.zeros(shape=(number_of_underlying, self.number_of_paths, self.number_of_fixings))
        self.variates_jump_bins = np.zeros(shape=(self.number_of_paths, self.number_of_fixings), dtype=np.int64)
        self.variates_jump_size = np.zeros(shape=(self.number_of_paths, self.number_of_fixings))

        beta_vec = []
        dividend_yield_vec = []
        kappa_vec = []
        rhoij_vec = []
        rhoix_vec = []
        self.gammai_vec = []
        for b, d, k, r, g in zip(betai, dividend_yield, kappai, rhoix, gammai):
            beta_vec.append(b.integral(time1=t1, time2=t2))
            dividend_yield_vec.append(d.integral(time1=t1, time2=t2))
            kappa_vec.append(k.integral(time1=t1, time2=t2))
            rhoix_vec.append(r.integral(time1=t1, time2=t2))
            self.gammai_vec.append(g.integral(time1=t1, time2=t2))

        for v in rhoij:
            rhoij_vec.append(v.integral(time1=t1, time2=t2))

        rhoij_vec = np.array(rhoij_vec).reshape(-1, 1)
        beta_vec = np.array(beta_vec).reshape(-1, 1)
        kappa_vec = np.array(kappa_vec).reshape(-1, 1)
        rhoix_vec = np.array(rhoix_vec).reshape(-1, 1)
        self.gammai_vec = np.array(self.gammai_vec).reshape(-1, 1)
        dividend_yield_vec = np.array(dividend_yield_vec).reshape(-1, 1)
        sigma = sigma.integral(t1, t2)

        rhoij_mat = get_corr_mat(rhoij_vec, number_of_underlying)
        self.L = np.linalg.cholesky(rhoij_mat)

        psi_variance = psi_vol(
            betai=beta_vec,
            kappai=kappa_vec,
            rhoix=rhoix_vec,
            sigma=sigma
        )**2

        zeta = (self.p_prob * self.eta1 / (self.eta1 - self.gammai_vec)) + (q * self.eta2 / (self.eta2 + self.gammai_vec)) - 1.

        # print(psi_variance.shape)
        drift_const = risk_free_rate.integral(t1, t2) - dividend_yield_vec - 0.5 * psi_variance - lamb.integral(t1, t2) * zeta

        t1, t2 = np.array(0), self.times[0]
        delta_t = t2 - t1
        self.drifts[:, :, 0] = drift_const * delta_t
        self.stddev[:, :, 0] = np.sqrt(psi_variance * delta_t)
        self.scaled_lamb[0] = lamb.integral(t1, t2)

        for i in range(1, self.number_of_fixings):
            t1, t2 = self.times[i - 1], self.times[i]
            delta_t = t2 - t1
            self.drifts[:, :, i] = drift_const * delta_t
            self.stddev[:, :, i] = np.sqrt(psi_variance * delta_t)
            self.scaled_lamb[i] = lamb.integral(t1, t2)

        self.log_spot = np.zeros(shape=(number_of_underlying, self.number_of_paths, self.number_of_fixings))
        self.log_spot[:, :, 0] = np.log(self.spot)

    def get_paths(self, spot_values: NDArray[np.float64]) -> None:
        self.generator_ptr.get_gaussian(self.variates_cont)
        self.generator_ptr.get_poisson(lamb=self.scaled_lamb.reshape(1, -1), variates=self.variates_jump_bins)
        self.generator_ptr.get_aded(
            variates=self.variates_jump_size,
            n=self.variates_jump_bins,
            eta1=self.eta1,
            eta2=self.eta2,
            pprob=self.p_prob
        )

        current_log_spot = self.log_spot[:, :, 0]

        for j in range(self.number_of_fixings):
            current_log_spot += self.drifts[:, :, j]
            current_log_spot += self.stddev[:, :, j] * np.dot(self.L, self.variates_cont[:, :, j])
            current_log_spot += self.gammai_vec * self.variates_jump_size[:, j]
            spot_values[:, :, j] = current_log_spot

        spot_values[:] = np.exp(spot_values)

    @property
    def generator_ptr(self) -> RandomBase:
        return self._generator_ptr

    @generator_ptr.setter
    def generator_ptr(self, generator_ptr: RandomBase) -> None:
        self._generator_ptr = generator_ptr.clone()

    @property
    def spot(self) -> NDArray[np.float64]:
        return self._spot

    @spot.setter
    def spot(self, value: Union[np.float64, NDArray[np.float64]]) -> None:
        self._spot = np.array([value]).reshape((-1, 1))
