import copy
import numpy as np

from typing import List
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from Library.Payoff import PayoffBase
from Library.Wrapper import Wrapper


class CashFlow:

    def __init__(
            self,
            amount: NDArray[np.float64] = np.zeros(shape=(1, 1)),
            time_idx: NDArray[np.float64] = np.uint64(0)
    ) -> None:
        self.amount = amount
        self.time_idx = time_idx

    @property
    def amount(self) -> NDArray[np.float64]:
        return self._amount

    @amount.setter
    def amount(self, amount: NDArray[np.float64]) -> None:
        self._amount = amount

    @property
    def time_idx(self) -> np.uint64:
        return self._time_idx

    @time_idx.setter
    def time_idx(self, time_idx: np.uint64) -> None:
        self._time_idx = np.uint64(time_idx)


class PathDependentBase(ABC):

    def __init__(self, fixings: NDArray[np.float64], number_of_assets: np.uint64) -> None:
        self.fixings = fixings
        self.number_of_assets = number_of_assets

    @abstractmethod
    def __deepcopy__(self, memodict={}) -> 'PathDependentBase':
        pass

    @abstractmethod
    def max_number_of_cash_flows(self) -> np.uint64:
        pass

    @abstractmethod
    def possible_cash_flow_times(self) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def cash_flows(self, spot_values: NDArray[np.float64], generated_flows: List[CashFlow]) -> np.uint64:
        pass

    @property
    def fixings(self) -> NDArray[np.float64]:
        return self._fixings

    @fixings.setter
    def fixings(self, fixings: NDArray[np.float64]) -> None:
        self._fixings = np.array(fixings).reshape((-1, 1))

    @property
    def number_of_assets(self) -> np.uint64:
        return self._number_of_assets

    @number_of_assets.setter
    def number_of_assets(self, number_of_assets: np.uint64) -> None:
        self._number_of_assets = np.uint64(number_of_assets)


class PathDependentAsianDiscrete(PathDependentBase):

    def __init__(
            self,
            fixing_times: NDArray[np.float64],
            delivery_time: NDArray[np.float64],
            the_payoff: PayoffBase,
            quantity_amount: NDArray[np.float64]
    ) -> None:
        super().__init__(fixings=fixing_times, number_of_assets=np.uint64(1))
        self.delivery_time = delivery_time
        self.the_payoff = the_payoff
        self.number_of_times = fixing_times.shape[0]
        self.quantity_amount = quantity_amount

    def __deepcopy__(self, memodict={}) -> PathDependentBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    def max_number_of_cash_flows(self) -> np.uint64:
        return np.uint64(1)

    def possible_cash_flow_times(self) -> NDArray[np.float64]:
        tmp = np.zeros(shape=(1, 1))
        tmp[0, 0] = self.delivery_time
        return tmp

    def cash_flows(self, spot_values: NDArray[np.float64], generated_flows: List[CashFlow]) -> np.uint64:
        tot = spot_values.sum(axis=2)
        avg = tot / self.number_of_times

        generated_flows[0].time_idx = np.uint64(0)
        generated_flows[0].amount = self.the_payoff(avg.reshape((-1, 1))) * self.quantity_amount

        return np.uint64(1)

    @property
    def delivery_time(self) -> np.float64:
        return self._delivery_time

    @delivery_time.setter
    def delivery_time(self, delivery_time: np.float64) -> None:
        self._delivery_time = delivery_time

    @property
    def number_of_times(self) -> np.uint64:
        return self._number_of_times

    @number_of_times.setter
    def number_of_times(self, number_of_times: np.uint64) -> None:
        self._number_of_times = np.uint64(number_of_times)

    @property
    def the_payoff(self) -> PayoffBase:
        return self._the_payoff

    @the_payoff.setter
    def the_payoff(self, the_payoff: PayoffBase) -> None:
        self._the_payoff = Wrapper(the_payoff)

    @property
    def quantity_amount(self) -> NDArray[np.float64]:
        return self._quantity_amount

    @quantity_amount.setter
    def quantity_amount(self, quantity_amount: NDArray[np.float64]) -> None:
        self._quantity_amount = np.array(quantity_amount).reshape((-1, 1))


class PathDependentContingentTwoAssets(PathDependentBase):

    def __init__(
            self,
            delivery_time: NDArray[np.float64],
            the_contingent: PayoffBase,
            the_und_payoff: PayoffBase,
            quantity_amount: NDArray[np.float64]
    ) -> None:
        super().__init__(fixings=delivery_time, number_of_assets=np.uint64(2))
        self.delivery_time = delivery_time
        self.the_contingent = the_contingent
        self.the_und_payoff = the_und_payoff
        self.number_of_times = self.fixings.shape[0]
        self.quantity_amount = quantity_amount

    def __deepcopy__(self, memodict={}) -> PathDependentBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    def max_number_of_cash_flows(self) -> np.uint64:
        return np.uint64(1)

    def possible_cash_flow_times(self) -> NDArray[np.float64]:
        tmp = np.zeros(shape=(1, 1))
        tmp[0, 0] = self.delivery_time
        return tmp

    def cash_flows(self, spot_values: NDArray[np.float64], generated_flows: List[CashFlow]) -> np.uint64:

        payoff  = self.the_contingent.__call__(spot_values[0])
        payoff *= self.the_und_payoff.__call__(spot_values[1]) * self.quantity_amount * .01
        # payoff  = self.the_contingent(spot_values[0])
        # payoff *= self.the_und_payoff(spot_values[1]) * self.quantity_amount * .01

        generated_flows[0].time_idx = np.uint64(0)
        generated_flows[0].amount = payoff

        return np.uint64(1)

    @property
    def delivery_time(self) -> np.float64:
        return self._delivery_time

    @delivery_time.setter
    def delivery_time(self, delivery_time: np.float64) -> None:
        self._delivery_time = delivery_time

    @property
    def number_of_times(self) -> np.uint64:
        return self._number_of_times

    @number_of_times.setter
    def number_of_times(self, number_of_times: np.uint64) -> None:
        self._number_of_times = np.uint64(number_of_times)

    @property
    def the_contingent(self) -> Wrapper[PayoffBase]:
        return self._the_contingent

    @the_contingent.setter
    def the_contingent(self, the_contingent: PayoffBase) -> None:
        self._the_contingent = Wrapper(the_contingent)

    @property
    def the_und_payoff(self) -> Wrapper[PayoffBase]:
        return self._the_und_payoff

    @the_und_payoff.setter
    def the_und_payoff(self, the_und_payoff: PayoffBase) -> None:
        self._the_und_payoff = Wrapper(the_und_payoff)

    @property
    def quantity_amount(self) -> NDArray[np.float64]:
        return self._quantity_amount

    @quantity_amount.setter
    def quantity_amount(self, quantity_amount: NDArray[np.float64]) -> None:
        self._quantity_amount = np.array(quantity_amount).reshape((-1, 1))


# class PathDependentDownOutOpt(PathDependentBase):
#
#     def __init__(self, **kwargs) -> None:
#         super().__init__(fixings=kwargs['expiry'], number_of_asset=np.uint64(1))
#         self.delivery_time: np.float64 = kwargs['expiry']
#         self.payoff_vanilla = kwargs['payoff_vanilla']
#         self.payoff_binary = kwargs['payoff_binary']
#         self.number_of_times = np.int64(self.get_fixings().shape[0])
#         self.amount: np.float64 = kwargs['amount']
#
#     def clone(self) -> PathDependentBase:
#         return copy.deepcopy(self)
#
#     def max_number_of_cash_flows(self) -> np.uint64:
#         return np.uint64(1)
#
#     def possible_cash_flow_times(self) -> np.ndarray:
#         tmp = np.zeros(shape=(1, 1))
#         tmp[0, 0] = self.delivery_time
#         return tmp
#
#     def cash_flows(self, spot_values: np.ndarray, generated_flows: list[CashFlow]) -> np.uint64:
#         spot_values = spot_values[0]
#         generated_flows[0].time_idx = np.uint64(0)
#         generated_flows[0].amount = self.amount * self.payoff_vanilla(spot_values) * self.payoff_binary(spot_values)
#         return np.uint64(1)
#
#     @property
#     def delivery_time(self) -> np.float64:
#         return self._delivery_time
#
#     @delivery_time.setter
#     def delivery_time(self, delivery_time: np.float64) -> None:
#         self._delivery_time = np.float64(delivery_time)
#
#     @property
#     def number_of_times(self) -> np.uint64:
#         return self._number_of_times
#
#     @number_of_times.setter
#     def number_of_times(self, number_of_times: np.uint64) -> None:
#         self._number_of_times = np.uint64(number_of_times)
#
#     @property
#     def amount(self) -> np.float64:
#         return self._amount
#
#     @amount.setter
#     def amount(self, amount: np.float64) -> None:
#         self._amount = np.float64(amount)


# class PathDependentUpAndOutCall(PathDependentBase):
#
#     def __init__(self, delivery_time: np.float64, call_payoff: PayoffBridge, binary_put_payoff: PayoffBridge) -> None:
#         super().__init__(np.array([delivery_time]), number_of_asset=np.uint64(1))
#         self.delivery_time = delivery_time
#         self.call_payoff = call_payoff
#         self.binary_put_payoff = binary_put_payoff
#         self.number_of_times = delivery_time
#         self.time_idx = np.uint64(0)
#
#     def clone(self) -> PathDependentBase:
#         return copy.deepcopy(self)
#
#     def max_number_of_cash_flows(self) -> np.uint64:
#         return np.uint64(1)
#
#     def possible_cash_flow_times(self) -> np.ndarray:
#         tmp = np.zeros(shape=(1, 1))
#         tmp[0, 0] = self.delivery_time
#         return tmp
#
#     def cash_flows(self, spot_values: np.ndarray, generated_flows: list[CashFlow]) -> np.uint64:
#         generated_flows[0].time_idx = np.uint64(0)
#         generated_flows[0].amount = self.call_payoff(spot_values) * self.binary_put_payoff(spot_values)
#         return np.uint64(1)
#
#     @property
#     def delivery_time(self) -> np.float64:
#         return self._delivery_time
#
#     @delivery_time.setter
#     def delivery_time(self, delivery_time: np.float64) -> None:
#         self._delivery_time = np.float64(delivery_time)
#
#     @property
#     def call_payoff(self) -> 'PayoffBridge':
#         return self._call_payoff
#
#     @call_payoff.setter
#     def call_payoff(self, call_payoff: 'PayoffBridge') -> None:
#         if not isinstance(call_payoff, PayoffBridge):
#             raise TypeError("arg 'call_payoff' must be type PayoffBridge")
#         self._call_payoff = call_payoff
#
#     @property
#     def binary_put_payoff(self) -> 'PayoffBridge':
#         return self._binary_put_payoff
#
#     @binary_put_payoff.setter
#     def binary_put_payoff(self, binary_put_payoff: 'PayoffBridge') -> None:
#         if not isinstance(binary_put_payoff, PayoffBridge):
#             raise TypeError("arg 'binary_put_payoff' must be type PayoffBridge")
#         self._binary_put_payoff = binary_put_payoff
#
#
# class PathDependentPutWorstOf(PathDependentBase):
#
#     def __init__(self, delivery_time: np.float64, put_payoff: PayoffBridge, number_of_asset: np.uint64) -> None:
#         super().__init__(np.array([delivery_time]), number_of_asset)
#         self.delivery_time = delivery_time
#         self.put_payoff = put_payoff
#         self.number_of_times = delivery_time
#         self.time_idx = np.uint64(0)
#
#     def clone(self) -> PathDependentBase:
#         return copy.deepcopy(self)
#
#     def max_number_of_cash_flows(self) -> np.uint64:
#         return np.uint64(1)
#
#     def possible_cash_flow_times(self) -> np.ndarray:
#         tmp = np.zeros(shape=(1, 1))
#         tmp[0, 0] = self.delivery_time
#         return tmp
#
#     def cash_flows(self, spot_values: np.ndarray, generated_flows: list[CashFlow]) -> np.uint64:
#         generated_flows[0].time_idx = np.uint64(0)
#         generated_flows[0].amount = self.put_payoff(spot_values.min(axis=0))
#         return np.uint64(1)
#
#     @property
#     def delivery_time(self) -> np.float64:
#         return self._delivery_time
#
#     @delivery_time.setter
#     def delivery_time(self, delivery_time: np.float64) -> None:
#         self._delivery_time = np.float64(delivery_time)
#
#     @property
#     def number_of_asset(self) -> np.uint64:
#         return self._number_of_asset
#
#     @number_of_asset.setter
#     def number_of_asset(self, number_of_asset: np.uint64) -> None:
#         self._number_of_asset = np.uint64(number_of_asset)
#
#     @property
#     def put_payoff(self) -> 'PayoffBridge':
#         return self._put_payoff
#
#     @put_payoff.setter
#     def put_payoff(self, put_payoff: 'PayoffBridge') -> None:
#         if not isinstance(put_payoff, PayoffBridge):
#             raise TypeError("arg 'put_payoff' must be type PayoffBridge")
#         self._put_payoff = put_payoff
#
#
# class PathDependentAutocallable(PathDependentBase):
#
#     def __init__(
#             self,
#             initial_spots: np.array,
#             fixings: np.ndarray,
#             delivery_time: np.float64,
#             cpn_barrier_payoff: PayoffBridge,
#             koc_barrier_payoff: PayoffBridge,
#             kip_barrier_payoff: PayoffBridge,
#             amount: np.float64,
#             coupon: np.float64,
#             coupon_cum: np.float64,
#             coupon_frequency: np.uint64,
#             number_of_asset: np.uint64
#     ) -> None:
#         super().__init__(fixings=fixings, number_of_asset=number_of_asset)
#         self.delivery_time = delivery_time
#         self.cpn_barrier_payoff = cpn_barrier_payoff
#         self.koc_barrier_payoff = koc_barrier_payoff
#         self.kip_barrier_payoff = kip_barrier_payoff
#         self.coupon = coupon / coupon_frequency
#         self.coupon_cum = coupon_cum / coupon_frequency
#         self.amount = amount
#         self.number_of_fixings = fixings.shape[0]
#         self.initial_spots = initial_spots
#
#     def clone(self) -> PathDependentBase:
#         return copy.deepcopy(self)
#
#     def max_number_of_cash_flows(self) -> np.uint64:
#         return np.uint64(1)
#
#     def possible_cash_flow_times(self) -> np.ndarray:
#         tmp = np.zeros(shape=(1, 1))
#         tmp[0, 0] = self.delivery_time
#         return tmp
#
#     def cash_flows(self, spot_values: np.ndarray, generated_flows: list[CashFlow]) -> np.uint64:
#         d_dims, m_sims, n_steps = spot_values.shape
#         kip_ratio = np.ones(shape=(d_dims, m_sims, 1))
#
#         initial_spots = np.array([np.tile(spot, (m_sims, n_steps)) for spot in self.initial_spots])
#         performance = spot_values / initial_spots
#         cpn_barrier_rslt = self.cpn_barrier_payoff(spot=performance).astype(dtype=bool)
#         koc_barrier_rslt = self.koc_barrier_payoff(spot=performance).astype(dtype=bool)
#         kip_barrier_rslt = self.kip_barrier_payoff(spot=performance).astype(dtype=bool)
#         mask0 = (kip_barrier_rslt==True)[:, :, -1]
#         kip_ratio[mask0] = performance[:, :, -1][mask0].reshape(-1, 1)
#         kip_ratio = kip_ratio.min(axis=0)
#
#         mask1 = np.any(koc_barrier_rslt!=True, axis=0) & np.any(cpn_barrier_rslt!=True, axis=0)
#         mask2 = np.all(koc_barrier_rslt==True, axis=0)
#
#         pay_list = []
#         for i in range(m_sims):
#             pay = np.float64(0)
#             for j in range(1, n_steps + 1):
#                 if ~mask1[i, j - 1]:
#                     pay += self.coupon
#                 elif mask2[i, j - 1]:
#                     pay = j * self.coupon_cum + pay
#                     break
#             pay_list.append(pay)
#
#         the_payoff = (np.float64(1) + np.array(pay_list).reshape(-1, 1)) * kip_ratio * self.amount
#
#         generated_flows[0].time_idx = np.uint64(0)
#         generated_flows[0].amount = the_payoff
#
#         return np.uint64(1)
#
#     @property
#     def delivery_time(self) -> np.float64:
#         return self._delivery_time
#
#     @delivery_time.setter
#     def delivery_time(self, delivery_time: np.float64) -> None:
#         self._delivery_time = np.float64(delivery_time)
#
#     @property
#     def number_of_asset(self) -> np.uint64:
#         return self._number_of_asset
#
#     @number_of_asset.setter
#     def number_of_asset(self, number_of_asset: np.uint64) -> None:
#         self._number_of_asset = np.uint64(number_of_asset)
#
#     @property
#     def coupon(self) -> np.float64:
#         return self._coupon
#
#     @coupon.setter
#     def coupon(self, coupon: np.float64) -> None:
#         self._coupon = np.float64(coupon)
#
#     @property
#     def coupon_cum(self) -> np.float64:
#         return self._coupon_cum
#
#     @coupon_cum.setter
#     def coupon_cum(self, value: np.float64) -> None:
#         self._coupon_cum = np.float64(value)
#
#     @property
#     def amount(self) -> np.float64:
#         return self._amount
#
#     @amount.setter
#     def amount(self, value: np.float64) -> None:
#         self._amount = np.float64(value)
#
#     @property
#     def initial_spots(self) -> np.ndarray:
#         return self._initial_spots
#
#     @initial_spots.setter
#     def initial_spots(self, initial_spots: np.ndarray) -> None:
#         self._initial_spots = initial_spots.reshape(-1, 1)


# class PathDependentCondVarSwap(PathDependentBase):
#
#     def __init__(
#             self,
#             fixings: NDArray[np.float64],
#             delivery_time: np.float64,
#             lower_barrier_payoff: PayoffBase,
#             upper_barrier_payoff: PayoffBase,
#             forward_payoff: PayoffBase,
#             realized_prices: NDArray[np.float64],
#             amount: np.float64,
#             exp_in_days: np.int64,
#             lag_in_days: np.int64,
#             base_days: np.int64
#     ) -> None:
#         super().__init__(fixings=fixings, number_of_asset=np.int64(1))
#         self.delivery_time = delivery_time
#         self.lower_barrier_payoff = lower_barrier_payoff
#         self.upper_barrier_payoff = upper_barrier_payoff
#         self.forward_payoff = forward_payoff
#         self.realized_prices = realized_prices
#         self.number_of_times = delivery_time
#         self.amount = amount
#         self.exp_in_days = exp_in_days
#         self.lag_in_days = lag_in_days
#         self.base_days = base_days
#         self.time_idx = np.uint64(0)
#
#     def clone(self) -> PathDependentBase:
#         return copy.deepcopy(self)
#
#     def max_number_of_cash_flows(self) -> np.uint64:
#         return np.uint64(1)
#
#     def possible_cash_flow_times(self) -> NDArray[np.float64]:
#         tmp = np.zeros(shape=(1, 1))
#         tmp[0, 0] = self.delivery_time
#         return tmp
#
#     def cash_flows(self, spot_values: NDArray[np.float64], generated_flows: list[CashFlow]) -> np.uint64:
#         _, m_sims, n_steps = spot_values.shape
#
#         spot_values = np.hstack(
#             (np.tile(self.realized_prices, reps=(1, m_sims)).T, spot_values[0])
#         )
#         lower_barrier_payoff = self.lower_barrier_payoff(spot_values).astype(bool)
#         upper_barrier_payoff = self.upper_barrier_payoff(spot_values).astype(bool)
#         log_return_squared = np.power(np.log(spot_values[:, 1:] / spot_values[:, :-1]), 2)
#         price_within_barrier_indicator = lower_barrier_payoff * upper_barrier_payoff
#         number_of_days_within_barrier = np.sum(price_within_barrier_indicator, axis=1)
#
#         annualized_ratio = (self.base_days / number_of_days_within_barrier)
#         lag_inverse = (1 / self.lag_in_days)
#         final_realized_volatility = np.sum(log_return_squared * price_within_barrier_indicator, axis=1)
#         final_realized_volatility *= annualized_ratio * lag_inverse
#         final_realized_volatility = np.sqrt(final_realized_volatility) * np.float64(100)
#
#         generated_flows[0].time_idx = np.uint64(0)
#         generated_flows[0].amount = self.forward_payoff(final_realized_volatility) * self.amount * (number_of_days_within_barrier / self.exp_in_days)
#
#         return np.uint64(1)
#
#     @property
#     def delivery_time(self) -> np.float64:
#         return self._delivery_time
#
#     @delivery_time.setter
#     def delivery_time(self, delivery_time: np.float64) -> None:
#         self._delivery_time = np.float64(delivery_time)
#
#     @property
#     def number_of_asset(self) -> np.uint64:
#         return self._number_of_asset
#
#     @number_of_asset.setter
#     def number_of_asset(self, number_of_asset: np.uint64) -> None:
#         self._number_of_asset = np.uint64(number_of_asset)
#
#     @property
#     def lower_barrier_payoff(self) -> PayoffBridge:
#         return self._lower_barrier_payoff
#
#     @lower_barrier_payoff.setter
#     def lower_barrier_payoff(self, value: PayoffBridge) -> None:
#         if not isinstance(value, PayoffBridge):
#             raise TypeError("arg 'lower_barrier_payoff' must be type PayoffBridge")
#         self._lower_barrier_payoff = value
#
#     @property
#     def upper_barrier_payoff(self) -> PayoffBridge:
#         return self._upper_barrier_payoff
#
#     @upper_barrier_payoff.setter
#     def upper_barrier_payoff(self, value: PayoffBridge) -> None:
#         if not isinstance(value, PayoffBridge):
#             raise TypeError("arg 'upper_barrier_payoff' must be type PayoffBridge")
#         self._upper_barrier_payoff = value
#
#     @property
#     def forward_payoff(self) -> PayoffBridge:
#         return self._forward_payoff
#
#     @forward_payoff.setter
#     def forward_payoff(self, value: PayoffBridge) -> None:
#         if not isinstance(value, PayoffBridge):
#             raise TypeError("arg 'forward_payoff' must be type PayoffBridge")
#         self._forward_payoff = value