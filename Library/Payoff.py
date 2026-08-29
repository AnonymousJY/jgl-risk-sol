import copy
import numpy as np
from Library.Wrapper import Wrapper
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class PayoffBase(ABC):

    @abstractmethod
    def __call__(self, spot: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def __deepcopy__(self, memodict={}) -> 'PayoffBase':
        pass


class PayoffCall(PayoffBase):

    def __init__(self, strike: NDArray[np.float64]) -> None:
        self.strike = strike

    def __call__(self, spot: NDArray[np.float64]) -> NDArray[np.float64]:
        spot = spot.reshape((-1, 1))
        return np.maximum(spot - self.strike, 0.)

    def __deepcopy__(self, memodict={}) -> PayoffBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def strike(self) -> NDArray[np.float64]:
        return self._strike

    @strike.setter
    def strike(self, strike: NDArray[np.float64]) -> None:
        self._strike = np.array(strike).reshape((-1, 1))


class PayoffPut(PayoffBase):

    def __init__(self, strike: NDArray[np.float64]) -> None:
        self.strike = strike

    def __call__(self, spot: NDArray[np.float64]) -> NDArray[np.float64]:
        spot = spot.reshape((-1, 1))
        return np.maximum(self.strike - spot, 0.)

    def __deepcopy__(self, memodict={}) -> PayoffBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def strike(self) -> NDArray[np.float64]:
        return self._strike

    @strike.setter
    def strike(self, strike: NDArray[np.float64]) -> None:
        self._strike = np.array(strike).reshape((-1, 1))


class PayoffBinaryCall(PayoffBase):

    def __init__(self, strike: NDArray[np.float64]) -> None:
        self.inner_ptr = PayoffCall(strike=strike)

    def __call__(self, spot: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(self.inner_ptr(spot=spot) > 0., 1., 0.)

    def __deepcopy__(self, memodict={}) -> PayoffBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result


class PayoffBinaryPut(PayoffBase):

    def __init__(self, strike: NDArray[np.float64]) -> None:
        self.inner_ptr = PayoffPut(strike=strike)

    def __call__(self, spot: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(self.inner_ptr(spot=spot) > 0., 1., 0.)

    def __deepcopy__(self, memodict={}) -> PayoffBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result


class PayoffSpread(PayoffBase):

    def __init__(
            self,
            payoff_lower: PayoffBase,
            payoff_upper: PayoffBase,
            volume_upper: np.float64 = np.float64(1),
            volume_lower: np.float64 = np.float64(1)
    ) -> None:
        self.payoff_lower = payoff_lower
        self.payoff_upper = payoff_upper
        self.volume_upper = volume_upper
        self.volume_lower = volume_lower

    def __call__(self, spot: NDArray[np.float64]) -> NDArray[np.float64]:
        return (self.volume_lower * self.payoff_lower(spot)) + (self.volume_upper * self.payoff_upper(spot))

    def __deepcopy__(self, memodict={}) -> PayoffBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def volume_lower(self) -> np.float64:
        return self._volume_lower

    @volume_lower.setter
    def volume_lower(self, value: np.float64) -> None:
        self._volume_lower = np.float64(value)

    @property
    def volume_upper(self) -> np.float64:
        return self._volume_upper

    @volume_upper.setter
    def volume_upper(self, value: np.float64) -> None:
        self._volume_upper = np.float64(value)

    @property
    def payoff_lower(self) -> PayoffBase:
        return self._payoff_lower

    @payoff_lower.setter
    def payoff_lower(self, value: PayoffBase) -> None:
        self._payoff_lower = Wrapper(value)

    @property
    def payoff_upper(self) -> PayoffBase:
        return self._payoff_lower

    @payoff_upper.setter
    def payoff_upper(self, value: PayoffBase) -> None:
        self._payoff_lower = Wrapper(value)


class PayoffForward(PayoffBase):

    def __init__(self, strike: NDArray[np.float64]) -> None:
        self.strike = strike

    def __call__(self, spot: NDArray[np.float64]) -> NDArray[np.float64]:
        return spot - self.strike

    def __deepcopy__(self, memodict={}) -> PayoffBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def strike(self) -> NDArray[np.float64]:
        return self._strike

    @strike.setter
    def strike(self, strike: NDArray[np.float64]) -> None:
        self._strike = np.array(strike).reshape((-1, 1))
