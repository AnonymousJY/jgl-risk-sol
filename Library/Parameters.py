import copy
import numpy as np

from numpy.typing import NDArray
from abc import ABC, abstractmethod


class ParametersBase(ABC):

    @abstractmethod
    def integral(self, time1: NDArray[np.float64], time2: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def integral_square(self, time1: NDArray[np.float64], time2: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def __deepcopy__(self, memodict={}) -> 'ParametersBase':
        pass


class ParametersConstant(ParametersBase):

    def __init__(self, constant: NDArray[np.float64]) -> None:
        self.constant = constant
        self.constant_square = constant**2

    def integral(self, time1: NDArray[np.float64], time2: NDArray[np.float64]) -> NDArray[np.float64]:
        time1 = time1.reshape((-1, 1))
        time2 = time2.reshape((-1, 1))
        return (time2 - time1) * self.constant

    def integral_square(self, time1: NDArray[np.float64], time2: NDArray[np.float64]) -> NDArray[np.float64]:
        time1 = time1.reshape((-1, 1))
        time2 = time2.reshape((-1, 1))
        return (time2 - time1) * self.constant_square

    def __deepcopy__(self, memodict={}) -> ParametersBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def constant(self) -> NDArray[np.float64]:
        return self._constant

    @constant.setter
    def constant(self, value: NDArray[np.float64]) -> None:
        self._constant = np.array(value).reshape((-1, 1))
