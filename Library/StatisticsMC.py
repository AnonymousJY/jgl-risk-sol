import copy
import numpy as np
from scipy import stats
from numpy.typing import NDArray
from abc import ABC, abstractmethod


def get_corr_mat(rhoij: NDArray[np.float64], number_of_assets: int) -> NDArray[np.float64]:
    rhoij_mat = np.eye(number_of_assets)
    rows, cols = np.triu_indices(n=number_of_assets, k=1)
    rhoij_mat[rows, cols] = rhoij.T
    rhoij_mat[cols, rows] = rhoij.T
    return rhoij_mat


class StatisticsMCBase(ABC):

    @abstractmethod
    def dump_result(self, result: NDArray[np.float64]) -> None:
        pass

    @abstractmethod
    def get_result_so_far(self) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def __deepcopy__(self, memodict={}) -> 'StatisticsMCBase':
        pass


class StatisticsMCMean(StatisticsMCBase):

    def __init__(self):
        self.running_sum = np.zeros(shape=(1, 1))

    def dump_result(self, result: NDArray[np.float64]) -> None:
        self.running_sum = result

    def get_result_so_far(self) -> NDArray[np.float64]:
        return np.mean(self.running_sum).reshape((-1, 1))

    def __deepcopy__(self, memodict={}) -> StatisticsMCBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def running_sum(self) -> NDArray[np.float64]:
        return self._running_sum

    @running_sum.setter
    def running_sum(self, value: NDArray[np.float64]) -> None:
        self._running_sum = np.array(value).reshape((-1, 1))


class StatisticsMCVariance(StatisticsMCBase):

    def __init__(self):
        self.running_sum = np.zeros(shape=(1, 1))

    def dump_result(self, result: NDArray[np.float64]) -> None:
        self.running_sum = result

    def get_result_so_far(self) -> NDArray[np.float64]:
        return np.var(self.running_sum).reshape((-1, 1))

    def __deepcopy__(self, memodict={}) -> StatisticsMCBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def running_sum(self) -> NDArray[np.float64]:
        return self._running_sum

    @running_sum.setter
    def running_sum(self, value: NDArray[np.float64]) -> None:
        self._running_sum = np.array(value).reshape((-1, 1))


class StatisticsMCConfidenceInterval(StatisticsMCBase):

    def __init__(self, confidence_level: NDArray[np.float64] = 0.975):
        self.inner_var = StatisticsMCVariance()
        self.n_paths = np.int64(0)
        self.confidence_level = confidence_level

    def dump_result(self, result: NDArray[np.float64]) -> None:
        self.n_paths = result.shape[0]
        self.inner_mean.dump_result(result)
        self.inner_var.dump_result(result)

    def get_result_so_far(self) -> NDArray[np.float64]:
        mean = self.inner_mean.get_result_so_far()
        var = self.inner_var.get_result_so_far()
        stddev_factor = self.confidence_level * np.sqrt(var / self.n_paths)
        lower = mean - stddev_factor
        upper = mean + stddev_factor
        return np.array([mean, lower, upper]).reshape(-1, 1)

    def __deepcopy__(self, memodict={}) -> StatisticsMCBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def confidence_level(self) -> np.float64:
        return self._confidence_level

    @confidence_level.setter
    def confidence_level(self, value: np.float64) -> None:
        self._confidence_level = stats.norm.ppf(q=np.float64(value))


class StatisticsMCQuantile(StatisticsMCBase):

    def __init__(self, alpha: NDArray[np.float64]):
        self.running_sum = np.zeros(shape=(1, 1))
        self.alpha = alpha

    def dump_result(self, result: NDArray[np.float64]) -> None:
        self.running_sum = result

    def get_result_so_far(self) -> NDArray[np.float64]:
        return np.quantile(self.running_sum, q=self.alpha)

    def __deepcopy__(self, memodict={}) -> StatisticsMCBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def running_sum(self) -> NDArray[np.float64]:
        return self._running_sum

    @running_sum.setter
    def running_sum(self, value: np.float64) -> None:
        self._running_sum = np.array(value).reshape((-1, 1))

    @property
    def alpha(self) -> NDArray[np.float64]:
        return self._alpha

    @alpha.setter
    def alpha(self, value: NDArray[np.float64]) -> None:
        self._alpha = np.array(value).reshape((-1, 1))


class StatisticsMCConditionalQuantile(StatisticsMCBase):

    def __init__(self, alpha: NDArray[np.float64]):
        self.running_sum = np.zeros(shape=(1, 1))
        self.alpha = alpha

    def dump_result(self, result: NDArray[np.float64]) -> None:
        self.running_sum = result

    def get_result_so_far(self) -> NDArray[np.float64]:
        x = np.sort(self.running_sum)
        n = x.shape[0]
        threshold = int(n * self.alpha)
        return np.mean(x[:threshold]) * -1.

    def __deepcopy__(self, memodict={}) -> StatisticsMCBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    @property
    def running_sum(self) -> NDArray[np.float64]:
        return self._running_sum

    @running_sum.setter
    def running_sum(self, value: np.float64) -> None:
        self._running_sum = np.array(value).reshape((-1, 1))

    @property
    def alpha(self) -> NDArray[np.float64]:
        return self._alpha

    @alpha.setter
    def alpha(self, value: NDArray[np.float64]) -> None:
        self._alpha = np.array(value).reshape((-1, 1))
