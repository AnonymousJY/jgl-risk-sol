import numpy as np
from typing import Callable
from numpy.typing import NDArray


UST_TENOR_MAP = {
    '1m': np.array(1/12),
    '1.5m': np.array(1.5/12),
    '2m': np.array(2/12),
    '3m': np.array(3/12),
    '4m': np.array(4/12),
    '6m': np.array(6/12),
    '1y': np.array(1.),
    '2y': np.array(2.),
    '3y': np.array(3.),
    '5y': np.array(5.),
    '7y': np.array(7.),
    '10y': np.array(10.),
    '20y': np.array(20.),
    '30y': np.array(30.)
}


def get_fixings_vec(number_of_fixings: np.uint64, expiry: np.float64) -> NDArray[np.float64]:
    return np.array([(i + 1.) * expiry / number_of_fixings for i in range(number_of_fixings)], dtype=np.float64)


def year_frac(base_days: np.int64) -> Callable:

    if base_days == np.int64(252):
        return _actual_252
    elif base_days == np.int64(360):
        return _actual_360
    elif base_days == np.int64(365):
        return _actual_365


def _actual_252(time1: NDArray[np.float64], time2: NDArray[np.float64]) -> NDArray[np.float64]:
    time1 = np.array(time1).reshape(-1, 1)
    time2 = np.array(time2).reshape(-1, 1)
    return (time2 - time1) / 252.


def _actual_360(time1: NDArray[np.float64], time2: NDArray[np.float64]) -> NDArray[np.float64]:
    time1 = np.array(time1).reshape(-1, 1)
    time2 = np.array(time2).reshape(-1, 1)
    return (time2 - time1) / 360.


def _actual_365(time1: NDArray[np.float64], time2: NDArray[np.float64]) -> NDArray[np.float64]:
    time1 = np.array(time1).reshape(-1, 1)
    time2 = np.array(time2).reshape(-1, 1)
    return (time2 - time1) / 365.