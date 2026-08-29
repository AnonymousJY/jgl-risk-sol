import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator


def pchip_interpolator2d(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        x1: NDArray[np.float64],
        x2: NDArray[np.float64]
) -> NDArray[np.float64]:

    x1 = np.array(x1).reshape(-1,)
    x2 = np.array(x2).reshape(-1,)

    x_min = np.min(x)
    x_max = np.max(x)

    for i, v in enumerate(x1):
        if v < x_min:
            x1[i] = x_min
        elif v > x_max:
            x1[i] = x_max

    y_min = np.min(y)
    y_max = np.max(y)

    for i, v in enumerate(x2):
        if v < y_min:
            x2[i] = y_min
        elif v > y_max:
            x2[i] = y_max

    y1 = PchipInterpolator(x=x, y=z, extrapolate=False, axis=0)(x1)
    y2 = PchipInterpolator(x=y, y=y1, extrapolate=False, axis=1)(x2)

    return np.array([y2])
