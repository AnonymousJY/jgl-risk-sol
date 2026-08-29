import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable


def max_iter_reached_msg(x: int) -> str:
    return f'Reached maxed iterations of {x} : may not have converged. The End!'


def bisection(
        func: Callable,
        lower_value: NDArray[np.float64],
        upper_value: NDArray[np.float64],
        target_value: NDArray[np.float64],
        initial_value: NDArray[np.float64],
        max_iter: np.int64 = 1_000,
        tolerance: np.float64 = 1e-8,
        is_verbose: bool = False
) -> NDArray[np.float64]:

    a = lower_value
    b = upper_value

    if (a > b).any():
        print('Lower value is larger than the upper value. The End!')
        return initial_value

    c = 0.5 * (a + b)
    fc = func(c)

    for i in range(max_iter):

        if is_verbose:
            print(f'iter = {i + 1} ; '
                  f'a = {a.reshape(-1,)[0]:.4f} ; '
                  f'b = {b.reshape(-1,)[0]:.4f} ; '
                  f'c = {c.reshape(-1,)[0]:.4f} ; '
                  f'b - c = {(b - c).reshape(-1,)[0]:.4f} ; '
                  f'f(c) = {fc.reshape(-1,)[0]:.4f}'
                  )

        if i == max_iter - 1:
            print(max_iter_reached_msg(i + 1))
            break

        if np.array(np.abs(fc - target_value) < tolerance).all():
            break

        mask = fc < target_value
        a[mask] = c[mask]
        b[~mask] = c[~mask]

        c = 0.5 * (a + b)
        fc = func(c)
        i += 1

    return c


def newton_raphson(
        func: Callable,
        func_deriv: Callable,
        target_value: NDArray[np.float64],
        initial_value: NDArray[np.float64],
        max_iter: np.int64 = 1_000,
        tolerance: np.float64 = 1e-8,
        is_verbose: bool = False
) -> NDArray[np.float64]:

    x0 = initial_value

    for i in range(max_iter):

        dydx = np.maximum(func_deriv(x0), 1e-8)

        x1 = x0  - (func(x0) - target_value) / dydx
        y_new = func(x1)
        error = y_new - target_value

        if is_verbose:
            print(f'iter = {i + 1} ; '
                  f'f_prime = {dydx.reshape(-1,)[0]:.4f} ; '
                  f'x_n = {x1.reshape(-1,)[0]:.4f} ; '
                  f'f(x_n) = {y_new.reshape(-1,)[0]:.4f} ; '
                  f'x_n - x_n-1 = {(x1 - x0).reshape(-1,)[0]:.4f} ; '
                  f'target_value - f(x_n) = {(target_value - y_new).reshape(-1,)[0]:.4f}'
                  )

        if np.array(np.abs(error) < tolerance).all():
            x0 = x1
            break
        elif i == max_iter:
            print(max_iter_reached_msg(i))
            break
        else:
            x0 = x1
            i += 1

    return x0


if __name__=='__main__':

    f = lambda x: x**6 - x - 1
    f_prime = lambda x: 6 * x**5 - 1

    # true_value = f(np.array(1.134724138))

    # x1 = newton_raphson(f, f_prime, target_value=true_value, initial_value=np.array(1.5), is_verbose=True)
    # x2 = bisection(f, lower_value=np.array(1.), upper_value=np.array(2.), target_value=true_value, initial_value=np.array(1.5), is_verbose=True)
    # print(x1 - x2)

    from OptionPricerBSM1973 import *

    vol_true = np.array([.5, .4])
    r = np.array(.05)
    d = np.array(0.)
    T = np.array(.5)
    S = np.array([95., 100.])
    K = np.array(200.)

    pricer_obj = BlackScholesMertonCall(und_price=S, und_strike=K, risk_free_rate=r, dividend_yield=d, time_to_expiry=T)
    true_price = pricer_obj.price(vol_true)
    true_vega = pricer_obj.vega(vol_true)

    print(true_vega.reshape(-1,), true_price.reshape(-1,))

    vol_calib = newton_raphson(
        func=pricer_obj.price,
        func_deriv=pricer_obj.vega,
        target_value=true_price,
        initial_value=np.array([2.] * vol_true.shape[0]).reshape(-1, 1),
        is_verbose=True
    )

    # vol_calib = bisection(
    #     func=pricer_obj.price,
    #     lower_value=np.array([0.01] * 2).reshape(-1, 1),
    #     upper_value=np.array([5.] * 2).reshape(-1, 1),
    #     target_value=true_price,
    #     initial_value=np.array([.1] * vol_true.shape[0]).reshape(-1, 1),
    #     is_verbose=True
    # )

    print(vol_true)
    print(vol_calib.reshape(-1,))