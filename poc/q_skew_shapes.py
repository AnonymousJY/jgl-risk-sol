"""Can gamma_i alone fit a name whose smile is shaped unlike the index's?

THE STRUCTURAL POINT FIRST. In the pricer gamma_i enters as eta1/gamma_i and
eta2/gamma_i - it divides BOTH decay rates by the same factor. So the mean up
jump and the mean down jump scale together and their RATIO, eta2/eta1, is
invariant to gamma_i. gamma_i is a volume knob on the jump, not a balance
knob. Whatever asymmetry the index's (p, eta1, eta2) carries, every name
inherits it and can only scale it up or down.

That predicts which of the four shapes a one-parameter name fit can reach,
and this script measures it rather than asserting it. Four 3M target smiles
are generated from the model itself:

    1  both wings up          p 0.50, eta1 12, eta2 12   (a symmetric smile)
    2  put wing up, call flat p 0.15, eta1 40, eta2 10   (the equity smirk)
    3  put flat, call wing up p 0.85, eta1 10, eta2 40   (the reverse smirk)
    4  both flat              lamb 0.5, eta1 40, eta2 40 (nearly Black-Scholes)

Each is then fitted twice: once with gamma_i alone, holding (p, lamb, eta1,
eta2) at shape 2's values - the production idiosyncratic setup, with an index
carrying the usual equity smirk - and once with all four jump parameters free,
which is what the systematic arm does.

    python poc/q_skew_shapes.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Library.SkewCalibrationKimYi2025 import (                        # noqa: E402
    KimYiSkewCalibrationSystematic, KimYiSkewCalibrationIdiosyncratic)

T, S0, R, Q, N = 0.25, 100.0, 0.04, 0.015, 15
SIGMA, PHII = 0.15, 0.30          # index diffusion, name diffusion
SHAPES = [("1 put^ call^", dict(pprob=0.50, lamb=10.0, eta1=12.0, eta2=12.0)),
          ("2 put^ call-", dict(pprob=0.15, lamb=10.0, eta1=40.0, eta2=10.0)),
          ("3 put- call^", dict(pprob=0.85, lamb=10.0, eta1=10.0, eta2=40.0)),
          ("4 put- call-", dict(pprob=0.50, lamb=0.5, eta1=40.0, eta2=40.0))]
INDEX = SHAPES[1][1]              # the index the names are calibrated against


def grid():
    width = 3.0 * 0.45 * np.sqrt(T)
    k = S0 * np.exp(np.linspace(-width, width, N))
    return k, k > S0 * np.exp((R - Q) * T)


def name_fitter(mkt, jump, phii):
    k, is_call = grid()
    col = lambda v: np.full(N, float(v))
    return KimYiSkewCalibrationIdiosyncratic(
        sigma=np.array(SIGMA), pprob=np.array(jump["pprob"]),
        lamb=np.array(jump["lamb"]), eta1=np.array(jump["eta1"]),
        eta2=np.array(jump["eta2"]), mkt_imp_vol=mkt, und_price=col(S0),
        und_strike=k, risk_free_rate=col(R), dividend_yield=col(Q),
        time_to_expiry=col(T), is_call_option=is_call,
        option_weights=np.ones((N, 1)) / N, phii=phii)


def wings(iv, k, is_call):
    """IV at 90% and 110% moneyness minus ATM, in vol points (put, call)."""
    order = np.concatenate([np.flatnonzero(~is_call), np.flatnonzero(is_call)])
    m, v = k[order] / S0, np.asarray(iv).reshape(-1)
    f = lambda x: float(np.interp(x, m, v))
    return 100 * (f(0.90) - f(1.0)), 100 * (f(1.10) - f(1.0))


def gamma_sweep():
    """What gamma_i actually moves: amplitude, and not the balance."""
    k, is_call = grid()
    print("  gamma_i sweep at the index's own jump parameters")
    print("    %7s %9s %9s %9s" % ("gamma_i", "put wg", "call wg", "put/call"))
    for g in (0.5, 1.0, 1.5, 2.0, 3.0):
        iv = np.asarray(name_fitter(np.zeros(N), INDEX, PHII).model_vol(
            np.array([g]))).reshape(-1)
        pw, cw = wings(iv, k, is_call)
        print("    %7.1f %9.2f %9.2f %9.3f" % (g, pw, cw, pw / cw))
    print()


def main():
    k, is_call = grid()
    print("3M, name phi_i %.2f, index sigma %.2f, %d strikes\n" % (PHII, SIGMA, N))
    gamma_sweep()
    print("  %-14s %8s %8s   %-28s %-28s"
          % ("target shape", "put wg", "call wg", "gamma_i only (production)",
             "all four jump parameters"))

    for label, jump in SHAPES:
        tgt = np.asarray(name_fitter(np.zeros(N), jump, PHII).model_vol(
            np.array([1.0]))).reshape(-1)                   # gamma_i = 1 truth
        pw, cw = wings(tgt, k, is_call)

        # (a) production: gamma_i free, jump parameters pinned at the index's
        f1 = name_fitter(tgt, INDEX, PHII)
        r1 = minimize(lambda u: f1.target(np.array([u[0] * 2.0])), x0=[0.5],
                      method="SLSQP", bounds=[(0.05, 10.0)], tol=1e-12)
        g = float(r1.x[0]) * 2.0
        iv1 = np.asarray(f1.model_vol(np.array([g]))).reshape(-1)
        e1 = 100 * np.sqrt(np.mean((tgt - iv1) ** 2))
        p1, c1 = wings(iv1, k, is_call)

        # (b) the systematic arm's freedom: all four jump parameters
        f2 = KimYiSkewCalibrationSystematic(
            mkt_imp_vol=tgt, und_price=np.full(N, S0), und_strike=k,
            risk_free_rate=np.full(N, R), dividend_yield=np.full(N, Q),
            time_to_expiry=np.full(N, T), is_call_option=is_call,
            option_weights=np.ones((N, 1)) / N, sigma=np.array(PHII))
        sc = np.array([0.5, 10.0, 25.0, 25.0])
        r2 = minimize(lambda u: f2.target(u * sc), x0=np.array([.4, .5, 1., .4]),
                      method="SLSQP", tol=1e-12, options={"maxiter": 60},
                      bounds=[(1e-3 / .5, 1 / .5), (1e-4 / 10, None),
                              (1.5 / 25, None), (.5 / 25, None)])
        iv2 = np.asarray(f2.model_vol(r2.x * sc)).reshape(-1)
        e2 = 100 * np.sqrt(np.mean((tgt - iv2) ** 2))
        p2, c2 = wings(iv2, k, is_call)

        print("  %-14s %7.2f %8.2f   gam %5.2f  rmse %5.2f  %5.2f/%5.2f"
              "   rmse %5.2f  %5.2f/%5.2f"
              % (label, pw, cw, g, e1, p1, c1, e2, p2, c2))

    print("\n  put wg / call wg are IV(90%) - IV(ATM) and IV(110%) - IV(ATM),")
    print("  vol points. The pairs after each rmse are the FITTED wings, to be")
    print("  read against the target's two columns on the left.")


if __name__ == "__main__":
    main()
