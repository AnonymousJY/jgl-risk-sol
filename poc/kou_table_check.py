"""Does the vectorised P/Q table build still agree with _P and _Q?

OptionPricerKou2002._pq_tables stopped calling _P and _Q term by term and now
builds each entry as one dot product over j, with the binomials coming from a
Pascal triangle instead of scipy's comb. That is a rearrangement of the same
sum, so it has to agree to rounding - and because the sum is accumulated in a
different order, exactly to rounding and not to the bit. This script pins both
halves of that claim: the tables against the definition, and the prices the
tables feed, with put-call parity as an independent check.

It also reports the speedup, since that is the whole reason for the change:
the optimizer drives lambda up when it is lost, bound rises with lambda*T, and
the old build was cubic in bound.

    python poc/kou_table_check.py
    NCASE=20 python poc/kou_table_check.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Library.OptionPricerKou2002 as K  # noqa: E402

TOL = 1e-13          # generous against the 1e-15 these actually land at


def definition_tables(bound, eta1, eta2, p):
    """What _pq_tables used to do: one _P and one _Q call per entry."""
    return ({n: {k: K._P(n, k, eta1, eta2, p) for k in range(1, n + 1)}
             for n in range(1, bound)},
            {n: {k: K._Q(n, k, eta1, eta2, p) for k in range(1, n + 1)}
             for n in range(1, bound)})


def main():
    ncase = int(os.environ.get("NCASE", 8))
    rng = np.random.default_rng(int(os.environ.get("SEED", 7)))

    worst = 0.
    for _ in range(ncase):
        e1, e2 = float(rng.uniform(1.6, 95)), float(rng.uniform(0.6, 95))
        p, bound = float(rng.uniform(.02, .98)), int(rng.integers(15, 62))
        Pf, Qf = K._pq_scalar(bound, e1, e2, p)
        Pr, Qr = definition_tables(bound, e1, e2, p)
        for n in range(1, bound):
            for k in range(1, n + 1):
                for ref, got in ((Pr[n][k], Pf[n][k]), (Qr[n][k], Qf[n][k])):
                    ref = float(np.asarray(ref).reshape(-1)[0])
                    worst = max(worst, abs(got - ref) / max(abs(ref), 1e-300))
    print("  tables vs _P/_Q over %d random (bound, eta1, eta2, p): %.3e"
          % (ncase, worst))

    col = lambda v, n=11: np.full((n, 1), float(v))
    strikes = (100 * np.exp(np.linspace(-.675, .675, 11))).reshape(-1, 1)
    mkt = dict(r=col(.04), d=col(.015), S0=col(100.), K=strikes, expiry=col(.25))
    # The corners the calibration studies actually visit: the four scenario
    # truths, plus two points the optimizer stalled on.
    cases = [(.3, 10., .50, 12., 12.), (.3, 10., .15, 40., 10.),
             (.3, 10., .85, 10., 40.), (.3, .5, .50, 40., 40.),
             (.3, 54., .30, 5.05, 48.9), (.3, 18., .55, 80., 9.9)]

    fast = K._pq_tables
    wpx, wpar, t_ref, t_fast = 0., 0., 0., 0.
    for sg, lam, p, e1, e2 in cases:
        a = dict(sigma=col(sg), lam=lam, p=p, eta1=e1, eta2=e2, **mkt)
        K._PQ_CACHE.clear(); K._pq_tables = definition_tables
        t0 = time.time()
        ref = np.asarray(K.kou_call(**a)).reshape(-1)
        t_ref += time.time() - t0
        K._PQ_CACHE.clear(); K._pq_tables = fast
        t0 = time.time()
        got = np.asarray(K.kou_call(**a)).reshape(-1)
        t_fast += time.time() - t0
        put = np.asarray(K.kou_put(**a)).reshape(-1)
        par = (got + strikes.reshape(-1) * np.exp(-.04 * .25)
               - 100 * np.exp(-.015 * .25) - put)
        wpx = max(wpx, float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12))))
        wpar = max(wpar, float(np.max(np.abs(par))))

    print("  prices vs the definition over %d parameter sets:            %.3e"
          % (len(cases), wpx))
    print("  put-call parity residual:                                  %.3e" % wpar)
    print("  cold-cache kou_call, 11 strikes: definition %.2f s, now %.2f s  (%.0fx)"
          % (t_ref, t_fast, t_ref / max(t_fast, 1e-9)))

    bad = [n for n, v in (("tables", worst), ("prices", wpx), ("parity", wpar))
           if v > TOL]
    print("\n  %s" % ("FAIL: " + ", ".join(bad) if bad
                      else "agrees to rounding on all three."))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
