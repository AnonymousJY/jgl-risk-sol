"""Do random starts recover the true Q jump parameters? Four scenarios.

WHY THIS REPLACES THE HAND-PICKED START LADDER. poc/q_recover_3m.py used five
fixed starts, and one of them turned out to BE scenario 2's true parameters -
(0.15, 10, 40, 10) exactly - with another carrying scenario 1's true p and
lamb. "Best of five starts recovers the truth" is worth nothing when a start
is standing on the answer. Random draws over the support cannot do that, and
this script prints the closest any draw came to the truth so the claim is
auditable rather than asserted.

WHAT IS MEASURED. For each scenario: simulate the 3M surface at the true
(p, lamb, eta1, eta2), draw NDRAWS random starts, calibrate from each, and
count how many come back to the truth. p is drawn uniform on its support; the
three positive scale parameters are drawn LOG-uniform, because they range over
orders of magnitude and a uniform draw would crowd the top of the range.

Three numbers decide the reading:

  reached      how many draws landed within 1% of the truth on all four
               parameters. This is the honest success rate of the calibration
               as a procedure, not of the model.
  objective    the value at the truth is 0 by construction, so any draw
               stopping well above 0 stopped early rather than finding a rival
               optimum. Draws that stop at a genuinely different parameter set
               with objective ~0 would mean the surface does not identify.
  nearest      the closest any START came to the truth, relative. If this is
               small the run is contaminated the way the old ladder was, and
               the reached count should be discarded.

    python poc/q_multistart.py
    NDRAWS=25 python poc/q_multistart.py
    SCENARIOS=2,4 NDRAWS=40 python poc/q_multistart.py

Budget roughly 30-40 seconds per draw: the default 10 draws x 4 scenarios is
about half an hour.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Library.SkewCalibrationKimYi2025 import KimYiSkewCalibrationSystematic  # noqa: E402

BOUNDS = [(0., 1.), (1e-4, None), (1.5, None), (0.5, None)]
SCALE = np.array([0.20, 5.00, 22.00, 7.00])      # production's x0, as the metric

# The range each start is drawn from. p is uniform; the rest log-uniform.
SUPPORT = [("pprob", 0.05, 0.95, "uniform"),
           ("lamb", 0.20, 50.0, "log"),
           ("eta1", 5.00, 80.0, "log"),
           ("eta2", 5.00, 80.0, "log")]

SCENARIOS = [("1 both wings up",    dict(pprob=0.50, lamb=10.0, eta1=12.0, eta2=12.0)),
             ("2 put up call down", dict(pprob=0.15, lamb=10.0, eta1=40.0, eta2=10.0)),
             ("3 put down call up", dict(pprob=0.85, lamb=10.0, eta1=10.0, eta2=40.0)),
             ("4 both flat",        dict(pprob=0.50, lamb=0.50, eta1=40.0, eta2=40.0))]


def draw_starts(rng, n):
    out = np.empty((n, 4))
    for j, (_, lo, hi, kind) in enumerate(SUPPORT):
        out[:, j] = (rng.uniform(lo, hi, n) if kind == "uniform"
                     else np.exp(rng.uniform(np.log(lo), np.log(hi), n)))
    return out


def fitter(T, n, phii, s0, r, q, mkt):
    width = 3.0 * 0.45 * np.sqrt(T)
    k = s0 * np.exp(np.linspace(-width, width, n))
    col = lambda v: np.full(n, float(v))
    return KimYiSkewCalibrationSystematic(
        mkt_imp_vol=mkt, und_price=col(s0), und_strike=k, risk_free_rate=col(r),
        dividend_yield=col(q), time_to_expiry=col(T),
        is_call_option=k > s0 * np.exp((r - q) * T),
        option_weights=np.ones((n, 1)) / n, sigma=np.array(phii))


def fit(f, x0):
    bnds = [(lo / s, None if hi is None else hi / s)
            for (lo, hi), s in zip(BOUNDS, SCALE)]
    r = minimize(lambda u: f.target(u * SCALE), x0=x0 / SCALE, method="SLSQP",
                 bounds=bnds, tol=1e-11, options={"maxiter": 150})
    return np.asarray(r.x, dtype=float) * SCALE, float(r.fun)


def main():
    env = lambda k, d: float(os.environ.get(k, d))
    T, n = env("TENOR", 0.25), int(env("NSTRIKES", 11))
    phii, s0, r, q = env("PHII", 0.30), env("SPOT", 100.), env("RATE", 0.04), env("DIVY", 0.015)
    ndraws = int(env("NDRAWS", 10))
    rng = np.random.default_rng(int(env("SEED", 20240114)))
    pick = os.environ.get("SCENARIOS")
    scen = ([s for s in SCENARIOS if s[0][0] in pick.split(",")] if pick else SCENARIOS)

    print("tenor %.3f yr, %d strikes, phi_i %.2f, %d random starts per scenario"
          % (T, n, phii, ndraws))
    print("start support: " + ", ".join(
        "%s %s(%g, %g)" % (nm, "U" if k == "uniform" else "logU", lo, hi)
        for nm, lo, hi, k in SUPPORT) + "\n")

    summary = []
    for label, jump in scen:
        true = np.array([jump["pprob"], jump["lamb"], jump["eta1"], jump["eta2"]])
        tgt = np.asarray(fitter(T, n, phii, s0, r, q, np.zeros(n))
                         .model_vol(true)).reshape(-1)
        f = fitter(T, n, phii, s0, r, q, tgt)
        starts = draw_starts(rng, ndraws)
        nearest = float(np.min(np.max(np.abs(starts - true) / true, axis=1)))

        print("%s   true p %.3f lam %.2f e1 %.2f e2 %.2f   objective at truth %.2e"
              % (label, *true, float(f.target(true))))
        print("  closest start was %.0f%% away on its worst coordinate" % (100 * nearest))
        print("  %5s %8s %8s %8s %8s %11s %8s"
              % ("draw", "p", "lamb", "eta1", "eta2", "objective", "err"))

        errs, objs, best, bobj = [], [], None, np.inf
        t0 = time.time()
        for i, x0 in enumerate(starts, 1):
            x, o = fit(f, x0)
            e = 100 * np.max(np.abs(x - true) / true)
            errs.append(e)
            objs.append(o)
            if o < bobj:
                best, bobj = x, o
            print("  %5d %8.3f %8.2f %8.2f %8.2f %11.3e %7.1f%%"
                  % (i, x[0], x[1], x[2], x[3], o, e), flush=True)

        errs, objs = np.array(errs), np.array(objs)
        got1 = int(np.sum(errs <= 1.0))
        got5 = int(np.sum(errs <= 5.0))
        print("  reached the truth: %d/%d within 1%%, %d/%d within 5%%   (%.0fs)"
              % (got1, ndraws, got5, ndraws, time.time() - t0))
        print("  best objective %.3e at p %.3f lam %.2f e1 %.2f e2 %.2f (err %.1f%%)\n"
              % (bobj, *best, 100 * np.max(np.abs(best - true) / true)))
        summary.append((label, got1, got5, ndraws, nearest, float(np.median(objs)), bobj))

    print("\n  %-20s %10s %10s %10s %12s"
          % ("scenario", "<=1% err", "<=5% err", "nearest", "median obj"))
    for label, g1, g5, nd, near, med, _ in summary:
        print("  %-20s %7d/%-3d %7d/%-3d %9.0f%% %12.2e"
              % (label, g1, nd, g5, nd, 100 * near, med))
    print("\n  'nearest' is the closest any random START came to that scenario's")
    print("  truth. A large value is what makes the reached count meaningful -")
    print("  it rules out a start having been handed the answer.")


if __name__ == "__main__":
    main()
