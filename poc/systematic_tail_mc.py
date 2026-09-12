"""Expected shortfall by simulation, from each year's estimated parameters.

Paths come from KimYiRiskEngine.random - the repo's own simulator, the same
one the VaR run uses. One simulation per year at that year's MEAN fitted
parameters, then the ES straight off the sample.

ES at level a averages the worst round(a*N) of N paths, so it carries sampling
error. se = sd(tail)/sqrt(a*N) is printed beside every figure.

    python poc/systematic_tail_mc.py
    python poc/systematic_tail_mc.py --paths 1000000
    python poc/systematic_tail_mc.py --horizon 10
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import PMLE_DIR                              # noqa: E402
from Library.Logging import report as _report                        # noqa: E402
from Library.Parameters import ParametersConstant                    # noqa: E402
from Library.Random import RandomMT19937                             # noqa: E402
from Library.RiskEngineKimYi2025 import (KimYiRiskEngine,            # noqa: E402
                                         SYSTEMATIC_PRIOR_SETS)
from poc.estimate_systematic import store_id                         # noqa: E402

SYSTEMATIC_ID = "^SPX"
BASE_DAYS = 252
SEED = 20240114
COLS = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]

_LOG = _report(__name__)


def simulate(row, dt, horizon, paths, seed):
    """One parameter set -> `paths` horizon-day returns.

    Systematic block, so beta = gamma = 1 and kappa = rho = mu = 0, exactly as
    simulate_shock_returns_systematic_helper builds it in run_var_kimyi2025.
    """
    k = lambda v: ParametersConstant(np.array(float(v)))             # noqa: E731
    ret, _, _ = KimYiRiskEngine(
        mui=[k(0.)], kappai=[k(0.)], gammai=[k(1.)],
        betai=[k(1.)], rhoix=[k(0.)],
        alpha=k(row["dALPHA"]), sigma=k(row["dSIGMA"]),
        pprob=k(row["dPPROB"]), lamb=k(row["dLAMB"]),
        eta1=k(row["dETA1"]), eta2=k(row["dETA2"]),
        end_dt=np.array(dt),
    ).random(rng=RandomMT19937(np.int64(seed)), size=(1, paths, horizon))

    r = ret[0][:, 1:]                              # drop the identity column
    if horizon == 1:
        return r[:, 0]
    return np.expm1(np.log(np.clip(1.0 + r, 1e-300, None)).sum(axis=1))


def es(x, level, lower=True):
    """(quantile, ES, se) from a simulated sample."""
    x = np.sort(np.asarray(x, dtype=float))
    m = max(int(round(level * x.size)), 2)
    tail = x[:m] if lower else x[-m:]
    return (float(x[m - 1] if lower else x[-m]), float(tail.mean()),
            float(tail.std(ddof=1) / np.sqrt(m)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priors", default="alpha-pprob-eta-flat")
    ap.add_argument("--lookback", type=int, default=504)
    ap.add_argument("--levels", default="0.025,0.01,0.005")
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    levels = [float(v) for v in a.levels.split(",") if v.strip()]
    dt = 1.0 / BASE_DAYS

    drawer = store_id(a.priors, SYSTEMATIC_PRIOR_SETS[a.priors], a.lookback)
    files = sorted(glob.glob(os.path.join(PMLE_DIR, drawer, "*.csv")))
    if not files:
        raise SystemExit("no fitted drawer %s" % drawer)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["year"] = pd.to_datetime(df["dtVALUATION_DATE"]).dt.year
    n_dates = df.groupby("year").size()
    par = df.groupby("year")[COLS].mean()

    _LOG.info("=" * 78)
    _LOG.info("EXPECTED SHORTFALL BY SIMULATION :: %s" % drawer)
    _LOG.info("%s paths per year   horizon %d day(s)   seed %d"
              % ("{:,}".format(a.paths), a.horizon, a.seed))
    _LOG.info("=" * 78)

    _LOG.info("\nParameters simulated (mean of that year's valuation dates)")
    _LOG.info("  %-6s %5s %8s %8s %8s %8s %8s %8s"
              % ("year", "n", "alpha", "sigma", "pprob", "lamb", "eta1", "eta2"))
    _LOG.info("  " + "-" * 64)
    for y, r in par.iterrows():
        _LOG.info("  %-6d %5d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
                  % (y, n_dates[y], r["dALPHA"], r["dSIGMA"], r["dPPROB"],
                     r["dLAMB"], r["dETA1"], r["dETA2"]))

    t0 = time.time()
    out = {}
    for y, r in par.iterrows():
        v = simulate(r, dt, a.horizon, a.paths, a.seed)
        out[y] = {lv: (es(v, lv, True), es(v, lv, False)) for lv in levels}
    _LOG.info("\n  simulated %d x %s paths in %.1f s"
              % (len(par), "{:,}".format(a.paths), time.time() - t0))

    for lv in levels:
        _LOG.info("\nLEVEL %.3f" % lv)
        _LOG.info("  %-6s %10s %8s %10s %8s %10s %10s"
                  % ("year", "ES down", "se", "ES up", "se",
                     "q down", "q up"))
        _LOG.info("  " + "-" * 68)
        for y in par.index:
            (qd, ed, sd_), (qu, eu, su) = out[y][lv]
            _LOG.info("  %-6d %9.3f%% %7.3f%% %9.3f%% %7.3f%% %9.3f%% %9.3f%%"
                      % (y, 100 * ed, 100 * sd_, 100 * eu, 100 * su,
                         100 * qd, 100 * qu))
        d = np.array([out[y][lv][0][1] for y in par.index])
        u = np.array([out[y][lv][1][1] for y in par.index])
        _LOG.info("  " + "-" * 68)
        _LOG.info("  %-6s %9.3f%% %8s %9.3f%%" % ("mean", 100 * d.mean(),
                                                  "", 100 * u.mean()))
        _LOG.info("  %-6s %9.3f%% %8s %9.3f%%   (%d / %d)"
                  % ("worst", 100 * d.min(), "", 100 * u.max(),
                     par.index[int(d.argmin())], par.index[int(u.argmax())]))
    _LOG.info("")


if __name__ == "__main__":
    main()
