"""Expected shortfall by simulation, from each year's estimated parameters,
read against what that year actually did.

Paths come from KimYiRiskEngine.random - the repo's own simulator, the same
one the VaR run uses. One simulation per year at that year's MEAN fitted
parameters, then the ES straight off the sample, printed beside the ES of the
realised SPX returns for the same year.

The comparison is the point. A simulated ES the year's own returns do not
support means the prescribed shock is being read against a distribution that
does not believe in it.

ES at level a averages the worst round(a*N) draws. From N simulated paths that
carries sampling error, so se = sd(tail)/sqrt(a*N) is printed; from a year of
~252 realised days the same tail is ~6 observations, so its count is printed
instead and its third decimal means nothing.

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

from Library.DataAccess import PMLE_DIR, get_aligned_price_panel     # noqa: E402
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
    """(quantile, ES, se, n) of the `level` tail of a sample.

    Used for both the simulated paths and the realised returns. On 200,000
    paths the se is the Monte Carlo error; on 252 realised days it is the
    sampling error of six observations and should be read as a warning rather
    than a precision.
    """
    x = np.sort(np.asarray(x, dtype=float))
    m = max(int(round(level * x.size)), 2)
    tail = x[:m] if lower else x[-m:]
    return (float(x[m - 1] if lower else x[-m]), float(tail.mean()),
            float(tail.std(ddof=1) / np.sqrt(m)), m)


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
    ap.add_argument("--beg", default="20070101")
    ap.add_argument("--end", default="20261231")
    a = ap.parse_args()

    levels = [float(v) for v in a.levels.split(",") if v.strip()]
    dt = 1.0 / BASE_DAYS

    # ---- realised ------------------------------------------------------
    panel, _ = get_aligned_price_panel([SYSTEMATIC_ID], reference=SYSTEMATIC_ID)
    r = panel[SYSTEMATIC_ID].pct_change().dropna()
    r = r[(r.index >= pd.to_datetime(a.beg, format="%Y%m%d")) &
          (r.index <= pd.to_datetime(a.end, format="%Y%m%d"))]
    if r.empty:
        raise SystemExit("no returns in %s -> %s" % (a.beg, a.end))
    ry = {y: v.values for y, v in r.groupby(r.index.year)}

    # ---- fitted --------------------------------------------------------
    drawer = store_id(a.priors, SYSTEMATIC_PRIOR_SETS[a.priors], a.lookback)
    files = sorted(glob.glob(os.path.join(PMLE_DIR, drawer, "*.csv")))
    if not files:
        raise SystemExit("no fitted drawer %s" % drawer)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["year"] = pd.to_datetime(df["dtVALUATION_DATE"]).dt.year
    n_dates = df.groupby("year").size()
    par = df.groupby("year")[COLS].mean()

    _LOG.info("=" * 78)
    _LOG.info("EXPECTED SHORTFALL BY SIMULATION, AGAINST THE REALISED TAIL")
    _LOG.info("%s   drawer %s" % (SYSTEMATIC_ID, drawer))
    _LOG.info("realised %s -> %s, %d trading days"
              % (r.index.min().date(), r.index.max().date(), len(r)))
    _LOG.info("%s paths per year   horizon %d day(s)   seed %d"
              % ("{:,}".format(a.paths), a.horizon, a.seed))
    _LOG.info("=" * 78)

    _LOG.info("\nRealised daily returns, pooled")
    _LOG.info("  mean %+.4f%%   sd %.4f%%   annualised sd %.2f%%"
              % (100 * r.mean(), 100 * r.std(), 100 * r.std() * np.sqrt(BASE_DAYS)))
    _LOG.info("  skew %+.3f   excess kurtosis %+.2f" % (r.skew(), r.kurtosis()))
    _LOG.info("  %-8s %12s %12s %7s    %12s %12s %7s"
              % ("level", "q down", "ES down", "n", "q up", "ES up", "n"))
    _LOG.info("  " + "-" * 80)
    for lv in levels:
        qd, ed, _, nd = es(r.values, lv, True)
        qu, eu, _, nu = es(r.values, lv, False)
        _LOG.info("  %-8.3f %11.3f%% %11.3f%% %7d    %11.3f%% %11.3f%% %7d"
                  % (lv, 100 * qd, 100 * ed, nd, 100 * qu, 100 * eu, nu))

    _LOG.info("\nTen worst and ten best days")
    _LOG.info("  %-12s %9s     %-12s %9s" % ("date", "worst", "date", "best"))
    for (dw, vw), (db, vb) in zip(r.nsmallest(10).items(), r.nlargest(10).items()):
        _LOG.info("  %-12s %8.3f%%     %-12s %8.3f%%"
                  % (dw.date(), 100 * vw, db.date(), 100 * vb))

    _LOG.info("\nHow often has a move of at least this size happened?")
    _LOG.info("  %-9s %10s %10s %14s"
              % ("size", "down days", "up days", "1 in N days"))
    for x in (0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20):
        nd, nu = int((r <= -x).sum()), int((r >= x).sum())
        tot = nd + nu
        _LOG.info("  %8.0f%% %10d %10d %14s"
                  % (100 * x, nd, nu, ("%d" % (len(r) / tot)) if tot else "never"))

    _LOG.info("\nParameters simulated (mean of that year's valuation dates)")
    _LOG.info("  %-6s %5s %8s %8s %8s %8s %8s %8s"
              % ("year", "n", "alpha", "sigma", "pprob", "lamb", "eta1", "eta2"))
    _LOG.info("  " + "-" * 64)
    for y, p in par.iterrows():
        _LOG.info("  %-6d %5d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
                  % (y, n_dates[y], p["dALPHA"], p["dSIGMA"], p["dPPROB"],
                     p["dLAMB"], p["dETA1"], p["dETA2"]))

    t0 = time.time()
    sim = {y: simulate(p, dt, a.horizon, a.paths, a.seed)
           for y, p in par.iterrows()}
    _LOG.info("\n  simulated %d x %s paths in %.1f s"
              % (len(par), "{:,}".format(a.paths), time.time() - t0))
    if a.horizon > 1:
        _LOG.info("  horizon > 1: the realised columns are still ONE-DAY and")
        _LOG.info("  are not comparable. Read the simulated side only.")

    # ---- simulated against realised, by year ---------------------------
    for lv in levels:
        _LOG.info("\nLEVEL %.3f   simulated vs realised, by year" % lv)
        _LOG.info("  %-6s %10s %7s %10s %6s %6s   %9s %7s %9s %6s"
                  % ("year", "sim ESdn", "se", "real ESdn", "n", "ratio",
                     "sim ESup", "se", "real ESup", "n"))
        _LOG.info("  " + "-" * 92)
        acc = []
        for y in par.index:
            _, sd_, sdse, _ = es(sim[y], lv, True)
            _, su, suse, _ = es(sim[y], lv, False)
            if y in ry and len(ry[y]) >= 20:
                _, rd, _, rnd = es(ry[y], lv, True)
                _, ru, _, rnu = es(ry[y], lv, False)
                acc.append((sd_, rd, su, ru))
                _LOG.info("  %-6d %9.3f%% %6.3f%% %9.3f%% %6d %6.2f   %8.3f%% "
                          "%6.3f%% %8.3f%% %6d"
                          % (y, 100 * sd_, 100 * sdse, 100 * rd, rnd,
                             sd_ / rd, 100 * su, 100 * suse, 100 * ru, rnu))
            else:
                _LOG.info("  %-6d %9.3f%% %6.3f%% %9s %6s %6s   %8.3f%% "
                          "%6.3f%% %9s %6s"
                          % (y, 100 * sd_, 100 * sdse, "-", "-", "-",
                             100 * su, 100 * suse, "-", "-"))
        if acc:
            A = np.array(acc)
            _LOG.info("  " + "-" * 92)
            _LOG.info("  %-6s %9.3f%% %7s %9.3f%% %6s %6.2f   %8.3f%% %7s "
                      "%8.3f%%"
                      % ("mean", 100 * A[:, 0].mean(), "",
                         100 * A[:, 1].mean(), "",
                         A[:, 0].mean() / A[:, 1].mean(),
                         100 * A[:, 2].mean(), "", 100 * A[:, 3].mean()))
            _LOG.info("  %-6s %9.3f%% %7s %9.3f%%"
                      % ("worst", 100 * A[:, 0].min(), "", 100 * A[:, 1].min()))
            _LOG.info("  ratio = simulated / realised. Above 1 the fit is more")
            _LOG.info("  severe than the year turned out; below 1 it is milder.")
    _LOG.info("")


if __name__ == "__main__":
    main()
