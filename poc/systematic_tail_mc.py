"""Expected shortfall by Monte Carlo, from the fitted parameters, year by year.

poc/systematic_tail.py already reports the one-day tail by QUADRATURE - it
integrates the Kou transition density on a grid. That is exact to nine decimals
and has no sampling error, so the obvious question is why simulate at all.

THREE REASONS, AND ONLY THE FIRST IS ABOUT CHECKING ARITHMETIC.

  1. INDEPENDENT CONFIRMATION. The quadrature evaluates a closed form I wrote
     by hand from the paper's page 42. Paths built by the repo's OWN engine -
     KimYiRiskEngine.random, the same code the VaR run uses - reproduce the
     same numbers through a completely different route. A referee who does not
     trust the closed form can read the simulation instead.

  2. THE LIKELIHOOD AND THE ENGINE ARE NOT THE SAME MODEL. The likelihood
     (KimYiLogLike.logp) is a TWO-TERM mixture,

         (1 - lamb dt) * Gaussian  +  lamb dt * (one-jump convolution)

     so it cannot produce a day with two jumps. The engine draws a full
     Poisson count and sums that many jumps. At the fitted lamb = 9.11 and
     dt = 1/252, P(N >= 2) = 6.4e-4 - negligible as a fraction of ALL days,
     but the 0.5% tail is only 0.5% of days, so two-jump days can be a real
     share of the tail that produces the ES. The gap is measured here rather
     than assumed small: --jumps bernoulli reruns with the count clipped at 1
     and the difference against --jumps poisson is the answer.

  3. HORIZON. The closed form is one-day only. A two-day return is a
     convolution with no tidy expression, and over h days the mean reversion
     -alpha Psi dt stops being negligible: at alpha = 0.75/yr a single day's
     pull is 0.30% of the level, but twenty-one days compound it. --horizon h
     compounds the path properly, which the quadrature cannot do at all.

WHAT IS SIMULATED. Exactly the paper's discretisation, via the repo engine:

    Psi_0     = 0                           (start at the long-run mean)
    Psi_{k+1} = (1 - alpha dt) Psi_k + sqrt(V dt) Z_k + gamma Y_k
    R_k       = Psi_{k+1} - Psi_k + drift dt
    h-day     = prod_k (1 + R_k) - 1

with the systematic constants beta = gamma = 1, kappa = rho = mu = 0, so
V = sigma^2 and drift = sigma^2/2. Y is ADED on the LEVEL - additive, not
exp(Y) - 1. That is what the code estimates and what the engine simulates; see
the note on the price floor below.

TWO THINGS TO READ THE OUTPUT WITH.

  THE DRIFT OFFSET. The quadrature integrates the density of dPsi. The engine
  returns R = dPsi + sigma^2/2 dt. At sigma = 0.135 that is +0.0036% a day, so
  MC ES should sit 0.0036% ABOVE the quadrature at horizon 1. A difference of
  that size is agreement, not disagreement.

  THE PRICE FLOOR. The jump is additive, so nothing stops 1 + R from going
  negative - a price below zero. It takes an additive jump under -1, which at
  eta2 = 37.8 has probability e^-37.8, i.e. never at the fitted parameters.
  The count is printed anyway, because it stops being zero at long horizons
  and it is the same page-41-versus-page-42 question that the idiosyncratic
  translation ran into: the SDE's jump is exp(Y) - 1 and cannot breach the
  floor; the density's jump is additive and can.

MONTE CARLO ERROR IS REPORTED, because it is the one thing quadrature does not
have and the reason not to replace the closed form with this. ES at level a
from N paths averages the worst round(aN) draws, so its standard error is
sd(tail)/sqrt(aN): at N = 200,000 the 2.5% ES rests on 5,000 draws and the
0.5% ES on 1,000. Read every MC number against its own se before comparing it
with anything.

    python poc/systematic_tail_mc.py
    python poc/systematic_tail_mc.py --paths 500000 --compare-quad
    python poc/systematic_tail_mc.py --jumps bernoulli
    python poc/systematic_tail_mc.py --horizon 10 --agg year
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
from poc.systematic_tail import model_tail                           # noqa: E402

SYSTEMATIC_ID = "^SPX"
BASE_DAYS = 252
SEED = 20240114


class BernoulliJumpRNG(RandomMT19937):
    """The engine with the LIKELIHOOD's jump law instead of its own.

    KimYiLogLike is a two-term mixture, so it prices at most one jump per step.
    Overriding the count draw - and nothing else - makes the simulation match
    the estimator exactly, which is the only clean way to ask what the second
    jump is worth. Everything downstream (the ADED draw, the recursion, the
    drift) is the engine's, untouched.
    """

    def get_poisson(self, variates, lamb=None):
        variates[:] = (self.generator.random(size=variates.shape)
                       < float(lamb)).astype(np.int64)


def _engine(row, dt):
    """The systematic block: beta = gamma = 1, kappa = rho = mu = 0.

    Copied field for field from simulate_shock_returns_systematic_helper in
    Scripts/run_var_kimyi2025.py, so this is the same object the VaR run
    builds, not a second opinion about what the systematic case means.
    """
    k = lambda v: ParametersConstant(np.array(float(v)))             # noqa: E731
    return KimYiRiskEngine(
        mui=[k(0.)], kappai=[k(0.)], gammai=[k(1.)],
        betai=[k(1.)], rhoix=[k(0.)],
        alpha=k(row["dALPHA"]), sigma=k(row["dSIGMA"]),
        pprob=k(row["dPPROB"]), lamb=k(row["dLAMB"]),
        eta1=k(row["dETA1"]), eta2=k(row["dETA2"]),
        end_dt=np.array(dt))


def simulate(row, dt, horizon, paths, seed, jumps="poisson"):
    """One valuation date's parameters -> `paths` h-day returns.

    Returns (total_return, n_breach) where a breach is a path on which some
    step drove 1 + R to zero or below. A breached path is reported as -100%
    rather than dropped: the model produced it, and dropping it would quietly
    improve the very tail the ES is measuring.
    """
    rng = (BernoulliJumpRNG(np.int64(seed)) if jumps == "bernoulli"
           else RandomMT19937(np.int64(seed)))
    ret, _, _ = _engine(row, dt).random(rng=rng, size=(1, paths, horizon))
    r = ret[0][:, 1:]                                # drop the identity column
    if horizon == 1:
        v = r[:, 0]
        return v, int((v <= -1.0).sum())
    g = 1.0 + r
    bust = (g <= 0.0).any(axis=1)
    tot = np.expm1(np.log(np.clip(g, 1e-300, None)).sum(axis=1))
    tot[bust] = -1.0
    return tot, int(bust.sum())


def es_mc(x, level, lower=True):
    """ES of a SIMULATED sample, with the standard error that goes with it.

    The tail is the worst m = round(level * N) draws, so the ES is a mean of m
    numbers and se = sd(tail)/sqrt(m). That se is conditional on where the cut
    fell; it understates slightly, and it is still the number that says whether
    two ES figures differ.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    m = max(int(round(level * n)), 2)
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
    ap.add_argument("--paths", type=int, default=200_000,
                    help="paths per valuation date (default 200,000)")
    ap.add_argument("--horizon", type=int, default=1,
                    help="days per path; >1 compounds and turns on the mean "
                         "reversion the one-day closed form cannot see")
    ap.add_argument("--jumps", choices=("poisson", "bernoulli"),
                    default="poisson",
                    help="poisson = the engine's law; bernoulli = the "
                         "LIKELIHOOD's one-jump-at-most approximation")
    ap.add_argument("--agg", choices=("date", "year"), default="date",
                    help="date: simulate every valuation date and average by "
                         "year (comparable with systematic_tail --by-year). "
                         "year: one simulation per year at that year's median "
                         "parameters - much faster, use it for long horizons")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-crn", action="store_true",
                    help="advance the seed per date instead of reusing it. "
                         "Common random numbers are the default so that "
                         "differences ACROSS years are parameter effects and "
                         "not sampling noise")
    ap.add_argument("--compare-quad", action="store_true",
                    help="also integrate the closed form at horizon 1")
    a = ap.parse_args()

    levels = [float(v) for v in a.levels.split(",") if v.strip()]
    dt = 1.0 / BASE_DAYS

    priors = SYSTEMATIC_PRIOR_SETS[a.priors]
    drawer = store_id(a.priors, priors, a.lookback)
    files = sorted(glob.glob(os.path.join(PMLE_DIR, drawer, "*.csv")))
    if not files:
        raise SystemExit("no fitted drawer %s" % drawer)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dtVALUATION_DATE"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["year"] = df["dt"].dt.year

    cols = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]
    if a.agg == "year":
        df = (df.groupby("year")[cols].median().reset_index())
        df["dt"] = pd.to_datetime(df["year"].astype(str) + "-07-01")

    _LOG = _report(__name__)
    _LOG.info("=" * 74)
    _LOG.info("MONTE CARLO EXPECTED SHORTFALL :: paths from the fitted parameters")
    _LOG.info("%s   drawer %s" % (SYSTEMATIC_ID, drawer))
    _LOG.info("%d parameter set(s), aggregated by %s" % (len(df), a.agg))
    _LOG.info("%s paths each   horizon %d day(s)   jumps %s   seed %d%s"
              % ("{:,}".format(a.paths), a.horizon, a.jumps, a.seed,
                 "" if a.no_crn else "  (common random numbers)"))
    _LOG.info("=" * 74)
    _LOG.info("")
    _LOG.info("  ES at level a averages the worst round(a*N) of N paths, so it")
    _LOG.info("  carries sampling error. se = sd(tail)/sqrt(a*N) is printed")
    _LOG.info("  beside every figure - at N = %s that is %d draws at a = 0.025"
              % ("{:,}".format(a.paths), round(0.025 * a.paths)))
    _LOG.info("  and %d at a = 0.005. Compare two ES only against their se."
              % round(0.005 * a.paths))
    _LOG.info("")
    if a.horizon == 1:
        _LOG.info("  The closed form integrates the density of dPsi; the engine")
        _LOG.info("  returns R = dPsi + sigma^2/2 dt. MC should therefore sit")
        _LOG.info("  ABOUT +0.004%% above the quadrature. That is agreement.")
        _LOG.info("")

    t0 = time.time()
    rows, breaches = [], 0
    for i, row in df.iterrows():
        seed = a.seed + (i if a.no_crn else 0)
        v, nb = simulate(row, dt, a.horizon, a.paths, seed, a.jumps)
        breaches += nb
        rec = {"dt": row["dt"], "year": int(row["dt"].year)}
        for lv in levels:
            _, ed, sd_, _ = es_mc(v, lv, lower=True)
            _, eu, su, _ = es_mc(v, lv, lower=False)
            rec["dn%g" % lv], rec["dnse%g" % lv] = ed, sd_
            rec["up%g" % lv], rec["upse%g" % lv] = eu, su
        rows.append(rec)
    out = pd.DataFrame(rows)
    _LOG.info("  simulated %d x %s paths in %.1f s"
              % (len(df), "{:,}".format(a.paths), time.time() - t0))
    _LOG.info("  paths breaching the zero-price floor: %d of %d  (%.2e)"
              % (breaches, len(df) * a.paths,
                 breaches / float(len(df) * a.paths)))
    _LOG.info("")

    # ---- cross-date distribution, and the closed form beside it -------
    _LOG.info("ACROSS ALL PARAMETER SETS")
    _LOG.info("  %-7s %10s %9s %10s %10s %10s %10s"
              % ("level", "", "min", "25%", "median", "75%", "max"))
    _LOG.info("  " + "-" * 72)
    for lv in levels:
        for side, tag in (("dn", "ES down"), ("up", "ES up")):
            s = out["%s%g" % (side, lv)]
            _LOG.info("  %-7.3f %10s %9.3f%% %9.3f%% %9.3f%% %9.3f%% %9.3f%%"
                      % (lv, tag, 100 * s.min(), 100 * s.quantile(.25),
                         100 * s.median(), 100 * s.quantile(.75),
                         100 * s.max()))
        i = int(out["dn%g" % lv].idxmin())
        _LOG.info("  %-7s %10s %s   ES %.3f%%  se %.3f%%"
                  % ("", "severest:", out.loc[i, "dt"].date(),
                     100 * out.loc[i, "dn%g" % lv],
                     100 * out.loc[i, "dnse%g" % lv]))
    _LOG.info("")

    if a.compare_quad and a.horizon == 1:
        med = {c: float(df[c].median()) for c in cols}
        tail, mass = model_tail(med["dSIGMA"], med["dLAMB"], med["dPPROB"],
                                med["dETA1"], med["dETA2"], dt, levels)
        _LOG.info("MC vs THE CLOSED FORM, at the cross-date median parameters")
        _LOG.info("  grid mass %.9f" % mass)
        mrow = df[cols].median()
        v, _ = simulate(mrow, dt, 1, a.paths, a.seed, a.jumps)
        _LOG.info("  %-7s %11s %8s %11s %9s %8s"
                  % ("level", "MC", "se", "quadrature", "diff", "in se"))
        _LOG.info("  " + "-" * 62)
        for lv in levels:
            for side, lo in (("down", True), ("up", False)):
                _, e, se, _ = es_mc(v, lv, lower=lo)
                q = tail[("lower" if lo else "upper", lv)][1]
                d = e - q
                _LOG.info("  %-7s %10.3f%% %7.3f%% %10.3f%% %8.3f%% %7.1f  %s"
                          % ("%.3f %s" % (lv, side), 100 * e, 100 * se,
                             100 * q, 100 * d, d / se if se else float("nan"),
                             side))
        _LOG.info("")
        _LOG.info("  'in se' is the difference in standard errors. Under 2 is")
        _LOG.info("  agreement. A LARGE and consistently NEGATIVE down-side")
        _LOG.info("  difference is the two-jump mass the closed form omits,")
        _LOG.info("  not an error - rerun with --jumps bernoulli to confirm.")
        _LOG.info("")

    # ---- by year -------------------------------------------------------
    _LOG.info("BY YEAR (mean over that year's parameter sets), horizon %d"
              % a.horizon)
    hdr = "  %-6s %4s" % ("year", "n")
    for lv in levels:
        hdr += " %11s %8s" % ("ESdn %.3f" % lv, "se")
    _LOG.info(hdr)
    _LOG.info("  " + "-" * (len(hdr) - 2))
    g = out.groupby("year")
    for y, blk in g:
        line = "  %-6d %4d" % (y, len(blk))
        for lv in levels:
            line += " %10.3f%% %7.3f%%" % (100 * blk["dn%g" % lv].mean(),
                                           100 * blk["dnse%g" % lv].mean())
        _LOG.info(line)
    _LOG.info("")
    hdr = "  %-6s %4s" % ("year", "n")
    for lv in levels:
        hdr += " %11s %8s" % ("ESup %.3f" % lv, "se")
    _LOG.info(hdr)
    _LOG.info("  " + "-" * (len(hdr) - 2))
    for y, blk in g:
        line = "  %-6d %4d" % (y, len(blk))
        for lv in levels:
            line += " %10.3f%% %7.3f%%" % (100 * blk["up%g" % lv].mean(),
                                           100 * blk["upse%g" % lv].mean())
        _LOG.info(line)
    _LOG.info("")
    _LOG.info("  Down/up ES ratio by year, level %.3f:" % levels[0])
    lv = levels[0]
    for y, blk in g:
        r = abs(blk["dn%g" % lv].mean()) / blk["up%g" % lv].mean()
        _LOG.info("    %d  %.3f" % (y, r))
    _LOG.info("")


if __name__ == "__main__":
    main()
