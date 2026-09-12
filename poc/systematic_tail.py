"""What is a defensible boundary for a prescribed one-day systematic shock?

The stress ladder prescribes x and the translation turns it into a name shock,
but nothing so far says which x are worth prescribing. A -40% one-day SPX move
is arithmetic, not a scenario; a -2% one is not a stress test. This puts a
number on both ends from the data the paper already uses.

TWO ANSWERS, AND THEY ARE NOT THE SAME NUMBER.

  EMPIRICAL      the realised distribution of daily SPX returns since 2007.
                 Expected shortfall at 2.5% is the mean of the worst 2.5% of
                 days - a severe day, not an extreme one, and the number a
                 regulator would recognise. The worst single day bounds what
                 has actually happened.

  MODEL-IMPLIED  the same quantiles under the fitted Kou density
                 f(u) = (1-lam dt) phi_s(u)/s + lam dt (I_up + I_dn),
                 which is what the translation is conditioning on. If the
                 model's tail is thinner than the realised one, every
                 prescribed shock is being read against a distribution that
                 does not believe in it, and the posterior split at that shock
                 is correspondingly overconfident.

The gap between the two IS the result. Quoting a shock the model assigns
1e-12 probability to is quoting an extrapolation, however sensible the number
looks in percent.

    python poc/systematic_tail.py
    python poc/systematic_tail.py --beg 20070101 --end 20261231
    python poc/systematic_tail.py --priors alpha-pprob-eta-flat --lookback 504

WHY EXPECTED SHORTFALL AND NOT VaR. ES averages the tail beyond the quantile,
so it is sensitive to how bad the bad days are rather than only to where the
cut falls. For a prescribed shock that is the relevant summary: the question
is "how large is a severe day", not "where does the 2.5% boundary sit".
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import get_aligned_price_panel               # noqa: E402
from Library.Logging import report as _report                        # noqa: E402

_LOG = _report(__name__)

SYSTEMATIC_ID = "^SPX"
BASE_DAYS = 252


# ---------------------------------------------------------------------------
# empirical
# ---------------------------------------------------------------------------
def es(x, level, lower=True):
    """Expected shortfall: the MEAN of the tail beyond the `level` quantile.

    lower=True  -> E[x | x <= q_level]        the loss tail
    lower=False -> E[x | x >= q_(1-level)]    the gain tail

    Returned with the quantile itself and the count, because an ES computed
    from four observations is a different object from one computed from a
    hundred and the reader has to be able to tell.
    """
    x = np.asarray(x, dtype=float)
    if lower:
        q = np.quantile(x, level)
        tail = x[x <= q]
    else:
        q = np.quantile(x, 1.0 - level)
        tail = x[x >= q]
    return float(q), float(tail.mean()), int(tail.size)


# ---------------------------------------------------------------------------
# model-implied
# ---------------------------------------------------------------------------
def kou_density(u, sg, lam, p, e1, e2, dt):
    """The one-day density the translation conditions on. Same expression as
    poc/shock_to_name._jump_posterior_1d builds, evaluated on a grid."""
    s = sg * np.sqrt(dt)
    s2 = s * s
    Iu = p * e1 * np.exp(0.5 * e1**2 * s2 - e1 * u) * norm.cdf((u - e1 * s2) / s)
    Id = (1 - p) * e2 * np.exp(0.5 * e2**2 * s2 + e2 * u) * norm.cdf(-(u + e2 * s2) / s)
    return (1 - lam * dt) * norm.pdf(u / s) / s + lam * dt * (Iu + Id)


def model_tail(sg, lam, p, e1, e2, dt, levels, lo=-0.60, hi=0.60, n=2_000_001):
    """Quantiles and ES of the fitted one-day density, by direct integration.

    A grid rather than a closed form because the ES of a Kou mixture has no
    tidy one, and because a grid makes the normalisation visible: if the mass
    on [lo, hi] is not 1 to several decimals the window is too narrow and
    every number below is wrong.
    """
    u = np.linspace(lo, hi, n)
    f = kou_density(u, sg, lam, p, e1, e2, dt)
    du = u[1] - u[0]
    mass = float(np.sum(f) * du)
    cdf = np.cumsum(f) * du / mass
    out = {}
    for a in levels:
        ql = float(np.interp(a, cdf, u))
        m = u <= ql
        out[("lower", a)] = (ql, float(np.sum(u[m] * f[m]) * du /
                                       max(np.sum(f[m]) * du, 1e-300)))
        qh = float(np.interp(1.0 - a, cdf, u))
        m = u >= qh
        out[("upper", a)] = (qh, float(np.sum(u[m] * f[m]) * du /
                                       max(np.sum(f[m]) * du, 1e-300)))
    return out, mass


def model_prob(x, sg, lam, p, e1, e2, dt, lo=-0.999, hi=1.0, n=2_000_001):
    """P(one-day move <= x) under the fitted density. The number that says
    whether a prescribed shock is a scenario or an extrapolation."""
    u = np.linspace(lo, hi, n)
    f = kou_density(u, sg, lam, p, e1, e2, dt)
    du = u[1] - u[0]
    tot = float(np.sum(f) * du)
    return float(np.sum(f[u <= x]) * du / tot)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beg", default="20070101")
    ap.add_argument("--end", default="20261231")
    ap.add_argument("--levels", default="0.025,0.01,0.005",
                    help="tail probabilities to report, both directions")
    ap.add_argument("--priors", default="alpha-pprob-eta-flat",
                    help="systematic arm whose fit to compare against")
    ap.add_argument("--lookback", type=int, default=504)
    ap.add_argument("--no-model", action="store_true",
                    help="empirical only; skips loading the fitted drawer")
    a = ap.parse_args()

    levels = [float(v) for v in a.levels.split(",") if v.strip()]
    beg = pd.to_datetime(a.beg, format="%Y%m%d")
    end = pd.to_datetime(a.end, format="%Y%m%d")

    panel, _ = get_aligned_price_panel([SYSTEMATIC_ID], reference=SYSTEMATIC_ID)
    r = panel[SYSTEMATIC_ID].pct_change().dropna()
    r = r[(r.index >= beg) & (r.index <= end)]
    if r.empty:
        raise SystemExit("no returns in %s -> %s" % (a.beg, a.end))

    _LOG.info("=" * 74)
    _LOG.info("One-day systematic shock :: what the tail supports")
    _LOG.info("%s   %s -> %s   %d trading days"
              % (SYSTEMATIC_ID, r.index.min().date(), r.index.max().date(), len(r)))
    _LOG.info("=" * 74)

    _LOG.info("\nRealised daily returns")
    _LOG.info("  mean %+.4f%%   sd %.4f%%   annualised sd %.2f%%"
              % (100 * r.mean(), 100 * r.std(), 100 * r.std() * np.sqrt(BASE_DAYS)))
    _LOG.info("  skew %+.3f   excess kurtosis %+.2f"
              % (r.skew(), r.kurtosis()))

    _LOG.info("\nEXPECTED SHORTFALL, empirical")
    _LOG.info("  %-8s %12s %12s %7s    %12s %12s %7s"
              % ("level", "q lower", "ES lower", "n", "q upper", "ES upper", "n"))
    _LOG.info("  " + "-" * 80)
    for lv in levels:
        ql, el, nl = es(r.values, lv, lower=True)
        qh, eh, nh = es(r.values, lv, lower=False)
        _LOG.info("  %-8.3f %11.3f%% %11.3f%% %7d    %11.3f%% %11.3f%% %7d"
                  % (lv, 100 * ql, 100 * el, nl, 100 * qh, 100 * eh, nh))

    _LOG.info("\nTen worst and ten best days")
    w = r.nsmallest(10)
    b = r.nlargest(10)
    _LOG.info("  %-12s %9s     %-12s %9s" % ("date", "worst", "date", "best"))
    for (dw, vw), (db, vb) in zip(w.items(), b.items()):
        _LOG.info("  %-12s %8.3f%%     %-12s %8.3f%%"
                  % (dw.date(), 100 * vw, db.date(), 100 * vb))

    _LOG.info("\nHow often has a move of at least this size happened?")
    _LOG.info("  %-9s %10s %10s %14s" % ("size", "down days", "up days", "1 in N days"))
    for x in (0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20):
        nd = int((r <= -x).sum())
        nu = int((r >= x).sum())
        tot = nd + nu
        _LOG.info("  %8.0f%% %10d %10d %14s"
                  % (100 * x, nd, nu, ("%d" % (len(r) / tot)) if tot else "never"))

    if a.no_model:
        return

    # ---- model-implied ------------------------------------------------
    from Library.RiskEngineKimYi2025 import SYSTEMATIC_PRIOR_SETS
    from poc.estimate_systematic import store_id
    from Library.DataAccess import PMLE_DIR
    import glob

    priors = SYSTEMATIC_PRIOR_SETS[a.priors]
    drawer = store_id(a.priors, priors, a.lookback)
    files = sorted(glob.glob(os.path.join(PMLE_DIR, drawer, "*.csv")))
    if not files:
        _LOG.info("\nNo fitted drawer %s - skipping the model comparison." % drawer)
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sg, lam = float(df["dSIGMA"].median()), float(df["dLAMB"].median())
    p = float(df["dPPROB"].median())
    e1, e2 = float(df["dETA1"].median()), float(df["dETA2"].median())
    dt = 1.0 / BASE_DAYS

    _LOG.info("\n" + "=" * 74)
    _LOG.info("MODEL-IMPLIED, cross-date medians of %s" % drawer)
    _LOG.info("  sigma %.4f  lambda %.4f  p %.4f  eta1 %.4f  eta2 %.4f"
              % (sg, lam, p, e1, e2))
    _LOG.info("  daily diffusion sd %.4f%%   P(jump on a day) %.4f"
              % (100 * sg * np.sqrt(dt), lam * dt))

    tail, mass = model_tail(sg, lam, p, e1, e2, dt, levels)
    _LOG.info("  grid mass %.9f  (must be 1.000000000)" % mass)
    _LOG.info("\n  %-8s %12s %12s    %12s %12s"
              % ("level", "q lower", "ES lower", "q upper", "ES upper"))
    _LOG.info("  " + "-" * 64)
    for lv in levels:
        ql, el = tail[("lower", lv)]
        qh, eh = tail[("upper", lv)]
        _LOG.info("  %-8.3f %11.3f%% %11.3f%%    %11.3f%% %11.3f%%"
                  % (lv, 100 * ql, 100 * el, 100 * qh, 100 * eh))

    _LOG.info("\nWhat the model thinks of a prescribed shock")
    _LOG.info("  %-9s %16s %18s" % ("x", "P(move <= x)", "1 in N days"))
    for x in (-0.02, -0.03, -0.05, -0.07, -0.10, -0.15, -0.20, -0.40):
        pr = model_prob(x, sg, lam, p, e1, e2, dt)
        _LOG.info("  %8.0f%% %16.3e %18s"
                  % (100 * x, pr, ("%.3g" % (1.0 / pr)) if pr > 0 else "never"))
    _LOG.info("")
    _LOG.info("  Read the last column against the empirical one above. Where the")
    _LOG.info("  model says 1 in 10^6 days and the sample holds three of them,")
    _LOG.info("  the prescribed shock is outside what the fit can price and the")
    _LOG.info("  posterior split at that shock is not to be trusted either.")
    _LOG.info("")


if __name__ == "__main__":
    main()
