"""Two window lengths, same priors, same dates: what does the extra data buy?

The question this answers is not "which estimate is right" - both are, for the
window each one uses. It is which PARAMETERS a longer window actually resolves.
alpha is the reason it exists: its posterior mean sits at 0.036 on the full
sample, a half-life of 19.3 years, against a 252-day window that cannot see
past its own end. If three years is enough to start pinning it, the
identification ratio falls below 1.00 here and nowhere else has to change. If
it does not, alpha is not a window-length problem.

Ratios are POSTERIOR 95% width over PRIOR 95% width, both equal-tailed. The
earlier form divided the posterior width by the prior SD, which assumes a
normal prior; under Uniform(0, L) that put the no-information floor at 0.84,
and an alpha posterior sitting exactly on its prior read as a 16% narrowing.
Width against width, an unmoved posterior reads 1.00 for any prior shape.

    python poc/window_compare.py --priors skew-tight --lb-b 756
    python poc/window_compare.py --priors skew-tight --lb-b 756 --beg 20080101 --end 20091231

WHAT A LONGER WINDOW COSTS. A 252-day window drops the GFC in late 2009; a
756-day window carries it to late 2011. The cross-sectional result - banks
loading on the 2008 gap and not the 2020 one, cruise reversing - is a
statement about what is INSIDE the window at a date, so it is not preserved
across window lengths and the two series must not be pooled to produce it.
Use the long window to ask what is identified, the one-year window for the
vintage contrast.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import PMLE_DIR                          # noqa: E402
from Library.PosteriorSummary import CI_PROB                     # noqa: E402

SYS_PARAMS = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]
_MAP = {"dSIGMA": "sigma", "dALPHA": "alpha_rv", "dPPROB": "pprob_rv",
        "dLAMB": "lamb", "dETA1": "eta1", "dETA2": "eta2"}


def load_drawer(drawer, beg=None, end=None):
    folder = os.path.join(PMLE_DIR, drawer)
    if not os.path.isdir(folder):
        raise SystemExit("no such drawer: %s\n  (looked in %s)"
                         % (drawer, PMLE_DIR))
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit("drawer %s holds no CSV files" % drawer)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dtVALUATION_DATE"] = pd.to_datetime(df["dtVALUATION_DATE"])
    df = df.sort_values("dtVALUATION_DATE").set_index("dtVALUATION_DATE")
    if beg:
        df = df[df.index >= pd.to_datetime(beg, format="%Y%m%d")]
    if end:
        df = df[df.index <= pd.to_datetime(end, format="%Y%m%d")]
    return df


def width(df, k):
    """95% interval width, from the stored endpoints or the _W column."""
    if k + "_W" in df:
        return df[k + "_W"].astype(float)
    lo, hi = df.get(k + "_CI_LOWER"), df.get(k + "_CI_UPPER")
    if lo is None or hi is None:
        return None
    return (hi.astype(float) - lo.astype(float))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priors", default="skew-tight")
    ap.add_argument("--lb-a", type=int, default=252,
                    help="the reference window (default 252, one year)")
    ap.add_argument("--lb-b", type=int, default=756,
                    help="the window under test (default 756, three years)")
    ap.add_argument("--beg", default=None, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    a = ap.parse_args()

    from Library.RiskEngineKimYi2025 import (
        SYSTEMATIC_PRIOR_SETS, prior_ci_width)
    from poc.estimate_systematic import store_id

    priors = SYSTEMATIC_PRIOR_SETS[a.priors]
    da = load_drawer(store_id(a.priors, priors, a.lb_a), a.beg, a.end)
    db = load_drawer(store_id(a.priors, priors, a.lb_b), a.beg, a.end)
    common = da.index.intersection(db.index)
    if not len(common):
        raise SystemExit(
            "the two drawers share no valuation date.\n"
            "  %d-day: %d dates %s -> %s\n  %d-day: %d dates %s -> %s\n"
            "  Run the same --beg/--end/--step under both windows first."
            % (a.lb_a, len(da), da.index.min().date(), da.index.max().date(),
               a.lb_b, len(db), db.index.min().date(), db.index.max().date()))
    da, db = da.loc[common], db.loc[common]

    print()
    print("=" * 78)
    print("window comparison :: %d vs %d days, priors %s"
          % (a.lb_a, a.lb_b, a.priors))
    print("=" * 78)
    print("  %d shared valuation date(s)  %s -> %s"
          % (len(common), common.min().date(), common.max().date()))
    print("  ratio = posterior %.0f%% width / prior %.0f%% width, both "
          "equal-tailed;" % (100 * CI_PROB, 100 * CI_PROB))
    print("  1.00 means the posterior is the prior. Below 0.70 is identified.")
    print()
    print("  %-8s %9s %9s %8s %9s %9s %8s   %s"
          % ("param", "mean A", "mean B", "d mean", "ratio A", "ratio B",
             "width B/A", "verdict"))
    print("  " + "-" * 88)

    for k in SYS_PARAMS:
        if k not in da or k not in db:
            continue
        pw = prior_ci_width(priors[_MAP[k]], CI_PROB)
        wa, wb = width(da, k), width(db, k)
        if wa is None or wb is None or not pw:
            continue
        ra, rb = float(wa.median()) / pw, float(wb.median()) / pw
        ma, mb = float(da[k].mean()), float(db[k].mean())
        shrink = float(wb.median()) / float(wa.median())

        if rb < 0.70 <= ra:
            verdict = "IDENTIFIED by the longer window"
        elif rb < 0.70:
            verdict = "identified under both"
        elif rb < 0.95 * ra:
            verdict = "narrower, still not identified"
        elif rb > 0.99:
            verdict = "flat - no information at either length"
        else:
            verdict = "no material gain"
        print("  %-8s %9.4f %9.4f %8.4f %9.3f %9.3f %8.3f   %s"
              % (k, ma, mb, mb - ma, ra, rb, shrink, verdict))

    print()
    print("  A longer window buys sqrt(3) = 1.73 on a parameter the likelihood")
    print("  already resolves, so width B/A near 0.58 is the most a purely")
    print("  statistical gain can look like. Near 1.00 means the extra two")
    print("  years told the sampler nothing it did not already have.")
    print()

    print("  Series stability (sd of the posterior mean across the shared dates):")
    print("  %-8s %11s %11s %8s" % ("param", "sd A", "sd B", "B/A"))
    print("  " + "-" * 42)
    for k in SYS_PARAMS:
        if k not in da or k not in db:
            continue
        sa, sb = float(da[k].std(ddof=1)), float(db[k].std(ddof=1))
        print("  %-8s %11.5f %11.5f %8.3f"
              % (k, sa, sb, sb / sa if sa else np.nan))
    print()
    print("  A 3-year window overlaps 755 of its 756 observations day to day, so")
    print("  the series is smoother by construction. That is not a better")
    print("  estimate of anything - it is the same crisis held in view three")
    print("  times as long, which is exactly what the vintage contrast needs")
    print("  NOT to happen. Read this column as a cost, not a gain.")
    print()


if __name__ == "__main__":
    main()
