"""One arm, several window lengths, one table. How much does the data say?

The question in its simplest form. Fit the same model on one year, two years
and three years of trailing returns, and for each parameter ask two things:

    where did the posterior end up      the mean at each window
    how much did the data move it       posterior 95% width / prior 95% width

The second number is the one that matters and it reads directly. 1.00 means
the posterior is the prior - the data said nothing. 0.50 means the data halved
the range. Below 0.70 the parameter is identified by that window; at or above
1.00 the likelihood is flat in that direction and the number reported is the
prior wearing a posterior's clothes.

Both widths are 95% equal-tailed, so the ratio carries no assumption about the
shape of either. Dividing a posterior WIDTH by a prior SD instead - which an
earlier version of this diagnostic did - silently assumes a normal prior and
puts the no-information floor anywhere from 0.84 to 1.00 depending on which
prior it is.

    python poc/window_ladder.py --priors pprob-flat
    python poc/window_ladder.py --priors pprob-flat --lookbacks 252,504,756

A longer window buys at most sqrt(n) on a parameter the likelihood already
resolves: three years against one is sqrt(3), so a width ratio of 0.58 between
them is the best a purely statistical gain can look like. Anything near 1.00
means the extra years told the sampler nothing it did not already have.
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

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)

SYS_PARAMS = ["dSIGMA", "dLAMB", "dPPROB", "dETA1", "dETA2", "dALPHA"]
_MAP = {"dSIGMA": "sigma", "dALPHA": "alpha_rv", "dPPROB": "pprob_rv",
        "dLAMB": "lamb", "dETA1": "eta1", "dETA2": "eta2"}


def load_drawer(drawer):
    folder = os.path.join(PMLE_DIR, drawer)
    if not os.path.isdir(folder):
        return None
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dtVALUATION_DATE"] = pd.to_datetime(df["dtVALUATION_DATE"])
    return df.sort_values("dtVALUATION_DATE").set_index("dtVALUATION_DATE")


def width(df, k):
    if k + "_W" in df:
        return df[k + "_W"].astype(float)
    lo, hi = df.get(k + "_CI_LOWER"), df.get(k + "_CI_UPPER")
    if lo is None or hi is None:
        return None
    return hi.astype(float) - lo.astype(float)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priors", default="pprob-flat")
    ap.add_argument("--lookbacks", default="252,504,756",
                    help="trailing days per fit, comma separated")
    ap.add_argument("--common-dates", action="store_true",
                    help="restrict every window to the dates all of them "
                         "share, so the means are over the same calendar")
    a = ap.parse_args()

    from Library.RiskEngineKimYi2025 import (
        SYSTEMATIC_PRIOR_SETS, prior_ci_width, prior_moments)
    from poc.estimate_systematic import store_id

    if a.priors not in SYSTEMATIC_PRIOR_SETS:
        raise SystemExit("unknown arm %r; have %s"
                         % (a.priors, sorted(SYSTEMATIC_PRIOR_SETS)))
    priors = SYSTEMATIC_PRIOR_SETS[a.priors]
    lbs = [int(x) for x in a.lookbacks.split(",") if x.strip()]

    cells, missing = {}, []
    for lb in lbs:
        drawer = store_id(a.priors, priors, lb)
        df = load_drawer(drawer)
        if df is None:
            missing.append((lb, drawer))
        else:
            cells[lb] = df
    if missing:
        _LOG.info("")
        _LOG.info("  NOT YET FITTED - run these first:")
        for lb, drawer in missing:
            _LOG.info("    python poc/estimate_systematic.py --priors %s "
                  "--step 21%s" % (a.priors,
                                   "" if lb == 252 else " --lookback %d" % lb))
            _LOG.info("      (would write to %s)" % drawer)
    if not cells:
        raise SystemExit("no drawers to read")
    lbs = [lb for lb in lbs if lb in cells]

    if a.common_dates and len(cells) > 1:
        common = None
        for df in cells.values():
            common = df.index if common is None else common.intersection(df.index)
        cells = {lb: df.loc[common] for lb, df in cells.items()}

    _LOG.info("")
    _LOG.info("=" * 74)
    _LOG.info("window ladder :: priors %s" % a.priors)
    _LOG.info("=" * 74)
    for lb in lbs:
        df = cells[lb]
        _LOG.info("  %4d days (%.1f yr)  %4d dates  %s -> %s"
              % (lb, lb / 252.0, len(df), df.index.min().date(),
                 df.index.max().date()))

    _LOG.info("")
    _LOG.info("  POSTERIOR MEAN")
    _LOG.info("  %-8s %11s %s" % ("param", "prior mean",
                              " ".join("%11s" % ("%d d" % lb) for lb in lbs)))
    _LOG.info("  " + "-" * (20 + 12 * len(lbs)))
    for k in SYS_PARAMS:
        pm_, _ = prior_moments(priors[_MAP[k]])
        row = " ".join("%11.4f" % cells[lb][k].mean()
                       if k in cells[lb] else "%11s" % "-" for lb in lbs)
        _LOG.info("  %-8s %11.4f %s" % (k, pm_, row))

    _LOG.info("")
    _LOG.info("  HOW MUCH THE DATA MOVED THE PRIOR")
    _LOG.info("  posterior %.0f%% width / prior %.0f%% width."
          % (100 * CI_PROB, 100 * CI_PROB))
    _LOG.info("  1.00 = the data said nothing.  below 0.70 = identified.")
    _LOG.info("  %-8s %10s %s   %s"
          % ("param", "prior W",
             " ".join("%9s" % ("%d d" % lb) for lb in lbs), "reads as"))
    _LOG.info("  " + "-" * (22 + 10 * len(lbs) + 34))
    for k in SYS_PARAMS:
        pw = prior_ci_width(priors[_MAP[k]], CI_PROB)
        if not pw:
            continue
        rs = []
        for lb in lbs:
            w = width(cells[lb], k)
            rs.append(float(w.median()) / pw if w is not None else np.nan)
        first = next((lb for lb, r in zip(lbs, rs) if r < 0.70), None)
        if first is None:
            reads = "never identified - this is the prior"
        elif first == lbs[0]:
            reads = "identified at every window here"
        else:
            reads = "identified from %d days (%.0f yr) on" % (first, first / 252.0)
        _LOG.info("  %-8s %10.4f %s   %s"
              % (k, pw, " ".join("%9.3f" % r for r in rs), reads))

    if len(lbs) > 1:
        _LOG.info("")
        _LOG.info("  WHAT THE EXTRA YEARS BOUGHT")
        _LOG.info("  width at the longer window / width at %d days. %.2f is the"
              % (lbs[0], np.sqrt(lbs[0] / lbs[-1])))
        _LOG.info("  most a purely statistical gain can give at the longest one.")
        _LOG.info("  %-8s %s" % ("param",
                             " ".join("%14s" % ("%d/%d" % (lb, lbs[0]))
                                      for lb in lbs[1:])))
        _LOG.info("  " + "-" * (10 + 15 * (len(lbs) - 1)))
        for k in SYS_PARAMS:
            w0 = width(cells[lbs[0]], k)
            if w0 is None:
                continue
            base = float(w0.median())
            row = []
            for lb in lbs[1:]:
                w = width(cells[lb], k)
                r = float(w.median()) / base if w is not None and base else np.nan
                n_eff = 1.0 / r ** 2 if r and np.isfinite(r) else np.nan
                row.append("%14s" % ("%.2f  (x%.1f)" % (r, n_eff)))
            _LOG.info("  %-8s %s" % (k, " ".join(row)))
        _LOG.info("  x is the effective sample multiple, 1/ratio^2 - how many")
        _LOG.info("  years' worth of information the longer window actually")
        _LOG.info("  delivered, against the %.1f it nominally contains."
              % (lbs[-1] / lbs[0]))
    _LOG.info("")


if __name__ == "__main__":
    main()
