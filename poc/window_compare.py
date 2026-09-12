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

--priors-b compares two ARMS instead of two windows. Leave --lb-b at --lb-a
and name a second arm, and the same table reads as what a prior was doing to
a posterior it disagreed with:

    python poc/window_compare.py --priors skew-tight \
        --priors-b skewtight-lamflat --lb-a 252 --lb-b 252
    python poc/window_compare.py --priors skew-tight \
        --priors-b skewtight-lamflat --lb-a 756 --lb-b 756

Absolute interval widths are printed beside the ratios, because two arms have
two denominators and the ratio columns are then not comparable to each other.

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

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)

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
    ap.add_argument("--priors", default="skew-tight",
                    help="the reference arm")
    ap.add_argument("--priors-b", default=None,
                    help="the arm under test. Defaults to --priors, which is "
                         "the window comparison. Naming a DIFFERENT arm and "
                         "leaving --lb-b at --lb-a compares two priors at one "
                         "window instead - the way to see what a prior was "
                         "doing to a posterior it disagreed with.")
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

    tag_a, tag_b = a.priors, a.priors_b or a.priors
    pri_a = SYSTEMATIC_PRIOR_SETS[tag_a]
    pri_b = SYSTEMATIC_PRIOR_SETS[tag_b]
    if (tag_a, a.lb_a) == (tag_b, a.lb_b):
        raise SystemExit("A and B are the same run (%s, %d days). Change "
                         "--priors-b or --lb-b." % (tag_a, a.lb_a))
    da = load_drawer(store_id(tag_a, pri_a, a.lb_a), a.beg, a.end)
    db = load_drawer(store_id(tag_b, pri_b, a.lb_b), a.beg, a.end)
    common = da.index.intersection(db.index)
    if not len(common):
        raise SystemExit(
            "the two drawers share no valuation date.\n"
            "  %d-day: %d dates %s -> %s\n  %d-day: %d dates %s -> %s\n"
            "  Run the same --beg/--end/--step under both windows first."
            % (a.lb_a, len(da), da.index.min().date(), da.index.max().date(),
               a.lb_b, len(db), db.index.min().date(), db.index.max().date()))
    da, db = da.loc[common], db.loc[common]

    _LOG.info("")
    _LOG.info("=" * 78)
    _LOG.info("comparison :: A = %s @ %d days   B = %s @ %d days"
          % (tag_a, a.lb_a, tag_b, a.lb_b))
    _LOG.info("=" * 78)
    _LOG.info("  %d shared valuation date(s)  %s -> %s"
          % (len(common), common.min().date(), common.max().date()))
    _LOG.info("  ratio = posterior %.0f%% width / prior %.0f%% width, both "
          "equal-tailed;" % (100 * CI_PROB, 100 * CI_PROB))
    _LOG.info("  1.00 means the posterior is the prior. Below 0.70 is identified.")
    _LOG.info("")
    _LOG.info("  %-8s %9s %9s %9s %9s %8s %8s %8s   %s"
          % ("param", "mean A", "mean B", "width A", "width B", "ratio A",
             "ratio B", "W B/A", "verdict"))
    _LOG.info("  " + "-" * 100)

    for k in SYS_PARAMS:
        if k not in da or k not in db:
            continue
        # Each side against its OWN prior. Two arms have two denominators, and
        # dividing both posteriors by one of them is how a lambda width got
        # reported here against a Gamma(10, 0.5) that no run has used since
        # the paper arm.
        pwa = prior_ci_width(pri_a[_MAP[k]], CI_PROB) if pri_a else None
        pwb = prior_ci_width(pri_b[_MAP[k]], CI_PROB) if pri_b else None
        wa, wb = width(da, k), width(db, k)
        if wa is None or wb is None or not pwa or not pwb:
            continue
        ra, rb = float(wa.median()) / pwa, float(wb.median()) / pwb
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
        _LOG.info("  %-8s %9.4f %9.4f %9.4f %9.4f %8.3f %8.3f %8.3f   %s"
              % (k, ma, mb, float(wa.median()), float(wb.median()),
                 ra, rb, shrink, verdict))

    if tag_a != tag_b:
        _LOG.info("")
        _LOG.info("  A and B are DIFFERENT ARMS, so ratio A and ratio B are each")
        _LOG.info("  against their own prior and are not comparable to each other.")
        _LOG.info("  The comparable columns are width A against width B, which is")
        _LOG.info("  what the prior was doing to the posterior, and W B/A.")
    _LOG.info("")
    _LOG.info("  A longer window buys sqrt(3) = 1.73 on a parameter the likelihood")
    _LOG.info("  already resolves, so width B/A near 0.58 is the most a purely")
    _LOG.info("  statistical gain can look like. Near 1.00 means the extra two")
    _LOG.info("  years told the sampler nothing it did not already have.")
    _LOG.info("")

    _LOG.info("  Series stability (sd of the posterior mean across the shared dates):")
    _LOG.info("  %-8s %11s %11s %8s" % ("param", "sd A", "sd B", "B/A"))
    _LOG.info("  " + "-" * 42)
    for k in SYS_PARAMS:
        if k not in da or k not in db:
            continue
        sa, sb = float(da[k].std(ddof=1)), float(db[k].std(ddof=1))
        _LOG.info("  %-8s %11.5f %11.5f %8.3f"
              % (k, sa, sb, sb / sa if sa else np.nan))
    _LOG.info("")
    _LOG.info("  Confounding - correlation of each parameter's series with dSIGMA")
    _LOG.info("  and dLAMB, at the date level, inside each drawer:")
    _LOG.info("  %-8s %9s %9s %9s %9s" % ("param", "r sigma A", "r sigma B",
                                      "r lamb A", "r lamb B"))
    _LOG.info("  " + "-" * 50)
    for k in SYS_PARAMS:
        if k in ("dSIGMA", "dLAMB") or k not in da or k not in db:
            continue
        r = []
        for d, ref in ((da, "dSIGMA"), (db, "dSIGMA"), (da, "dLAMB"),
                       (db, "dLAMB")):
            r.append(float(np.corrcoef(d[k].astype(float),
                                       d[ref].astype(float))[0, 1])
                     if ref in d and d[k].std() > 0 and d[ref].std() > 0
                     else float("nan"))
        _LOG.info("  %-8s %+9.3f %+9.3f %+9.3f %+9.3f" % (k, *r))
    _LOG.info("")
    _LOG.info("  This is the test a parameter that starts MOVING under the longer")
    _LOG.info("  window has to pass. A posterior mean that wanders while its width")
    _LOG.info("  stays put is being pushed, not measured, and the usual thing")
    _LOG.info("  pushing it is the diffusion/jump split: sigma falls and lambda")
    _LOG.info("  rises as the window stops being one regime. A correlation with")
    _LOG.info("  dSIGMA that appears only in column B is that, not information.")
    _LOG.info("  The dates overlap heavily, so read the SIGN and the SIZE; there")
    _LOG.info("  are nothing like %d independent observations behind it." % len(common))
    _LOG.info("")
    _LOG.info("  A 3-year window overlaps 755 of its 756 observations day to day, so")
    _LOG.info("  the series is smoother by construction. That is not a better")
    _LOG.info("  estimate of anything - it is the same crisis held in view three")
    _LOG.info("  times as long, which is exactly what the vintage contrast needs")
    _LOG.info("  NOT to happen. Read this column as a cost, not a gain.")
    _LOG.info("")


if __name__ == "__main__":
    main()
