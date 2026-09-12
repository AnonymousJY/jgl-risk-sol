"""Does a fitted arm reproduce the window it was fitted on?

The skewtight-lamflat runs showed lambda is not identified: freeing its prior
moved the posterior mean from 8.8 to 31.0 at 252 days and from 19.6 to 70.0 at
756, with an interval as wide as the level itself, while eta1 and eta2 rose
with it and sigma fell. That is a RIDGE - many small jumps and a smaller
diffusion look, to this likelihood, much like few large jumps and a larger one.

Which leaves one question that the identification table cannot answer and that
decides what to do about it. A ridge means the data is INDIFFERENT along it,
so every point on it fits equally well and the prior is choosing harmlessly.
A prior climbing OFF the ridge into a region the data does not support is a
different thing entirely, and would say the flat arm is wrong rather than
liberated.

So: take each arm's posterior means, build the one-day return density the
model implies - shock_to_name._density_h, the code's own machinery, not an
algebraic shortcut past it - and compare its sd and its 2.5% quantile against
what the trailing window actually did. An arm on the ridge matches the window
it was fitted on. An arm off it does not.

    python poc/variance_check.py
    python poc/variance_check.py --cells skew-tight:252,skewtight-lamflat:252
    python poc/variance_check.py --beg 20080101 --end 20091231

Read the sd ratio first: it is the second moment, which any arm that fits at
all has to get right. The quantile ratio is the tail, which is what the paper
is actually about, and an arm can match the variance while splitting it
between diffusion and jumps in a way that moves the 2.5% point a long way.

TWO THINGS THIS IS NOT, both of which bound what a result here can support.

It evaluates the density at the POSTERIOR MEAN, not over the posterior. Under
skewtight-lamflat lambda's interval is as wide as its level, so its mean is a
poor summary of it and a ratio computed there is a point estimate of a ratio,
not the ratio of the posterior predictive. Reading a 5% miss as meaningful
would be over-reading; a 25% one is past what that can explain.

And _density_h is the UNCONDITIONAL h-day density - the object the shock and
ES work uses - which is not by construction the same as the one-day density
the P-MLE likelihood evaluates. If the two differ, a uniform bias appears in
every cell at once. What survives that is the DIFFERENCE between arms on the
same dates, which is why the table prints the cells side by side rather than
each one's ratio against 1.00 on its own.
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import PMLE_DIR, get_price_panel        # noqa: E402

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)

DEFAULT_CELLS = ("skew-tight:252,skewtight-lamflat:252,"
                 "skew-tight:756,skewtight-lamflat:756")
EPISODES = {"GFC 2008-09": ("2008-01-01", "2009-12-31"),
            "Covid 2020":  ("2020-02-01", "2020-12-31"),
            "calm 2015-17": ("2015-01-01", "2017-12-31")}


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


def model_moments(row, base_days):
    """(sd, q025) of the model's ONE-DAY return, at this date's posterior means."""
    import poc.shock_to_name as stn

    f = stn._density_h(float(row["dSIGMA"]), float(row["dLAMB"]),
                       float(row["dPPROB"]), float(row["dETA1"]),
                       float(row["dETA2"]), float(row["dALPHA"]),
                       1.0 / base_days)
    # _GRID is built lazily inside _conv_stack, so it must be read off the
    # MODULE after the call. Importing the name binds None once and for all.
    x = stn._GRID
    dx = float(x[1] - x[0])
    m = f * dx
    tot = m.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.nan, np.nan
    m = m / tot                                    # guard grid truncation
    mu = float((x * m).sum())
    var = float(((x - mu) ** 2 * m).sum())
    cdf = np.cumsum(m)
    q025 = float(np.interp(0.025, cdf, x))
    return np.sqrt(var), q025


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=DEFAULT_CELLS,
                    help="comma-separated arm:lookback, e.g. "
                         "'skew-tight:252,skewtight-lamflat:756'")
    ap.add_argument("--beg", default=None, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--base-days", type=int, default=252)
    a = ap.parse_args()

    from Library.RiskEngineKimYi2025 import SYSTEMATIC_PRIOR_SETS
    from poc.estimate_systematic import store_id, SYSTEMATIC_ID

    px = get_price_panel([SYSTEMATIC_ID])
    rets = px.pct_change().dropna()[SYSTEMATIC_ID]

    _LOG.info("")
    _LOG.info("=" * 78)
    _LOG.info("variance check :: model-implied one-day moments vs the fitted window")
    _LOG.info("=" * 78)
    _LOG.info("  A ratio of 1.00 means the arm reproduces the window it was fitted")
    _LOG.info("  on. Systematically above 1.00 means the arm asserts more risk than")
    _LOG.info("  the data showed; below, less.")
    _LOG.info("")
    _LOG.info("  %-26s %6s %10s %10s %10s %10s"
          % ("cell", "dates", "sd ratio", "within 10%", "q2.5 ratio", "q2.5 hit"))
    _LOG.info("  " + "-" * 78)

    keep = {}
    for cell in [c.strip() for c in a.cells.split(",") if c.strip()]:
        tag, _, lb = cell.partition(":")
        lb = int(lb or a.base_days)
        if tag not in SYSTEMATIC_PRIOR_SETS:
            raise SystemExit("unknown arm %r; have %s"
                             % (tag, sorted(SYSTEMATIC_PRIOR_SETS)))
        drawer = store_id(tag, SYSTEMATIC_PRIOR_SETS[tag], lb)
        try:
            df = load_drawer(drawer, a.beg, a.end)
        except SystemExit as exc:
            _LOG.info("  %-26s  SKIPPED - %s" % (cell, exc))
            continue

        rows = []
        for dt, row in df.iterrows():
            win = rets.loc[rets.index <= dt]
            if len(win) < lb:
                continue
            win = win.iloc[-lb:]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                msd, mq = model_moments(row, a.base_days)
            rows.append((dt, msd, mq, float(win.std(ddof=1)),
                         float(win.quantile(0.025))))
        if not rows:
            _LOG.info("  %-26s  no usable dates" % cell)
            continue
        r = pd.DataFrame(rows, columns=["dt", "msd", "mq", "rsd", "rq"]
                         ).set_index("dt")
        r["sd_ratio"] = r.msd / r.rsd
        r["q_ratio"] = r.mq / r.rq
        keep[cell] = r
        _LOG.info("  %-26s %6d %10.3f %9.0f%% %10.3f %9.0f%%"
              % (cell, len(r), r.sd_ratio.median(),
                 100 * ((r.sd_ratio - 1).abs() < 0.10).mean(),
                 r.q_ratio.median(),
                 100 * ((r.q_ratio - 1).abs() < 0.10).mean()))

    if not keep:
        return
    _LOG.info("")
    _LOG.info("  Median sd ratio by episode - a ridge point holds across regimes,")
    _LOG.info("  a mis-specified one drifts with the regime:")
    _LOG.info("  %-26s %14s %14s %14s"
          % ("cell", *EPISODES.keys()))
    _LOG.info("  " + "-" * 72)
    for cell, r in keep.items():
        vals = []
        for lo, hi in EPISODES.values():
            w = r.loc[lo:hi, "sd_ratio"]
            vals.append(w.median() if len(w) else np.nan)
        _LOG.info("  %-26s %14.3f %14.3f %14.3f" % (cell, *vals))
    _LOG.info("")
    _LOG.info("  The second moment is the low bar. If two arms both clear it while")
    _LOG.info("  disagreeing about lambda by a factor of three, the data is")
    _LOG.info("  indifferent along that ridge and the prior is choosing something")
    _LOG.info("  the returns cannot - which has to be SAID, not estimated. If one")
    _LOG.info("  arm misses it, that arm is off the ridge and the comparison of")
    _LOG.info("  their identification ratios was never meaningful.")
    _LOG.info("")


if __name__ == "__main__":
    main()
