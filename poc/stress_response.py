"""Does lambda rise in stress - and does it rise MORE than sigma does?

That is the economic claim, and it is a different question from the level of
lambda. The level is set jointly with eta and sigma along a ridge the returns
are nearly indifferent to, so a prior picks it; what no prior can manufacture
is a lambda that is high in 2008 and low in 2017, because the prior is the
same at every valuation date. Movement across dates is data.

So the cells are compared on PERCENTILE within each cell's own lambda series,
not on level. A cell whose lambda sits at the 95th percentile of its own
history during the GFC has made the claim, whether its lambda is 8 or 80.
That normalisation is the whole point: it lets the tight-prior and flat-prior
arms be held against each other despite a factor of three in level, and it is
robust to a lambda distribution with a long right tail.

The second table is the one that decides whether the jump channel EARNS its
place. If lambda merely tracks the diffusion, a one-parameter volatility model
says everything this says and the decomposition is decoration. So sigma gets
the same treatment, and the third table is the difference: where lambda's
percentile clears sigma's in a stress episode, the jump intensity carried
something the diffusion did not.

    python poc/stress_response.py
    python poc/stress_response.py --cells skew-tight:252,skewtight-lamflat:252

WINDOW COVERAGE. An episode label names when the market moved, not what the
trailing window held. At 252 days the GFC is inside the window from late 2008
to late 2009 and gone by 2010; at 756 it is still inside in 2011, which is why
a three-year series can show its largest lambda at "Euro 2011" and a merely
average one in 2008H2. The coverage line under each episode says how much of
the episode the window at that date actually contains, so a spike can be read
against what produced it.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import PMLE_DIR                          # noqa: E402

DEFAULT_CELLS = ("skew-tight:252,skewtight-lamflat:252,"
                 "skew-tight:756,skewtight-lamflat:756")

STRESS = {
    "GFC 2008H2":   ("2008-07-01", "2008-12-31"),
    "GFC 2009H1":   ("2009-01-01", "2009-06-30"),
    "Euro 2011":    ("2011-07-01", "2011-12-31"),
    "Covid 2020":   ("2020-02-01", "2020-06-30"),
    "SVB 2023":     ("2023-03-01", "2023-06-30"),
    "Tariffs 2025": ("2025-04-01", "2025-07-31"),
}
CALM = {
    "calm 2013":    ("2013-01-01", "2013-12-31"),
    "calm 2017":    ("2017-01-01", "2017-12-31"),
    "calm 2019H2":  ("2019-07-01", "2019-12-31"),
}


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
    return df.sort_values("dtVALUATION_DATE").set_index("dtVALUATION_DATE")


def pctile(series, window):
    """Where the episode's median sits in the cell's OWN distribution, 0-100."""
    w = series.loc[window[0]:window[1]]
    if not len(w):
        return np.nan, np.nan
    med = float(w.median())
    return 100.0 * float((series <= med).mean()), med


def table(title, cells, episodes, col, note):
    print()
    print("  %s" % title)
    print("  %-14s %s" % ("episode",
                          " ".join("%18s" % c for c in cells)))
    print("  " + "-" * (14 + 19 * len(cells)))
    out = {}
    for name, win in episodes.items():
        row, cellvals = [], {}
        for c, df in cells.items():
            if col not in df:
                row.append("%18s" % "-"); continue
            p, med = pctile(df[col], win)
            cellvals[c] = p
            row.append("%18s" % ("%5.0f%%  (%7.2f)" % (p, med)
                                 if np.isfinite(p) else "-"))
        out[name] = cellvals
        print("  %-14s %s" % (name, " ".join(row)))
    print("  %s" % note)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    a = ap.parse_args()

    from Library.RiskEngineKimYi2025 import SYSTEMATIC_PRIOR_SETS
    from poc.estimate_systematic import store_id

    cells, lbs = {}, {}
    for cell in [c.strip() for c in a.cells.split(",") if c.strip()]:
        tag, _, lb = cell.partition(":")
        lb = int(lb or 252)
        if tag not in SYSTEMATIC_PRIOR_SETS:
            raise SystemExit("unknown arm %r" % tag)
        try:
            cells[cell] = load_drawer(
                store_id(tag, SYSTEMATIC_PRIOR_SETS[tag], lb))
            lbs[cell] = lb
        except SystemExit as exc:
            print("  %s SKIPPED - %s" % (cell, exc))
    if not cells:
        raise SystemExit("no drawers loaded")

    print()
    print("=" * 78)
    print("stress response :: percentile within each cell's OWN series")
    print("=" * 78)
    for c, df in cells.items():
        print("  %-26s %4d dates  %s -> %s   lambda median %7.2f"
              % (c, len(df), df.index.min().date(), df.index.max().date(),
                 float(df["dLAMB"].median())))
    print()
    print("  Each entry is percentile (median level). Percentile is what")
    print("  compares across cells; the level in brackets does not, because")
    print("  the arms differ by a factor of three on it by construction.")

    lam = table("LAMBDA - jump intensity", cells, STRESS, "dLAMB",
                "A prior is identical at every date and cannot put lambda at "
                "the 90th\n  percentile in 2008 and the 10th in 2017. "
                "High percentiles here are data.")
    table("LAMBDA in calm periods - the control", cells, CALM, "dLAMB",
          "These should be LOW. A cell that puts calm years at high "
          "percentiles too\n  is not responding to stress, it is wandering.")
    sig = table("SIGMA - diffusion, same treatment", cells, STRESS, "dSIGMA",
                "The comparison, not the point. If lambda only does what "
                "sigma does,\n  the jump channel is decoration.")

    print()
    print("  LAMBDA percentile MINUS SIGMA percentile, stress episodes only.")
    print("  Positive means the jump intensity moved further into its own tail")
    print("  than the diffusion did - the decomposition earning its place.")
    print("  %-14s %s" % ("episode", " ".join("%18s" % c for c in cells)))
    print("  " + "-" * (14 + 19 * len(cells)))
    for name in STRESS:
        row = []
        for c in cells:
            d = lam[name].get(c, np.nan) - sig[name].get(c, np.nan)
            row.append("%18s" % ("%+6.0f pts" % d if np.isfinite(d) else "-"))
        print("  %-14s %s" % (name, " ".join(row)))

    print()
    print("  WINDOW COVERAGE - share of the episode inside the trailing window")
    print("  at the episode's own dates, which is what the estimate could see:")
    print("  %-14s %s" % ("episode", " ".join("%18s" % ("%d-day" % lbs[c])
                                              for c in cells)))
    print("  " + "-" * (14 + 19 * len(cells)))
    for name, (lo, hi) in STRESS.items():
        row = []
        for c in cells:
            df = cells[c]
            w = df.loc[lo:hi]
            if not len(w):
                row.append("%18s" % "-"); continue
            # at the episode's LAST date, how far back does the window reach
            back = w.index.max() - pd.Timedelta(days=lbs[c] * 365.0 / 252.0)
            row.append("%18s" % ("from %s" % back.date()))
        print("  %-14s %s" % (name, " ".join(row)))
    print()
    print("  Read a spike against that date. A 756-day window in late 2011")
    print("  still holds the whole of 2008-09, so a 'Euro 2011' peak there may")
    print("  be the GFC that has not yet left rather than anything in 2011.")
    print()


if __name__ == "__main__":
    main()
