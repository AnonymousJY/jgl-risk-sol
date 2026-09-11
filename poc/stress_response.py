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

sigma gets the same treatment, but NOT as a hurdle for lambda to clear. sigma
should rise in stress as well - that is its own economic content, and in a
prolonged episode like 2008-09 it is the channel one would expect to carry
most of it. So "lambda beats sigma" is the wrong null. What would make the
decomposition decoration is the two being REDUNDANT, and that is a question
about correlation and timing, not about levels:

  - corr(lambda, sigma) across dates, printed directly. Near 1.00 and one of
    them is spare.
  - lambda orthogonalised to sigma - the residual from regressing one on the
    other - put through the same percentile treatment. If the residual still
    lands high in stress, lambda is carrying something the diffusion is not.
  - onset against prolonged. GFC 2008H2 and GFC 2009H1 are kept as separate
    episodes because the expected signatures differ: an event RATE should
    spike on the gap days and can fall back while a volatility LEVEL stays
    elevated through the whole episode.

The jump-shape block asks the other half. In stress the down branch should
carry more of the jumps; whether it also carries bigger ones is a separate
question with a separate answer, and the two are reported separately rather
than collapsed into "negative skew".

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
    sig = table("SIGMA - diffusion", cells, STRESS, "dSIGMA",
                "Expected to rise too, and in a prolonged episode to carry "
                "most of it.\n  Not a hurdle for lambda - see the "
                "orthogonalised block below.")
    table("PPROB - P(up jump); should FALL in stress", cells, STRESS, "dPPROB",
          "Below 50% means the down branch carries more than half the jumps.")
    table("ETA1 - up-jump decay; FALLING means bigger up jumps",
          cells, STRESS, "dETA1", "  Mean up jump is 1/eta1.")
    table("ETA2 - down-jump decay; FALLING means bigger down jumps",
          cells, STRESS, "dETA2", "  Mean down jump is 1/eta2.")

    # --- is lambda redundant given sigma? -----------------------------------
    print()
    print("  REDUNDANCY - lambda against sigma across all dates, and lambda")
    print("  orthogonalised to sigma put through the stress percentiles again:")
    print("  %-14s %s" % ("", " ".join("%18s" % c for c in cells)))
    print("  " + "-" * (14 + 19 * len(cells)))
    row = []
    resid = {}
    for c, df in cells.items():
        x = df["dSIGMA"].astype(float).to_numpy()
        y = df["dLAMB"].astype(float).to_numpy()
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else np.nan
        row.append("%18s" % ("%+.3f" % r))
        b = np.polyfit(x, y, 1) if np.isfinite(r) else (0.0, 0.0)
        resid[c] = pd.Series(y - (b[0] * x + b[1]), index=df.index)
    print("  %-14s %s" % ("corr", " ".join(row)))
    for name, win in STRESS.items():
        row = []
        for c in cells:
            p_, _ = pctile(resid[c], win)
            row.append("%18s" % ("%5.0f%%" % p_ if np.isfinite(p_) else "-"))
        print("  %-14s %s" % (name, " ".join(row)))
    print("  A residual that still lands high in stress is lambda carrying")
    print("  something sigma does not. Near 50% everywhere and the jump")
    print("  intensity is a restatement of the diffusion.")

    # --- what the jump distribution looks like, stress vs calm ---------------
    print()
    print("  JUMP SHAPE, at the median parameters of stress vs calm dates:")
    print("  %-26s %8s %8s %8s %8s %9s"
          % ("cell / regime", "P(down)", "up size", "dn size", "dn/up",
             "jump skew"))
    print("  " + "-" * 74)
    for c, df in cells.items():
        for regime, eps in (("stress", STRESS), ("calm", CALM)):
            idx = pd.Index([])
            for lo, hi in eps.values():
                idx = idx.union(df.loc[lo:hi].index)
            if not len(idx):
                continue
            w = df.loc[idx]
            pp = float(w["dPPROB"].median())
            e1 = float(w["dETA1"].median())
            e2 = float(w["dETA2"].median())
            m1 = pp / e1 - (1 - pp) / e2
            m2 = 2 * (pp / e1 ** 2 + (1 - pp) / e2 ** 2)
            m3 = 6 * (pp / e1 ** 3 - (1 - pp) / e2 ** 3)
            v = m2 - m1 ** 2
            sk = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / v ** 1.5
            print("  %-26s %8.3f %7.2f%% %7.2f%% %8.2f %+9.3f"
                  % ("%s / %s" % (c, regime), 1 - pp, 100 / e1, 100 / e2,
                     (1 / e2) / (1 / e1), sk))
    print("  Frequency asymmetry and SIZE asymmetry are different claims and")
    print("  can move in opposite directions. On the 756 skew-tight by-year")
    print("  means they do: P(down) rises 0.41 -> 0.50 from calm to stress")
    print("  while the down/up SIZE ratio compresses 1.83 -> 1.38 and the jump")
    print("  skew goes -1.25 -> -0.67. Less skewed, not more - which is the")
    print("  index-option stylised fact, where skew is steepest in quiet")
    print("  markets and flattens when ATM vol spikes.")

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
