"""
Pull daily adjusted-close history for the spot backfill study and freeze it as
per-symbol snapshots under data/snapshots/.

Uses FinanceDataReader through the repository's existing data layer, so the
study draws on the same source as the published paper.

    python poc/fetch_prices.py            # from the repo root

Snapshot, not live, is the point. Vendor history is revised over time, so a
live pull is not reproducible. Freeze once, commit the snapshots, and every
later rerun reproduces exactly. Same philosophy as Library/DataAccess.py and
Scripts/export_snapshots.py.
"""

import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.DataAccess import write_snapshot, _safe_symbol  # noqa: E402

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

# Systematic factor. Same ticker the paper uses.
SYSTEMATIC_ID = "^SPX"

# Sector ETFs - the incumbent proxy benchmark. All launched Dec 1998, so all
# have full 2007 history.
ETFS = ["XLK", "XLF", "XLP", "XLE", "XLI", "XLV", "XLY", "XLU"]

# Test names mapped to their sector ETF. Selection is ex ante on sector and
# size, NOT on 2008 behaviour, which would be selection leakage. Bucket by
# post-crisis estimated liquidity beta after estimation.
#
# GE and BAC are the negative controls: both had large firm-specific events in
# 2008 (GE Capital; the Countrywide and Merrill acquisitions plus capital
# raises). The model is expected to do poorly on them. Report that.
NAMES = {
    # Technology
    "MSFT": "XLK", "ORCL": "XLK", "INTC": "XLK", "AAPL": "XLK", "NVDA": "XLK",
    # Financials
    "JPM":  "XLF", "GS":   "XLF", "BAC":  "XLF", "C":    "XLF", "WFC":  "XLF",
    # Consumer staples
    "PG":   "XLP", "KO":   "XLP", "WMT":  "XLP", "PEP":  "XLP", "CL":   "XLP",
    # Energy
    "XOM":  "XLE", "CVX":  "XLE", "COP":  "XLE", "SLB":  "XLE", "OXY":  "XLE",
    # Industrials
    "CAT":  "XLI", "GE":   "XLI", "BA":   "XLI", "HON":  "XLI", "UNP":  "XLI",
    # Health care
    "JNJ":  "XLV", "PFE":  "XLV", "MRK":  "XLV", "ABT":  "XLV", "UNH":  "XLV",
    # Consumer discretionary
    "HD":   "XLY", "SBUX": "XLY", "AMZN": "XLY", "MCD":  "XLY", "NKE":  "XLY",
    # Utilities
    "SO":   "XLU", "D":    "XLU", "AEP":  "XLU", "XEL":  "XLU", "ED":   "XLU",
}

# A name is unusable as a test case if its history does not reach back to here.
REQUIRED_START = pd.Timestamp("2007-01-15")


def fetch_one(symbol):
    """Adjusted-close series for one symbol, straight from FinanceDataReader."""
    import FinanceDataReader as fdr

    series = fdr.DataReader(symbol)["Adj Close"]
    series.index = pd.to_datetime(series.index)
    series.index.name = "sVALUATION_DATE"
    series.name = symbol
    return series.dropna()


def _largest_moves(series, n=3):
    """The n largest absolute daily log returns, as (date, pct) pairs.

    A split or reverse-split that the vendor adjusted badly shows up here as an
    implausible one-day move. Citigroup did a 1-for-10 reverse split in May
    2011, so it is the most likely candidate in this panel. Eyeball these before
    trusting any downstream number: a fabricated 900% day would dominate every
    tail statistic in the study.
    """
    import numpy as np

    r = np.log(series / series.shift(1)).dropna()
    top = r.reindex(r.abs().sort_values(ascending=False).index[:n])
    return [(d.date(), float(np.expm1(v) * 100)) for d, v in top.items()]


def main():
    symbols = [SYSTEMATIC_ID] + ETFS + sorted(NAMES)
    print("=" * 72)
    print("jgl-risk-sol :: spot backfill study - snapshot export")
    print("=" * 72)
    print("%d symbols\n" % len(symbols))

    coverage = {}
    extremes = {}
    failed = []

    for symbol in symbols:
        try:
            series = fetch_one(symbol)
        except Exception as exc:                                  # noqa: BLE001
            print("  FAILED  %-6s %s" % (symbol, exc))
            failed.append(symbol)
            continue

        write_snapshot(series.to_frame(), "prices_%s.csv" % _safe_symbol(symbol))
        coverage[symbol] = series.index.min()
        extremes[symbol] = _largest_moves(series)
        print("  wrote   %-6s %5d rows  %s -> %s"
              % (symbol, len(series), series.index.min().date(),
                 series.index.max().date()))

    late = {s: d for s, d in coverage.items() if d > REQUIRED_START}
    print("\n" + "-" * 72)
    if failed:
        print("COULD NOT FETCH: %s" % ", ".join(failed))
    if late:
        print("HISTORY STARTS TOO LATE (unusable as test names):")
        for s, start in sorted(late.items()):
            print("   %-6s first observation %s" % (s, start.date()))
        print("   Drop or replace these in NAMES before running the study.")
    if not failed and not late:
        print("All symbols reach back to %s. No substitutions needed."
              % REQUIRED_START.date())

    print("\nLargest daily moves per symbol - check for split-adjustment artifacts:")
    for sym in sorted(extremes):
        moves = ", ".join("%s %+.0f%%" % (d, p) for d, p in extremes[sym])
        flag = "  <-- CHECK" if any(abs(p) > 60 for _, p in extremes[sym]) else ""
        print("   %-6s %s%s" % (sym, moves, flag))

    print("-" * 72)
    print("Snapshots in data/snapshots/. Commit them - the study is then")
    print("reproducible from committed inputs, independent of vendor revisions.")


if __name__ == "__main__":
    main()
