"""
export_snapshots.py — freeze live data sources into committed snapshots.

Run this ONCE, in an environment that has network access, to turn the
repository into a self-contained replication package. After it runs, every
analysis script and notebook works in the default ``snapshot`` data mode with
no network dependency, and the published tables and figures become exactly
reproducible.

    python Scripts/export_snapshots.py

Outputs (written under ``data/snapshots/``)
-------------------------------------------
    prices_SPX.csv      adjusted-close history for the systematic proxy
    prices_COIN.csv     adjusted-close history for each idiosyncratic asset

Notes
-----
* Price history is pulled from FinanceDataReader. Vendor history is revised
  over time, so the committed snapshot is the single source of truth for
  reproducibility once frozen.
* The P-MLE parameter estimates are not exported here — they already ship as
  per-date CSVs under ``Study/Estimated Parameters PMLE/`` and are produced
  directly by ``Scripts/run_pmle_kimyi2025.py``.
* The portfolio definition is produced by ``Scripts/load_portfolio.py``
  (``build_portfolio()`` / ``data/snapshots/portfolio.csv``); it requires no
  external data source.
"""

# Ensure package imports resolve regardless of cwd.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import os
import sys

# allow "python Scripts/export_snapshots.py" from the repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.DataAccess import write_snapshot, _safe_symbol
from Scripts.load_portfolio import get_idiosyncratic_ids

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)

# --- configuration -----------------------------------------------------------
SYSTEMATIC_ID = "^SPX"
IDIOSYNCRATIC_IDS = get_idiosyncratic_ids()  # e.g. ["COIN"]


def export_prices(extra=None, only=None):
    """Pull adjusted-close history for every symbol and write one CSV each.

    `only` replaces the portfolio list outright, `extra` appends to it. Both
    exist because a study basket is not always the portfolio: comparing the
    banks against a cruise/airline basket needs CCL, RCL and LUV frozen, and
    editing the portfolio definition to get them would change what the
    replication package claims to be about.
    """
    import FinanceDataReader as fdr

    if only:
        symbols = [SYSTEMATIC_ID] + list(only)
    else:
        symbols = [SYSTEMATIC_ID] + list(IDIOSYNCRATIC_IDS) + list(extra or [])
    # Two statements, not one tuple assignment: the right-hand side of
    # "seen, symbols = set(), [... seen ...]" is evaluated in full before
    # either name is bound, so the comprehension cannot see `seen`.
    seen = set()
    symbols = [x for x in symbols if not (x in seen or seen.add(x))]
    _LOG.info(f"Exporting price snapshots for: {symbols}")

    ends = {}
    for symbol in symbols:
        series = fdr.DataReader(symbol)["Adj Close"]
        series.index.name = "sVALUATION_DATE"
        series.name = symbol
        out_name = f"prices_{_safe_symbol(symbol)}.csv"
        path = write_snapshot(series.to_frame(), out_name)
        ends[symbol] = series.index.max()
        _LOG.info(f"  wrote {path}  ({len(series)} rows, "
              f"{series.index.min().date()} -> {series.index.max().date()})")
    _warn_stale(ends)


def _warn_stale(ends):
    """Flag any snapshot that now ends earlier than the freshest one.

    A PARTIAL refresh is the trap. Exporting ^SPX plus a new basket moves the
    reference calendar forward while every symbol not named stays where it
    was, and get_aligned_price_panel then reports the difference as a gap in
    those symbols - forward-filled up to ffill_limit and NaN past it. It looks
    like a data defect in the untouched names and is nothing of the kind.

    Only the symbols exported in THIS run are checked, which is the point:
    the ones not exported are exactly the ones that will fall behind, and this
    cannot see them. So the message names the fix rather than the symptom.
    """
    if len(ends) < 2:
        return
    newest = max(ends.values())
    stale = {k: v for k, v in ends.items() if (newest - v).days > 3}
    if not stale:
        return
    _LOG.info("")
    _LOG.info("  NOTE these are behind the freshest symbol (%s):" % newest.date())
    for k, v in sorted(stale.items(), key=lambda kv: kv[1]):
        _LOG.info("    %-8s %s  (%d days)" % (k, v.date(), (newest - v).days))
    _LOG.info("  A snapshot that lags the reference calendar shows up as a gap in")
    _LOG.info("  the panel report, not as staleness. Export every symbol the study")
    _LOG.info("  uses in one run rather than adding a basket on its own.")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Freeze live price history into committed snapshots.")
    ap.add_argument("--symbols", default=None,
                    help="comma-separated tickers to export INSTEAD of the "
                         "portfolio, e.g. CCL,RCL,LUV. ^SPX is always "
                         "included.")
    ap.add_argument("--add", default=None,
                    help="comma-separated tickers to export IN ADDITION to "
                         "the portfolio.")
    a = ap.parse_args()
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None

    _LOG.info("=" * 70)
    _LOG.info("mkt-depth-n-resiliency :: snapshot export")
    _LOG.info("=" * 70)
    export_prices(extra=split(a.add), only=split(a.symbols))
    _LOG.info("Done. Commit the files under data/snapshots/ to make the "
          "repository a self-contained replication package.")


if __name__ == "__main__":
    main()
