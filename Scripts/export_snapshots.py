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

# --- configuration -----------------------------------------------------------
SYSTEMATIC_ID = "^SPX"
IDIOSYNCRATIC_IDS = get_idiosyncratic_ids()  # e.g. ["COIN"]


def export_prices():
    """Pull adjusted-close history for every symbol and write one CSV each."""
    import FinanceDataReader as fdr

    symbols = [SYSTEMATIC_ID] + list(IDIOSYNCRATIC_IDS)
    print(f"Exporting price snapshots for: {symbols}")

    for symbol in symbols:
        series = fdr.DataReader(symbol)["Adj Close"]
        series.index.name = "sVALUATION_DATE"
        series.name = symbol
        out_name = f"prices_{_safe_symbol(symbol)}.csv"
        path = write_snapshot(series.to_frame(), out_name)
        print(f"  wrote {path}  ({len(series)} rows, "
              f"{series.index.min().date()} -> {series.index.max().date()})")


def main():
    print("=" * 70)
    print("mkt-depth-n-resiliency :: snapshot export")
    print("=" * 70)
    export_prices()
    print("Done. Commit the files under data/snapshots/ to make the "
          "repository a self-contained replication package.")


if __name__ == "__main__":
    main()
