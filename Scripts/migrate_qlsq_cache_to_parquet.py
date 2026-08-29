"""
Scripts/migrate_qlsq_cache_to_parquet.py — one-off migration of the QLSQ
skew-calibration cache from the old pickle format (a dict of
"{ticker}-{YYYYMMDD}" -> scipy OptimizeResult, as read/written by
Scripts/skew_calibration_kimyi2025.ipynb and the pre-Parquet versions of
Scripts/skew_calibrate_systematic.py / skew_calibrate_idiosyncratic.py) into
the wide Parquet table those scripts now use (see
Scripts/skew_calibrate_systematic.py's QLSQ_PARAM_ORDER / CACHE_COLUMNS).

Idiosyncratic entries in the old pickle only carry the four idiosyncratic
parameters (kappai, gammai, betai, rhoix); this migration looks up that same
date's systematic entry to denormalize sigma/pprob/lamb/eta1/eta2 onto the
row, matching what the new scripts do at calibration time.

The old pickle's OptimizeResult objects don't record how many option
observations went into each fit, so iN_OBS is left null for migrated rows;
the new scripts always fill it in for calibrations run going forward.

Run once from the repository root:

    python Scripts/migrate_qlsq_cache_to_parquet.py [old_pickle_path]

Defaults to config_skew.OUTPUT_DIR / "kimyi2025_vol_calibration.pkl" and
writes to config_skew.OUTPUT_DIR / OUTPUT_PARQUET_NAME. Refuses to overwrite
an existing Parquet file at that path -- delete it first if you need to
re-run the migration.
"""

# Ensure package imports resolve regardless of cwd.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import os
import pickle
import sys

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config_skew as cfg  # Scripts/config_skew.py
from Scripts.skew_calibrate_systematic import (
    CACHE_COLUMNS,
    IDIOSYNCRATIC_PARAM_NAMES,
    SYSTEMATIC_FIXED_PARAMS,
    SYSTEMATIC_PARAM_NAMES,
    cache_path,
)


def migrate(old_pickle_path: str) -> str:
    with open(old_pickle_path, "rb") as f:
        old_cache = pickle.load(f)

    # Pass 1: systematic rows, indexed by date for the idiosyncratic lookup.
    rows = {}
    systematic_by_date = {}
    for key, result in old_cache.items():
        ticker, date_str = key.rsplit("-", 1)
        if ticker != cfg.SYSTEMATIC_TICKER:
            continue
        row = {"sTICKER": ticker, "sVALUATION_DATE": date_str}
        row.update(SYSTEMATIC_FIXED_PARAMS)
        row.update(dict(zip(SYSTEMATIC_PARAM_NAMES, result.x)))
        row["bCONVERGED"] = bool(result.success)
        row["dOBJECTIVE"] = float(result.fun)
        row["iN_OBS"] = None
        rows[(ticker, date_str)] = row
        systematic_by_date[date_str] = row

    # Pass 2: idiosyncratic rows, denormalizing that date's systematic params.
    skipped = []
    for key, result in old_cache.items():
        ticker, date_str = key.rsplit("-", 1)
        if ticker == cfg.SYSTEMATIC_TICKER:
            continue
        sys_row = systematic_by_date.get(date_str)
        if sys_row is None:
            skipped.append(key)
            continue
        row = {"sTICKER": ticker, "sVALUATION_DATE": date_str}
        row.update(dict(zip(IDIOSYNCRATIC_PARAM_NAMES, result.x)))
        row.update({p: sys_row[p] for p in SYSTEMATIC_PARAM_NAMES})
        row["bCONVERGED"] = bool(result.success)
        row["dOBJECTIVE"] = float(result.fun)
        row["iN_OBS"] = None
        rows[(ticker, date_str)] = row

    if skipped:
        print(
            f"Skipped {len(skipped)} idiosyncratic entr{'y' if len(skipped) == 1 else 'ies'} "
            f"with no matching systematic date: {skipped}"
        )

    out_path = cache_path()
    if os.path.exists(out_path):
        raise FileExistsError(
            f"{out_path} already exists; refusing to overwrite. "
            f"Remove it first if you want to re-run the migration."
        )

    df = pd.DataFrame(list(rows.values()), columns=CACHE_COLUMNS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    default_old_path = os.path.join(cfg.OUTPUT_DIR, "kimyi2025_vol_calibration.pkl")
    old_path = sys.argv[1] if len(sys.argv) > 1 else default_old_path

    if not os.path.exists(old_path):
        raise FileNotFoundError(f"No pickle cache found at {old_path}")

    written_path = migrate(old_path)
    migrated_df = pd.read_parquet(written_path)
    print(f"Migrated {len(migrated_df)} rows from {old_path}\n  -> {written_path}")
