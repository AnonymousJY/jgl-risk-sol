"""
migrate_pkl_to_parquet.py - one-shot conversion utility.

Walks ``Study/Estimated Parameters QLSQ/`` and ``Study/Vol Surface From Model/``
and converts each legacy ``.pkl`` file to the parquet format understood by
``Library.Serialization``. After a successful conversion the original ``.pkl``
is left in place; delete it manually once you have verified the pipeline runs
against the parquet output.

Files handled:
  - ``kimyi2025_vol_surface[_].pkl``     -> ``kimyi2025_vol_surface[_].parquet``
  - ``kimyi2025_vol_calibration[_].pkl`` -> ``kimyi2025_vol_calibration[_].parquet``
  - ``heston1993_vol_calibration.pkl``   -> ``heston1993_vol_calibration.parquet``

Run once, in an environment with scipy installed (needed to unpickle the
existing calibration results that contain scipy.optimize.OptimizeResult
objects):

    PYTHONPATH=. python Scripts/migrate_pkl_to_parquet.py

Idempotent: existing parquet outputs are skipped unless ``--force`` is passed.
"""

import argparse
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Library.Logging import setup_logging  # noqa: E402
from Library.Serialization import (  # noqa: E402
    save_calibration_results,
    save_vol_surface,
)

logger = setup_logging(__name__)


VOL_SURFACE_STEMS = ("kimyi2025_vol_surface", "kimyi2025_vol_surface_")
CALIB_STEMS = (
    "kimyi2025_vol_calibration",
    "kimyi2025_vol_calibration_",
    "heston1993_vol_calibration",
)


def _convert_vol_surface(pkl_path: Path, force: bool) -> None:
    parquet_path = pkl_path.with_suffix(".parquet")
    if parquet_path.exists() and not force:
        logger.info("Skip (parquet exists): %s", parquet_path)
        return
    logger.info("Converting vol surface: %s -> %s", pkl_path, parquet_path)
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    save_vol_surface(obj, parquet_path)
    logger.info("Wrote %s (%d entries)", parquet_path, len(obj))


def _convert_calibration(pkl_path: Path, force: bool) -> None:
    parquet_path = pkl_path.with_suffix(".parquet")
    if parquet_path.exists() and not force:
        logger.info("Skip (parquet exists): %s", parquet_path)
        return
    logger.info("Converting calibration results: %s -> %s", pkl_path, parquet_path)
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    save_calibration_results(obj, parquet_path)
    logger.info("Wrote %s (%d entries)", parquet_path, len(obj))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing parquet outputs.",
    )
    args = ap.parse_args()

    study_dir = _REPO_ROOT / "Study"

    # Vol surfaces.
    for stem in VOL_SURFACE_STEMS:
        pkl = study_dir / "Vol Surface From Model" / f"{stem}.pkl"
        if pkl.exists():
            _convert_vol_surface(pkl, force=args.force)
        else:
            logger.info("Not present: %s", pkl)

    # Calibration results.
    for stem in CALIB_STEMS:
        pkl = study_dir / "Estimated Parameters QLSQ" / f"{stem}.pkl"
        if pkl.exists():
            _convert_calibration(pkl, force=args.force)
        else:
            logger.info("Not present: %s", pkl)

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
