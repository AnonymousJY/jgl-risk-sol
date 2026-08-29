"""
reproduce_paper.py - single entry point that reproduces every table and
figure in the paper "Systematic Liquidity Risk Management: A Novel
Perspective on Derivatives" (Yi and Kim).

Orchestrates the four pipeline scripts below, each of which owns a specific
subset of the paper's outputs. Toggle any step in the CONFIG dict to skip
it (useful when re-running just one figure without redoing multi-hour
calibrations).

    python Scripts/reproduce_paper.py

Environment variables honoured:
    LIQUIDITY_LOG_LEVEL    INFO | DEBUG | WARNING | ERROR
    LIQUIDITY_LOG_FILE     optional path; log lines duplicated to this file
    LIQUIDITY_CBOE_DATA_DIR  path to raw Cboe option chain data (skew step)
    MKTDEPTH_DATA_MODE     snapshot (default) | live (refresh from vendor)

Outputs produced (per configured steps):
    Table 1 -> Study/Estimated Parameters PMLE/{TICKER}/estimated_params_pmle_*.csv
    Table 2 -> Study/Collar Asian/final_output_1d.csv, final_output_1d_base.csv
    All figures (paper Figures 1-7 + diagnostics) -> Figures/*.pdf
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure Library imports resolve regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Library.Logging import setup_logging  # noqa: E402

logger = setup_logging(__name__)


# ============================================================================
# CONFIG - toggle each step to True (run) or False (skip)
# ============================================================================
CONFIG = {
    # P-MLE calibration -> Table 1 raw estimates for every (date, ticker).
    # Long-running (~3 minutes with fork parallelism, but incremental: skips
    # any date already present in Study/Estimated Parameters PMLE/).
    "table_1_pmle_calibration": True,

    # Volatility skew calibration -> Figures 4-5.
    # Requires raw Cboe option data via LIQUIDITY_CBOE_DATA_DIR; skips cleanly
    # if data is not present.
    "figures_4_5_skew_calibration": True,

    # Collar / Asian portfolio pipeline -> Figures 1, 2, 6 + Table 2 CSVs.
    # This is the authoritative source for those figures/tables.
    "figures_1_2_6_and_table_2": True,

    # VaR surface and term structure -> Figures 3, 7.
    # This is the authoritative source for those two figures only. Duplicated
    # Figure 1/2/6 outputs from this script are cosmetic diagnostics; skip
    # the flag above if you only want them from this script.
    "figures_3_7_var_surface_and_termstruct": True,
}


# ============================================================================
# Step definitions - script path + short human label
# ============================================================================
STEPS = [
    (
        "table_1_pmle_calibration",
        "Table 1: P-MLE parameter estimates",
        _REPO_ROOT / "Scripts" / "run_pmle_kimyi2025.py",
    ),
    # Skew calibration is a three-step pipeline:
    #   (a) Kim-Yi QLSQ  -> Study/Estimated Parameters QLSQ/kimyi2025_vol_calibration.parquet
    #   (b) Heston QLSQ  -> Study/Estimated Parameters QLSQ/heston1993_vol_calibration.parquet
    #   (c) plot         -> Figures/Figure_4_*.pdf, Figure_5_*.pdf
    # All three are gated by the same CONFIG flag; running only step (c) would
    # crash if the two parquets are missing.
    (
        "figures_4_5_skew_calibration",
        "Figures 4-5 (a): Kim-Yi QLSQ skew calibration",
        _REPO_ROOT / "Scripts" / "skew_calibration_kimyi2025.py",
    ),
    (
        "figures_4_5_skew_calibration",
        "Figures 4-5 (b): Heston (1993) QLSQ skew calibration",
        _REPO_ROOT / "Scripts" / "skew_calibration_heston1993.py",
    ),
    (
        "figures_4_5_skew_calibration",
        "Figures 4-5 (c): render market vs. calibrated skew plots",
        _REPO_ROOT / "Scripts" / "skew_calibration_main.py",
    ),
    (
        "figures_1_2_6_and_table_2",
        "Figures 1, 2, 6 + Table 2 CSVs: Collar / Asian portfolio pipeline",
        _REPO_ROOT / "Study" / "Collar Asian" / "report_collar_asian.py",
    ),
    (
        "figures_3_7_var_surface_and_termstruct",
        "Figures 3, 7: VaR surface and term structure",
        _REPO_ROOT / "Scripts" / "run_var_kimyi2025.py",
    ),
]


def _run_step(label: str, script_path: Path) -> None:
    """Execute one pipeline script as a subprocess, streaming its output."""
    logger.info("=" * 72)
    logger.info("STEP: %s", label)
    logger.info("      %s", script_path)
    logger.info("=" * 72)
    t0 = time.perf_counter()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(_REPO_ROOT),
        env=env,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        logger.error(
            "STEP FAILED (returncode=%d, elapsed=%.1fs): %s",
            result.returncode, elapsed, label,
        )
        raise SystemExit(result.returncode)
    logger.info("STEP OK (elapsed=%.1fs): %s", elapsed, label)


def main():
    logger.info("Reproducing paper outputs from %s", _REPO_ROOT)
    logger.info("CONFIG: %s", CONFIG)

    enabled = [(key, label, path) for (key, label, path) in STEPS if CONFIG.get(key, False)]
    skipped = [key for key, _, _ in STEPS if not CONFIG.get(key, False)]

    if not enabled:
        logger.warning("All steps are disabled in CONFIG. Nothing to do.")
        return

    logger.info("Will run %d step(s); skipping %d.", len(enabled), len(skipped))
    if skipped:
        logger.info("Skipped: %s", skipped)

    t_total = time.perf_counter()
    for _key, label, path in enabled:
        if not path.exists():
            logger.error("Missing script for step %r: %s", _key, path)
            raise SystemExit(2)
        _run_step(label, path)

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "Reproduction complete: %d step(s) in %.1fs total.",
        len(enabled), total_elapsed,
    )


if __name__ == "__main__":
    main()
