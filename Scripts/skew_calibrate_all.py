"""
Scripts/skew_calibrate_all.py -- one-shot orchestrator that runs the
systematic-stage skew calibration then the idiosyncratic-stage calibration
back-to-back, sharing CLI flags between the two.

Equivalent to running the pair in sequence:

    python Scripts/skew_calibrate_systematic.py       [flags]
    python Scripts/skew_calibrate_idiosyncratic.py    [flags + --tickers ...]

but with a single argparse invocation, unified logging, and the
correct execution order (systematic first -- idiosyncratic depends on
its cached parameters).

Flag semantics
--------------
All shared flags (data-source, valuation-date-*, tenor-*, n-jobs,
checkpoint-every, overwrite) forward to BOTH stages. The idiosyncratic-
only flag --tickers forwards only to that stage; the systematic stage
always fits cfg.SYSTEMATIC_UNDERLYING["ticker"] regardless.

Orchestrator-specific flags
---------------------------
--skip-systematic       Skip stage 1 (useful when the systematic cache
                        is already up to date and you just want to
                        (re)compute idiosyncratic).
--skip-idiosyncratic    Skip stage 2 (equivalent to running only the
                        systematic script directly).

Run from the repository root:

    # Full run, defaults
    python Scripts/skew_calibrate_all.py

    # 8-DTE nearest match across the full window, 8 workers, just COIN
    python Scripts/skew_calibrate_all.py \\
        --tenor-mode list --tenors 8 --tenor-tolerance 3 \\
        --valuation-date-beg 2025-03-18 \\
        --valuation-date-end 2025-04-17 \\
        --n-jobs 8 --checkpoint-every 10 \\
        --tickers COIN

    # Only re-run the idiosyncratic stage (systematic cache is fresh)
    python Scripts/skew_calibrate_all.py --skip-systematic --tickers COIN
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config_skew as cfg  # Scripts/config_skew.py
from Scripts.skew_calibrate_systematic import (
    LOADERS,
    configure_logging,
    main as run_systematic,
)
from Scripts.skew_calibrate_idiosyncratic import main as run_idiosyncratic

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Runs the systematic then idiosyncratic Kim-Yi (2025) "
                    "skew calibration in sequence with shared CLI flags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Shared flags (forwarded to both stages) ---
    p.add_argument("--data-source", choices=sorted(LOADERS.keys()),
                   default=cfg.DATA_SOURCE)
    p.add_argument("--valuation-date-beg", default=cfg.VALUATION_DATE_BEG,
                   help="ISO date, inclusive.")
    p.add_argument("--valuation-date-end", default=cfg.VALUATION_DATE_END,
                   help="ISO date, inclusive.")
    p.add_argument("--tenor-mode", choices=["all", "range", "list"],
                   default=cfg.TENOR_MODE)
    p.add_argument("--tenors", nargs="+", type=int,
                   default=cfg.TENOR_LIST_DAYS, metavar="DAYS",
                   help="Target tenors when --tenor-mode list.")
    p.add_argument("--tenor-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                   default=[cfg.TENOR_RANGE_MIN, cfg.TENOR_RANGE_MAX],
                   help="Inclusive tenor bounds when --tenor-mode range.")
    p.add_argument("--tenor-tolerance", type=int,
                   default=cfg.TENOR_TOLERANCE_DAYS, metavar="DAYS")
    p.add_argument("--n-jobs", type=int, default=cfg.N_JOBS_DEFAULT, metavar="N")
    p.add_argument("--checkpoint-every", type=int,
                   default=cfg.CHECKPOINT_EVERY_DEFAULT, metavar="N")
    p.add_argument("--overwrite", action="store_true",
                   default=cfg.OVERWRITE_EXISTING)

    # --- Idiosyncratic-only flag (forwarded to that stage only) ---
    p.add_argument("--tickers", nargs="+", default=None, metavar="TICKER",
                   help="Subset of cfg.IDIOSYNCRATIC_UNDERLYINGS to calibrate. "
                        "Systematic stage always runs on cfg.SYSTEMATIC_"
                        "UNDERLYING['ticker'] regardless.")

    # --- Orchestrator-specific flags ---
    p.add_argument("--skip-systematic", action="store_true",
                   help="Skip stage 1 (systematic). Useful when the "
                        "systematic cache is already up to date.")
    p.add_argument("--skip-idiosyncratic", action="store_true",
                   help="Skip stage 2 (idiosyncratic).")

    return p


def _build_common_forwarded_args(args: argparse.Namespace) -> List[str]:
    """Reconstruct the shared subset of CLI flags for forwarding to each
    stage's main(). We explicitly reconstruct rather than pass args.__dict__
    so that flags with defaults are always explicit downstream (avoids
    surprises when downstream defaults drift)."""
    forwarded: List[str] = [
        "--data-source", args.data_source,
        "--valuation-date-beg", args.valuation_date_beg,
        "--valuation-date-end", args.valuation_date_end,
        "--tenor-mode", args.tenor_mode,
        "--tenor-tolerance", str(args.tenor_tolerance),
        "--n-jobs", str(args.n_jobs),
        "--checkpoint-every", str(args.checkpoint_every),
    ]
    if args.tenor_mode == "list":
        forwarded += ["--tenors"] + [str(t) for t in args.tenors]
    elif args.tenor_mode == "range":
        forwarded += ["--tenor-range", str(args.tenor_range[0]), str(args.tenor_range[1])]
    if args.overwrite:
        forwarded.append("--overwrite")
    return forwarded


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.skip_systematic and args.skip_idiosyncratic:
        raise SystemExit("--skip-systematic and --skip-idiosyncratic cannot both be set.")

    common = _build_common_forwarded_args(args)

    if not args.skip_systematic:
        logger.info("=" * 70)
        logger.info("Stage 1/2: SYSTEMATIC calibration (%s)", cfg.SYSTEMATIC_UNDERLYING["ticker"])
        logger.info("=" * 70)
        run_systematic(common)
    else:
        logger.info("Stage 1/2: SYSTEMATIC skipped (--skip-systematic).")

    if not args.skip_idiosyncratic:
        tickers = args.tickers or list(cfg.IDIOSYNCRATIC_UNDERLYINGS.keys())
        logger.info("=" * 70)
        logger.info("Stage 2/2: IDIOSYNCRATIC calibration (%s)", ", ".join(tickers))
        logger.info("=" * 70)
        idio_args = list(common)
        if args.tickers:
            idio_args += ["--tickers"] + args.tickers
        run_idiosyncratic(idio_args)
    else:
        logger.info("Stage 2/2: IDIOSYNCRATIC skipped (--skip-idiosyncratic).")

    logger.info("Orchestrator done.")


if __name__ == "__main__":
    configure_logging()
    main()
