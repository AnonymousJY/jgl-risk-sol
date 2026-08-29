"""
Scripts/config_skew.py — inputs for Scripts/skew_calibrate_systematic.py and
Scripts/skew_calibrate_idiosyncratic.py.

Mirrors the systematic- and idiosyncratic-stage calibration cells in
Scripts/skew_calibration_kimyi2025.ipynb, but as a set of editable knobs
instead of hard-coded notebook constants, so the calibration can be re-run
from the command line and incrementally extended to new dates.

This file is the *source of truth* for which underlyings the pipeline
calibrates. Adding a new idiosyncratic name is a one-place edit:

    IDIOSYNCRATIC_UNDERLYINGS["AAPL"] = {
        "dividend_yield_pct": 0.5,
        "data_path": None,
        "display_name": "Apple Inc.",
    }

Scripts/load_portfolio.get_idiosyncratic_ids() reads from
IDIOSYNCRATIC_UNDERLYINGS below, so downstream P-MLE / VaR / collar-analysis
code picks up the new name automatically -- no other file has to change.

This module has NO imports from application code (Scripts/, Library/, Study/):
those modules import from here, not the other way round. That inversion is
what lets IDIOSYNCRATIC_UNDERLYINGS be the single source of truth.

Edit these values to point at real data / a new sample window; the
calibration logic in the scripts themselves should not need to change.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives in Scripts/, so the repo root is one level up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root folder holding one subfolder per underlying (e.g. "SPX/", "COIN/"),
# each containing daily implied-vol CSV snapshots (quote_date, expiration,
# strike, option_type, active_underlying_price_1545,
# implied_volatility_1545, trade_volume, underlying_symbol, ...).
# Same layout / location as DATA_PATH_STR in skew_calibration_kimyi2025.ipynb.
#
# Only consulted when a given underlying's "data_path" is None -- see
# SYSTEMATIC_UNDERLYING / IDIOSYNCRATIC_UNDERLYINGS below.
DATA_PATH_IMPLIED_VOL = os.path.expanduser(
    "~/Data/ImpliedVolatility"
)

# Where calibration results are cached: a wide Parquet table, one row per
# (ticker, valuation date, tenor), columns = QLSQ_PARAM_ORDER (see
# skew_calibrate_systematic.py) plus bCONVERGED / dOBJECTIVE / iN_OBS. Mirrors
# the wide-table convention Library/DataAccess.py already uses for the P-MLE
# CSVs (Study/Estimated Parameters PMLE/): a plain, typed, columnar table
# rather than a pickle of scipy OptimizeResult objects -- readable from R /
# Julia / Spark, stable across scipy/numpy versions, and free of pickle's
# arbitrary-code-on-load risk. This script and skew_calibrate_idiosyncratic.py
# share this file incrementally.
OUTPUT_DIR = os.path.join(_REPO_ROOT, "Study", "Estimated Parameters QLSQ")
OUTPUT_PARQUET_NAME = "kimyi2025_vol_calibration.parquet"

# Used instead of OUTPUT_PARQUET_NAME whenever USE_SYNTHETIC_DATA is True, so
# a synthetic smoke test can never read from or write into the real cache
# above (real cache keys could otherwise collide with synthetic test dates
# and be mistaken for "already calibrated").
OUTPUT_PARQUET_NAME_SYNTHETIC = "kimyi2025_vol_calibration_synthetic_smoketest.parquet"

# False (default): skip any (ticker, date, tenor) already present in the
# cache -- a rerun only fills in triples that haven't been calibrated yet.
# True: recalibrate and overwrite every (ticker, date, tenor) in the
# requested window, even ones already cached. Applies to both
# skew_calibrate_systematic.py and skew_calibrate_idiosyncratic.py.
OVERWRITE_EXISTING = False

# ---------------------------------------------------------------------------
# Data source (input schema)
# ---------------------------------------------------------------------------
# Which loader converts the raw files at each underlying's data_path into
# the canonical DataFrame the calibrators consume. All loaders return the
# same columns (see Scripts/templates/canonical_option_data.csv for the
# schema), so the rest of the pipeline is agnostic to what's on disk.
#
#   "canonical": read CSV or Parquet whose columns already match the
#                canonical schema exactly. Simplest path -- the user (or
#                an upstream ETL) is responsible for producing that shape.
#   "cboe":      read one folder per underlying of daily CBOE CSV snapshots
#                and reshape to canonical (the notebook's original layout;
#                DATA_PATH_IMPLIED_VOL / <ticker>/*.csv).
#
# Extend LOADERS in skew_calibrate_systematic.py with additional adapters
# (OptionMetrics, Databento, ORATS, ...) as needed.
DATA_SOURCE = "cboe"

# ---------------------------------------------------------------------------
# Underlyings
# ---------------------------------------------------------------------------
# Exactly one systematic underlying (the market factor). Fitting the
# systematic stage against a different index (e.g. NDX for a Nasdaq-heavy
# idiosyncratic basket) is a one-line change here.
#
# Keys:
#   ticker              CBOE / OptionMetrics ticker as it appears in the data.
#   dividend_yield      Continuous dividend yield as a decimal (e.g. 0.0125
#                       for 1.25%). Matches DIVIDEND_YIELDS in
#                       skew_calibration_kimyi2025.ipynb after the notebook's
#                       "/ 100.0" step. Same convention as FLAT_RATE below,
#                       so no unit-mismatch bugs across the config.
#   data_path           If None: fall back to <DATA_PATH_IMPLIED_VOL>/<ticker
#                       without-caret> for the CBOE adapter, or raise for
#                       the canonical adapter. If set: an explicit path
#                       (a folder for CBOE, a file for canonical CSV/Parquet).
#   display_name        Human-readable label (used in log lines).
SYSTEMATIC_UNDERLYING = {
    "ticker": "^SPX",
    "dividend_yield": 0.0125,   # 1.25% as decimal
    "data_path": None,
    "display_name": "S&P 500 Index",
}

# Any number of idiosyncratic underlyings, calibrated conditional on the
# systematic parameters cached at the (SYSTEMATIC_UNDERLYING["ticker"], date,
# tenor) key. To add a new name, add an entry here -- Scripts/load_portfolio.
# get_idiosyncratic_ids() reads from this dict, so downstream code (P-MLE,
# VaR, collar analysis) picks it up automatically.
#
# Same schema as SYSTEMATIC_UNDERLYING, minus the "ticker" field (the dict
# key is the ticker).
IDIOSYNCRATIC_UNDERLYINGS = {
    "COIN": {
        "dividend_yield": 0.0,
        "data_path": None,
        "display_name": "Coinbase Global",
    },
    # Example additions (commented so the smoke test doesn't fail if the
    # data isn't present on disk). Dividend yields are decimals, not
    # percentage points -- 0.005 = 0.5%, matches FLAT_RATE convention.
    # "AAPL": {"dividend_yield": 0.005, "data_path": None, "display_name": "Apple Inc."},
    # "NVDA": {"dividend_yield": 0.0,   "data_path": None, "display_name": "NVIDIA"},
    # "TSLA": {"dividend_yield": 0.0,   "data_path": None, "display_name": "Tesla"},
}

# --- Flat-name shims exposing the underlying-config dicts as top-level
# constants, so migrate_qlsq_cache_to_parquet.py, run_pmle_kimyi2025.py, and
# the calibration scripts can keep using cfg.SYSTEMATIC_TICKER /
# cfg.IDIOSYNCRATIC_TICKERS without knowing about the nested dict shape.
# Derived from the dicts above; edit the dicts, not the shims.
#
# The dividend-yield shims are decimals (0.0125 = 1.25%), matching FLAT_RATE
# below. They replaced the older DIVIDEND_YIELD_PCT / DIVIDEND_YIELD_PCT_
# IDIOSYNCRATIC constants, which were percentage points and forced the
# calibration scripts to /100.0 at every call site -- a cross-file unit
# mismatch that was overdue to be cleaned up. ---
SYSTEMATIC_TICKER = SYSTEMATIC_UNDERLYING["ticker"]
DIVIDEND_YIELD = SYSTEMATIC_UNDERLYING["dividend_yield"]
IDIOSYNCRATIC_TICKERS = list(IDIOSYNCRATIC_UNDERLYINGS.keys())
DIVIDEND_YIELD_IDIOSYNCRATIC = {
    ticker: meta["dividend_yield"]
    for ticker, meta in IDIOSYNCRATIC_UNDERLYINGS.items()
}

# ---------------------------------------------------------------------------
# Sample window
# ---------------------------------------------------------------------------
VALUATION_DATE_BEG = "2025-03-18"
VALUATION_DATE_END = "2025-04-17"

# ---------------------------------------------------------------------------
# Option filters (applied before calibration, matching the notebook)
# ---------------------------------------------------------------------------
MIN_TRADE_VOLUME = 0     # keep rows with trade_volume > MIN_TRADE_VOLUME
MIN_EXPIRY_DAYS = 6      # keep rows with iEXPIRY > MIN_EXPIRY_DAYS
MONEYNESS_PUT_RANGE = (50.0, 100.0)   # puts:  50% <= moneyness <= 100%
MONEYNESS_CALL_RANGE = (100.0, 150.0)  # calls: 100% <= moneyness <= 150%

# ---------------------------------------------------------------------------
# Tenor selection
# ---------------------------------------------------------------------------
# Which tenors to calibrate on each valuation date. Applied AFTER the option
# filters above and BEFORE the MIN_OPTIONS_PER_TENOR check -- so a chosen
# tenor is still dropped if too few option points survive filtering to fit
# stably.
#
#   "all"    -- every distinct iEXPIRY in that day's filtered chain (default).
#   "range"  -- every iEXPIRY within [TENOR_RANGE_MIN, TENOR_RANGE_MAX]
#               (inclusive, calendar days).
#   "list"   -- for each target in TENOR_LIST_DAYS, pick that day's tenor
#               with iEXPIRY closest to the target within
#               TENOR_TOLERANCE_DAYS. Target skipped for that day if nothing
#               is within tolerance. Prevents jagged term-structure series on
#               days without an exact-match expiry (e.g. asking for "30-day"
#               on a Friday whose nearest expiry is 28 days).
#
# The cache key stores the *actual* iEXPIRY that was calibrated, not the
# target. If you ask for 30 and the nearest match is 28, the row records
# iEXPIRY = 28.
TENOR_MODE = "all"

TENOR_RANGE_MIN = 7          # used only when TENOR_MODE == "range"
TENOR_RANGE_MAX = 365

TENOR_LIST_DAYS = [30, 90]   # used only when TENOR_MODE == "list"
TENOR_TOLERANCE_DAYS = 7     # +/- days around each target considered a match

# Minimum surviving option rows for a tenor to be calibrated. Applies in all
# three tenor modes.
MIN_OPTIONS_PER_TENOR = 5

# ---------------------------------------------------------------------------
# Risk-free rate curve
# ---------------------------------------------------------------------------
# "ust_curve": pull the UST par curve via ustreasurycurve and NSS-fit it
#              per valuation date (requires network + nelson_siegel_svensson).
# "flat":      use FLAT_RATE for every date/expiry (no network; for dev/test
#              or when a curve fetch isn't available).
#
# Set to "flat" for now: the installed ustreasurycurve's nominalRates()
# converts its result to pandas (df = df.to_pandas()) and then tries to
# filter that pandas object with polars syntax (df.filter(pl.col(...) >= ...))
# a few lines later -- a genuine bug in that package (a pandas/polars
# migration left half-finished), not an environment issue. Switch back to
# "ust_curve" if/when that's fixed upstream or pinned to a working version.
RATE_SOURCE = "flat"
FLAT_RATE = 0.05

# ---------------------------------------------------------------------------
# Kim-Yi (2025) systematic calibration: sigma, pprob, lamb, eta1, eta2
# ---------------------------------------------------------------------------
# Multi-start baselines representing different market regimes. calibrate_
# date_systematic() runs SLSQP from each in turn and keeps the best result.
# Necessary because SLSQP is a local solver, and a single fixed initial
# guess drops into the Newton IV-inversion's numerical-failure region for
# some (date, tenor) combinations -- returning the 1e6 penalty and getting
# stuck. Multi-start gives SLSQP multiple entry points so at least one
# lands in a well-behaved basin.
INITIAL_VALUES_SYSTEMATIC_MULTISTART = [
    np.array([0.15, 0.05,  1.0, 30.0, 10.0]),  # low vol / few jumps  (2013-17)
    np.array([0.20, 0.20,  5.0, 22.0,  7.0]),  # baseline (paper's April 2025 window)
    np.array([0.35, 0.40, 15.0, 15.0,  5.0]),  # high vol regime      (2008, 2020)
    np.array([0.25, 0.30, 10.0, 12.0,  3.0]),  # heavy-tail regime
    np.array([0.10, 0.90, 30.0,  5.0,  2.0]),  # extreme jump-dominated (crisis / ultra-short-dated)
]

# Back-compat alias -- kept so the migrate script and any other consumer
# that imports the singular name still works. Points at the baseline.
INITIAL_VALUES_SYSTEMATIC = INITIAL_VALUES_SYSTEMATIC_MULTISTART[1]

BOUNDS_SYSTEMATIC = {
    "dSIGMA": (1e-4, None),
    "dPPROB": (0.0, 1.0),
    "dLAMB": (1e-4, None),
    "dETA1": (1.5, None),
    "dETA2": (0.5, None),
}

# ---------------------------------------------------------------------------
# Kim-Yi (2025) idiosyncratic calibration: kappai, gammai, betai, rhoix
#
# Fit per idiosyncratic underlying, *conditional* on that (date, tenor)'s
# systematic parameters (sigma, pprob, lamb, eta1, eta2) -- see
# Scripts/skew_calibrate_idiosyncratic.py, which reads those systematic
# parameters from the cache Scripts/skew_calibrate_systematic.py writes to
# (OUTPUT_DIR / OUTPUT_PARQUET_NAME). A (date, tenor) must be calibrated
# systematically before it can be calibrated idiosyncratically.
# ---------------------------------------------------------------------------
# Multi-start baselines for the idiosyncratic (kappai, gammai, betai,
# rhoix) fit -- same rationale as INITIAL_VALUES_SYSTEMATIC_MULTISTART.
# calibrate_date_idiosyncratic() iterates over these and keeps the best
# converged result.
INITIAL_VALUES_IDIOSYNCRATIC_MULTISTART = [
    np.array([1.0, 2.0, 1.0,  0.5]),   # baseline (paper's April 2025 window)
    np.array([0.3, 1.5, 0.8,  0.0]),   # low idio, uncorrelated
    np.array([2.0, 3.0, 1.5, -0.3]),   # high idio, negative rho
    np.array([0.5, 1.2, 0.5,  0.3]),   # low idio, positive rho
]

# Back-compat alias -- kept for consumers that import the singular name.
INITIAL_VALUES_IDIOSYNCRATIC = INITIAL_VALUES_IDIOSYNCRATIC_MULTISTART[0]

BOUNDS_IDIOSYNCRATIC = {
    "dKAPPAI": (1e-4, None),
    "dGAMMAI": (1e-1, None),
    "dBETAI": (1e-1, None),
    "dRHOIX": (-1.0, 1.0),
}

OPTIMIZER_METHOD = "SLSQP"
OPTIMIZER_TOL = 1e-6
OPTIMIZER_MAXITER = 10_000

# ---------------------------------------------------------------------------
# Objective weighting
# ---------------------------------------------------------------------------
# How to weight the per-strike IV residuals in the least-squares objective:
#     objective = 0.5 * sum( weight_i * (mkt_iv_i - mod_iv_i)^2 )
#
#   "vega"      -- BSM put vega at each strike (dPnL / dsigma). Approximately
#                  converts the IV residuals into price residuals. Standard
#                  in academic vol-surface calibration, but concentrates
#                  ~90% of the loss in the narrow ATM band (vega drops off
#                  exponentially in log-moneyness), so SLSQP can find a
#                  global minimum that nails ATM at the cost of deep-OTM.
#   "sqrt_vega" -- sqrt(vega). Halves the ATM dominance (in log-space).
#                  Compromise between the "prices are what matter"
#                  argument (vega) and the "smile should look right
#                  everywhere" argument (uniform). Recommended default:
#                  restores visually balanced fits without abandoning
#                  the price-space justification.
#   "uniform"   -- every strike counts equally. Best visual coverage of
#                  the wings; may slightly degrade ATM fit and is more
#                  sensitive to noisy deep-OTM quotes (thin liquidity).
OBJECTIVE_WEIGHTING = "sqrt_vega"

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
# Number of worker processes used by both calibration scripts.
#   1  -- sequential path (no ProcessPoolExecutor; easiest to attach pdb).
#   0  -- default to os.cpu_count() - 1 at runtime (leave one core for OS).
#   >1 -- explicit worker count.
#
# The --n-jobs CLI flag overrides this value. Each worker forces
# single-threaded BLAS (OMP_NUM_THREADS=1 etc.) via a ProcessPoolExecutor
# initializer to avoid the classic MKL/OpenBLAS oversubscription trap --
# without that, running N worker processes each with N BLAS threads makes
# parallel calibration SLOWER than sequential.
N_JOBS_DEFAULT = 1

# How many completed calibrations to accumulate in memory before checkpoint-
# saving to Parquet. Only the main process ever writes -- workers return row
# dicts, the main loop aggregates and saves -- so this is a crash-resilience
# knob, not a correctness one. Lower = more frequent I/O; higher = fewer
# saves, more work at risk from a mid-run crash.
CHECKPOINT_EVERY_DEFAULT = 50

# ---------------------------------------------------------------------------
# Development / testing
# ---------------------------------------------------------------------------
# If True, the scripts skip DATA_PATH_IMPLIED_VOL and RATE_SOURCE entirely
# and calibrate against synthetic option data generated from a known set of
# "true" parameters (via the real KimYiSkewCalibration* models), so the
# optimization path can be validated end-to-end without market data.
#
# Set to False to run against the real data. Verified against the real
# SPX/COIN CSVs (columns quote_date, expiration, strike, option_type,
# active_underlying_price_1545, implied_volatility_1545, trade_volume,
# underlying_symbol match load_cboe() exactly) -- not executed end-to-end
# in this environment since a single day's SPX file is ~28k rows, too
# large to shuttle through the sandbox's file-read bridge. Run for real
# locally: `python Scripts/skew_calibrate_systematic.py`.
USE_SYNTHETIC_DATA = False
RANDOM_SEED = 20250409

# The Newton IV-inversion inside KimYiSkewCalibrationSystematic.model_vol is
# not cheap per point: a handful of dates/strikes/tenors is enough to
# smoke-test the optimizer end-to-end. Kept separate from
# VALUATION_DATE_BEG/END so a synthetic smoke test never has to touch the
# production window.
SYNTHETIC_VALUATION_DATE_BEG = "2019-01-02"
SYNTHETIC_VALUATION_DATE_END = "2019-01-04"
SYNTHETIC_N_MONEYNESS = 9

# Days-to-expiry generated per synthetic valuation date, so the multi-tenor
# calibration path (one fit per (date, tenor)) is actually exercised.
SYNTHETIC_TENORS_DAYS = [7, 30, 90]
