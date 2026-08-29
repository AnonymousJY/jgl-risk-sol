"""
Scripts/skew_calibrate_systematic.py — systematic-stage skew calibration
(Kim-Yi 2025 model), scripted from Scripts/skew_calibration_kimyi2025.ipynb.

For each valuation date in the configured window and each tenor selected
under cfg.TENOR_MODE, fits the five common Kim-Yi (2025) parameters (sigma,
pprob, lamb, eta1, eta2) to the systematic underlying's implied-vol smile
via vega-weighted least squares (Library.SkewCalibrationKimYi2025.
KimYiSkewCalibrationSystematic + scipy.optimize.minimize). A tenor is
dropped if it has fewer than cfg.MIN_OPTIONS_PER_TENOR rows.

Data input is *vendor-agnostic*: LOADERS[cfg.DATA_SOURCE] chooses the
adapter that reshapes the raw files at each underlying's data_path into a
canonical DataFrame the rest of the pipeline consumes. Two adapters ship
built in:

    "cboe"      -- one folder per underlying of daily CBOE CSV snapshots
                   (columns quote_date, expiration, strike, option_type,
                   active_underlying_price_1545, implied_volatility_1545,
                   trade_volume, underlying_symbol). Original notebook layout.
    "canonical" -- a single CSV or Parquet file whose columns already match
                   the canonical schema (see Scripts/templates/
                   canonical_option_data.csv). No vendor-specific reshaping.

Add more (OptionMetrics, Databento, ORATS, ...) by writing a function that
returns a canonical DataFrame and registering it in LOADERS.

Results are cached, incrementally, to a wide Parquet table (one row per
(ticker, valuation date, tenor); see QLSQ_PARAM_ORDER for the parameter
columns) at cfg.OUTPUT_DIR / OUTPUT_PARQUET_NAME. Cached triples are
skipped unless --overwrite is passed (or cfg.OVERWRITE_EXISTING is True).

Parallelism: --n-jobs > 1 runs calibrations across a ProcessPoolExecutor
whose workers force single-threaded BLAS (OMP_NUM_THREADS=1 etc.) to avoid
the MKL/OpenBLAS oversubscription trap. --n-jobs 1 (the default) keeps the
plain sequential path -- easiest to attach pdb, no worker-side pickling
overhead, log messages ordered as fits complete.

The idiosyncratic stage (KimYiSkewCalibrationIdiosyncratic, conditional on
these systematic parameters) is intentionally out of scope for this script;
see Scripts/skew_calibrate_idiosyncratic.py.

Run from the repository root:

    # All defaults
    python Scripts/skew_calibrate_systematic.py

    # Vendor-agnostic input, 8 parallel workers, 30d + 90d only
    python Scripts/skew_calibrate_systematic.py \\
        --data-source canonical \\
        --data-path ~/data/spx_2007_2025.parquet \\
        --tenor-mode list --tenors 30 90 --tenor-tolerance 5 \\
        --n-jobs 8 --checkpoint-every 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config_skew as cfg  # Scripts/config_skew.py
from Library.OptionPricerBSM1973 import BlackScholesMertonPut
from Library.SkewCalibrationKimYi2025 import KimYiSkewCalibrationSystematic

# tqdm is nice-to-have; the fits still run without it. Kept optional so a
# fresh checkout doesn't hard-crash on missing packages before the user
# activates the conda env.
try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    def _tqdm(iterable, **kwargs):
        return iterable

# Module logger only -- no handler/level is configured here. Configuration
# (level, format, handlers) is the entry point's job; see configure_logging()
# and the __main__ guard below. Importing this module elsewhere (tests,
# notebooks, other scripts) must not silently install a root handler.
logger = logging.getLogger(__name__)

SYSTEMATIC_PARAM_NAMES = ["dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]

# Kim-Yi (2025) parameters fixed (not calibrated) for the systematic proxy --
# see KimYiSkewCalibrationSystematic.model_vol, which hardcodes exactly these
# values. Stored on every systematic row so the cache's schema is self-
# describing without cross-referencing the model code.
SYSTEMATIC_FIXED_PARAMS = {"dKAPPAI": 0.0, "dGAMMAI": 1.0, "dBETAI": 1.0, "dRHOIX": 0.0}

# Owns the shared wide-table schema: skew_calibrate_idiosyncratic.py imports
# IDIOSYNCRATIC_PARAM_NAMES and QLSQ_PARAM_ORDER from here rather than
# redefining them, so there is one source of truth for the cache's columns.
# Ordering mirrors Library.DataAccess.PMLE_PARAM_ORDER: idiosyncratic-only
# parameters first, then the parameters shared with the systematic proxy.
IDIOSYNCRATIC_PARAM_NAMES = ["dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX"]
QLSQ_PARAM_ORDER = IDIOSYNCRATIC_PARAM_NAMES + SYSTEMATIC_PARAM_NAMES
CACHE_COLUMNS = (
    ["sTICKER", "sVALUATION_DATE", "iEXPIRY", "dEXPIRY"]
    + QLSQ_PARAM_ORDER
    + ["bCONVERGED", "dOBJECTIVE", "iN_OBS"]
)


# ---------------------------------------------------------------------------
# Canonical schema (what every loader has to return)
# ---------------------------------------------------------------------------
# Required columns of the canonical DataFrame consumed by the rest of the
# pipeline. Loader adapters (load_cboe, load_canonical, ...) are responsible
# for producing this shape -- whatever the input file looks like, the output
# has these columns and units.
#
# Yields and rates are decimals (0.0125 = 1.25%), matching config_skew.py.
# See Scripts/templates/canonical_option_data.csv for a headers-only file
# users can populate to run the "canonical" adapter without any vendor code.
CANONICAL_COLUMNS = [
    "quote_date",         # pd.Timestamp    valuation date of the smile
    "underlying_symbol",  # str             ticker (e.g. "^SPX", "COIN")
    "dUND_PRICE",         # float           spot price
    "dUND_STRIKE",        # float           strike price
    "iEXPIRY",            # int             days to expiry
    "dEXPIRY",            # float           years to expiry (iEXPIRY / 365)
    "bIS_CALL_OPTION",    # bool            True for calls, False for puts
    "dMKT_IMP_VOL",       # float           implied vol, decimal (0.20 = 20%)
    "dDIVIDEND_YIELD",    # float           decimal (0.0125 = 1.25%)
]
# dRISK_FREE_RATE is optional in the raw input: the CBOE adapter doesn't set
# it (attach_risk_free_rate() populates it downstream from RATE_SOURCE),
# whereas the canonical adapter expects it to be already in the file.
# dMONEYNESS and dVEGA are always computed by the pipeline, never required
# in the input.


def _apply_option_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the config's option filters (min volume, min expiry, moneyness
    bounds). Shared by every loader so filter semantics are identical
    regardless of the input vendor."""
    if "dMONEYNESS" not in df.columns:
        df["dMONEYNESS"] = df["dUND_STRIKE"] / df["dUND_PRICE"] * 100.0

    # trade_volume is optional -- CBOE has it, canonical typically doesn't.
    # Absent = no volume filter (equivalent to trade_volume = +inf).
    if "trade_volume" in df.columns:
        volume_ok = df["trade_volume"] > cfg.MIN_TRADE_VOLUME
    else:
        volume_ok = pd.Series(True, index=df.index)

    mask = volume_ok & (df["iEXPIRY"] > cfg.MIN_EXPIRY_DAYS) & (df["dMKT_IMP_VOL"] > 0)
    df = df.loc[mask]

    put_lo, put_hi = cfg.MONEYNESS_PUT_RANGE
    call_lo, call_hi = cfg.MONEYNESS_CALL_RANGE
    is_put_in_range = (~df["bIS_CALL_OPTION"]) & df["dMONEYNESS"].between(put_lo, put_hi)
    is_call_in_range = df["bIS_CALL_OPTION"] & df["dMONEYNESS"].between(call_lo, call_hi)
    return df.loc[is_put_in_range | is_call_in_range].copy()


def _resolve_data_path(underlying_meta: dict, ticker: str, fallback_folder: bool) -> str:
    """Return the on-disk data path for one underlying, resolving the
    convention that ``data_path=None`` falls back to
    ``<DATA_PATH_IMPLIED_VOL>/<ticker-without-caret>/``. Only meaningful for
    the CBOE adapter -- the canonical adapter has no folder convention to
    fall back to and rejects None."""
    data_path = underlying_meta.get("data_path")
    if data_path is not None:
        return os.path.expanduser(data_path)
    if not fallback_folder:
        raise ValueError(
            f"data_path is None for {ticker!r} and this adapter has no "
            f"folder fallback. Set 'data_path' explicitly in "
            f"config_skew.SYSTEMATIC_UNDERLYING or IDIOSYNCRATIC_UNDERLYINGS."
        )
    safe_name = ticker.split("^")[-1]
    return os.path.join(cfg.DATA_PATH_IMPLIED_VOL, safe_name)


# ---------------------------------------------------------------------------
# Loader adapters
# ---------------------------------------------------------------------------
def load_cboe(underlying_meta: dict, ticker: str) -> pd.DataFrame:
    """Read a folder of daily CBOE CSV snapshots for one ticker and reshape
    to the canonical schema. Preserves the same transformations
    Scripts/skew_calibration_kimyi2025.ipynb applied inline in its data
    cell (via load_market_vol_data)."""
    path = _resolve_data_path(underlying_meta, ticker, fallback_folder=True)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"No implied-vol data folder at {path}. Point "
            f"config_skew.DATA_PATH_IMPLIED_VOL (or this underlying's "
            f"data_path) at your data, or set USE_SYNTHETIC_DATA = True to "
            f"run against synthetic data."
        )

    files = sorted(os.listdir(path))
    df = pd.concat((pd.read_csv(os.path.join(path, f)) for f in files), ignore_index=True)

    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["iEXPIRY"] = (df["expiration"] - df["quote_date"]).dt.days
    df["dEXPIRY"] = df["iEXPIRY"] / 365.0
    df["dUND_PRICE"] = df["active_underlying_price_1545"]
    df["dUND_STRIKE"] = df["strike"]
    df["dMONEYNESS"] = df["dUND_STRIKE"] / df["dUND_PRICE"] * 100.0
    df["dMKT_IMP_VOL"] = df["implied_volatility_1545"]
    df["bIS_CALL_OPTION"] = df["option_type"] == "C"
    df["dDIVIDEND_YIELD"] = underlying_meta["dividend_yield"]   # decimal

    return _apply_option_filters(df)


def load_canonical(underlying_meta: dict, ticker: str) -> pd.DataFrame:
    """Read a CSV or Parquet whose columns already match the canonical
    schema (see Scripts/templates/canonical_option_data.csv). No
    vendor-specific reshaping -- the user (or an upstream ETL step) is
    responsible for producing this shape."""
    path = _resolve_data_path(underlying_meta, ticker, fallback_folder=False)
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["quote_date"])
    df["quote_date"] = pd.to_datetime(df["quote_date"])

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Canonical file {path} missing required columns: {missing}. "
            f"See Scripts/templates/canonical_option_data.csv for the "
            f"expected header."
        )

    # underlying_symbol scoping: canonical files might mix multiple tickers.
    df = df.loc[df["underlying_symbol"] == ticker].copy()
    if df.empty:
        raise ValueError(
            f"Canonical file {path} contains no rows with "
            f"underlying_symbol == {ticker!r}. Check the ticker string "
            f"(e.g. '^SPX' vs 'SPX')."
        )

    return _apply_option_filters(df)


# Dispatch table for --data-source / cfg.DATA_SOURCE. Add new adapters here.
LOADERS = {
    "cboe": load_cboe,
    "canonical": load_canonical,
}


# ---------------------------------------------------------------------------
# Risk-free rate and vega attachment
# ---------------------------------------------------------------------------
def attach_risk_free_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Populate df['dRISK_FREE_RATE'] per row, by valuation date + expiry.
    Idempotent: if the column already exists (e.g. a canonical input file
    already carries it), this is a no-op."""
    if "dRISK_FREE_RATE" in df.columns and df["dRISK_FREE_RATE"].notna().all():
        return df

    if cfg.RATE_SOURCE == "flat":
        df = df.copy()
        df["dRISK_FREE_RATE"] = cfg.FLAT_RATE
        return df

    if cfg.RATE_SOURCE != "ust_curve":
        raise ValueError(f"Unknown RATE_SOURCE: {cfg.RATE_SOURCE!r}")

    import ustreasurycurve as ustcurve
    from nelson_siegel_svensson.calibrate import calibrate_nss_ols

    from Library.Utility import UST_TENOR_MAP

    df = df.copy()
    dates = df["quote_date"].unique()
    rates_df = ustcurve.nominalRates(
        pd.Timestamp(dates.min()).strftime("%Y-%m-%d"),
        pd.Timestamp(dates.max()).strftime("%Y-%m-%d"),
    ).set_index("date")
    rates_df = (
        pd.DataFrame(index=pd.date_range(rates_df.index.min(), rates_df.index.max()))
        .join(rates_df)
        .ffill(axis=1)
        .bfill(axis=1)
        .ffill()
        .bfill()
    )
    tenors = np.array([UST_TENOR_MAP[c] for c in rates_df.columns])

    for quote_date in dates:
        curve_fit, _ = calibrate_nss_ols(tenors, rates_df.xs(quote_date).to_numpy() / 100.0)
        mask = df["quote_date"] == quote_date
        df.loc[mask, "dRISK_FREE_RATE"] = df.loc[mask, "dEXPIRY"].apply(curve_fit)

    return df


def attach_vega_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Populate df['dVEGA'] with per-row Black-Scholes-Merton put vega
    (used as the option_weights passed into KimYiSkewCalibrationSystematic).
    Idempotent: skipped if the column already exists and has no NaNs."""
    if "dVEGA" in df.columns and df["dVEGA"].notna().all():
        return df

    vegas = np.empty(len(df))
    for i, row in enumerate(df.itertuples()):
        vega = BlackScholesMertonPut(
            und_price=np.array(row.dUND_PRICE),
            und_strike=np.array(row.dUND_STRIKE),
            risk_free_rate=np.array(row.dRISK_FREE_RATE),
            dividend_yield=np.array(row.dDIVIDEND_YIELD),
            time_to_expiry=np.array(row.dEXPIRY),
        ).vega(np.array(row.dMKT_IMP_VOL))
        vegas[i] = vega.reshape(-1)[0] / 100.0
    df = df.copy()
    df["dVEGA"] = vegas
    return df


# ---------------------------------------------------------------------------
# Tenor selection (all / range / list)
# ---------------------------------------------------------------------------
def select_tenors(
    available: Iterable[int],
    mode: str,
    range_min: int,
    range_max: int,
    target_days: List[int],
    tolerance_days: int,
) -> List[int]:
    """Return the iEXPIRY values to calibrate on one valuation date.

    Modes:
      "all"    every distinct iEXPIRY in ``available``.
      "range"  every iEXPIRY within [range_min, range_max] inclusive.
      "list"   for each target in target_days, the available iEXPIRY closest
               to it within tolerance_days (nearest match). Ties broken by
               the smaller iEXPIRY (dict-order in the sort). A target with
               no available tenor within tolerance is skipped. Duplicate
               picks (two targets both mapping to the same actual iEXPIRY)
               are deduped so the same fit isn't queued twice.
    """
    available = sorted(int(t) for t in available)
    if not available:
        return []

    if mode == "all":
        return available

    if mode == "range":
        return [t for t in available if range_min <= t <= range_max]

    if mode == "list":
        picked: List[int] = []
        seen: set = set()
        for target in target_days:
            # Nearest available within tolerance.
            best = None
            best_dist = tolerance_days + 1
            for t in available:
                dist = abs(t - target)
                if dist <= tolerance_days and dist < best_dist:
                    best = t
                    best_dist = dist
            if best is not None and best not in seen:
                picked.append(best)
                seen.add(best)
        return picked

    raise ValueError(
        f"Unknown tenor mode: {mode!r}. Expected one of 'all', 'range', 'list'."
    )


# ---------------------------------------------------------------------------
# Synthetic data (dev / test, no market data or network required)
# ---------------------------------------------------------------------------
TRUE_PARAMS_SYNTHETIC = {
    "dSIGMA": 0.18, "dPPROB": 0.25, "dLAMB": 6.0, "dETA1": 20.0, "dETA2": 8.0,
}


def _make_synthetic_market_vol_data(ticker: str) -> pd.DataFrame:
    """Synthesize an implied-vol smile per (valuation date, tenor), generated
    from TRUE_PARAMS_SYNTHETIC via the real KimYiSkewCalibrationSystematic
    model (plus noise), so the per-tenor calibration path can be validated
    end-to-end. Tenors come from config_skew.SYNTHETIC_TENORS_DAYS."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    dates = pd.bdate_range(cfg.SYNTHETIC_VALUATION_DATE_BEG, cfg.SYNTHETIC_VALUATION_DATE_END)

    S0, r, q = 100.0, 0.04, cfg.DIVIDEND_YIELD   # q as decimal, matches r's units
    moneyness = np.linspace(60.0, 140.0, cfg.SYNTHETIC_N_MONEYNESS)
    strikes = moneyness / 100.0 * S0
    is_call = moneyness >= 100.0
    x_true = np.array([TRUE_PARAMS_SYNTHETIC[p] for p in SYSTEMATIC_PARAM_NAMES])

    rows = []
    for quote_date in dates:
        for expiry_days in cfg.SYNTHETIC_TENORS_DAYS:
            T = expiry_days / 365.0

            fitter = KimYiSkewCalibrationSystematic(
                mkt_imp_vol=np.zeros_like(strikes),
                und_price=np.full_like(strikes, S0),
                und_strike=strikes,
                risk_free_rate=np.full_like(strikes, r),
                dividend_yield=np.full_like(strikes, q),
                time_to_expiry=np.full_like(strikes, T),
                is_call_option=is_call,
                option_weights=np.ones_like(strikes),
            )
            true_vol = fitter.model_vol(x=x_true).reshape(-1)
            noisy_vol = true_vol + rng.normal(0, 0.002, size=true_vol.shape[0])

            # model_vol() stacks puts then calls (see SkewCalibrationKimYi2025.py)
            ordered_is_call = np.concatenate([is_call[~is_call], is_call[is_call]])
            ordered_strike = np.concatenate([strikes[~is_call], strikes[is_call]])

            for k, c, v in zip(ordered_strike, ordered_is_call, noisy_vol):
                rows.append({
                    "quote_date": quote_date,
                    "underlying_symbol": ticker,
                    "iEXPIRY": expiry_days,
                    "dEXPIRY": T,
                    "dUND_PRICE": S0,
                    "dUND_STRIKE": k,
                    "dMONEYNESS": k / S0 * 100.0,
                    "dMKT_IMP_VOL": max(v, 1e-4),
                    "bIS_CALL_OPTION": bool(c),
                    "dRISK_FREE_RATE": r,
                    "dDIVIDEND_YIELD": q,
                    "dVEGA": 1.0,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _compute_option_weights(vega: np.ndarray) -> np.ndarray:
    """Build the per-strike objective weights from cfg.OBJECTIVE_WEIGHTING.
    Returns weights that sum to 1 (the calibration classes expect
    normalized weights)."""
    if cfg.OBJECTIVE_WEIGHTING == "vega":
        raw = vega
    elif cfg.OBJECTIVE_WEIGHTING == "sqrt_vega":
        raw = np.sqrt(np.maximum(vega, 0.0))
    elif cfg.OBJECTIVE_WEIGHTING == "uniform":
        raw = np.ones_like(vega)
    else:
        raise ValueError(
            f"Unknown OBJECTIVE_WEIGHTING: {cfg.OBJECTIVE_WEIGHTING!r}. "
            f"Expected 'vega', 'sqrt_vega', or 'uniform'."
        )
    total = raw.sum()
    if total <= 0:
        raise ValueError(
            f"Option weights sum to zero under OBJECTIVE_WEIGHTING="
            f"{cfg.OBJECTIVE_WEIGHTING!r}; check the input smile."
        )
    return raw / total


def calibrate_date_systematic(df_date: pd.DataFrame):
    """Fit the systematic Kim-Yi (2025) parameters to one date's smile.

    Runs SLSQP from each starting point in cfg.INITIAL_VALUES_SYSTEMATIC_
    MULTISTART and returns the best result. Multi-start is necessary
    because SLSQP is a local solver and any single fixed initial guess
    can land in the Newton IV-inversion's 1e6-penalty region for some
    (date, tenor) combinations -- once there, the gradient is
    numerically zero and SLSQP quits early at x0.

    'Best' = converged AND lowest objective. Falls back to lowest
    objective overall if none converged.
    """
    weights = _compute_option_weights(df_date["dVEGA"].to_numpy())

    fitter = KimYiSkewCalibrationSystematic(
        mkt_imp_vol=df_date["dMKT_IMP_VOL"].to_numpy(),
        und_price=df_date["dUND_PRICE"].to_numpy(),
        und_strike=df_date["dUND_STRIKE"].to_numpy(),
        risk_free_rate=df_date["dRISK_FREE_RATE"].to_numpy(),
        dividend_yield=df_date["dDIVIDEND_YIELD"].to_numpy(),
        time_to_expiry=df_date["dEXPIRY"].to_numpy(),
        is_call_option=df_date["bIS_CALL_OPTION"].to_numpy(),
        option_weights=weights,
    )

    bounds = [cfg.BOUNDS_SYSTEMATIC[p] for p in SYSTEMATIC_PARAM_NAMES]

    best_result = None
    for x0 in cfg.INITIAL_VALUES_SYSTEMATIC_MULTISTART:
        result = minimize(
            fitter.target,
            x0=x0,
            method=cfg.OPTIMIZER_METHOD,
            bounds=bounds,
            tol=cfg.OPTIMIZER_TOL,
            options={"maxiter": cfg.OPTIMIZER_MAXITER},
        )
        if best_result is None:
            best_result = result
            continue
        # Prefer converged over not-converged; among like-flagged, prefer
        # lower objective. This avoids picking a converged-but-worse fit
        # over a not-converged-but-lower-objective one that just needed
        # more iterations.
        if result.success and not best_result.success:
            best_result = result
        elif result.success == best_result.success and result.fun < best_result.fun:
            best_result = result

    return best_result


def systematic_row(ticker: str, date_str: str, tenor: int, result, n_obs: int) -> dict:
    """Build one cache row from a calibrate_date_systematic() result.
    dKAPPAI/dGAMMAI/dBETAI/dRHOIX are fixed (SYSTEMATIC_FIXED_PARAMS), not
    calibrated, for a systematic row."""
    row = {
        "sTICKER": ticker, "sVALUATION_DATE": date_str,
        "iEXPIRY": int(tenor), "dEXPIRY": tenor / 365.0,
    }
    row.update(SYSTEMATIC_FIXED_PARAMS)
    row.update(dict(zip(SYSTEMATIC_PARAM_NAMES, result.x)))
    row["bCONVERGED"] = bool(result.success) and float(result.fun) < 1e5
    row["dOBJECTIVE"] = float(result.fun)
    row["iN_OBS"] = int(n_obs)
    return row


# ---------------------------------------------------------------------------
# Worker function (module-level, picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------
def _worker_init() -> None:
    """Cap BLAS threads inside each worker process to 1 -- the #1 way
    parallel SciPy code becomes slower than sequential is running N worker
    processes each spawning N MKL/OpenBLAS threads. Set BEFORE the worker
    imports NumPy for the environment variables to take effect."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"


def _calibrate_one(work_item: dict) -> dict:
    """Calibrate one (ticker, date, tenor). Returns a row dict for the
    Parquet cache. Runs a fresh KimYiSkewCalibrationSystematic per call --
    safe default when the calibrator's stateless-across-calls property
    isn't verified. If profiling later shows setup cost dominates the fit,
    switch to a per-worker persistent instance."""
    result = calibrate_date_systematic(work_item["df_date"])
    return systematic_row(
        ticker=work_item["ticker"],
        date_str=work_item["date_str"],
        tenor=work_item["tenor"],
        result=result,
        n_obs=work_item["n_obs"],
    )


# ---------------------------------------------------------------------------
# Result cache (wide Parquet table, shared with skew_calibrate_idiosyncratic.py)
# ---------------------------------------------------------------------------
def cache_path() -> str:
    name = cfg.OUTPUT_PARQUET_NAME_SYNTHETIC if cfg.USE_SYNTHETIC_DATA else cfg.OUTPUT_PARQUET_NAME
    return os.path.join(cfg.OUTPUT_DIR, name)


def load_cache() -> dict:
    """Read the Parquet cache into {(ticker, date_str, tenor): row_dict}."""
    path = cache_path()
    if not os.path.exists(path):
        return {}
    df = pd.read_parquet(path)
    return {
        (row["sTICKER"], row["sVALUATION_DATE"], int(row["iEXPIRY"])): row.to_dict()
        for _, row in df.iterrows()
    }


def save_cache(cache: dict) -> str:
    """Write {(ticker, date_str, tenor): row_dict} out as the wide Parquet table."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = cache_path()
    df = pd.DataFrame(list(cache.values()), columns=CACHE_COLUMNS)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Work-item construction
# ---------------------------------------------------------------------------
def build_work_items(
    mkt_vol_df: pd.DataFrame,
    ticker: str,
    valuation_dates: pd.DatetimeIndex,
    cache: dict,
    tenor_mode: str,
    tenor_range: Tuple[int, int],
    tenor_list: List[int],
    tenor_tolerance: int,
    overwrite: bool,
) -> List[dict]:
    """Flatten the (date, tenor) Cartesian product into a list of independent
    work items, applying the tenor selection and cache-skip rules up front.
    Doing selection here (in the main process) means workers only ever see
    fits that actually need to run -- no wasted pickling of already-cached
    triples."""
    available_dates = set(pd.to_datetime(mkt_vol_df["quote_date"].unique()))

    items: List[dict] = []
    for valuation_date in valuation_dates:
        if valuation_date not in available_dates:
            continue

        date_str = valuation_date.strftime("%Y%m%d")
        date_mask = (mkt_vol_df["underlying_symbol"] == ticker) & (mkt_vol_df["quote_date"] == valuation_date)
        chain = mkt_vol_df.loc[date_mask]
        if chain.empty:
            continue

        selected_tenors = select_tenors(
            available=chain["iEXPIRY"].unique(),
            mode=tenor_mode,
            range_min=tenor_range[0],
            range_max=tenor_range[1],
            target_days=tenor_list,
            tolerance_days=tenor_tolerance,
        )
        if not selected_tenors:
            continue

        for tenor in selected_tenors:
            key = (ticker, date_str, int(tenor))
            if key in cache and not overwrite:
                continue

            df_date = chain.loc[chain["iEXPIRY"] == tenor].sort_values("dMONEYNESS")
            if len(df_date) < cfg.MIN_OPTIONS_PER_TENOR:
                logger.info(
                    "%s tenor=%d: only %d row(s) (< MIN_OPTIONS_PER_TENOR=%d), skipping",
                    date_str, tenor, len(df_date), cfg.MIN_OPTIONS_PER_TENOR,
                )
                continue

            items.append({
                "ticker": ticker,
                "date_str": date_str,
                "tenor": int(tenor),
                "df_date": df_date,
                "n_obs": len(df_date),
                "already_cached": key in cache,
            })

    return items


# ---------------------------------------------------------------------------
# Execution (sequential and parallel paths)
# ---------------------------------------------------------------------------
def _resolve_n_jobs(n_jobs: int) -> int:
    """Interpret the --n-jobs flag: 0 = os.cpu_count() - 1, else the value."""
    if n_jobs == 0:
        return max(1, (os.cpu_count() or 1) - 1)
    return n_jobs


def _log_completed(row: dict, param_names: List[str], action: str) -> None:
    params = {p: row[p] for p in param_names}
    param_str = "  ".join(f"{k}={v:.4f}" for k, v in params.items())
    logger.info(
        "%s tenor=%d  %s  converged=%s  n_obs=%d  %s",
        row["sVALUATION_DATE"], row["iEXPIRY"], action,
        row["bCONVERGED"], row["iN_OBS"], param_str,
    )


def _run_sequential(work_items: List[dict], cache: dict, checkpoint_every: int) -> int:
    n_done = 0
    for item in _tqdm(work_items, desc="calibrating", unit="fit"):
        row = _calibrate_one(item)
        key = (row["sTICKER"], row["sVALUATION_DATE"], row["iEXPIRY"])
        cache[key] = row
        n_done += 1
        _log_completed(row, SYSTEMATIC_PARAM_NAMES, "overwrote" if item["already_cached"] else "calibrated")
        if n_done % checkpoint_every == 0:
            save_cache(cache)
    return n_done


def _run_parallel(work_items: List[dict], cache: dict, n_jobs: int, checkpoint_every: int) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_done = 0
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_worker_init) as pool:
        futures = {pool.submit(_calibrate_one, item): item for item in work_items}
        for f in _tqdm(as_completed(futures), total=len(futures), desc=f"calibrating ({n_jobs} workers)", unit="fit"):
            item = futures[f]
            row = f.result()   # will re-raise the worker's exception with context
            key = (row["sTICKER"], row["sVALUATION_DATE"], row["iEXPIRY"])
            cache[key] = row
            n_done += 1
            _log_completed(row, SYSTEMATIC_PARAM_NAMES, "overwrote" if item["already_cached"] else "calibrated")
            if n_done % checkpoint_every == 0:
                save_cache(cache)
                logger.info("Checkpoint: %d fits saved to %s", n_done, cache_path())
    return n_done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Systematic-stage Kim-Yi (2025) skew calibration. "
                    "CLI flags override defaults in Scripts/config_skew.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-source", choices=sorted(LOADERS.keys()),
                   default=cfg.DATA_SOURCE,
                   help="Loader adapter for the raw options data.")
    p.add_argument("--data-path", default=None,
                   help="Override cfg.SYSTEMATIC_UNDERLYING['data_path']. "
                        "Folder path for --data-source cboe, single-file path "
                        "for --data-source canonical.")
    p.add_argument("--valuation-date-beg", default=cfg.VALUATION_DATE_BEG,
                   help="ISO date, inclusive.")
    p.add_argument("--valuation-date-end", default=cfg.VALUATION_DATE_END,
                   help="ISO date, inclusive.")
    p.add_argument("--tenor-mode", choices=["all", "range", "list"],
                   default=cfg.TENOR_MODE,
                   help='"all": every tenor in the chain. "range": tenors '
                        'in [MIN, MAX]. "list": nearest-match to --tenors '
                        'within --tenor-tolerance.')
    p.add_argument("--tenors", nargs="+", type=int,
                   default=cfg.TENOR_LIST_DAYS, metavar="DAYS",
                   help="Target tenors when --tenor-mode list.")
    p.add_argument("--tenor-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                   default=[cfg.TENOR_RANGE_MIN, cfg.TENOR_RANGE_MAX],
                   help="Inclusive tenor bounds when --tenor-mode range.")
    p.add_argument("--tenor-tolerance", type=int,
                   default=cfg.TENOR_TOLERANCE_DAYS, metavar="DAYS",
                   help="+/- days accepted around each target in --tenor-mode list.")
    p.add_argument("--n-jobs", type=int, default=cfg.N_JOBS_DEFAULT, metavar="N",
                   help="Worker processes. 1 = sequential (no pool); "
                        "0 = os.cpu_count() - 1.")
    p.add_argument("--checkpoint-every", type=int,
                   default=cfg.CHECKPOINT_EVERY_DEFAULT, metavar="N",
                   help="Save the cache every N completed fits.")
    p.add_argument("--overwrite", action="store_true",
                   default=cfg.OVERWRITE_EXISTING,
                   help="Recalibrate keys already present in the cache.")
    return p


def configure_logging(level: int = logging.INFO) -> None:
    """Entry-point logging setup. Only called from the __main__ guard, so
    importing this module never has the side effect of installing handlers
    on the root logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    ticker = cfg.SYSTEMATIC_UNDERLYING["ticker"]
    underlying_meta = dict(cfg.SYSTEMATIC_UNDERLYING)
    if args.data_path:
        underlying_meta["data_path"] = args.data_path

    if cfg.USE_SYNTHETIC_DATA:
        logger.warning(
            "config_skew.USE_SYNTHETIC_DATA is True: calibrating against "
            "synthetic data, not market data."
        )
        mkt_vol_df = _make_synthetic_market_vol_data(ticker)
        valuation_dates = pd.bdate_range(
            cfg.SYNTHETIC_VALUATION_DATE_BEG, cfg.SYNTHETIC_VALUATION_DATE_END
        )
    else:
        loader = LOADERS[args.data_source]
        logger.info("Loading data for %s via %r adapter", ticker, args.data_source)
        mkt_vol_df = loader(underlying_meta, ticker)
        mkt_vol_df = attach_risk_free_rate(mkt_vol_df)
        mkt_vol_df = attach_vega_weights(mkt_vol_df)
        valuation_dates = pd.bdate_range(args.valuation_date_beg, args.valuation_date_end)

    cache = load_cache()
    work_items = build_work_items(
        mkt_vol_df=mkt_vol_df,
        ticker=ticker,
        valuation_dates=valuation_dates,
        cache=cache,
        tenor_mode=args.tenor_mode,
        tenor_range=(args.tenor_range[0], args.tenor_range[1]),
        tenor_list=args.tenors,
        tenor_tolerance=args.tenor_tolerance,
        overwrite=args.overwrite,
    )

    if not work_items:
        logger.info("No (date, tenor) pairs to calibrate; %d rows already in cache.", len(cache))
        return

    logger.info(
        "Built %d work item(s); %s mode; %d already in cache.",
        len(work_items), args.tenor_mode, len(cache),
    )

    n_jobs = _resolve_n_jobs(args.n_jobs)
    if n_jobs == 1:
        n_done = _run_sequential(work_items, cache, args.checkpoint_every)
    else:
        logger.info("Running in parallel across %d worker process(es)", n_jobs)
        n_done = _run_parallel(work_items, cache, n_jobs, args.checkpoint_every)

    save_cache(cache)
    verb = "Overwrote/calibrated" if args.overwrite else "Calibrated"
    logger.info("%s %d (date, tenor) pair(s); %d total in cache.", verb, n_done, len(cache))
    logger.info("Cache: %s", cache_path())


if __name__ == "__main__":
    configure_logging()
    main()
