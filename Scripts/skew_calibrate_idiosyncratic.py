"""
Scripts/skew_calibrate_idiosyncratic.py — idiosyncratic-stage skew
calibration (Kim-Yi 2025 model), scripted from the "idiosyncratic" cells of
Scripts/skew_calibration_kimyi2025.ipynb.

For each (idiosyncratic ticker, valuation date, tenor), fits the four
asset-specific Kim-Yi (2025) parameters (kappai, gammai, betai, rhoix) to
that underlying's implied-vol smile at that tenor, *conditional* on the five
systematic parameters (sigma, pprob, lamb, eta1, eta2) already calibrated
for that same (date, tenor) (Library.SkewCalibrationKimYi2025.
KimYiSkewCalibrationIdiosyncratic + scipy.optimize.minimize). Tenor
selection follows cfg.TENOR_MODE ("all" / "range" / "list"), same semantics
as skew_calibrate_systematic.py.

Which tickers are calibrated
-----------------------------
By default, every entry in cfg.IDIOSYNCRATIC_UNDERLYINGS. --tickers filters
to a subset for one-off runs. Adding a new name (with its dividend yield
and data path) is a one-place edit in Scripts/config_skew.py.

Dependency on the systematic stage
-----------------------------------
This script does not calibrate the systematic parameters itself. It reads
them from the cache Scripts/skew_calibrate_systematic.py writes to
(cfg.OUTPUT_DIR / OUTPUT_PARQUET_NAME, keyed by (SYSTEMATIC_UNDERLYING[
"ticker"], YYYYMMDD, iEXPIRY)). Run Scripts/skew_calibrate_systematic.py
first: a (date, tenor) with no matching cached systematic entry is skipped
here with a warning, not calibrated as a side effect. If the cache
contains *no* systematic rows for the current cfg.SYSTEMATIC_UNDERLYING[
"ticker"], this script fails loudly at load time rather than emitting one
warning per (ticker, date, tenor) triple.

Data loading, tenor selection, multiprocessing, and cache format are all
shared with skew_calibrate_systematic.py (LOADERS, select_tenors,
_worker_init, cache_path/load_cache/save_cache, CACHE_COLUMNS) so both
stages march to the same drum.

Run from the repository root, after skew_calibrate_systematic.py:

    # Every configured ticker, defaults
    python Scripts/skew_calibrate_systematic.py
    python Scripts/skew_calibrate_idiosyncratic.py

    # Just COIN, 30d + 90d only, 8 parallel workers
    python Scripts/skew_calibrate_idiosyncratic.py \\
        --tickers COIN \\
        --tenor-mode list --tenors 30 90 --tenor-tolerance 5 \\
        --n-jobs 8 --checkpoint-every 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config_skew as cfg  # Scripts/config_skew.py
from Library.SkewCalibrationKimYi2025 import KimYiSkewCalibrationIdiosyncratic
from Scripts.skew_calibrate_systematic import (
    IDIOSYNCRATIC_PARAM_NAMES,
    LOADERS,
    SYSTEMATIC_FIXED_PARAMS,
    SYSTEMATIC_PARAM_NAMES,
    TRUE_PARAMS_SYNTHETIC as TRUE_PARAMS_SYSTEMATIC_SYNTHETIC,
    _resolve_n_jobs,
    _worker_init,
    attach_risk_free_rate,
    attach_vega_weights,
    cache_path,
    configure_logging,
    load_cache,
    save_cache,
    select_tenors,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    def _tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Systematic dependency
# ---------------------------------------------------------------------------
def get_systematic_params(
    cache: dict, systematic_ticker: str, valuation_date_str: str, tenor: int,
) -> Optional[np.ndarray]:
    row = cache.get((systematic_ticker, valuation_date_str, tenor))
    if row is None:
        return None
    # Reject penalty-region rows: SLSQP reported success but really got stuck.
    # Idiosyncratic calibration conditional on nonsense systematic params
    # produces nonsense idiosyncratic params (see COIN 20250411).
    if row["dOBJECTIVE"] > 1e5:
        return None
    return np.array([row[p] for p in SYSTEMATIC_PARAM_NAMES])


def diagnose_systematic_availability(cache: dict, systematic_ticker: str) -> None:
    """Fail loudly at load time if the cache has no systematic rows for the
    current systematic ticker -- catches the two common misconfigurations
    (never ran the systematic stage; ran it against a different index) in
    one clear error message rather than one warning per (date, tenor)."""
    matching = [k for k in cache if k[0] == systematic_ticker]
    if matching:
        return

    tickers_in_cache = sorted({k[0] for k in cache})
    if not tickers_in_cache:
        raise RuntimeError(
            f"No cached calibrations found at {cache_path()}. Run "
            f"`python Scripts/skew_calibrate_systematic.py` first."
        )
    raise RuntimeError(
        f"No cached systematic parameters for "
        f"cfg.SYSTEMATIC_UNDERLYING['ticker'] = {systematic_ticker!r} in the "
        f"cache at {cache_path()}. Cache contains rows for: {tickers_in_cache}. "
        f"Either run `python Scripts/skew_calibrate_systematic.py` for "
        f"{systematic_ticker!r}, or set cfg.SYSTEMATIC_UNDERLYING['ticker'] "
        f"to a ticker that is already cached."
    )


def idiosyncratic_row(
    ticker: str, date_str: str, tenor: int, result,
    systematic_params: np.ndarray, n_obs: int,
) -> dict:
    """Build one cache row from a calibrate_date_idiosyncratic() result. The
    systematic parameters this fit was conditioned on are denormalized onto
    the row (see module docstring)."""
    row = {
        "sTICKER": ticker, "sVALUATION_DATE": date_str,
        "iEXPIRY": int(tenor), "dEXPIRY": tenor / 365.0,
    }
    row.update(dict(zip(IDIOSYNCRATIC_PARAM_NAMES, result.x)))
    row.update(dict(zip(SYSTEMATIC_PARAM_NAMES, systematic_params)))
    row["bCONVERGED"] = bool(result.success) and float(result.fun) < 1e5
    row["dOBJECTIVE"] = float(result.fun)
    row["iN_OBS"] = int(n_obs)
    return row


# ---------------------------------------------------------------------------
# Synthetic data (dev / test, no market data or network required)
# ---------------------------------------------------------------------------
TRUE_PARAMS_SYNTHETIC = {
    "dKAPPAI": 0.30, "dGAMMAI": 1.20, "dBETAI": 0.80, "dRHOIX": -0.30,
}


def _seed_synthetic_systematic_cache(
    cache: dict, systematic_ticker: str, dates: pd.DatetimeIndex,
) -> None:
    """Inject a systematic cache row for each (synthetic date, tenor), built
    from TRUE_PARAMS_SYSTEMATIC_SYNTHETIC (the same "true" values
    skew_calibrate_systematic.py's own synthetic mode targets), so this
    script's synthetic run can exercise the real dependency lookup without
    re-running the (slow) systematic optimizer here."""
    for valuation_date in dates:
        date_str = valuation_date.strftime("%Y%m%d")
        for tenor in cfg.SYNTHETIC_TENORS_DAYS:
            key = (systematic_ticker, date_str, int(tenor))
            if key in cache:
                continue
            row = {
                "sTICKER": systematic_ticker, "sVALUATION_DATE": date_str,
                "iEXPIRY": int(tenor), "dEXPIRY": tenor / 365.0,
            }
            row.update(SYSTEMATIC_FIXED_PARAMS)
            row.update(TRUE_PARAMS_SYSTEMATIC_SYNTHETIC)
            row["bCONVERGED"] = True
            row["dOBJECTIVE"] = 0.0
            row["iN_OBS"] = 0
            cache[key] = row


def _make_synthetic_market_vol_data(
    ticker: str, dividend_yield: float, systematic_params: np.ndarray,
) -> pd.DataFrame:
    """Synthesize an implied-vol smile per (valuation date, tenor), generated
    from TRUE_PARAMS_SYNTHETIC via the real KimYiSkewCalibrationIdiosyncratic
    model (plus noise), conditional on systematic_params (assumed constant
    across tenors here, matching _seed_synthetic_systematic_cache). Tenors
    come from config_skew.SYNTHETIC_TENORS_DAYS."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    dates = pd.bdate_range(cfg.SYNTHETIC_VALUATION_DATE_BEG, cfg.SYNTHETIC_VALUATION_DATE_END)

    sigma, pprob, lamb, eta1, eta2 = systematic_params
    S0, r = 100.0, 0.04
    q = dividend_yield   # decimal, matches r
    moneyness = np.linspace(60.0, 140.0, cfg.SYNTHETIC_N_MONEYNESS)
    strikes = moneyness / 100.0 * S0
    is_call = moneyness >= 100.0
    x_true = np.array([TRUE_PARAMS_SYNTHETIC[p] for p in IDIOSYNCRATIC_PARAM_NAMES])

    rows = []
    for quote_date in dates:
        for expiry_days in cfg.SYNTHETIC_TENORS_DAYS:
            T = expiry_days / 365.0

            fitter = KimYiSkewCalibrationIdiosyncratic(
                sigma=np.array(sigma), pprob=np.array(pprob), lamb=np.array(lamb),
                eta1=np.array(eta1), eta2=np.array(eta2),
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
def calibrate_date_idiosyncratic(df_date: pd.DataFrame, systematic_params: np.ndarray):
    """Fit the idiosyncratic Kim-Yi (2025) parameters to one date's smile,
    conditional on that date's systematic parameters.

    Multi-start over cfg.INITIAL_VALUES_IDIOSYNCRATIC_MULTISTART; keep the
    best (converged, lowest-objective) result. See the systematic version's
    docstring for the full rationale.
    """
    # Same weighting scheme as the systematic stage (see cfg.OBJECTIVE_WEIGHTING).
    from Scripts.skew_calibrate_systematic import _compute_option_weights
    weights = _compute_option_weights(df_date["dVEGA"].to_numpy())

    sigma, pprob, lamb, eta1, eta2 = systematic_params

    fitter = KimYiSkewCalibrationIdiosyncratic(
        sigma=np.array(sigma), pprob=np.array(pprob), lamb=np.array(lamb),
        eta1=np.array(eta1), eta2=np.array(eta2),
        mkt_imp_vol=df_date["dMKT_IMP_VOL"].to_numpy(),
        und_price=df_date["dUND_PRICE"].to_numpy(),
        und_strike=df_date["dUND_STRIKE"].to_numpy(),
        risk_free_rate=df_date["dRISK_FREE_RATE"].to_numpy(),
        dividend_yield=df_date["dDIVIDEND_YIELD"].to_numpy(),
        time_to_expiry=df_date["dEXPIRY"].to_numpy(),
        is_call_option=df_date["bIS_CALL_OPTION"].to_numpy(),
        option_weights=weights,
    )

    bounds = [cfg.BOUNDS_IDIOSYNCRATIC[p] for p in IDIOSYNCRATIC_PARAM_NAMES]

    best_result = None
    for x0 in cfg.INITIAL_VALUES_IDIOSYNCRATIC_MULTISTART:
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
        if result.success and not best_result.success:
            best_result = result
        elif result.success == best_result.success and result.fun < best_result.fun:
            best_result = result

    return best_result


# ---------------------------------------------------------------------------
# Worker function (module-level, picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------
def _calibrate_one(work_item: dict) -> dict:
    """Calibrate one (ticker, date, tenor) idiosyncratic fit. Returns a row
    dict for the Parquet cache. Systematic parameters are attached to the
    work item so no shared cache access from workers is needed."""
    result = calibrate_date_idiosyncratic(work_item["df_date"], work_item["systematic_params"])
    return idiosyncratic_row(
        ticker=work_item["ticker"],
        date_str=work_item["date_str"],
        tenor=work_item["tenor"],
        result=result,
        systematic_params=work_item["systematic_params"],
        n_obs=work_item["n_obs"],
    )


# ---------------------------------------------------------------------------
# Work-item construction
# ---------------------------------------------------------------------------
def build_work_items(
    per_ticker_data: dict,          # {ticker: canonical DataFrame}
    valuation_dates: pd.DatetimeIndex,
    cache: dict,
    systematic_ticker: str,
    tenor_mode: str,
    tenor_range: Tuple[int, int],
    tenor_list: List[int],
    tenor_tolerance: int,
    overwrite: bool,
) -> List[dict]:
    """Flatten (ticker, date, tenor) into a list of independent work items,
    applying tenor selection, cache-skip, and systematic-availability rules
    up front. Doing all filtering here (in the main process) means workers
    only ever see fits that actually need to run -- no wasted pickling of
    already-cached or systematic-less triples."""
    items: List[dict] = []
    n_skipped_no_sys = 0
    n_skipped_thin = 0

    for ticker, mkt_vol_df in per_ticker_data.items():
        available_dates = set(pd.to_datetime(mkt_vol_df["quote_date"].unique()))

        for valuation_date in valuation_dates:
            if valuation_date not in available_dates:
                continue

            date_str = valuation_date.strftime("%Y%m%d")
            date_mask = (
                (mkt_vol_df["underlying_symbol"] == ticker)
                & (mkt_vol_df["quote_date"] == valuation_date)
            )
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

            for tenor in selected_tenors:
                key = (ticker, date_str, int(tenor))
                if key in cache and not overwrite:
                    continue

                systematic_params = get_systematic_params(
                    cache, systematic_ticker, date_str, int(tenor),
                )
                if systematic_params is None:
                    logger.warning(
                        "%s %s tenor=%d: no cached systematic parameters for "
                        "%s at that (date, tenor); skipping. Run "
                        "skew_calibrate_systematic.py for this window first.",
                        ticker, date_str, tenor, systematic_ticker,
                    )
                    n_skipped_no_sys += 1
                    continue

                df_date = chain.loc[chain["iEXPIRY"] == tenor].sort_values("dMONEYNESS")
                if len(df_date) < cfg.MIN_OPTIONS_PER_TENOR:
                    logger.info(
                        "%s %s tenor=%d: only %d row(s) (< MIN_OPTIONS_PER_TENOR=%d), skipping",
                        ticker, date_str, tenor, len(df_date), cfg.MIN_OPTIONS_PER_TENOR,
                    )
                    n_skipped_thin += 1
                    continue

                items.append({
                    "ticker": ticker,
                    "date_str": date_str,
                    "tenor": int(tenor),
                    "df_date": df_date,
                    "n_obs": len(df_date),
                    "systematic_params": systematic_params,
                    "already_cached": key in cache,
                })

    if n_skipped_no_sys:
        logger.warning("%d (ticker, date, tenor) triple(s) skipped for missing systematic parameters.", n_skipped_no_sys)
    if n_skipped_thin:
        logger.info("%d triple(s) skipped for too few option points.", n_skipped_thin)

    return items


# ---------------------------------------------------------------------------
# Execution (sequential and parallel paths)
# ---------------------------------------------------------------------------
def _log_completed(row: dict, action: str) -> None:
    params = {p: row[p] for p in IDIOSYNCRATIC_PARAM_NAMES}
    param_str = "  ".join(f"{k}={v:.4f}" for k, v in params.items())
    logger.info(
        "%s %s tenor=%d  %s  converged=%s  n_obs=%d  %s",
        row["sTICKER"], row["sVALUATION_DATE"], row["iEXPIRY"], action,
        row["bCONVERGED"], row["iN_OBS"], param_str,
    )


def _run_sequential(work_items: List[dict], cache: dict, checkpoint_every: int) -> int:
    n_done = 0
    for item in _tqdm(work_items, desc="calibrating", unit="fit"):
        row = _calibrate_one(item)
        key = (row["sTICKER"], row["sVALUATION_DATE"], row["iEXPIRY"])
        cache[key] = row
        n_done += 1
        _log_completed(row, "overwrote" if item["already_cached"] else "calibrated")
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
            row = f.result()
            key = (row["sTICKER"], row["sVALUATION_DATE"], row["iEXPIRY"])
            cache[key] = row
            n_done += 1
            _log_completed(row, "overwrote" if item["already_cached"] else "calibrated")
            if n_done % checkpoint_every == 0:
                save_cache(cache)
                logger.info("Checkpoint: %d fits saved to %s", n_done, cache_path())
    return n_done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Idiosyncratic-stage Kim-Yi (2025) skew calibration. "
                    "CLI flags override defaults in Scripts/config_skew.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-source", choices=sorted(LOADERS.keys()),
                   default=cfg.DATA_SOURCE,
                   help="Loader adapter for the raw options data.")
    p.add_argument("--tickers", nargs="+", default=None, metavar="TICKER",
                   help="Subset of cfg.IDIOSYNCRATIC_UNDERLYINGS to calibrate. "
                        "All values must be keys in that dict; unknown tickers "
                        "error out. Default: every configured ticker.")
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


def _resolve_tickers(requested: Optional[List[str]]) -> List[str]:
    """Return the list of idiosyncratic tickers to calibrate. Errors out if
    --tickers references a name not in cfg.IDIOSYNCRATIC_UNDERLYINGS -- we
    require the dividend yield / data path / display name to be configured
    before a ticker can be calibrated, rather than silently defaulting to
    zero yield or the fallback data folder."""
    configured = list(cfg.IDIOSYNCRATIC_UNDERLYINGS.keys())
    if requested is None:
        return configured

    unknown = [t for t in requested if t not in cfg.IDIOSYNCRATIC_UNDERLYINGS]
    if unknown:
        raise SystemExit(
            f"--tickers references name(s) not configured: {unknown}. "
            f"Add them to config_skew.IDIOSYNCRATIC_UNDERLYINGS with their "
            f"dividend_yield / data_path / display_name, then re-run. "
            f"Currently configured: {configured}"
        )
    return list(requested)


def _load_per_ticker(
    tickers: List[str], data_source: str,
) -> dict:
    """Load one canonical DataFrame per ticker via the LOADERS dispatch,
    then attach the risk-free rate and vega weights. Returns
    {ticker: DataFrame}."""
    loader = LOADERS[data_source]
    out = {}
    for ticker in tickers:
        meta = cfg.IDIOSYNCRATIC_UNDERLYINGS[ticker]
        logger.info(
            "Loading data for %s (%s) via %r adapter",
            ticker, meta.get("display_name", ticker), data_source,
        )
        df = loader(meta, ticker)
        df = attach_risk_free_rate(df)
        df = attach_vega_weights(df)
        out[ticker] = df
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    tickers = _resolve_tickers(args.tickers)
    systematic_ticker = cfg.SYSTEMATIC_UNDERLYING["ticker"]

    cache = load_cache()

    if cfg.USE_SYNTHETIC_DATA:
        logger.warning(
            "config_skew.USE_SYNTHETIC_DATA is True: calibrating against "
            "synthetic data, not market data."
        )
        valuation_dates = pd.bdate_range(
            cfg.SYNTHETIC_VALUATION_DATE_BEG, cfg.SYNTHETIC_VALUATION_DATE_END
        )
        _seed_synthetic_systematic_cache(cache, systematic_ticker, valuation_dates)

        # Systematic params are constant across synthetic dates/tenors here
        # (seeded above), so any seeded entry works to build the smile.
        seed_key = (
            systematic_ticker,
            valuation_dates[0].strftime("%Y%m%d"),
            int(cfg.SYNTHETIC_TENORS_DAYS[0]),
        )
        seed_params = np.array([cache[seed_key][p] for p in SYSTEMATIC_PARAM_NAMES])

        per_ticker_data = {}
        for ticker in tickers:
            q = cfg.IDIOSYNCRATIC_UNDERLYINGS[ticker]["dividend_yield"]
            per_ticker_data[ticker] = _make_synthetic_market_vol_data(ticker, q, seed_params)
    else:
        diagnose_systematic_availability(cache, systematic_ticker)
        valuation_dates = pd.bdate_range(args.valuation_date_beg, args.valuation_date_end)
        per_ticker_data = _load_per_ticker(tickers, args.data_source)

    work_items = build_work_items(
        per_ticker_data=per_ticker_data,
        valuation_dates=valuation_dates,
        cache=cache,
        systematic_ticker=systematic_ticker,
        tenor_mode=args.tenor_mode,
        tenor_range=(args.tenor_range[0], args.tenor_range[1]),
        tenor_list=args.tenors,
        tenor_tolerance=args.tenor_tolerance,
        overwrite=args.overwrite,
    )

    if not work_items:
        logger.info("No (ticker, date, tenor) triples to calibrate; %d rows already in cache.", len(cache))
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
    logger.info("%s %d (ticker, date, tenor) triple(s); %d total in cache.", verb, n_done, len(cache))
    logger.info("Cache: %s", cache_path())


if __name__ == "__main__":
    configure_logging()
    main()
