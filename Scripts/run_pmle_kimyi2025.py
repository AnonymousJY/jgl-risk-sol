"""
run_pmle_kimyi2025.py — P-MLE calibration driver (database-free).

Estimates the liquidity-adjusted jump-diffusion model parameters for each
valuation date in the configured window, in two stages:

  1. Systematic stage  — calibrate the common parameters from the systematic
     proxy (^SPX).
  2. Idiosyncratic stage — calibrate the asset-specific parameters for each
     idiosyncratic asset (COIN), conditional on the systematic parameters.

Results are written as wide-format CSVs to ``Study/Estimated Parameters PMLE/``
via ``Library.DataAccess.save_pmle_params``. Dates that already have a CSV are
skipped, so the script is incremental. This replaces the original PostgreSQL
round-trip: there is no database dependency.

Price history is read through the data-access layer (committed snapshots by
default; live FinanceDataReader when ``MKTDEPTH_DATA_MODE=live``).

Run from the repository root:

    python Scripts/run_pmle_kimyi2025.py
"""

import time
import numpy as np
import pandas as pd
import os
import multiprocessing
# forkserver, not fork. fork() in a process that has already started
# threads is unsafe; Python 3.12+ warns and 3.14 changes the Linux
# default for this reason. PyMC with a numba backend does start
# threads, and the failure mode is a hang that looks like slow
# sampling. Set JGL_MP_START to override.
multiprocessing.set_start_method(
    os.environ.get("JGL_MP_START", "forkserver"), force=True)

from concurrent.futures import ProcessPoolExecutor
from typing import List

from Scripts.load_portfolio import get_idiosyncratic_ids
from Library.DataAccess import (
    get_price_panel,
    get_pmle_params,
    get_pmle_params_dict,
    pmle_params_exists,
    save_pmle_params,
)
from Library.RiskEngineKimYi2025 import (
    pmle_kimyirisk_systematic,
    pmle_kimyirisk_idiosyncratic,
)

# The six common parameters carried from the systematic stage into the
# idiosyncratic stage.
SYSTEMATIC_PARAMS = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]


# ----------------------------------------------------------------------------
# Worker helpers (run inside ProcessPoolExecutor)
# ----------------------------------------------------------------------------
def pmle_kimyirisk_systematic_helper(args) -> tuple:
    """Estimate the systematic parameters for one valuation date.

    Returns ``(valuation_dt, systematic_id, results)`` where ``results`` is the
    ``{param: ParamsResults}`` dict produced by ``pmle_kimyirisk_systematic``.
    """
    (valuation_dt, return_vector, delta_t, seed_number, n_mc_paths,
     systematic_id, *rest) = args
    priors = rest[0] if rest else None
    results = pmle_kimyirisk_systematic(
        priors=priors,
        sys_returns=return_vector,
        delta_t=delta_t,
        seed_number=seed_number,
        n_mc_paths=n_mc_paths,
    )
    return valuation_dt, systematic_id, results


def pmle_kimyirisk_idiosyncratic_helper(args) -> tuple:
    """Estimate the idiosyncratic parameters for one (valuation date, asset),
    conditional on the systematic parameters.

    Returns ``(valuation_dt, idiosyncratic_id, results)``.
    """
    valuation_dt, params_sys, return_vector, delta_t, seed_number, n_mc_paths, idiosyncratic_id = args
    results = pmle_kimyirisk_idiosyncratic(
        params_sys=params_sys,
        idi_returns=return_vector,
        delta_t=delta_t,
        seed_number=seed_number,
        n_mc_paths=n_mc_paths,
    )
    return valuation_dt, idiosyncratic_id, results


# ----------------------------------------------------------------------------
# Result assembly: MCMC output -> full eleven-parameter CSV row
# ----------------------------------------------------------------------------
def _triples_from_results(results: dict) -> dict:
    """Convert a ``{param: ParamsResults}`` dict into a
    ``{param: (mean, ci_lower, ci_upper)}`` dict."""
    return {k: (v.dMEAN, v.dCI_LOWER, v.dCI_UPPER) for k, v in results.items()}


def assemble_systematic_params(results: dict) -> dict:
    """Build the full eleven-parameter dict for a systematic underlying.

    The systematic proxy has, by definition, ``mu_i = kappa_i = rho_iX = 0`` and
    ``gamma_i = beta_i = 1`` (with degenerate confidence intervals); the
    remaining six parameters come from the systematic MCMC fit.
    """
    params = {
        "dMUI": (0.0, 0.0, 0.0),
        "dKAPPAI": (0.0, 0.0, 0.0),
        "dGAMMAI": (1.0, 1.0, 1.0),
        "dBETAI": (1.0, 1.0, 1.0),
        "dRHOIX": (0.0, 0.0, 0.0),
    }
    params.update(_triples_from_results(results))
    return params


def assemble_idiosyncratic_params(results: dict, systematic_series: pd.Series) -> dict:
    """Build the full eleven-parameter dict for an idiosyncratic asset.

    The five asset-specific parameters come from the idiosyncratic MCMC fit;
    the six common parameters are inherited from the systematic estimate
    (mean and confidence bounds) read back from its CSV.
    """
    params = _triples_from_results(results)
    for k in SYSTEMATIC_PARAMS:
        params[k] = (
            systematic_series[k],
            systematic_series[f"{k}_CI_LOWER"],
            systematic_series[f"{k}_CI_UPPER"],
        )
    return params


if __name__ == "__main__":

    beg_time = time.perf_counter()

    # --- configuration -------------------------------------------------------
    valuation_beg_dt = "20250331"
    valuation_end_dt = "20250417"
    date_format = "%Y%m%d"
    valuation_window = pd.bdate_range(
        pd.to_datetime(arg=valuation_beg_dt, format=date_format),
        pd.to_datetime(arg=valuation_end_dt, format=date_format),
    )
    valuation_window_str = [dt.strftime(date_format) for dt in valuation_window]

    lookback_period = 252
    base_days = 252
    delta_t = np.array(1 / base_days)
    seed_number = np.uint64(20240114)
    n_mc_paths = int(10_000)

    systematic_id = "^SPX"
    idiosyncratic_ids = get_idiosyncratic_ids()

    # --- price history (snapshot by default; see Library/DataAccess.py) ------
    price_ts = get_price_panel([systematic_id] + idiosyncratic_ids)
    return_ts = price_ts.pct_change().dropna()

    # --- incremental work set: skip dates that already have a CSV ------------
    set_to_valuate_systematic = [
        dt for dt in valuation_window_str
        if not pmle_params_exists(dt, systematic_id)
    ]
    set_to_valuate_idiosyncratic = [
        (dt, idi_id)
        for dt in valuation_window_str
        for idi_id in idiosyncratic_ids
        if not pmle_params_exists(dt, idi_id)
    ]
    print(f"Systematic dates to estimate:   {len(set_to_valuate_systematic)}")
    print(f"Idiosyncratic (date, id) pairs: {len(set_to_valuate_idiosyncratic)}")

    # --- P-MLE systematic stage ---------------------------------------------
    if set_to_valuate_systematic:
        systematic_arg_list = []
        for dt in set_to_valuate_systematic:
            return_vector = (
                return_ts.loc[return_ts.index <= dt, systematic_id]
                .iloc[-lookback_period:]
                .to_numpy()
            )
            systematic_arg_list.append(
                (dt, return_vector, delta_t, seed_number, n_mc_paths, systematic_id)
            )

        with ProcessPoolExecutor() as executor:
            for valuation_dt, sys_id, results in executor.map(
                pmle_kimyirisk_systematic_helper, systematic_arg_list
            ):
                path = save_pmle_params(
                    valuation_dt, sys_id, assemble_systematic_params(results)
                )
                print(f"  systematic  {valuation_dt} {sys_id}  -> {path}")

    # --- P-MLE idiosyncratic stage ------------------------------------------
    # Built after the systematic stage so every required systematic CSV exists
    # (whether just estimated above or already cached from a previous run).
    if set_to_valuate_idiosyncratic:
        idiosyncratic_arg_list = []
        for dt, idi_id in set_to_valuate_idiosyncratic:
            params_sys = get_pmle_params_dict(dt, systematic_id, params=SYSTEMATIC_PARAMS)
            return_vector = (
                return_ts.loc[return_ts.index <= dt, idi_id]
                .iloc[-lookback_period:]
                .to_numpy()
            )
            idiosyncratic_arg_list.append(
                (dt, params_sys, return_vector, delta_t, seed_number, n_mc_paths, idi_id)
            )

        with ProcessPoolExecutor() as executor:
            for valuation_dt, idi_id, results in executor.map(
                pmle_kimyirisk_idiosyncratic_helper, idiosyncratic_arg_list
            ):
                systematic_series = get_pmle_params(valuation_dt, systematic_id)
                path = save_pmle_params(
                    valuation_dt,
                    idi_id,
                    assemble_idiosyncratic_params(results, systematic_series),
                )
                print(f"  idiosyncratic {valuation_dt} {idi_id}  -> {path}")

    elapsed_time = time.perf_counter() - beg_time
    print(f"Time taken: {elapsed_time:.6f} seconds")
