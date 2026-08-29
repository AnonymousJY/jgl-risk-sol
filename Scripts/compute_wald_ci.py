"""
Scripts/compute_wald_ci.py -- Wald (frequentist) 95% and 99% confidence
intervals for the P-MLE parameter estimates.

Rationale
---------
The v13 paper reports Bayesian credible intervals from an MCMC posterior
(Table 1). The referee's "too wide" complaint partly reflects a mis-match
between the referee's frequentist expectation and the Bayesian estimator.
This script produces the standard frequentist alternative -- Wald
confidence intervals derived from the observed Fisher information at the
MLE point estimate:

    SE       = sqrt(diag(-H^{-1}))
    Wald CI  = theta_hat +/- z_alpha * SE

with H the log-likelihood Hessian at theta_hat (computed numerically via
numdifftools), z_{95%} = 1.96, z_{99%} = 2.576.

Approach: transformed parameter space (default)
------------------------------------------------
For bounded / positive parameters, the Wald normal approximation is
much better on an unconstrained real-line transform than on the
constrained natural parameters:

    natural           unconstrained transform
    -------           -----------------------
    x in (0, 1)   ->  logit(x)  = log(x / (1 - x))
    x in (-1, 1)  ->  atanh(x)  = 0.5 * log((1 + x) / (1 - x))
    x > 0         ->  log(x)
    x in R        ->  identity

The Hessian is computed with respect to the transformed parameters, and
Wald intervals are constructed on the transformed real line (where the
normal approximation is well-defined):

    lo_tilde, hi_tilde = theta_tilde_hat +/- z_alpha * SE_tilde

The interval endpoints are then back-transformed to the natural scale:

    lo_natural = inverse_transform(lo_tilde)
    hi_natural = inverse_transform(hi_tilde)

This is *interval-preserving*: the resulting natural-scale interval
automatically respects the parameter's bounds, never produces sigma < 0
or alpha > 1. This fixes both the "SE > point estimate" pathology and
the boundary-crossing issue of the naive natural-space Wald CI.

A natural-scale standard error is also reported via the delta method
(|d(natural)/d(tilde)| * SE_tilde at theta_hat), for the compact
"point +/- SE" style used in Bayesian posterior tables.

Legacy natural-space computation is available via --legacy-natural
for cross-comparison.

Approach
--------
For each cached (ticker, valuation_date) row in Study/Estimated Parameters
PMLE/:

  1. Load point estimates via Library.DataAccess.get_pmle_params.
  2. Reconstruct the return vector used for that estimation (same
     lookback + same price source as run_pmle_kimyi2025.py).
  3. Build a compiled log-likelihood evaluator over the transformed
     parameter space using Library.RiskEngineKimYi2025.KimYiLogLike.
  4. Compute the numerical Hessian at the point estimate (in transformed
     coordinates) via numdifftools.Hessian.
  5. Wald CI on the transformed real line; back-transform endpoints to
     the natural parameter scale.
  6. Write results as a wide Parquet keyed by (ticker, date, param).

If the transformed-space Hessian is still ill-conditioned for a given
parameter, the script retries with a coarser finite-difference step
before falling back to reporting NaN for that parameter.

Requires: numdifftools. Install via
    conda install -c conda-forge numdifftools
or
    pip install numdifftools

Run from the repository root:

    # All cached (ticker, date) rows in the window (default: transformed)
    python Scripts/compute_wald_ci.py \\
        --valuation-date-beg 2025-04-02 \\
        --valuation-date-end 2025-04-16

    # Legacy natural-space Wald CI (for cross-comparison)
    python Scripts/compute_wald_ci.py --legacy-natural \\
        --valuation-date-beg 2025-04-02 --valuation-date-end 2025-04-16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from Scripts.load_portfolio import get_idiosyncratic_ids
from Library.DataAccess import get_price_panel, get_pmle_params, pmle_params_exists
from Library.RiskEngineKimYi2025 import KimYiLogLike

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Z_95 = 1.959963984540054   # Two-sided 95% (0.975 quantile of N(0,1))
Z_99 = 2.5758293035489004  # Two-sided 99% (0.995 quantile of N(0,1))

SYSTEMATIC_PARAM_NAMES = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]
IDIOSYNCRATIC_PARAM_NAMES = ["dMUI", "dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX"]

# Per-parameter transform to unconstrained real line. Chosen to match the
# parameter's natural support:
#     logit   : bounded (0, 1)                   -> R
#     atanh   : bounded (-1, 1)                  -> R
#     log     : positive real (0, inf)           -> R
#     None    : unconstrained real               -> R  (identity)
PARAM_TRANSFORMS = {
    "dALPHA":  "logit",
    "dSIGMA":  "log",
    "dPPROB":  "logit",
    "dLAMB":   "log",
    "dETA1":   "log",
    "dETA2":   "log",
    "dMUI":    None,
    "dKAPPAI": "log",
    "dGAMMAI": "log",
    "dBETAI":  "log",
    "dRHOIX":  "atanh",
}

# For legacy natural-space reporting: whether the CI extends past the
# parameter's admissible region.
PARAM_BOUNDS = {
    "dALPHA":  (0.0, 1.0),
    "dSIGMA":  (0.0, np.inf),
    "dPPROB":  (0.0, 1.0),
    "dLAMB":   (0.0, np.inf),
    "dETA1":   (0.0, np.inf),
    "dETA2":   (0.0, np.inf),
    "dMUI":    (-np.inf, np.inf),
    "dKAPPAI": (0.0, np.inf),
    "dGAMMAI": (0.0, np.inf),
    "dBETAI":  (0.0, np.inf),
    "dRHOIX":  (-1.0, 1.0),
}

# Fallback finite-difference step sizes tried in order if the initial step
# yields an ill-conditioned Hessian.
HESSIAN_STEP_FALLBACKS = (1e-4, 1e-3, 1e-2)


# ---------------------------------------------------------------------------
# Transforms + inverse + derivative
# ---------------------------------------------------------------------------
def to_unconstrained(x: float, transform: Optional[str]) -> float:
    """Natural parameter -> unconstrained real line."""
    if transform is None:
        return x
    if transform == "log":
        return float(np.log(x))
    if transform == "logit":
        return float(np.log(x / (1.0 - x)))
    if transform == "atanh":
        return float(np.arctanh(x))
    raise ValueError(f"Unknown transform: {transform!r}")


def from_unconstrained(x_tilde: float, transform: Optional[str]) -> float:
    """Unconstrained real line -> natural parameter."""
    if transform is None:
        return x_tilde
    if transform == "log":
        return float(np.exp(x_tilde))
    if transform == "logit":
        return float(1.0 / (1.0 + np.exp(-x_tilde)))
    if transform == "atanh":
        return float(np.tanh(x_tilde))
    raise ValueError(f"Unknown transform: {transform!r}")


def d_natural_d_tilde(x_natural: float, transform: Optional[str]) -> float:
    """dx / dx_tilde evaluated at x_natural. Used for delta-method SE on
    the natural scale: SE(x) = |dx/dx_tilde| * SE(x_tilde)."""
    if transform is None:
        return 1.0
    if transform == "log":
        # x_tilde = log(x) => dx/dx_tilde = x
        return float(x_natural)
    if transform == "logit":
        # x_tilde = log(x/(1-x)) => dx/dx_tilde = x * (1 - x)
        return float(x_natural * (1.0 - x_natural))
    if transform == "atanh":
        # x_tilde = atanh(x) => dx/dx_tilde = 1 - x^2
        return float(1.0 - x_natural ** 2)
    raise ValueError(f"Unknown transform: {transform!r}")


def _pytensor_from_unconstrained(x_tilde, transform: Optional[str]):
    """PyTensor version of from_unconstrained, used to build the compiled
    log-likelihood evaluator that accepts unconstrained inputs."""
    import pytensor.tensor as pt
    if transform is None:
        return x_tilde
    if transform == "log":
        return pt.exp(x_tilde)
    if transform == "logit":
        return pt.sigmoid(x_tilde)
    if transform == "atanh":
        return pt.tanh(x_tilde)
    raise ValueError(f"Unknown transform: {transform!r}")


# ---------------------------------------------------------------------------
# Compiled log-likelihood evaluators
# ---------------------------------------------------------------------------
def build_systematic_loglik(
    returns: np.ndarray,
    delta_t: float,
    transformed: bool = True,
) -> Callable:
    """Compile a log-likelihood callable f(theta) -> scalar for the
    systematic (SPX-only) model.

    If ``transformed`` (default), input is in the unconstrained transformed
    space (theta_tilde with logit/log applied per parameter). Otherwise,
    input is in natural space.
    """
    import pytensor
    import pytensor.tensor as pt

    theta = pt.dvector("theta")
    param_slots = {}
    for i, name in enumerate(SYSTEMATIC_PARAM_NAMES):
        entry = theta[i]
        if transformed:
            entry = _pytensor_from_unconstrained(entry, PARAM_TRANSFORMS[name])
        param_slots[name] = entry

    y_data = np.cumsum(returns).reshape((-1, 1))

    ll = KimYiLogLike(
        mui=np.array(0.0),
        kappai=np.array(0.0),
        gammai=np.array(1.0),
        betai=np.array(1.0),
        rhoix=np.array(0.0),
        alpha=param_slots["dALPHA"],
        sigma=param_slots["dSIGMA"],
        pprob=param_slots["dPPROB"],
        lamb=param_slots["dLAMB"],
        eta1=param_slots["dETA1"],
        eta2=param_slots["dETA2"],
        dt=np.array(delta_t),
    ).logp(y=pt.as_tensor(y_data))

    return pytensor.function([theta], pt.sum(ll))


def build_idiosyncratic_loglik(
    idi_returns: np.ndarray,
    params_sys: dict,
    delta_t: float,
    transformed: bool = True,
) -> Callable:
    """Compile a log-likelihood callable f(theta) -> scalar for the
    idiosyncratic model, conditional on the systematic parameters."""
    import pytensor
    import pytensor.tensor as pt

    theta = pt.dvector("theta")
    param_slots = {}
    for i, name in enumerate(IDIOSYNCRATIC_PARAM_NAMES):
        entry = theta[i]
        if transformed:
            entry = _pytensor_from_unconstrained(entry, PARAM_TRANSFORMS[name])
        param_slots[name] = entry

    y_data = np.cumsum(idi_returns).reshape((-1, 1))

    ll = KimYiLogLike(
        mui=param_slots["dMUI"],
        kappai=param_slots["dKAPPAI"],
        gammai=param_slots["dGAMMAI"],
        betai=param_slots["dBETAI"],
        rhoix=param_slots["dRHOIX"],
        alpha=np.array(params_sys["dALPHA"]),
        sigma=np.array(params_sys["dSIGMA"]),
        pprob=np.array(params_sys["dPPROB"]),
        lamb=np.array(params_sys["dLAMB"]),
        eta1=np.array(params_sys["dETA1"]),
        eta2=np.array(params_sys["dETA2"]),
        dt=np.array(delta_t),
    ).logp(y=pt.as_tensor(y_data))

    return pytensor.function([theta], pt.sum(ll))


# ---------------------------------------------------------------------------
# Hessian + Wald CI
# ---------------------------------------------------------------------------
def compute_numerical_hessian(
    f: Callable, x_star: np.ndarray, step: float = 1e-4,
) -> np.ndarray:
    """Numerical Hessian of scalar f at x_star via numdifftools."""
    try:
        import numdifftools as nd
    except ImportError as e:
        raise ImportError(
            "numdifftools is required for Wald CI computation. Install via:\n"
            "  conda install -c conda-forge numdifftools\n"
            "  or  pip install numdifftools"
        ) from e

    def f_plain(x):
        return float(f(x))

    return nd.Hessian(f_plain, step=step)(x_star)


def compute_hessian_with_fallback(
    f: Callable, x_star: np.ndarray,
    steps: Tuple[float, ...] = HESSIAN_STEP_FALLBACKS,
) -> Tuple[np.ndarray, float]:
    """Compute Hessian, retrying with coarser step sizes if the first
    attempt yields a non-positive-definite -H (some diag(cov) <= 0).

    Returns (hessian, step_used). The best Hessian is whichever gives
    the fewest non-PSD parameter directions (or the first if all tie).
    """
    best_hess = None
    best_step = None
    best_bad_count = np.inf
    for step in steps:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                hess = compute_numerical_hessian(f, x_star, step=step)
            except Exception as e:
                logger.warning("Hessian at step=%g failed: %s", step, e)
                continue

        try:
            cov = -np.linalg.inv(hess)
            bad_count = int((np.diag(cov) <= 0).sum())
        except np.linalg.LinAlgError:
            bad_count = len(x_star)

        if bad_count < best_bad_count:
            best_bad_count = bad_count
            best_hess = hess
            best_step = step
            if bad_count == 0:
                break  # perfect; no need to try coarser steps

    if best_hess is None:
        raise RuntimeError("All Hessian step sizes failed.")

    if best_bad_count > 0:
        logger.warning(
            "Best Hessian (step=%g) still has %d non-PSD direction(s).",
            best_step, best_bad_count,
        )

    return best_hess, best_step


def wald_ci_transformed(
    theta_hat_natural: np.ndarray,
    hessian_tilde: np.ndarray,
    z: float,
    param_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transformed-space Wald CI, back-transformed to natural scale.

    - theta_hat_natural: point estimates on the natural parameter scale.
    - hessian_tilde: log-likelihood Hessian on the transformed scale.
    - Returns (se_natural, lower_natural, upper_natural).

    SE(natural) is computed via the delta method:
        SE(x) = |dx/dx_tilde|_{x_hat} * SE(x_tilde)
    Lower / upper are back-transformed interval endpoints (interval-
    preserving; automatically respects bounds).
    """
    n = len(theta_hat_natural)
    se_natural = np.full(n, np.nan)
    lower_natural = np.full(n, np.nan)
    upper_natural = np.full(n, np.nan)

    try:
        cov_tilde = -np.linalg.inv(hessian_tilde)
    except np.linalg.LinAlgError:
        logger.warning("Transformed Hessian is singular; all SEs NaN.")
        return se_natural, lower_natural, upper_natural

    diag_cov = np.diag(cov_tilde)
    for i, name in enumerate(param_names):
        transform = PARAM_TRANSFORMS[name]
        x_hat_nat = float(theta_hat_natural[i])
        if diag_cov[i] <= 0:
            logger.warning(
                "Non-positive-definite Hessian at %s (diag(cov_tilde)[%d]=%.4g). "
                "SE and CI set to NaN.", name, i, diag_cov[i],
            )
            continue

        se_tilde = float(np.sqrt(diag_cov[i]))
        # Delta-method SE on the natural scale
        se_natural[i] = d_natural_d_tilde(x_hat_nat, transform) * se_tilde

        # Wald CI on the transformed scale, then back-transform endpoints
        x_hat_tilde = to_unconstrained(x_hat_nat, transform)
        lower_natural[i] = from_unconstrained(x_hat_tilde - z * se_tilde, transform)
        upper_natural[i] = from_unconstrained(x_hat_tilde + z * se_tilde, transform)

    return se_natural, lower_natural, upper_natural


def wald_ci_natural(
    theta_hat: np.ndarray,
    hessian: np.ndarray,
    z: float,
    param_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy natural-space Wald CI (may cross parameter bounds; may
    return NaN for parameters where -H is not PSD)."""
    n = len(theta_hat)
    se = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    try:
        cov = -np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        logger.warning("Hessian singular; cannot invert.")
        return se, lower, upper

    diag_cov = np.diag(cov)
    for i, name in enumerate(param_names):
        if diag_cov[i] <= 0:
            logger.warning(
                "Non-PSD at %s (diag(cov)[%d]=%.4g); SE=NaN.",
                name, i, diag_cov[i],
            )
            continue
        se[i] = np.sqrt(diag_cov[i])
        lower[i] = theta_hat[i] - z * se[i]
        upper[i] = theta_hat[i] + z * se[i]

    return se, lower, upper


# ---------------------------------------------------------------------------
# Per-date computation
# ---------------------------------------------------------------------------
def _compute_one(
    ticker: str,
    valuation_date: str,
    build_loglik: Callable,
    theta_hat_natural: np.ndarray,
    param_names: List[str],
    use_transformed: bool,
    hessian_step: Optional[float],
) -> pd.DataFrame:
    """Shared driver: build compiled log-likelihood, compute Hessian, get CI."""
    if use_transformed:
        theta_hat_input = np.array([
            to_unconstrained(x, PARAM_TRANSFORMS[name])
            for x, name in zip(theta_hat_natural, param_names)
        ])
    else:
        theta_hat_input = theta_hat_natural.copy()

    f_ll = build_loglik()

    if hessian_step is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hess = compute_numerical_hessian(f_ll, theta_hat_input, step=hessian_step)
        step_used = hessian_step
    else:
        hess, step_used = compute_hessian_with_fallback(f_ll, theta_hat_input)

    if use_transformed:
        se95, lo95, hi95 = wald_ci_transformed(theta_hat_natural, hess, Z_95, param_names)
        _,   lo99, hi99 = wald_ci_transformed(theta_hat_natural, hess, Z_99, param_names)
    else:
        se95, lo95, hi95 = wald_ci_natural(theta_hat_natural, hess, Z_95, param_names)
        _,   lo99, hi99 = wald_ci_natural(theta_hat_natural, hess, Z_99, param_names)

    return _rows_to_df(ticker, valuation_date, param_names,
                       theta_hat_natural, se95, lo95, hi95, lo99, hi99,
                       step_used=step_used, use_transformed=use_transformed)


def compute_systematic(
    ticker: str,
    valuation_date: str,
    returns: np.ndarray,
    delta_t: float,
    hessian_step: Optional[float],
    use_transformed: bool,
) -> pd.DataFrame:
    """Compute Wald CIs for one systematic (ticker, date)."""
    point = get_pmle_params(valuation_date, ticker)
    theta_hat = np.array([float(point[name]) for name in SYSTEMATIC_PARAM_NAMES])

    scheme = "transformed" if use_transformed else "natural"
    logger.info("[%s %s] Building systematic log-likelihood (%s space) ...",
                ticker, valuation_date, scheme)

    def build():
        return build_systematic_loglik(returns, delta_t, transformed=use_transformed)

    return _compute_one(
        ticker, valuation_date,
        build_loglik=build,
        theta_hat_natural=theta_hat,
        param_names=SYSTEMATIC_PARAM_NAMES,
        use_transformed=use_transformed,
        hessian_step=hessian_step,
    )


def compute_idiosyncratic(
    ticker: str,
    valuation_date: str,
    idi_returns: np.ndarray,
    systematic_ticker: str,
    delta_t: float,
    hessian_step: Optional[float],
    use_transformed: bool,
) -> pd.DataFrame:
    """Compute Wald CIs for one idiosyncratic (ticker, date), conditional
    on the same-date systematic parameter point estimates."""
    point = get_pmle_params(valuation_date, ticker)
    theta_hat = np.array([float(point[name]) for name in IDIOSYNCRATIC_PARAM_NAMES])

    sys_point = get_pmle_params(valuation_date, systematic_ticker)
    params_sys = {k: float(sys_point[k]) for k in SYSTEMATIC_PARAM_NAMES}

    scheme = "transformed" if use_transformed else "natural"
    logger.info("[%s %s] Building idiosyncratic log-likelihood conditional on %s (%s space) ...",
                ticker, valuation_date, systematic_ticker, scheme)

    def build():
        return build_idiosyncratic_loglik(idi_returns, params_sys, delta_t,
                                          transformed=use_transformed)

    return _compute_one(
        ticker, valuation_date,
        build_loglik=build,
        theta_hat_natural=theta_hat,
        param_names=IDIOSYNCRATIC_PARAM_NAMES,
        use_transformed=use_transformed,
        hessian_step=hessian_step,
    )


def _rows_to_df(ticker, valuation_date, param_names,
                theta_hat, se, lo95, hi95, lo99, hi99,
                step_used, use_transformed) -> pd.DataFrame:
    """Assemble per-parameter results with bound-violation flags."""
    rows = []
    for i, name in enumerate(param_names):
        lo, hi = PARAM_BOUNDS.get(name, (-np.inf, np.inf))
        bnd_95 = (
            (not np.isnan(lo95[i]) and lo95[i] < lo)
            or (not np.isnan(hi95[i]) and hi95[i] > hi)
        )
        bnd_99 = (
            (not np.isnan(lo99[i]) and lo99[i] < lo)
            or (not np.isnan(hi99[i]) and hi99[i] > hi)
        )
        rows.append({
            "sTICKER": ticker,
            "sVALUATION_DATE": valuation_date,
            "sPARAM": name,
            "dPOINT": theta_hat[i],
            "dSE": se[i],
            "dCI_LOWER_95": lo95[i],
            "dCI_UPPER_95": hi95[i],
            "dCI_LOWER_99": lo99[i],
            "dCI_UPPER_99": hi99[i],
            "bCI_95_EXCEEDS_BOUND": bnd_95,
            "bCI_99_EXCEEDS_BOUND": bnd_99,
            "sTRANSFORM": PARAM_TRANSFORMS[name] or "none",
            "bUSED_TRANSFORMED_SPACE": use_transformed,
            "dHESSIAN_STEP_USED": step_used,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Wald 95%% and 99%% CIs for P-MLE parameter "
                    "estimates using numerical Hessian at the MLE. "
                    "Default: transformed parameter space (interval-preserving, "
                    "bound-aware). --legacy-natural for the naive natural-space "
                    "computation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--valuation-date-beg", required=True,
                   help='First valuation date, YYYYMMDD or ISO. E.g. 2025-04-02')
    p.add_argument("--valuation-date-end", required=True,
                   help='Last valuation date (inclusive).')
    p.add_argument("--systematic-ticker", default="^SPX",
                   help="Systematic ticker (must match the P-MLE cache).")
    p.add_argument("--idiosyncratic-tickers", nargs="+", default=None,
                   help="Which idiosyncratic tickers to process. "
                        "Default: every ticker returned by get_idiosyncratic_ids().")
    p.add_argument("--systematic-only", action="store_true",
                   help="Skip idiosyncratic stage.")
    p.add_argument("--idiosyncratic-only", action="store_true",
                   help="Skip systematic stage.")
    p.add_argument("--lookback-days", type=int, default=252,
                   help="Return-history lookback window (business days). "
                        "Must match run_pmle_kimyi2025.py's setting.")
    p.add_argument("--base-days", type=int, default=252,
                   help="Base days in a year (for delta_t = 1 / base_days).")
    p.add_argument("--hessian-step", type=float, default=None,
                   help="Explicit finite-difference step for numerical Hessian. "
                        "Default: try %s in order and pick the best." % (HESSIAN_STEP_FALLBACKS,))
    p.add_argument("--legacy-natural", action="store_true",
                   help="Use naive natural-space Wald CI instead of transformed-space "
                        "(default). Retained for cross-comparison.")
    p.add_argument("--out", default=None,
                   help="Output Parquet path. "
                        "Default: Study/Estimated Parameters PMLE/wald_ci_kimyi2025.parquet")
    return p


def _to_yyyymmdd(date_str: str) -> str:
    return pd.Timestamp(date_str).strftime("%Y%m%d")


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.systematic_only and args.idiosyncratic_only:
        raise SystemExit("Cannot pass both --systematic-only and --idiosyncratic-only.")

    use_transformed = not args.legacy_natural

    valuation_beg = _to_yyyymmdd(args.valuation_date_beg)
    valuation_end = _to_yyyymmdd(args.valuation_date_end)
    valuation_window = pd.bdate_range(
        pd.to_datetime(valuation_beg, format="%Y%m%d"),
        pd.to_datetime(valuation_end, format="%Y%m%d"),
    )
    valuation_dates = [dt.strftime("%Y%m%d") for dt in valuation_window]

    delta_t = 1.0 / args.base_days
    systematic_id = args.systematic_ticker
    idiosyncratic_ids = args.idiosyncratic_tickers or get_idiosyncratic_ids()

    logger.info("Loading price panel for %s + %s ...", systematic_id, idiosyncratic_ids)
    price_ts = get_price_panel([systematic_id] + idiosyncratic_ids)
    return_ts = price_ts.pct_change().dropna()

    out_path = args.out or os.path.join(
        _REPO_ROOT, "Study", "Estimated Parameters PMLE", "wald_ci_kimyi2025.parquet"
    )

    all_rows: List[pd.DataFrame] = []

    if not args.idiosyncratic_only:
        for dt in valuation_dates:
            if not pmle_params_exists(dt, systematic_id):
                logger.warning("No P-MLE cache for %s %s -- skipping.", systematic_id, dt)
                continue
            return_vec = (
                return_ts.loc[return_ts.index <= dt, systematic_id]
                .iloc[-args.lookback_days:]
                .to_numpy()
            )
            try:
                df = compute_systematic(
                    ticker=systematic_id, valuation_date=dt,
                    returns=return_vec, delta_t=delta_t,
                    hessian_step=args.hessian_step,
                    use_transformed=use_transformed,
                )
                all_rows.append(df)
                _log_row_summary(df)
            except Exception as e:
                logger.exception("Failed for %s %s: %s", systematic_id, dt, e)

    if not args.systematic_only:
        for idi_id in idiosyncratic_ids:
            for dt in valuation_dates:
                if not pmle_params_exists(dt, idi_id):
                    logger.warning("No P-MLE cache for %s %s -- skipping.", idi_id, dt)
                    continue
                if not pmle_params_exists(dt, systematic_id):
                    logger.warning(
                        "Idiosyncratic %s %s needs cached systematic %s -- skipping.",
                        idi_id, dt, systematic_id,
                    )
                    continue
                return_vec = (
                    return_ts.loc[return_ts.index <= dt, idi_id]
                    .iloc[-args.lookback_days:]
                    .to_numpy()
                )
                try:
                    df = compute_idiosyncratic(
                        ticker=idi_id, valuation_date=dt,
                        idi_returns=return_vec,
                        systematic_ticker=systematic_id,
                        delta_t=delta_t,
                        hessian_step=args.hessian_step,
                        use_transformed=use_transformed,
                    )
                    all_rows.append(df)
                    _log_row_summary(df)
                except Exception as e:
                    logger.exception("Failed for %s %s: %s", idi_id, dt, e)

    if not all_rows:
        logger.warning("No results produced.")
        return

    result = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(result), out_path)

    csv_path = out_path.replace(".parquet", ".csv")
    result.to_csv(csv_path, index=False, float_format="%.6g")
    logger.info("Also wrote CSV: %s", csv_path)


def _log_row_summary(df: pd.DataFrame) -> None:
    ticker = df["sTICKER"].iloc[0]
    date = df["sVALUATION_DATE"].iloc[0]
    parts = []
    for _, row in df.iterrows():
        se_str = f"{row['dSE']:.4g}" if not np.isnan(row["dSE"]) else "NA"
        parts.append(f"{row['sPARAM']}={row['dPOINT']:.4g}±{se_str}")
    logger.info("  %s %s  %s", ticker, date, "  ".join(parts))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
