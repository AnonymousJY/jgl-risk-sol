"""
Scripts/plot_skew_calibration.py -- regenerate the paper's Figure 4 (SPX)
and Figure 5 (COIN) style plots: market implied vs Kim-Yi (2025) liquidity-
adjusted volatility skew for one (ticker, valuation date, tenor).

Reads calibrated parameters from the QLSQ Parquet cache (produced by
Scripts/skew_calibrate_{systematic,idiosyncratic,all}.py), reconstructs
the market smile via the same LOADERS + attach_risk_free_rate +
attach_vega_weights pipeline the calibration uses, computes the model
IV via KimYiSkewCalibration*.model_vol(x=cached_params), and plots both
overlaid in the paper's style.

Run from the repository root:

    # Paper's Figure 4 (SPX, April 9 2025, 8-DTE)
    python Scripts/plot_skew_calibration.py \\
        --ticker '^SPX' --valuation-date 2025-04-09 --tenor 8

    # Paper's Figure 5 (COIN, April 9 2025, 8-DTE)
    python Scripts/plot_skew_calibration.py \\
        --ticker COIN --valuation-date 2025-04-09 --tenor 8

    # Save to a specific path (PDF preserves vector quality for LaTeX)
    python Scripts/plot_skew_calibration.py \\
        --ticker '^SPX' --valuation-date 2025-04-09 --tenor 8 \\
        --out Figures/SPX_VOL_SKEW_20250409.pdf
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config_skew as cfg  # Scripts/config_skew.py
from Library.SkewCalibrationKimYi2025 import (
    KimYiSkewCalibrationIdiosyncratic,
    KimYiSkewCalibrationSystematic,
)
from Scripts.skew_calibrate_systematic import (
    IDIOSYNCRATIC_PARAM_NAMES,
    LOADERS,
    SYSTEMATIC_PARAM_NAMES,
    attach_risk_free_rate,
    attach_vega_weights,
    cache_path,
    configure_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache + market data lookup
# ---------------------------------------------------------------------------
def load_calibrated_params(ticker: str, valuation_date_str: str, tenor: int) -> pd.Series:
    """Read one cached calibration row for (ticker, date, tenor)."""
    df = pd.read_parquet(cache_path())
    hit = df[
        (df["sTICKER"] == ticker)
        & (df["sVALUATION_DATE"] == valuation_date_str)
        & (df["iEXPIRY"] == int(tenor))
    ]
    if len(hit) == 0:
        available = df.loc[df["sTICKER"] == ticker, ["sVALUATION_DATE", "iEXPIRY"]].to_string(index=False)
        raise ValueError(
            f"No cached calibration for ({ticker!r}, {valuation_date_str!r}, tenor={tenor}). "
            f"Available for {ticker!r}:\n{available}"
        )
    return hit.iloc[0]


def load_market_smile(ticker: str, valuation_date: str, tenor: int) -> pd.DataFrame:
    """Load and filter one date's option chain for one tenor, matching the
    calibration pipeline exactly (same loader, same filters, same vega weights)."""
    if ticker == cfg.SYSTEMATIC_UNDERLYING["ticker"]:
        meta = cfg.SYSTEMATIC_UNDERLYING
    elif ticker in cfg.IDIOSYNCRATIC_UNDERLYINGS:
        meta = cfg.IDIOSYNCRATIC_UNDERLYINGS[ticker]
    else:
        raise KeyError(
            f"{ticker!r} not configured. Add to SYSTEMATIC_UNDERLYING or "
            f"IDIOSYNCRATIC_UNDERLYINGS in config_skew.py."
        )

    loader = LOADERS[cfg.DATA_SOURCE]
    df = loader(meta, ticker)
    df = attach_risk_free_rate(df)
    df = attach_vega_weights(df)

    valuation_ts = pd.Timestamp(valuation_date)
    smile = df[
        (df["underlying_symbol"] == ticker)
        & (df["quote_date"] == valuation_ts)
        & (df["iEXPIRY"] == int(tenor))
    ].sort_values("dMONEYNESS").reset_index(drop=True)

    if len(smile) == 0:
        raise ValueError(
            f"No market smile for ({ticker!r}, {valuation_date!r}, tenor={tenor})."
        )
    return smile


# ---------------------------------------------------------------------------
# Model IV reconstruction
# ---------------------------------------------------------------------------
def compute_model_iv(smile: pd.DataFrame, params: pd.Series, is_systematic: bool) -> np.ndarray:
    """Call KimYiSkewCalibration*.model_vol(x=cached_params) to get the
    model-implied vol at each strike in the smile. Returns an array aligned
    with the model_vol() output ordering (puts first, then calls -- see
    Library/SkewCalibrationKimYi2025.py)."""
    common_kwargs = dict(
        mkt_imp_vol=smile["dMKT_IMP_VOL"].to_numpy(),
        und_price=smile["dUND_PRICE"].to_numpy(),
        und_strike=smile["dUND_STRIKE"].to_numpy(),
        risk_free_rate=smile["dRISK_FREE_RATE"].to_numpy(),
        dividend_yield=smile["dDIVIDEND_YIELD"].to_numpy(),
        time_to_expiry=smile["dEXPIRY"].to_numpy(),
        is_call_option=smile["bIS_CALL_OPTION"].to_numpy(),
        option_weights=smile["dVEGA"].to_numpy() / smile["dVEGA"].sum(),
    )

    if is_systematic:
        fitter = KimYiSkewCalibrationSystematic(**common_kwargs)
        x = np.array([params[p] for p in SYSTEMATIC_PARAM_NAMES])
    else:
        fitter = KimYiSkewCalibrationIdiosyncratic(
            sigma=np.array(params["dSIGMA"]),
            pprob=np.array(params["dPPROB"]),
            lamb=np.array(params["dLAMB"]),
            eta1=np.array(params["dETA1"]),
            eta2=np.array(params["dETA2"]),
            **common_kwargs,
        )
        x = np.array([params[p] for p in IDIOSYNCRATIC_PARAM_NAMES])

    return fitter.model_vol(x=x).reshape(-1)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(ticker: str, valuation_date: str, tenor: int, out_path: Optional[str]) -> None:
    """Regenerate the paper's Figure 4 (SPX) / Figure 5 (COIN) style plot for
    the given (ticker, date, tenor). Saves to out_path if provided; also
    shows interactively if matplotlib backend supports it."""
    # Heavy imports deferred so the file's docstring/argparse loads fast.
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    valuation_date_str = pd.Timestamp(valuation_date).strftime("%Y%m%d")
    is_systematic = (ticker == cfg.SYSTEMATIC_UNDERLYING["ticker"])

    logger.info("Loading calibrated params for %s %s tenor=%d", ticker, valuation_date_str, tenor)
    params = load_calibrated_params(ticker, valuation_date_str, tenor)

    logger.info("Loading market smile ...")
    smile = load_market_smile(ticker, valuation_date, tenor)
    logger.info("Smile has %d option points", len(smile))

    logger.info("Computing model IV via cached parameters ...")
    model_iv = compute_model_iv(smile, params, is_systematic)

    # model_vol() stacks puts first then calls (see SkewCalibrationKimYi2025.py).
    # Reorder the smile the same way so model_iv aligns row-for-row.
    puts = smile[~smile["bIS_CALL_OPTION"]].sort_values("dMONEYNESS")
    calls = smile[smile["bIS_CALL_OPTION"]].sort_values("dMONEYNESS")
    plot_df = pd.concat([puts, calls]).reset_index(drop=True)
    plot_df["dMKT_IV_PCT"] = plot_df["dMKT_IMP_VOL"] * 100.0
    plot_df["dMOD_IV_PCT"] = model_iv * 100.0

    expiry_date = pd.Timestamp(valuation_date) + pd.Timedelta(days=int(tenor))

    # Paper's style: arviz-darkgrid theme, red '+' markers for market,
    # line with '^' markers for the liquidity-adjusted model, bold labels.
    try:
        import arviz as az
        az.style.use("arviz-darkgrid")
    except ImportError:
        # arviz is optional here; fall back to seaborn's grid style.
        sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(15, 7))
    sns.scatterplot(data=plot_df, x="dMONEYNESS", y="dMKT_IV_PCT",
                    label="Market", ax=ax, color="red", marker="+")
    sns.lineplot(data=plot_df, x="dMONEYNESS", y="dMOD_IV_PCT",
                 label="Liquidity Adjusted", ax=ax, marker="^")

    ax.set(xlabel=r"$\bf Moneyness$ (%)", ylabel=r"$\bf \sigma_{\text{implied}}$ (%)")
    ax.legend(title=r"$\bf \sigma_{\text{implied}}$")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.set_title(
        f'{pd.Timestamp(valuation_date).strftime("%b %d, %Y")} Market Implied versus '
        f'Calibrated Volatility Skew of {ticker} '
        f'Expiring on {expiry_date.strftime("%b %d, %Y")}',
        weight="bold",
    )
    plt.setp(ax.get_legend().get_title(), fontsize="16", fontweight="bold")

    # Report the residual so the user can eyeball fit quality at a glance.
    # NaN-aware: some strikes may return NaN model IV when the Kim-Yi model
    # produces prices outside the BSM-invertible range; those points are
    # excluded from the residual average and reported separately.
    mkt_iv = plot_df["dMKT_IMP_VOL"].to_numpy()
    vega   = plot_df["dVEGA"].to_numpy()
    resid_sq = (mkt_iv - model_iv) ** 2
    finite = np.isfinite(resid_sq)
    n_nan = int((~finite).sum())
    if finite.any():
        w = vega[finite] / vega[finite].sum()
        weighted_resid = np.sqrt(np.average(resid_sq[finite], weights=w))
        logger.info(
            "RMS residual (vega-weighted, decimal IV units): %.4f  (=%.2f vol points)",
            weighted_resid, weighted_resid * 100.0,
        )
    else:
        logger.warning("All strikes returned NaN model IV; cannot compute RMS residual.")
    if n_nan:
        logger.warning(
            "%d of %d strikes had non-finite model IV (Kim-Yi price outside BSM-invertible range); "
            "excluded from residual average and dropped from the plot.",
            n_nan, len(model_iv),
        )
    logger.info("Cached dOBJECTIVE: %.6f", float(params["dOBJECTIVE"]))

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("Saved figure to %s", out_path)

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regenerate the paper's Figure 4 (SPX) / Figure 5 (COIN) "
                    "skew plot from the QLSQ Parquet cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", required=True,
                   help='e.g. "^SPX" or "COIN"')
    p.add_argument("--valuation-date", required=True,
                   help='ISO date, e.g. "2025-04-09"')
    p.add_argument("--tenor", type=int, required=True,
                   help="iEXPIRY in calendar days (must match a cached row)")
    p.add_argument("--out", default=None,
                   help="Output path (.pdf recommended for LaTeX). "
                        "Default: Figures/<TICKER>_VOL_SKEW_<YYYYMMDD>.pdf")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out_path = args.out
    if out_path is None:
        safe_ticker = args.ticker.replace("^", "")
        yyyymmdd = pd.Timestamp(args.valuation_date).strftime("%Y%m%d")
        out_path = os.path.join(_REPO_ROOT, "Figures", f"{safe_ticker}_VOL_SKEW_{yyyymmdd}.pdf")
    make_plot(args.ticker, args.valuation_date, args.tenor, out_path)


if __name__ == "__main__":
    configure_logging()
    main()
