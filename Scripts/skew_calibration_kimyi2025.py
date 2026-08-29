"""
skew_calibration_kimyi2025 - Kim-Yi (2025) liquidity-adjusted jump-diffusion
calibration helper for the volatility skew calibration pipeline.

Auto-converted from skew_calibration_kimyi2025.ipynb.
"""

# Ensure package imports resolve regardless of cwd.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# --- logging ---
from Library.Logging import setup_logging
logger = setup_logging(__name__)
logger.info("Starting %s", __name__)

# %% cell 1
logger.info("[cell %d/14] compute", 1)
import os
import pickle
import numpy as np
import arviz as az
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import ustreasurycurve as ustcurve

from Library.Utility import UST_TENOR_MAP
from Library.OptionPricerBSM1973 import BlackScholesMertonPut
from Library.SkewCalibrationKimYi2025 import KimYiSkewCalibrationSystematic, KimYiSkewCalibrationIdiosyncratic, kimyi_vol_surface, _kimyi_imp_vol_call
from scipy.optimize import minimize
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

az.style.use("arviz-darkgrid")

# Path to raw Cboe implied-vol history. Override via LIQUIDITY_CBOE_DATA_DIR.
DATA_PATH_STR = os.environ.get(
    "LIQUIDITY_CBOE_DATA_DIR",
    str(_REPO_ROOT / "data" / "options"),
)
_REQUIRED_SUBDIRS = ["SPX", "COIN"]
_missing = [d for d in _REQUIRED_SUBDIRS if not os.path.isdir(os.path.join(DATA_PATH_STR, d))]
if _missing:
    logger.warning(
        "Raw Cboe implied-vol data not found at %s (missing subdirs: %s). "
        "Set LIQUIDITY_CBOE_DATA_DIR to your local Cboe dataset or see "
        "DATA_AVAILABILITY.md. Skipping this step.",
        DATA_PATH_STR, _missing,
    )
    raise SystemExit(0)
DATA_PATH_CALIB_RSLT_STR = str(_REPO_ROOT / "Study" / "Estimated Parameters QLSQ")
UND_TICKERS_DICT = {'systematic': '^SPX', 'idiosyncratic': ['COIN']}
DIVIDEND_YIELDS = {'^SPX': 1.25, 'COIN': 0.}
EXPIRY_MAP = {0: 11, 1: 10, 2: 9, 3: 8, 4: 7}

# %% cell 2
logger.info("[cell %d/14] compute valuation_date_array", 2)
valuation_date_array = pd.bdate_range('2025-03-18', '2025-04-17')
# https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield%27
rates_data_df = ustcurve.nominalRates(valuation_date_array[0].strftime("%Y-%m-%d"), valuation_date_array[-1].strftime("%Y-%m-%d")).set_index('date')
rates_data_df = pd.DataFrame(index=pd.date_range(rates_data_df.index.min(), rates_data_df.index.max())).join(rates_data_df).ffill(axis=1).bfill(axis=1).ffill().bfill()
rates_tenors = np.array([UST_TENOR_MAP[x] for x in rates_data_df.columns])
rates_data_df.head()

# %% cell 3
logger.info("[cell %d/14] compute PATH_STR", 3)
PATH_STR = os.path.join(DATA_PATH_STR, UND_TICKERS_DICT['systematic'].split("^")[-1])

mkt_vol_df1 = pd.concat([pd.read_csv(os.path.join(PATH_STR, file)) for file in sorted(os.listdir(PATH_STR))])

mkt_vol_df2 = []
for underlying_name in UND_TICKERS_DICT['idiosyncratic']:

    PATH_STR = os.path.join(DATA_PATH_STR, underlying_name)

    tmp_df = pd.concat([pd.read_csv(os.path.join(PATH_STR, file)) for file in sorted(os.listdir(PATH_STR))])

    mkt_vol_df2.append(tmp_df)

mkt_vol_df2 = pd.concat(mkt_vol_df2)
mkt_vol_df = pd.concat([mkt_vol_df1, mkt_vol_df2])

mkt_vol_df.quote_date = pd.to_datetime(mkt_vol_df.quote_date)
mkt_vol_df.expiration = pd.to_datetime(mkt_vol_df.expiration)

mkt_vol_df['iEXPIRY'] = (mkt_vol_df.expiration - mkt_vol_df.quote_date).dt.days
mkt_vol_df['dEXPIRY'] = mkt_vol_df.iEXPIRY / 365

mkt_vol_df['dUND_PRICE'] = mkt_vol_df.active_underlying_price_1545
mkt_vol_df['dUND_STRIKE'] = mkt_vol_df.strike
mkt_vol_df['dMONEYNESS'] = mkt_vol_df.dUND_STRIKE / mkt_vol_df.dUND_PRICE * 100.
mkt_vol_df['dMKT_IMP_VOL'] = mkt_vol_df.implied_volatility_1545

mask = mkt_vol_df.option_type=='C'
mkt_vol_df['bIS_CALL_OPTION'] = False
mkt_vol_df.loc[mask, 'bIS_CALL_OPTION'] = True

mask  = (mkt_vol_df.trade_volume > 0) & (mkt_vol_df.iEXPIRY > 5) & (mkt_vol_df.dMKT_IMP_VOL > 0)
mkt_vol_df = mkt_vol_df.loc[mask]

mask = (mkt_vol_df.option_type=='P') & (mkt_vol_df.dMONEYNESS <= 100.) & (mkt_vol_df.dMONEYNESS >= 50.)
mkt_vol_df1 = mkt_vol_df.loc[mask]
mask = (mkt_vol_df.option_type=='C') & (mkt_vol_df.dMONEYNESS >= 100.) & (mkt_vol_df.dMONEYNESS <= 150.)
mkt_vol_df2 = mkt_vol_df.loc[mask]

mkt_vol_df = pd.concat([mkt_vol_df1, mkt_vol_df2])

for t in mkt_vol_df.quote_date.unique():
    rate_fitter, _ = calibrate_nss_ols(rates_tenors, rates_data_df.xs(t).to_numpy() / 100)
    mask = (mkt_vol_df.quote_date==t)
    mkt_vol_df.loc[mask, 'dRISK_FREE_RATE'] = mkt_vol_df.loc[mask, 'dEXPIRY'].apply(rate_fitter)

mkt_vol_df['dDIVIDEND_YIELD'] = DIVIDEND_YIELDS[mkt_vol_df.underlying_symbol.iloc[0]] / 100.

vegas = []
for tup in mkt_vol_df.itertuples():
    vega = BlackScholesMertonPut(
        und_price=np.array(tup.dUND_PRICE),
        und_strike=np.array(tup.dUND_STRIKE),
        risk_free_rate=np.array(tup.dRISK_FREE_RATE),
        dividend_yield=np.array(tup.dDIVIDEND_YIELD),
        time_to_expiry=np.array(tup.dEXPIRY)
    ).vega(np.array(tup.dMKT_IMP_VOL)) / 100.
    vegas.append(vega[0, 0])

mkt_vol_df['dVEGA'] = vegas

# %% cell 4
logger.info("[cell %d/14] compute initial_values_systematic", 4)
initial_values_systematic = np.array([.2, .2, 5., 22., 7.])
initial_values_idiosyncratic = np.array([1., 2., 1., .5])

method = 'SLSQP'

bounds_systematic = {
    'dSIGMA': (1e-4, None),
    'dPPROB': (0., 1.),
    'dLAMB': (1e-4, None),
    'dETA1': (1.5, None),
    'dETA2': (0.5, None)
}

bounds_idiosyncratic = {
    'dKAPPAI': (1e-4, None),
    'dGAMMAI': (1e-1, None),
    'dBETAI': (1e-1, None),
    'dRHOIX': (-1., 1.)
}

# %% cell 5
logger.info("[cell %d/14] load cached kimyi2025 calibration", 5)
from Library.Serialization import load_calibration_results, save_calibration_results
_kimyi_calib_path = os.path.join(DATA_PATH_CALIB_RSLT_STR, "kimyi2025_vol_calibration")
_kimyi_calib_parquet = _kimyi_calib_path + ".parquet"
try:
    calib_results = load_calibration_results(_kimyi_calib_path)
    logger.info("Resuming from cached kimyi2025 calibration (%d keys)", len(calib_results))
except FileNotFoundError:
    calib_results = {}
    logger.info("No existing kimyi2025 calibration cache; starting fresh.")

# Warm-start ("x_hint") table for tricky (underlying, date) fits where the
# default SLSQP init lands in a shallow secondary basin. Values are the
# published Q-measure parameters from the paper's Figure 5 caption; using
# them as x0 nudges the optimizer into the correct basin without changing
# the objective or bounds. Keys are (underlying, YYYYMMDD).
IDIOSYNCRATIC_X0_HINTS = {
    ('COIN', '20250409'): np.array([0.61, 1.61, 0.94, 0.41]),
}

# %% cell 6
logger.info("[cell %d/14] Kim-Yi skew calibration loop (systematic + idiosyncratic)", 6)
systematic_name = UND_TICKERS_DICT['systematic']

# Optional single-date targeting. Set via --only-date YYYYMMDD or the
# LIQUIDITY_SKEW_ONLY_DATE env var to (a) restrict the loops to that one
# valuation date and (b) FORCE re-fit of that date's keys even if they are
# already present in the parquet cache. Handy for one-off reruns like
# repairing the COIN 2025-04-09 fit without redoing all dates.
_only_date = os.environ.get("LIQUIDITY_SKEW_ONLY_DATE")
import argparse as _argparse
_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument("--only-date", default=_only_date)
_args, _ = _p.parse_known_args()
_only_date = _args.only_date
if _only_date:
    logger.info("--only-date=%s: restricting to that valuation date and forcing re-fit.", _only_date)
    # Purge any cached entries for the targeted date so the loop re-runs them.
    _drop = [k for k in calib_results if k.endswith(f"-{_only_date}")]
    for _k in _drop:
        calib_results.pop(_k)
    logger.info("purged %d stale cache entries for %s: %s", len(_drop), _only_date, _drop)


def _skip_date(valuation_date):
    """True if --only-date is set and this valuation_date doesn't match."""
    return _only_date is not None and valuation_date.strftime("%Y%m%d") != _only_date


def _calibrate_systematic_for(valuation_date):
    valuation_date_str = valuation_date.strftime("%Y%m%d")
    key = f'{systematic_name}-{valuation_date_str}'
    logger.info('-------- %s --------', key)
    if key in calib_results:
        logger.info("skip (cached)")
        return

    T = EXPIRY_MAP[valuation_date.weekday()]
    if (valuation_date >= pd.to_datetime('20250407')) and (valuation_date < pd.to_datetime('20250414')):
        T = T - 1

    mask  = (mkt_vol_df.underlying_symbol==systematic_name)
    mask &= (mkt_vol_df.quote_date==valuation_date)
    mask &= (mkt_vol_df.iEXPIRY==T)
    mask &= (mkt_vol_df.dMONEYNESS >= 50.)
    mask &= (mkt_vol_df.dMONEYNESS <= 150.)
    tmp_df = mkt_vol_df.loc[mask].sort_values('dMONEYNESS')
    if len(tmp_df) == 0:
        logger.warning("no market vols for %s; skipping", key)
        return

    vol_fitter = KimYiSkewCalibrationSystematic(
        mkt_imp_vol=tmp_df.dMKT_IMP_VOL.to_numpy(),
        und_price=tmp_df.dUND_PRICE.to_numpy(),
        und_strike=tmp_df.dUND_STRIKE.to_numpy(),
        risk_free_rate=tmp_df.dRISK_FREE_RATE.to_numpy(),
        dividend_yield=tmp_df.dDIVIDEND_YIELD.to_numpy(),
        time_to_expiry=tmp_df.dEXPIRY.to_numpy(),
        is_call_option=tmp_df.bIS_CALL_OPTION.to_numpy(),
        option_weights=tmp_df.dVEGA.to_numpy() / tmp_df.dVEGA.sum(),
    )
    minimizer_results = minimize(
        vol_fitter.target,
        x0=initial_values_systematic,
        method=method,
        bounds=bounds_systematic.values(),
        tol=1e-6,
        options={'maxiter': 1e4},
    )
    logger.info("minimize result: %s", minimizer_results)
    calib_results[key] = minimizer_results
    save_calibration_results(calib_results, _kimyi_calib_parquet)


def _calibrate_idiosyncratic_for(underlying_name, valuation_date):
    valuation_date_str = valuation_date.strftime("%Y%m%d")
    key = f'{underlying_name}-{valuation_date_str}'
    logger.info('-------- %s --------', key)
    if key in calib_results:
        logger.info("skip (cached)")
        return

    T = EXPIRY_MAP[valuation_date.weekday()]
    if (valuation_date >= pd.to_datetime('20250407')) and (valuation_date < pd.to_datetime('20250414')):
        T = T - 1

    mask  = (mkt_vol_df.underlying_symbol==underlying_name)
    mask &= (mkt_vol_df.quote_date==valuation_date)
    mask &= (mkt_vol_df.iEXPIRY==T)
    mask &= (mkt_vol_df.dMONEYNESS >= 50.)
    mask &= (mkt_vol_df.dMONEYNESS <= 150.)
    tmp_df = mkt_vol_df.loc[mask].sort_values('dMONEYNESS')
    if len(tmp_df) == 0:
        logger.warning("no market vols for %s; skipping", key)
        return

    sys_key = f"{systematic_name}-{valuation_date_str}"
    if sys_key not in calib_results:
        logger.warning("systematic %s not calibrated yet; skipping %s", sys_key, key)
        return
    sigma, pprob, lamb, eta1, eta2 = calib_results[sys_key].x

    vol_fitter = KimYiSkewCalibrationIdiosyncratic(
        sigma=np.array(sigma),
        pprob=np.array(pprob),
        lamb=np.array(lamb),
        eta1=np.array(eta1),
        eta2=np.array(eta2),
        mkt_imp_vol=tmp_df.dMKT_IMP_VOL.to_numpy(),
        und_price=tmp_df.dUND_PRICE.to_numpy(),
        und_strike=tmp_df.dUND_STRIKE.to_numpy(),
        risk_free_rate=tmp_df.dRISK_FREE_RATE.to_numpy(),
        dividend_yield=tmp_df.dDIVIDEND_YIELD.to_numpy(),
        time_to_expiry=tmp_df.dEXPIRY.to_numpy(),
        is_call_option=tmp_df.bIS_CALL_OPTION.to_numpy(),
        option_weights=tmp_df.dVEGA.to_numpy() / tmp_df.dVEGA.sum(),
    )
    x0 = IDIOSYNCRATIC_X0_HINTS.get(
        (underlying_name, valuation_date_str),
        initial_values_idiosyncratic,
    )
    minimizer_results = minimize(
        vol_fitter.target,
        x0=x0,
        method=method,
        bounds=bounds_idiosyncratic.values(),
        tol=1e-6,
        options={'maxiter': 1e4},
    )
    logger.info("minimize result: %s", minimizer_results)
    calib_results[key] = minimizer_results
    save_calibration_results(calib_results, _kimyi_calib_parquet)


# Systematic first (idiosyncratic step consumes systematic params).
for valuation_date in valuation_date_array:
    if valuation_date in mkt_vol_df.quote_date.unique() and not _skip_date(valuation_date):
        _calibrate_systematic_for(valuation_date)

for underlying_name in UND_TICKERS_DICT['idiosyncratic']:
    for valuation_date in valuation_date_array:
        if valuation_date in mkt_vol_df.quote_date.unique() and not _skip_date(valuation_date):
            _calibrate_idiosyncratic_for(underlying_name, valuation_date)

# Final safety-net save (incremental saves already happened inside the helpers).
save_calibration_results(calib_results, _kimyi_calib_parquet)
logger.info("Saved kimyi2025 calibration to %s", _kimyi_calib_parquet)

# %% cell 7
logger.info("[cell %d/14] compute valuation_date", 7)
valuation_date = '20250409'
T = EXPIRY_MAP[pd.to_datetime(valuation_date).weekday()]
if (pd.to_datetime(valuation_date) >= pd.to_datetime('20250407')) and (pd.to_datetime(valuation_date) < pd.to_datetime('20250414')):
    T = T - 1
mask  = (mkt_vol_df.underlying_symbol==UND_TICKERS_DICT['systematic'])
mask &= (mkt_vol_df.quote_date==valuation_date) & (mkt_vol_df.iEXPIRY==T) & (mkt_vol_df.dMONEYNESS >= 50.) & (mkt_vol_df.dMONEYNESS <= 150.)
tmp_df = mkt_vol_df.loc[mask].sort_values('dMONEYNESS')
tmp_df.dMKT_IMP_VOL *= 100.
expiry_date = tmp_df.expiration.iloc[0]

# %% cell 8
logger.info("[cell %d/14] compute vol_fitter", 8)
vol_fitter = KimYiSkewCalibrationSystematic(
    mkt_imp_vol=tmp_df.dMKT_IMP_VOL.to_numpy() / 100.,
    und_price=tmp_df.dUND_PRICE.to_numpy(),
    und_strike=tmp_df.dUND_STRIKE.to_numpy(),
    risk_free_rate=tmp_df.dRISK_FREE_RATE.to_numpy(),
    dividend_yield=tmp_df.dDIVIDEND_YIELD.to_numpy(),
    time_to_expiry=tmp_df.dEXPIRY.to_numpy(),
    is_call_option=tmp_df.bIS_CALL_OPTION.to_numpy(),
    option_weights=tmp_df.dVEGA.to_numpy() / tmp_df.dVEGA.sum()
)

tmp_df['dMOD_IMP_VOL'] = vol_fitter.model_vol(x=calib_results[f"{UND_TICKERS_DICT['systematic']}-{valuation_date}"].x) * 100.

# %% cell 9
logger.info("[cell %d/14] fig, ax = plt.subplots(figsize=(15, 7))", 9)
fig, ax = plt.subplots(figsize=(15, 7))

sns.scatterplot(data=tmp_df.loc[~tmp_df.bIS_CALL_OPTION], x='dMONEYNESS', y='dMKT_IMP_VOL', label='Market Put',  ax=ax)
sns.scatterplot(data=tmp_df.loc[tmp_df.bIS_CALL_OPTION],  x='dMONEYNESS', y='dMKT_IMP_VOL', label='Market Call', ax=ax)
sns.lineplot(data=tmp_df.loc[~tmp_df.bIS_CALL_OPTION], x='dMONEYNESS', y='dMOD_IMP_VOL', label='Model Put',  ax=ax, color='r', marker='D')
sns.lineplot(data=tmp_df.loc[tmp_df.bIS_CALL_OPTION],  x='dMONEYNESS', y='dMOD_IMP_VOL', label='Model Call', ax=ax, color='purple', marker='D')

ax.set(xlabel=rf'$\bf Moneyness$ (%)', ylabel=r'$\bf \sigma_{\text{implied}}$')
ax.legend(title=r'$\bf \sigma_{\text{implied}}$')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax.set_title(
    f'{pd.to_datetime(valuation_date, format="%Y%m%d").strftime("%b %d, %Y")} Calibrated v Market Implied Volatility Skew of {UND_TICKERS_DICT["systematic"].split("^")[-1]} ' +
    f'Expiring on {pd.to_datetime(expiry_date, format="%Y%m%d").strftime("%b %d, %Y")}', weight='bold'
)
# plt.savefig(f"SPX_VOL_SKEW_{valuation_date_str}.pdf", dpi=300)
plt.show();

# %% cell 10
logger.info("[cell %d/14] valuation_date = '20250403'", 10)
# valuation_date = '20250403'
T = EXPIRY_MAP[pd.to_datetime(valuation_date).weekday()]
if (pd.to_datetime(valuation_date) >= pd.to_datetime('20250407')) and (pd.to_datetime(valuation_date) < pd.to_datetime('20250414')):
    T = T - 1
mask  = (mkt_vol_df.underlying_symbol==UND_TICKERS_DICT['idiosyncratic'][0])
mask &= (mkt_vol_df.quote_date==valuation_date) & (mkt_vol_df.iEXPIRY==T) & (mkt_vol_df.dMONEYNESS >= 50.) & (mkt_vol_df.dMONEYNESS <= 150.)
tmp_df = mkt_vol_df.loc[mask].sort_values('dMONEYNESS')
tmp_df.dMKT_IMP_VOL *= 100.
expiry_date = tmp_df.expiration.iloc[0]

# %% cell 11
logger.info("[cell %d/14] sigma, pprob, lamb, eta1, eta2 = calib_results[f'{UND_TICKER", 11)
sigma, pprob, lamb, eta1, eta2 = calib_results[f"{UND_TICKERS_DICT['systematic']}-{valuation_date}"].x

vol_fitter = KimYiSkewCalibrationIdiosyncratic(
    sigma=np.array(sigma),
    pprob=np.array(pprob),
    lamb=np.array(lamb),
    eta1=np.array(eta1),
    eta2=np.array(eta2),
    mkt_imp_vol=tmp_df.dMKT_IMP_VOL.to_numpy(),
    und_price=tmp_df.dUND_PRICE.to_numpy(),
    und_strike=tmp_df.dUND_STRIKE.to_numpy(),
    risk_free_rate=tmp_df.dRISK_FREE_RATE.to_numpy(),
    dividend_yield=tmp_df.dDIVIDEND_YIELD.to_numpy(),
    time_to_expiry=tmp_df.dEXPIRY.to_numpy(),
    is_call_option=tmp_df.bIS_CALL_OPTION.to_numpy(),
    option_weights=tmp_df.dVEGA.to_numpy() / tmp_df.dVEGA.sum()
)

tmp_df['dMOD_IMP_VOL'] = vol_fitter.model_vol(x=calib_results[f"{UND_TICKERS_DICT['idiosyncratic'][0]}-{valuation_date}"].x) * 100.

# %% cell 12
logger.info("[cell %d/14] fig, ax = plt.subplots(figsize=(15, 7))", 12)
fig, ax = plt.subplots(figsize=(15, 7))

sns.scatterplot(data=tmp_df.loc[~tmp_df.bIS_CALL_OPTION], x='dMONEYNESS', y='dMKT_IMP_VOL', label='Market Put',  ax=ax)
sns.scatterplot(data=tmp_df.loc[tmp_df.bIS_CALL_OPTION],  x='dMONEYNESS', y='dMKT_IMP_VOL', label='Market Call', ax=ax)
sns.lineplot(data=tmp_df.loc[~tmp_df.bIS_CALL_OPTION], x='dMONEYNESS', y='dMOD_IMP_VOL', label='Model Put',  ax=ax, color='r', marker='D')
sns.lineplot(data=tmp_df.loc[tmp_df.bIS_CALL_OPTION],  x='dMONEYNESS', y='dMOD_IMP_VOL', label='Model Call', ax=ax, color='purple', marker='D')

ax.set(xlabel=rf'$\bf Moneyness$ (%)', ylabel=r'$\bf \sigma_{\text{implied}}$')
ax.legend(title=r'$\bf \sigma_{\text{implied}}$')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax.set_title(
    f'{pd.to_datetime(valuation_date, format="%Y%m%d").strftime("%b %d, %Y")} Calibrated v Market Implied Volatility Skew of {UND_TICKERS_DICT["idiosyncratic"][0]} ' +
    f'Expiring on {pd.to_datetime(expiry_date, format="%Y%m%d").strftime("%b %d, %Y")}', weight='bold'
)
# plt.savefig(f"SPX_VOL_SKEW_{valuation_date_str}.pdf", dpi=300)
plt.show();

# %% cell 13
logger.info("[cell %d/14] compute underlying_name", 13)
underlying_name = UND_TICKERS_DICT['idiosyncratic'][0]
expiry_in_days = np.array([1, 7, 14, 30, 60, 90, 180, 365])
und_strikes = np.linspace(50, 150, 101).reshape((-1, 1))
und_spot = np.array(100.)

vol_surface_rslts = {}
# Only build the vol surface for dates that have both a systematic and an
# idiosyncratic calibration result -- otherwise --only-date runs (or partial
# reruns) crash on a missing key. Missing dates are simply skipped with a
# warning; the previously-saved vol surface parquet still contains them.
_sys_prefix = f"{UND_TICKERS_DICT['systematic']}-"
_idio_prefix = f"{underlying_name}-"
_available_dates = sorted({
    k.split("-", 1)[1] for k in calib_results
    if k.startswith(_sys_prefix)
} & {
    k.split("-", 1)[1] for k in calib_results
    if k.startswith(_idio_prefix)
})
_all_dates = [x.strftime("%Y%m%d") for x in valuation_date_array]
_missing = sorted(set(_all_dates) - set(_available_dates))
if _missing:
    logger.warning(
        "vol-surface loop: skipping %d date(s) missing systematic or "
        "idiosyncratic calibration: %s",
        len(_missing), _missing,
    )
for valuation_date in _available_dates:

    sigma, pprob, lamb, eta1, eta2 = calib_results[f"{UND_TICKERS_DICT['systematic']}-{valuation_date}"].x
    kappai, gammai, betai, rhoix = calib_results[f"{underlying_name}-{valuation_date}"].x
    rates = rates_data_df.xs(valuation_date).to_numpy().squeeze() * .01
    curve_fit, _ = calibrate_nss_ols(rates_tenors, rates)

    skew = []

    for expiry in expiry_in_days:

        expiry_in_years = expiry / 365

        vol_surface = kimyi_vol_surface(
            kappai=kappai,
            gammai=gammai,
            betai=betai,
            rhoix=rhoix,
            sigma=sigma,
            pprob=pprob,
            lamb=lamb,
            eta1=eta1,
            eta2=eta2,
            und_price=und_spot,
            und_strike=und_strikes,
            risk_free_rate=np.array(curve_fit(expiry_in_years)),
            dividend_yield=np.array(DIVIDEND_YIELDS[underlying_name]),
            time_to_expiry=expiry_in_years
        )

        skew.append(vol_surface.reshape((-1, )))

    vol_surface_rslts[f"{underlying_name}-{valuation_date}"] = {'iEXPIRY': expiry_in_days, 'dMONEYNESS': (und_strikes / und_spot).reshape((-1,)), 'dVOL': np.array(skew)}

from Library.Serialization import save_vol_surface
file_path = str(_REPO_ROOT / "Study" / "Vol Surface From Model" / "kimyi2025_vol_surface.parquet")
save_vol_surface(vol_surface_rslts, file_path)
logger.info("Saved kimyi2025 vol surface to %s", file_path)

# %% cell 14
logger.info("[cell %d/14] compute glrtvaluation_date", 14)
glrtvaluation_date = '20250409'
underlying_name = UND_TICKERS_DICT['idiosyncratic'][0]
tmp_dict = vol_surface_rslts[f"{underlying_name}-{valuation_date}"]
iEXPIRY = tmp_dict['iEXPIRY']
dMONEYNESS = tmp_dict['dMONEYNESS']
dVOL = tmp_dict['dVOL']

fig, ax = plt.subplots(figsize=(15, 7))

for i, d in enumerate(iEXPIRY):
    sns.lineplot(x=dMONEYNESS.reshape((-1,)) * 100., y=dVOL[i], label=f"{d}")

ax.set(xlabel=rf'$\bf Moneyness$ (%)', ylabel=r'$\bf \sigma_{\text{implied}}$')
ax.legend(title=r'$\bf Expiry (Days)$')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
ax.set_title(
    f'{pd.to_datetime(valuation_date, format="%Y%m%d").strftime("%b %d, %Y")} Model Implied Volatility Skew of {underlying_name}', weight='bold'
)
# plt.savefig(f"SPX_VOL_SKEW_{valuation_date_str}.pdf", dpi=300)
plt.show();
