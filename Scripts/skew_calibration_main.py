"""
skew_calibration_main - main driver for SPX/COIN volatility skew calibration.

Produces the calibrated model parameters and volatility skew plots (Figures 4 and 5
of the paper).

Auto-converted from skew_calibration_main.ipynb.
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

# --- centralised figures output directory ---
_FIGURES_DIR = _REPO_ROOT / "Figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% cell 1
logger.info("[cell %d/11] compute", 1)
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
from Library.SkewCalibrationHeston1993 import HestonSkewCalibration, feller_condition
from Library.SkewCalibrationKimYi2025 import KimYiSkewCalibrationSystematic, KimYiSkewCalibrationIdiosyncratic
from scipy.optimize import minimize
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

az.style.use("arviz-darkgrid")

# Path to the raw Cboe implied-volatility history. Configurable via the
# LIQUIDITY_CBOE_DATA_DIR environment variable so replicators can point at
# their own dataset without editing the script. Cboe data is proprietary
# and not shipped in this replication package; see DATA_AVAILABILITY.md.
DATA_PATH_INPUT_VOLS_STR = os.environ.get(
    "LIQUIDITY_CBOE_DATA_DIR",
    str(_REPO_ROOT / "data" / "options"),
)
# Verify the per-ticker subdirectories exist; the parent alone is not enough.
_REQUIRED_SUBDIRS = ["SPX", "COIN"]
_missing = [
    d for d in _REQUIRED_SUBDIRS
    if not os.path.isdir(os.path.join(DATA_PATH_INPUT_VOLS_STR, d))
]
if _missing:
    logger.warning(
        "Raw Cboe implied-vol data not found at %s (missing subdirs: %s). "
        "This script needs the Cboe LiveVol option-chain history to reproduce "
        "Figures 4-5. See DATA_AVAILABILITY.md for how to obtain it, or set "
        "the environment variable LIQUIDITY_CBOE_DATA_DIR to your local copy. "
        "The pre-generated PDFs already ship at "
        "Figures/Figure_4_SPX_VOL_SKEW_*.pdf and "
        "Figures/Figure_5_COIN_VOL_SKEW_*.pdf. Skipping this step.",
        DATA_PATH_INPUT_VOLS_STR, _missing,
    )
    # Exit code 0 so the reproduce_paper.py orchestrator treats it as skipped
    # rather than failed.
    raise SystemExit(0)

DATA_PATH_CALIB_RSLT_STR = str(_REPO_ROOT / "Study" / "Estimated Parameters QLSQ")
UND_TICKERS_DICT = {'systematic': '^SPX', 'idiosyncratic': ['COIN']}
DIVIDEND_YIELDS = {'^SPX': 1.25}
EXPIRY_MAP = {0: 11, 1: 10, 2: 9, 3: 8, 4: 7}

# %% cell 2
logger.info("[cell %d/11] compute _kimyi_calib_path", 2)
from Library.Serialization import load_calibration_results
_kimyi_calib_path = os.path.join(DATA_PATH_CALIB_RSLT_STR, "kimyi2025_vol_calibration")
_heston_calib_path = os.path.join(DATA_PATH_CALIB_RSLT_STR, "heston1993_vol_calibration")
calib_rslts_kimyi2025 = load_calibration_results(_kimyi_calib_path)
logger.info("Loaded kimyi2025 calibration from %s(.parquet|.pkl)", _kimyi_calib_path)
calib_rslts_heston1993 = load_calibration_results(_heston_calib_path)
logger.info("Loaded heston1993 calibration from %s(.parquet|.pkl)", _heston_calib_path)

# %% cell 3
logger.info("[cell %d/11] compute valuation_date_array", 3)
valuation_date_array = pd.bdate_range('2025-03-18', '2025-04-17')
# https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield%27
rates_data_df = ustcurve.nominalRates(valuation_date_array[0].strftime("%Y-%m-%d"), valuation_date_array[-1].strftime("%Y-%m-%d")).set_index('date')
rates_data_df = pd.DataFrame(index=pd.date_range(rates_data_df.index.min(), rates_data_df.index.max())).join(rates_data_df).ffill(axis=1).bfill(axis=1).ffill().bfill()
rates_tenors = np.array([UST_TENOR_MAP[x] for x in rates_data_df.columns])
rates_data_df.head()

# %% cell 4
logger.info("[cell %d/11] compute PATH_STR", 4)
PATH_STR = os.path.join(DATA_PATH_INPUT_VOLS_STR, UND_TICKERS_DICT['systematic'].split("^")[-1])

mkt_vol_df1 = pd.concat([pd.read_csv(os.path.join(PATH_STR, file)) for file in sorted(os.listdir(PATH_STR))])

mkt_vol_df2 = []
for underlying_name in UND_TICKERS_DICT['idiosyncratic']:

    PATH_STR = os.path.join(DATA_PATH_INPUT_VOLS_STR, underlying_name)

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

# %% cell 5
logger.info("[cell %d/11] compute valuation_date", 5)
valuation_date = '20250409'
systematic_name = UND_TICKERS_DICT['systematic']
T = EXPIRY_MAP[pd.to_datetime(valuation_date).weekday()]
if (pd.to_datetime(valuation_date) >= pd.to_datetime('20250407')) and (pd.to_datetime(valuation_date) < pd.to_datetime('20250414')):
    T = T - 1
mask  = (mkt_vol_df.underlying_symbol==systematic_name)
mask &= (mkt_vol_df.quote_date==valuation_date) & (mkt_vol_df.iEXPIRY==T) & (mkt_vol_df.dMONEYNESS >= 50.) & (mkt_vol_df.dMONEYNESS <= 150.)
tmp_df = mkt_vol_df.loc[mask].sort_values('dMONEYNESS')
tmp_df.dMKT_IMP_VOL *= 100.
expiry_date = tmp_df.expiration.iloc[0]

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

tmp_df['dMOD_IMP_VOL_KY'] = vol_fitter.model_vol(x=calib_rslts_kimyi2025[f"{systematic_name}-{valuation_date}"].x) * 100.

vol_fitter = HestonSkewCalibration(
    mkt_imp_vol=tmp_df.dMKT_IMP_VOL.to_numpy() / 100.,
    und_price=tmp_df.dUND_PRICE.to_numpy(),
    und_strike=tmp_df.dUND_STRIKE.to_numpy(),
    risk_free_rate=tmp_df.dRISK_FREE_RATE.to_numpy(),
    dividend_yield=tmp_df.dDIVIDEND_YIELD.to_numpy(),
    time_to_expiry=tmp_df.dEXPIRY.to_numpy(),
    is_call_option=tmp_df.bIS_CALL_OPTION.to_numpy(),
    option_weights=tmp_df.dVEGA.to_numpy() / tmp_df.dVEGA.sum()
)

tmp_df['dMOD_IMP_VOL_H'] = vol_fitter.model_vol(x=calib_rslts_heston1993[f"{systematic_name}-{valuation_date}"].x) * 100.

# %% cell 6
logger.info("[cell %d/11] calib_rslts_kimyi2025[f'{systematic_name}-{valuation_date}']", 6)
calib_rslts_kimyi2025[f"{systematic_name}-{valuation_date}"].x

# %% cell 7
logger.info("[cell %d/11] fig, ax = plt.subplots(figsize=(15, 7))", 7)
fig, ax = plt.subplots(figsize=(15, 7))

sns.scatterplot(data=tmp_df, x='dMONEYNESS', y='dMKT_IMP_VOL', label='Market', ax=ax, color='red', marker='+')
sns.lineplot(data=tmp_df,  x='dMONEYNESS', y='dMOD_IMP_VOL_KY', label='Liquidity Adjusted', ax=ax, marker='^')
# sns.lineplot(data=tmp_df,  x='dMONEYNESS', y='dMOD_IMP_VOL_H', label='Heston', ax=ax, marker='v')

ax.set(xlabel=rf'$\bf Moneyness$ (%)', ylabel=r'$\bf \sigma_{\text{implied}}$')
ax.legend(title=r'$\bf \sigma_{\text{implied}}$')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax.set_title(
    f'{pd.to_datetime(valuation_date, format="%Y%m%d").strftime("%b %d, %Y")} Market Implied versus Calibrated Volatility Skew of {systematic_name} ' +
    f'Expiring on {pd.to_datetime(expiry_date, format="%Y%m%d").strftime("%b %d, %Y")}', weight='bold'
)

plt.setp(ax.get_legend().get_title(), fontsize='16', fontweight='bold')
plt.savefig(str(_FIGURES_DIR / f"Figure_4_SPX_VOL_SKEW_{valuation_date}.pdf"), dpi=300)
plt.show();

# %% cell 8
logger.info("[cell %d/11] compute valuation_date", 8)
valuation_date = '20250409'
name = UND_TICKERS_DICT['idiosyncratic'][0]
T = EXPIRY_MAP[pd.to_datetime(valuation_date).weekday()]
if (pd.to_datetime(valuation_date) >= pd.to_datetime('20250407')) and (pd.to_datetime(valuation_date) < pd.to_datetime('20250414')):
    T = T - 1
mask  = (mkt_vol_df.underlying_symbol==name)
mask &= (mkt_vol_df.quote_date==valuation_date) & (mkt_vol_df.iEXPIRY==T) & (mkt_vol_df.dMONEYNESS >= 50.) & (mkt_vol_df.dMONEYNESS <= 150.)
tmp_df = mkt_vol_df.loc[mask].sort_values('dMONEYNESS')
tmp_df.dMKT_IMP_VOL *= 100.
expiry_date = tmp_df.expiration.iloc[0]

sigma, pprob, lamb, eta1, eta2 = calib_rslts_kimyi2025[f"{systematic_name}-{valuation_date}"].x

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

tmp_df['dMOD_IMP_VOL_KY'] = vol_fitter.model_vol(x=calib_rslts_kimyi2025[f"{name}-{valuation_date}"].x) * 100.

vol_fitter = HestonSkewCalibration(
    mkt_imp_vol=tmp_df.dMKT_IMP_VOL.to_numpy() / 100.,
    und_price=tmp_df.dUND_PRICE.to_numpy(),
    und_strike=tmp_df.dUND_STRIKE.to_numpy(),
    risk_free_rate=tmp_df.dRISK_FREE_RATE.to_numpy(),
    dividend_yield=tmp_df.dDIVIDEND_YIELD.to_numpy(),
    time_to_expiry=tmp_df.dEXPIRY.to_numpy(),
    is_call_option=tmp_df.bIS_CALL_OPTION.to_numpy(),
    option_weights=tmp_df.dVEGA.to_numpy() / tmp_df.dVEGA.sum()
)

tmp_df['dMOD_IMP_VOL_H'] = vol_fitter.model_vol(x=calib_rslts_heston1993[f"{name}-{valuation_date}"].x) * 100.

# %% cell 9
logger.info("[cell %d/11] calib_rslts_kimyi2025[f'{name}-{valuation_date}'].x", 9)
calib_rslts_kimyi2025[f"{name}-{valuation_date}"].x

# %% cell 10
logger.info("[cell %d/11] fig, ax = plt.subplots(figsize=(15, 7))", 10)
fig, ax = plt.subplots(figsize=(15, 7))

sns.scatterplot(data=tmp_df, x='dMONEYNESS', y='dMKT_IMP_VOL', label='Market', ax=ax, color='red', marker='+')
sns.lineplot(data=tmp_df,  x='dMONEYNESS', y='dMOD_IMP_VOL_KY', label='Liquidity Adjusted', ax=ax, marker='^')
sns.lineplot(data=tmp_df,  x='dMONEYNESS', y='dMOD_IMP_VOL_H', label='Heston (1993)', ax=ax, marker='v')

ax.set(xlabel=rf'$\bf Moneyness$ (%)', ylabel=r'$\bf \sigma_{\text{implied}}$')
ax.legend(title=r'$\bf \sigma_{\text{implied}}$')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax.set_title(
    f'{pd.to_datetime(valuation_date, format="%Y%m%d").strftime("%b %d, %Y")} Market Implied versus Calibrated Volatility Skew of {UND_TICKERS_DICT["idiosyncratic"][0]} ' +
    f'Expiring on {pd.to_datetime(expiry_date, format="%Y%m%d").strftime("%b %d, %Y")}', weight='bold'
)

plt.setp(ax.get_legend().get_title(), fontsize='16', fontweight='bold')
plt.savefig(str(_FIGURES_DIR / f"Figure_5_COIN_VOL_SKEW_{valuation_date}.pdf"), dpi=300)
plt.show();

# %% cell 11
logger.info("[cell %d/11] ", 11)

