"""
skew_calibration_heston1993 - Heston (1993) baseline calibration helper for the
volatility skew calibration pipeline.

Auto-converted from skew_calibration_heston1993.ipynb.
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
logger.info("[cell %d/11] compute", 1)
import os
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
UND_TICKERS_DICT = {'systematic': '^SPX', 'idiosyncratic': ['COIN']}
DIVIDEND_YIELDS = {'^SPX': 1.25}
EXPIRY_MAP = {0: 11, 1: 10, 2: 9, 3: 8, 4: 7}

# %% cell 2
logger.info("[cell %d/11] compute valuation_date_array", 2)
valuation_date_array = pd.bdate_range('2025-03-18', '2025-04-17')
# https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield%27
rates_data_df = ustcurve.nominalRates(valuation_date_array[0].strftime("%Y-%m-%d"), valuation_date_array[-1].strftime("%Y-%m-%d")).set_index('date')
rates_data_df = pd.DataFrame(index=pd.date_range(rates_data_df.index.min(), rates_data_df.index.max())).join(rates_data_df).ffill(axis=1).bfill(axis=1).ffill().bfill()
rates_tenors = np.array([UST_TENOR_MAP[x] for x in rates_data_df.columns])
rates_data_df.head()

# %% cell 3
logger.info("[cell %d/11] compute PATH_STR", 3)
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
logger.info("[cell %d/11] compute initial_values_systematic", 4)
initial_values_systematic = np.array([.25, 3., .25, .5, -.5, .1])

method = 'SLSQP'

bounds_systematic = {
    'dV0': (1e-3, 5.),
    'dKAPPA': (1e-3, 5.),
    'dTHETA': (1e-3, 5.),
    'dSIGMA': (1e-2, 10.),
    'dRHOSV': (-1., 1.),
    'dLAMBD': (-1., 1.)
}

# Load any previously-persisted calibration results so we can resume after
# a crash instead of redoing hours of SLSQP work.
from Library.Serialization import save_calibration_results, load_calibration_results
_heston_calib_path = str(_REPO_ROOT / "Study" / "Estimated Parameters QLSQ" / "heston1993_vol_calibration")
try:
    calib_results = load_calibration_results(_heston_calib_path)
    logger.info("Resuming from cached heston1993 calibration (%d keys)", len(calib_results))
except FileNotFoundError:
    calib_results = {}
    logger.info("No prior heston1993 calibration cache; starting fresh.")

_heston_calib_parquet = _heston_calib_path + ".parquet"


def _calibrate_heston_for(name, valuation_date):
    """Run SLSQP Heston calibration for one (ticker, date) and update cache.

    Saves calib_results to parquet after every successful minimize so a later
    crash cannot lose completed work.
    """
    valuation_date_str = valuation_date.strftime("%Y%m%d")
    key = f'{name}-{valuation_date_str}'
    logger.info('-------- %s --------', key)
    if key in calib_results:
        logger.info("skip (cached)")
        return

    T = EXPIRY_MAP[valuation_date.weekday()]
    if (valuation_date >= pd.to_datetime('20250407')) and (valuation_date < pd.to_datetime('20250414')):
        T = T - 1

    mask  = (mkt_vol_df.underlying_symbol==name)
    mask &= (mkt_vol_df.quote_date==valuation_date)
    mask &= (mkt_vol_df.iEXPIRY==T)
    mask &= (mkt_vol_df.dMONEYNESS >= 50.)
    mask &= (mkt_vol_df.dMONEYNESS <= 150.)
    tmp_df = mkt_vol_df.loc[mask].sort_values('dMONEYNESS')
    if len(tmp_df) == 0:
        logger.warning("no market vols for %s; skipping", key)
        return

    vol_fitter = HestonSkewCalibration(
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
        constraints=({'type': 'ineq', 'fun': feller_condition}),
    )
    logger.info("minimize result: %s", minimizer_results)
    calib_results[key] = minimizer_results
    # Persist after every successful key so a later crash cannot lose work.
    save_calibration_results(calib_results, _heston_calib_parquet)


# %% cell 5
logger.info("[cell %d/11] compute systematic_name", 5)
systematic_name = UND_TICKERS_DICT['systematic']

# Calibrate the systematic underlying (SPX) and every idiosyncratic
# underlying listed in UND_TICKERS_DICT (e.g. COIN). Figure 5 in the paper
# overlays the Heston fit against COIN market vols, so both are required.
for _name in [systematic_name, *UND_TICKERS_DICT['idiosyncratic']]:
    for valuation_date in valuation_date_array:
        if valuation_date in mkt_vol_df.quote_date.unique():
            _calibrate_heston_for(_name, valuation_date)

# %% cell 6
logger.info("[cell %d/11] compute valuation_date", 6)
valuation_date = '20250409'
name = 'COIN'
T = EXPIRY_MAP[pd.to_datetime(valuation_date).weekday()]
if (pd.to_datetime(valuation_date) >= pd.to_datetime('20250407')) and (pd.to_datetime(valuation_date) < pd.to_datetime('20250414')):
    T = T - 1
mask  = (mkt_vol_df.underlying_symbol==name)
mask &= (mkt_vol_df.quote_date==valuation_date) & (mkt_vol_df.iEXPIRY==T) & (mkt_vol_df.dMONEYNESS >= 50.) & (mkt_vol_df.dMONEYNESS <= 150.)
tmp_df = mkt_vol_df.loc[mask].sort_values('dMONEYNESS')
tmp_df.dMKT_IMP_VOL *= 100.
expiry_date = tmp_df.expiration.iloc[0]

# %% cell 7
logger.info("[cell %d/11] compute vol_fitter", 7)
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

tmp_df['dMOD_IMP_VOL'] = vol_fitter.model_vol(x=calib_results[f"{name}-{valuation_date}"].x) * 100.

tmp_df

# %% cell 8
logger.info("[cell %d/11] calib_results[f'{name}-{valuation_date}']", 8)
calib_results[f"{name}-{valuation_date}"]

# %% cell 9
logger.info("[cell %d/11] fig, ax = plt.subplots(figsize=(15, 7))", 9)
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
    f'{pd.to_datetime(valuation_date, format="%Y%m%d").strftime("%b %d, %Y")} Calibrated v Market Implied Volatility Skew of {systematic_name} ' +
    f'Expiring on {pd.to_datetime(expiry_date, format="%Y%m%d").strftime("%b %d, %Y")}', weight='bold'
)
# plt.savefig(f"SPX_VOL_SKEW_{valuation_date_str}.pdf", dpi=300)
plt.show();

# %% cell 10
logger.info("[cell %d/11] compute file_path", 10)
# Incremental saves happen inside _calibrate_heston_for(); this is a final
# safety-net save that also covers the (unlikely) case of no calibrations
# having been added this run (e.g. all keys already cached).
save_calibration_results(calib_results, _heston_calib_parquet)
logger.info("Saved heston1993 calibration to %s", _heston_calib_parquet)

# %% cell 11
logger.info("[cell %d/11] ", 11)

