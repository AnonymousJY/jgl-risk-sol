"""
run_var_kimyi2025 - Liquidity-Adjusted Value-at-Risk simulation and figure
generation for the paper's COIN collar / Asian derivative portfolio.

Produces:
  - Table 2: Comparison of Liquidity-Adjusted (VaR_LA) and Baseline (VaR_BS)
  - Figure 3: VaR surface (sensitivity to gamma_i, beta_i, rho_i,X)
  - Figure 7: VaR term structure over risk horizons

Data mode: uses committed snapshots via Library.DataAccess.

P-MLE parameters are loaded from cached per-date CSVs under
``Study/Estimated Parameters PMLE/``. Run Scripts/run_pmle_kimyi2025.py first
to populate the cache for any missing dates.

Auto-converted from run_var_kimyi2025.ipynb with FinanceDataReader and
psycopg2 dependencies removed. The original notebook queried a PostgreSQL
database for P-MLE parameters and portfolio definitions; these are now
served by Library.DataAccess (per-date CSVs) and Scripts.load_portfolio
(build_portfolio()).

Portions of the original notebook that queried or wrote to the database
have been REPLACED with the equivalent DataAccess calls. Any TODOs are
flagged in the code.
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

# --- warning filters ---
# Silence two known-benign warnings that would otherwise flood the log:
#   (a) ustreasurycurve uses BeautifulSoup's HTML parser on an XML feed.
#   (b) pandas .ffill/.bfill downcasting is a cosmetic deprecation notice.
import warnings as _warnings
try:
    from bs4 import XMLParsedAsHTMLWarning as _XMLParsedAsHTMLWarning
    _warnings.filterwarnings("ignore", category=_XMLParsedAsHTMLWarning)
except ImportError:
    pass
_warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated",
    category=FutureWarning,
)

# --- output paths (absolute, anchored at repo root) ---
_OUTPUT_DIR = _REPO_ROOT / "Study" / "Collar Asian"
_SCENARIOS_DIR = _REPO_ROOT / "Scripts" / "Scenarios"
_CURRENT_PV_DIR_TEMPLATE = _OUTPUT_DIR / "CurrentPV"
_SCENARIO_PV_DIR_TEMPLATE = _OUTPUT_DIR / "ScenarioPV"
for _d in (_OUTPUT_DIR, _SCENARIOS_DIR, _CURRENT_PV_DIR_TEMPLATE, _SCENARIO_PV_DIR_TEMPLATE):
    _d.mkdir(parents=True, exist_ok=True)
# --- centralised figures output directory ---
_FIGURES_DIR = _REPO_ROOT / "Figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
_REPO_ROOT_STR = str(_REPO_ROOT) + "/"

# %% cell 1
logger.info("[cell %d/27] compute", 1)
import os
import pickle
import numpy as np
import pandas as pd
from Library.DataAccess import get_pmle_params_dict, get_price_panel, get_price_series, pmle_params_exists

import ustreasurycurve as ustcurve

from typing import List
from collections import namedtuple
from itertools import repeat, product
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

from load_portfolio import get_idiosyncratic_ids
from Library.PayoffFactory import payoff_mc_factory
from Library.Utility import year_frac, UST_TENOR_MAP, get_fixings_vec
from Library.Interpolation import pchip_interpolator2d
from Library.Parameters import ParametersConstant
from Library.Random import RandomMT19937
from Library.StatisticsMC import StatisticsMCMean
from Library.ExoticEngine import ExoticEngineBlackScholesMerton
from Library.OptionPricerBSM1973 import BlackScholesMertonCall, BlackScholesMertonPut
from Library.RiskEngineKimYi2025 import pmle_kimyirisk_systematic, pmle_kimyirisk_idiosyncratic, KimYiRiskEngine
from Library.PathDependent import PathDependentAsianDiscrete

import matplotlib
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

az.style.use("arviz-darkgrid")

# %% cell 2
logger.info("[cell %d/27] compute valuation_beg_dt", 2)
valuation_beg_dt = '20250331'
valuation_end_dt = '20250416'
date_format = '%Y%m%d'
valuation_window = pd.bdate_range(pd.to_datetime(arg=valuation_beg_dt, format=date_format), pd.to_datetime(arg=valuation_end_dt, format=date_format))
valuation_window_str = [dt.strftime(date_format) for dt in valuation_window]

DIVIDEND_YIELDS = {'^SPX': 1.25, 'COIN': 0.}

lookback_period = 252
base_days = 252
delta_t = np.array(1 / base_days)
seed_number = np.uint64(20240114)
n_mc_paths = int(10_000)
m_steps = int(base_days)

rng = RandomMT19937(seed=seed_number)

systematic_id = '^SPX'
id_dict = {'systematic_id': systematic_id, 'idiosyncratic_ids': get_idiosyncratic_ids()}

# Price panel via Library.DataAccess (snapshot mode by default).
price_ts = get_price_panel([id_dict['systematic_id']] + id_dict['idiosyncratic_ids'])
price_ts.index = pd.to_datetime(price_ts.index)
return_ts = price_ts.pct_change().dropna()

# https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield%27
rates_data_df = ustcurve.nominalRates(pd.to_datetime(valuation_window_str[0]).strftime("%Y-%m-%d"), pd.to_datetime(valuation_window_str[-1]).strftime("%Y-%m-%d")).set_index('date')
rates_data_df = pd.DataFrame(index=pd.bdate_range(rates_data_df.index.min(), rates_data_df.index.max())).join(rates_data_df).ffill(axis=1).bfill(axis=1).ffill().bfill()

# %% cell 3
logger.info("[cell %d/27] # MIGRATION: originally the ostensibly-first 'try:' cell mix", 3)
# MIGRATION: originally the ostensibly-first "try:" cell mixed a DB connection
# check with a pickle load of the vol surface. Only the pickle load is
# actually needed by downstream cells (vol_surf_df). Path is now anchored at
# the repo root instead of os.getcwd()+".."/, so it works from any CWD.
from Library.Serialization import load_vol_surface
try:
    _vol_surface_path = _REPO_ROOT / "Study" / "Vol Surface From Model" / "kimyi2025_vol_surface"
    vol_surf_df = load_vol_surface(_vol_surface_path)
    logger.info("Loaded vol surface from %s(.parquet|.pkl)", _vol_surface_path)
except FileNotFoundError as e:
    logger.error("Vol surface not found: %s", e)
    raise

# %% cell 4
logger.info("[cell %d/27] # MIGRATION: this cell built the 'set_to_valuate_final_*' li", 4)
# MIGRATION: this cell built the 'set_to_valuate_final_*' lists by querying
# pmle_params_kim_yi. Replaced by cached-file check via pmle_params_exists().
set_to_valuate_final_systematic = [
    (dt, id_dict['systematic_id'], id_dict['systematic_id'])
    for dt in valuation_window_str
    if not pmle_params_exists(dt, id_dict['systematic_id'])
]
set_to_valuate_final_idiosyncratic = [
    (dt, idi_id, id_dict['systematic_id'])
    for dt in valuation_window_str
    for idi_id in id_dict['idiosyncratic_ids']
    if not pmle_params_exists(dt, idi_id)
]

# LEGACY: conn = None
# LEGACY: sys_df = None
# LEGACY: idi_df = None
# LEGACY: set_to_valuate_final_systematic = []
# LEGACY: set_to_valuate_final_idiosyncratic = []
#
# LEGACY: try:
# LEGACY:     conn = psycopg2.connect(
# LEGACY:         dbname="postgres",
# LEGACY:         user="postgres",
# LEGACY:         password="postgres",
# LEGACY:         host="localhost",
# LEGACY:         port="5432"
# LEGACY:     )
# LEGACY:     cur = conn.cursor()
# LEGACY:     print("Database connected successfully")
#
# LEGACY:     sql_str = f"SELECT * FROM pmle_params_kim_yi WHERE sSYSTEMATIC_ID='{systematic_id}' AND sSYSTEMATIC_ID=sIDIOSYNCRATIC_ID;"
#
# LEGACY:     sys_df = pd.read_sql(sql_str, conn)
# LEGACY:     sys_df = sys_df.set_index(['svaluation_date', 'sidiosyncratic_id', 'ssystematic_id'])
#
# LEGACY:     array_of_idi_ids = None
#
# LEGACY:     if len(id_dict['idiosyncratic_ids']) > 1:
# LEGACY:         array_of_idi_ids = tuple(id_dict['idiosyncratic_ids'])
# LEGACY:         sql_str = f"SELECT * FROM pmle_params_kim_yi WHERE sIDIOSYNCRATIC_ID IN '{array_of_idi_ids}';"
# LEGACY:     else:
# LEGACY:         array_of_idi_ids = id_dict['idiosyncratic_ids'][0]
# LEGACY:         sql_str = f"SELECT * FROM pmle_params_kim_yi WHERE sIDIOSYNCRATIC_ID = '{array_of_idi_ids}';"
#
#
# LEGACY:     idi_df = pd.read_sql(sql_str, conn)
# LEGACY:     idi_df = idi_df.set_index(['svaluation_date', 'sidiosyncratic_id', 'ssystematic_id'])
#
# LEGACY:     print("Successfully fetched data")
#
# LEGACY:     set_to_valuate_total_systematic = list(product(valuation_window_str, [id_dict['systematic_id']], [id_dict['systematic_id']]))
# LEGACY:     for tuple_to_valuate in set_to_valuate_total_systematic:
# LEGACY:         if not tuple_to_valuate in sys_df.index:
# LEGACY:             set_to_valuate_final_systematic.append(tuple_to_valuate)
#
# LEGACY:     set_to_valuate_total_idiosyncratic = list(product(valuation_window_str, id_dict['idiosyncratic_ids'], [id_dict['systematic_id']]))
# LEGACY:     for tuple_to_valuate in set_to_valuate_total_idiosyncratic:
# LEGACY:         if not tuple_to_valuate in idi_df.index:
# LEGACY:             set_to_valuate_final_idiosyncratic.append(tuple_to_valuate)
#
# LEGACY: except psycopg2.OperationalError as e:
# LEGACY:     print(f"Database not connected successfully: {e}")
#
# LEGACY: finally:
# LEGACY:     conn.close()
# LEGACY:     print("Database connection closed")

# %% cell 5
logger.info("[cell %d/27] define pmle_kimyirisk_systematic_helper()", 5)
def pmle_kimyirisk_systematic_helper(args) -> List[tuple]:

    valuation_dt, return_vector, delta_t, seed_number, n_mc_paths, systematic_id, idiosyncratic_id = args

    results = pmle_kimyirisk_systematic(
        sys_returns=return_vector,
        delta_t=delta_t,
        seed_number=seed_number,
        n_mc_paths=n_mc_paths
    )

    results_list = []
    for k, v in results.items():
        stats_name1, stats_name2, stats_name3 = v._asdict().keys()
        stats1, stats2, stats3 = v._asdict().values()
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name1, float(stats1)))
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name2, float(stats2)))
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name3, float(stats3)))

    return results_list


def pmle_kimyirisk_idiosyncratic_helper(args) -> List[tuple]:

    valuation_dt, params_sys, return_vector, delta_t, seed_number, n_mc_paths, systematic_id, idiosyncratic_id = args

    results = pmle_kimyirisk_idiosyncratic(
        params_sys=params_sys,
        idi_returns=return_vector,
        delta_t=delta_t,
        seed_number=seed_number,
        n_mc_paths=n_mc_paths
    )

    results_list = []
    for k, v in results.items():
        stats_name1, stats_name2, stats_name3 = v._asdict().keys()
        stats1, stats2, stats3 = v._asdict().values()
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name1, float(stats1)))
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name2, float(stats2)))
        results_list.append((valuation_dt, idiosyncratic_id, systematic_id, k, stats_name3, float(stats3)))

    return results_list

# %% cell 6
logger.info("[cell %d/27] # LEGACY: if len(set_to_valuate_final_systematic) > 0:", 6)

# LEGACY: if len(set_to_valuate_final_systematic) > 0:
# LEGACY:     systematic_arg_list = []
# LEGACY:     for dt, idi_id, sys_id in set_to_valuate_final_systematic:
# LEGACY:         return_vector = return_ts.loc[return_ts.index <= dt, idi_id].iloc[-lookback_period:].to_numpy()
# LEGACY:         systematic_arg_list.append((dt, return_vector, delta_t, seed_number, n_mc_paths, sys_id, idi_id))

# %% cell 7
logger.info("[cell %d/27] # MIGRATION: this cell calibrated missing systematic paramet", 7)
# MIGRATION: this cell calibrated missing systematic parameters and INSERTED
# them into the DB. For a clean run, call Scripts/run_pmle_kimyi2025.py
# beforehand to populate all cached parameter CSVs. If you need to
# re-calibrate programmatically from this script, use save_pmle_params()
# in place of the SQL INSERT.

# LEGACY: if __name__=='__main__':
# LEGACY:     import multiprocessing
# LEGACY:     from concurrent.futures import ProcessPoolExecutor
# LEGACY:     multiprocessing.set_start_method("fork", force=True)
#
# LEGACY:     if len(set_to_valuate_final_systematic) > 0:
#
# LEGACY:         valuation_dates = [dt for dt, _, _ in set_to_valuate_final_systematic]
# LEGACY:         return_vectors = [return_ts.loc[return_ts.index <= dt, systematic_id].iloc[-lookback_period:].to_numpy() for dt, _, _ in set_to_valuate_final_systematic]
# LEGACY:         delta_t_list = repeat(delta_t, len(set_to_valuate_final_systematic))
# LEGACY:         seed_number_list = repeat(seed_number, len(set_to_valuate_final_systematic))
# LEGACY:         n_mc_paths_list = repeat(n_mc_paths, len(set_to_valuate_final_systematic))
# LEGACY:         systematic_id_list = repeat(systematic_id, len(set_to_valuate_final_systematic))
# LEGACY:         idiosyncratic_id_list = repeat(systematic_id, len(set_to_valuate_final_systematic))
#
# LEGACY:         insert_query = ("INSERT INTO pmle_params_kim_yi ("
# LEGACY:                         "sVALUATION_DATE, sIDIOSYNCRATIC_ID, sSYSTEMATIC_ID, sPARAMETER, sVALUE_STATISTICS_DESC, sVALUE_STATISTICS"
# LEGACY:                         ") VALUES (%s, %s, %s, %s, %s, %s)")
#
# LEGACY:         with ProcessPoolExecutor() as executor:
#
# LEGACY:             results = executor.map(pmle_kimyirisk_systematic_helper, valuation_dates, return_vectors, delta_t_list, seed_number_list, n_mc_paths_list, systematic_id_list, idiosyncratic_id_list)
#
# LEGACY:             try:
# LEGACY:                 conn = psycopg2.connect(
# LEGACY:                     dbname="postgres",
# LEGACY:                     user="postgres",
# LEGACY:                     password="postgres",
# LEGACY:                     host="localhost",
# LEGACY:                     port="5432"
# LEGACY:                 )
# LEGACY:                 cur = conn.cursor()
# LEGACY:                 print("Database connected successfully")
#
# LEGACY:                 for result in results:
# LEGACY:                     cur.executemany(insert_query, result)
# LEGACY:                     num_ser_committed = cur.rowcount
# LEGACY:                     print(f"Number of ser affected by the operation: {num_ser_committed}")
# LEGACY:                     conn.commit()
# LEGACY:                     print("Transaction committed successfully.")
#
# LEGACY:             except psycopg2.OperationalError as e:
# LEGACY:                 print(f"Database not connected successfully: {e}")
#
# LEGACY:             finally:
# LEGACY:                 conn.close()
# LEGACY:                 print("Database connection closed")

# %% cell 8
logger.info("[cell %d/27] # MIGRATION: this cell re-queried the systematic parameters ", 8)
# MIGRATION: this cell re-queried the systematic parameters after cell 6.
# With cached CSVs, no re-query is needed — subsequent cells read directly
# via get_pmle_params_dict(date, ticker).

# LEGACY: try:
# LEGACY:     conn = psycopg2.connect(
# LEGACY:         dbname="postgres",
# LEGACY:         user="postgres",
# LEGACY:         password="postgres",
# LEGACY:         host="localhost",
# LEGACY:         port="5432"
# LEGACY:     )
# LEGACY:     cur = conn.cursor()
# LEGACY:     print("Database connected successfully")
#
# LEGACY:     sql_str = f"SELECT * FROM pmle_params_kim_yi WHERE sSYSTEMATIC_ID='{systematic_id}' AND sSYSTEMATIC_ID=sIDIOSYNCRATIC_ID;"
#
# LEGACY:     sys_df = pd.read_sql(sql_str, conn)
# LEGACY:     sys_df = sys_df.set_index(['svaluation_date', 'sidiosyncratic_id', 'ssystematic_id'])
#
# LEGACY:     print("Successfully fetched data")
#
# LEGACY: except psycopg2.OperationalError as e:
# LEGACY:     print(f"Database not connected successfully: {e}")
#
# LEGACY: finally:
# LEGACY:     conn.close()
# LEGACY:     print("Database connection closed")

# %% cell 9
logger.info("[cell %d/27] # MIGRATION: this cell calibrated missing idiosyncratic para", 9)
# MIGRATION: this cell calibrated missing idiosyncratic parameters and
# INSERTED them into the DB. Same replacement as cell 6.

# LEGACY: if __name__=='__main__':
# LEGACY:     import multiprocessing
# LEGACY:     from concurrent.futures import ProcessPoolExecutor
# LEGACY:     multiprocessing.set_start_method("fork", force=True)
#
# LEGACY:     idiosyncratic_arg_list = []
# LEGACY:     for dt, idi_id, sys_id in set_to_valuate_final_idiosyncratic:
# LEGACY:         mask  = (sys_df.index.get_level_values('svaluation_date')==dt)
# LEGACY:         mask &= (sys_df.index.get_level_values('ssystematic_id')==sys_id)
# LEGACY:         mask &= (sys_df.svalue_statistics_desc=='dMEAN')
# LEGACY:         sparameter, svalue = sys_df.loc[mask, ['sparameter', 'svalue_statistics']].to_numpy().T
# LEGACY:         params_sys = {k: v for k, v in zip(sparameter, svalue)}
# LEGACY:         return_vector = return_ts.loc[return_ts.index <= dt, idi_id].iloc[-lookback_period:].to_numpy()
# LEGACY:         idiosyncratic_arg_list.append((dt, params_sys, return_vector, delta_t, seed_number, n_mc_paths, sys_id, idi_id))
#
# LEGACY:     insert_query = ("INSERT INTO pmle_params_kim_yi ("
# LEGACY:                     "sVALUATION_DATE, sIDIOSYNCRATIC_ID, sSYSTEMATIC_ID, sPARAMETER, sVALUE_STATISTICS_DESC, sVALUE_STATISTICS"
# LEGACY:                     ") VALUES (%s, %s, %s, %s, %s, %s)")
#
# LEGACY:     with ProcessPoolExecutor() as executor:
#
# LEGACY:         results = executor.map(pmle_kimyirisk_idiosyncratic_helper, idiosyncratic_arg_list)
#
# LEGACY:         try:
# LEGACY:             conn = psycopg2.connect(
# LEGACY:                 dbname="postgres",
# LEGACY:                 user="postgres",
# LEGACY:                 password="postgres",
# LEGACY:                 host="localhost",
# LEGACY:                 port="5432"
# LEGACY:             )
# LEGACY:             cur = conn.cursor()
# LEGACY:             print("Database connected successfully")
#
# LEGACY:             for result in results:
# LEGACY:                 cur.executemany(insert_query, result)
# LEGACY:                 num_ser_committed = cur.rowcount
# LEGACY:                 conn.commit()
# LEGACY:                 print("Transaction committed successfully.")
#
# LEGACY:         except psycopg2.OperationalError as e:
# LEGACY:             print(f"Database not connected successfully: {e}")
#
# LEGACY:         finally:
# LEGACY:             conn.close()
# LEGACY:             print("Database connection closed")

# %% cell 10
logger.info("[cell %d/27] # MIGRATION: this cell built the params_df dataframe from a ", 10)
# MIGRATION: this cell built the params_df dataframe from a SELECT. Replaced
# by a loop that assembles the same long-form structure from the cached CSVs.
# The cached CSVs store all 11 parameters for both systematic and
# idiosyncratic underlyings (with dummy values on the "wrong side" per
# assemble_systematic_params/assemble_idiosyncratic_params). To match the
# original DB schema, we emit only the systematic parameters for the
# systematic row and only the idiosyncratic parameters for the idiosyncratic
# row, so downstream concat() has no duplicate index.
from Library.DataAccess import get_pmle_params
_SYS_PARAMS = ['dALPHA', 'dSIGMA', 'dPPROB', 'dLAMB', 'dETA1', 'dETA2']
_IDI_PARAMS = ['dMUI', 'dKAPPAI', 'dGAMMAI', 'dBETAI', 'dRHOIX']
_rows = []
for dt in valuation_window_str:
    for _ticker in [id_dict['systematic_id']] + id_dict['idiosyncratic_ids']:
        _series = get_pmle_params(dt, _ticker)
        _param_names = (
            _SYS_PARAMS if _ticker == id_dict['systematic_id'] else _IDI_PARAMS
        )
        for _param in _param_names:
            _base = {
                'svaluation_date': dt,
                'sidiosyncratic_id': _ticker,
                'ssystematic_id': id_dict['systematic_id'],
                'sparameter': _param,
            }
            _rows.append({**_base, 'svalue_statistics_desc': 'dMEAN',
                          'svalue_statistics': float(_series[_param])})
            _rows.append({**_base, 'svalue_statistics_desc': 'dCI_LOWER',
                          'svalue_statistics': float(_series[f'{_param}_CI_LOWER'])})
            _rows.append({**_base, 'svalue_statistics_desc': 'dCI_UPPER',
                          'svalue_statistics': float(_series[f'{_param}_CI_UPPER'])})
params_df = pd.DataFrame(_rows)

# LEGACY: try:
# LEGACY:     conn = psycopg2.connect(
# LEGACY:         dbname="postgres",
# LEGACY:         user="postgres",
# LEGACY:         password="postgres",
# LEGACY:         host="localhost",
# LEGACY:         port="5432"
# LEGACY:     )
# LEGACY:     print("Database connected successfully")
#
# LEGACY:     select_query = f"SELECT * FROM pmle_params_kim_yi;"
#
# LEGACY:     params_df = pd.read_sql(select_query, con=conn)
#
# LEGACY: except psycopg2.OperationalError as e:
# LEGACY:     print(f"Database not connected successfully: {e}")
#
# LEGACY: finally:
# LEGACY:     conn.close()
# LEGACY:     print("Database connection closed")

# %% cell 11
logger.info("[cell %d/27] define simulate_shock_returns_systematic_helper()", 11)
def simulate_shock_returns_systematic_helper(inputs: dict) -> dict:
    id = inputs["sID"]
    valuation_date = inputs["sVALUATION_DATE"]
    scenario_number = inputs["sSCENARIO"]
    params = inputs["oPARAMS"]
    rng = inputs["oRNG"]
    delta_time = inputs["dDELTA_TIME"]
    size = inputs["oSIZE"]

    ret, _, _ = KimYiRiskEngine(
        mui=[ParametersConstant(np.array(0.))],
        kappai=[ParametersConstant(np.array(0.))],
        gammai=[ParametersConstant(np.array(1.))],
        betai=[ParametersConstant(np.array(1.))],
        rhoix=[ParametersConstant(np.array(0.))],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=delta_time
    ).random(rng=rng, size=size)
    return {f"{valuation_date}-{id}-{scenario_number}": ret[0]}


def simulate_shock_returns_idiosyncratic_helper(inputs: dict) -> dict:
    id = inputs["sID"]
    valuation_date = inputs["sVALUATION_DATE"]
    scenario_number = inputs["sSCENARIO"]
    params = inputs["oPARAMS"]
    rng = inputs["oRNG"]
    delta_time = inputs["dDELTA_TIME"]
    size = inputs["oSIZE"]

    ret, _, _ = KimYiRiskEngine(
        mui=[ParametersConstant(params.dMUI)],
        kappai=[ParametersConstant(params.dKAPPAI)],
        gammai=[ParametersConstant(params.dGAMMAI)],
        betai=[ParametersConstant(params.dBETAI)],
        rhoix=[ParametersConstant(params.dRHOIX)],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=delta_time
    ).random(rng=rng, size=size)
    return {f"{id}-{valuation_date}-{scenario_number}": ret[0]}


def simulate_shock_returns_idiosyncratic_base_helper(inputs: dict) -> dict:
    id = inputs["sID"]
    valuation_date = inputs["sVALUATION_DATE"]
    scenario_number = inputs["sSCENARIO"]
    params = inputs["oPARAMS"]
    rng = inputs["oRNG"]
    delta_time = inputs["dDELTA_TIME"]
    size = inputs["oSIZE"]

    ret, _, _ = KimYiRiskEngine(
        mui=[ParametersConstant(params.dMUI)],
        kappai=[ParametersConstant(params.dKAPPAI)],
        gammai=[ParametersConstant(np.array(0.))],
        betai=[ParametersConstant(np.array(0.))],
        rhoix=[ParametersConstant(np.array(0.))],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=delta_time
    ).random(rng=rng, size=size)
    return {f"{id}-{valuation_date}-{scenario_number}": ret[0]}


def est_liquidity_process_sys_helper(inputs: dict) -> dict:
    id = inputs["sID"]
    valuation_date = inputs["sVALUATION_DATE"]
    scenario_number = inputs["sSCENARIO"]
    params = inputs["oPARAMS"]
    delta_time = inputs["dDELTA_TIME"]

    ret = KimYiRiskEngine(
        mui=[ParametersConstant(np.array(0.))],
        kappai=[ParametersConstant(np.array(0.))],
        gammai=[ParametersConstant(np.array(1.))],
        betai=[ParametersConstant(np.array(1.))],
        rhoix=[ParametersConstant(np.array(0.))],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=delta_time
    ).est_liquidity_process(return_ts.loc[return_ts.index <= valuation_date, id].iloc[-lookback_period:].to_numpy().reshape((-1, 1)))

    return {f"{id}-{valuation_date}-{scenario_number}": ret}


def est_liquidity_process_idi_helper(inputs: dict) -> dict:
    id = inputs["sID"]
    valuation_date = inputs["sVALUATION_DATE"]
    scenario_number = inputs["sSCENARIO"]
    params = inputs["oPARAMS"]
    delta_time = inputs["dDELTA_TIME"]

    ret = KimYiRiskEngine(
        mui=[ParametersConstant(params.dMUI)],
        kappai=[ParametersConstant(params.dKAPPAI)],
        gammai=[ParametersConstant(params.dGAMMAI)],
        betai=[ParametersConstant(params.dBETAI)],
        rhoix=[ParametersConstant(params.dRHOIX)],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=delta_time
    ).est_liquidity_process(return_ts.loc[return_ts.index <= valuation_date, id].iloc[-lookback_period:].to_numpy().reshape((-1, 1)))

    return {f"{id}-{valuation_date}-{scenario_number}": ret}

# %% cell 12
logger.info("[cell %d/27] compute point_in_time_dt", 12)
# Default: last valuation date (ensures Figure 6 timeline includes all
# annotated events -- Apr 2 and Apr 9 pause).
# Override via LIQUIDITY_POINT_IN_TIME_DT=YYYYMMDD for the Figure 7
# term-structure dates (paper uses 20250403 and 20250409). Run the script
# once per date; each run appends to Study/Collar Asian/ScenarioPV/{date}/.
point_in_time_dt = os.environ.get("LIQUIDITY_POINT_IN_TIME_DT", "20250416")
logger.info("point_in_time_dt = %s", point_in_time_dt)

inputs_sys = []
inputs_idi = []
inputs_idi_base = []

# mask = (params_df.svaluation_date==point_in_time_dt) & (params_df.sidiosyncratic_id=='COIN') & (params_df.svalue_statistics_desc=='dMEAN')
# series1 = params_df.loc[mask, ['sparameter', 'svalue_statistics']].set_index('sparameter').svalue_statistics
#
# range_gammai = np.sort(np.concat((np.linspace(0., 3., 7), np.array([series1.dGAMMAI]))))
# range_betai = np.sort(np.concat((np.linspace(0., 3., 7), np.array([series1.dBETAI]))))
# range_rhoix = np.sort(np.concat((np.linspace(-1., 1., 5), np.array([series1.dRHOIX]))))
# what_if_scenarios_dict = {f"SCEN_{k+1}": v for k, v in enumerate(product(range_gammai, range_betai, range_rhoix))}

range_gammai = np.linspace(0., 3., 7)
range_betai = np.linspace(0., 3., 7)
range_rhoix = np.linspace(-1., 1., 5)
what_if_scenarios_dict = {f"SCEN_{k+1}": v for k, v in enumerate(product(range_gammai, range_betai, range_rhoix))}

what_if_scenario_df = pd.DataFrame(what_if_scenarios_dict).T
what_if_scenario_df.columns = ['dGAMMAI', 'dBETAI', 'dRHOIX']
what_if_scenario_df.index.name = 'sSCENARIO'
what_if_scenario_df.to_csv(str(_SCENARIOS_DIR / 'what_if_scenarios.csv'))

do_what_if_scenario = True

for dt, idi_id in product([pd.to_datetime(point_in_time_dt)], get_idiosyncratic_ids()):
    dt = dt.strftime(date_format)

    rng.reset()

    mask = (params_df.svaluation_date==dt) & (params_df.sidiosyncratic_id==systematic_id) & (params_df.svalue_statistics_desc=='dMEAN')
    series1 = params_df.loc[mask, ['sparameter', 'svalue_statistics']].set_index('sparameter').svalue_statistics
    inputs_sys.append(
        {"sID": systematic_id, "sVALUATION_DATE": dt, "sSCENARIO": "SCEN_0", "oPARAMS": series1, "oRNG": rng, "dDELTA_TIME": delta_t, "oSIZE": (1, n_mc_paths, m_steps)}
    )

    rng.reset()

    mask = (params_df.svaluation_date==dt) & (params_df.sidiosyncratic_id==idi_id) & (params_df.svalue_statistics_desc=='dMEAN')
    series2 = params_df.loc[mask, ['sparameter', 'svalue_statistics']].set_index('sparameter').svalue_statistics
    series0 = pd.concat([series2, series1])
    inputs = {"sID": idi_id, "sVALUATION_DATE": dt, "sSCENARIO": "SCEN_0", "oPARAMS": series0, "oRNG": rng, "dDELTA_TIME": delta_t, "oSIZE": (1, n_mc_paths, m_steps)}
    inputs_idi.append(inputs)
    inputs_idi_base.append(inputs)

    # what-if scenarios
    if do_what_if_scenario:
        what_if_scenarios_dict["SCEN_0"] = (series2.dGAMMAI, series2.dBETAI, series2.dRHOIX)

        for scen_no, v in what_if_scenarios_dict.items():
            rng.reset()

            gammai, betai, rhoix = v

            series2.dGAMMAI = gammai
            series2.dBETAI = betai
            series2.dRHOIX = rhoix

            series0 = pd.concat([series2, series1])
            inputs_idi.append(
                {"sID": idi_id, "sVALUATION_DATE": dt, "sSCENARIO": scen_no, "oPARAMS": series0, "oRNG": rng, "dDELTA_TIME": delta_t, "oSIZE": (1, n_mc_paths, m_steps)}
            )

# %% cell 13
logger.info("[cell %d/27] multiprocessing.set_start_method('fork', force=True)", 13)
if __name__=='__main__':
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    multiprocessing.set_start_method("fork", force=True)

    with ProcessPoolExecutor() as executor:

        results1 = executor.map(simulate_shock_returns_systematic_helper, inputs_sys)
        results2 = executor.map(simulate_shock_returns_idiosyncratic_helper, inputs_idi)
        results3 = executor.map(simulate_shock_returns_idiosyncratic_base_helper, inputs_idi_base)
        results4 = executor.map(est_liquidity_process_sys_helper, inputs_sys)
        results5 = executor.map(est_liquidity_process_idi_helper, inputs_idi)

        shock_returns_sys_dict = {}
        shock_returns_idi_dict = {}
        shock_returns_idi_base_dict = {}
        liquidity_process_sys_dict = {}
        liquidity_process_idi_dict = {}

        for result1 in results1:
            for date_key, inner_dict in result1.items():
                shock_returns_sys_dict[date_key] = inner_dict

        for result2 in results2:
            for date_key, inner_dict in result2.items():
                shock_returns_idi_dict[date_key] = inner_dict

        for result3 in results3:
            for date_key, inner_dict in result3.items():
                shock_returns_idi_base_dict[date_key] = inner_dict

        for result4 in results4:
            for date_key, inner_dict in result4.items():
                liquidity_process_sys_dict[date_key] = inner_dict

        for result5 in results5:
            for date_key, inner_dict in result5.items():
                liquidity_process_idi_dict[date_key] = inner_dict

# Cells 14-18 removed: the return/P&L density and filtered-liquidity plots
# they produced are duplicates of paper Figures 1, 2, and 6, which are the
# authoritative outputs of Study/Collar Asian/report_collar_asian.py.
# The Monte Carlo shock_returns_* and liquidity_process_* dicts computed by
# cell 13 remain in scope for the downstream VaR pricing cells.

# %% cell 14
logger.info("[cell %d/27] # MIGRATION: this cell queried the portfolio table. Replaced", 14)
# MIGRATION: this cell queried the portfolio table. Replaced by
# build_portfolio() from Scripts.load_portfolio, materialised for every
# (valuation date, idiosyncratic asset) combination in the window. Column
# names are lower-cased to mirror the DB schema the downstream cells expect.
from Scripts.load_portfolio import build_portfolio
_portfolio_rows = []
for _dt in valuation_window_str:
    for _idi in id_dict['idiosyncratic_ids']:
        for _p in build_portfolio(_dt, _idi):
            _portfolio_rows.append(_p._asdict())
pos_df = pd.DataFrame(_portfolio_rows)
pos_df.columns = [c.lower() for c in pos_df.columns]

# LEGACY: try:
# LEGACY:     conn = psycopg2.connect(
# LEGACY:         dbname="postgres",
# LEGACY:         user="postgres",
# LEGACY:         password="postgres",
# LEGACY:         host="localhost",
# LEGACY:         port="5432"
# LEGACY:     )
# LEGACY:     print("Database connected successfully")
#
# LEGACY:     select_query = f"SELECT * FROM portfolio;"
#
# LEGACY:     pos_df = pd.read_sql(select_query, con=conn)
#
# LEGACY: except psycopg2.OperationalError as e:
# LEGACY:     print(f"Database not connected successfully: {e}")
#
# LEGACY: finally:
# LEGACY:     conn.close()
# LEGACY:     print("Database connection closed")

# %% cell 15
logger.info("[cell %d/27] define get_pv()", 15)
def get_pv(args_dict: dict) -> dict:

    sTYPOLOGY = args_dict["sTYPOLOGY"]

    if sTYPOLOGY.lower() == 'vanilla':

        return _get_pv_vanilla(args_dict)

    else:

        return _get_pv_exotic(args_dict)


def _get_pv_vanilla(args: dict) -> dict:

    dSPOT_PRICE = args["dSPOT_PRICE"]
    dSTRIKE_PRICE = args["dSTRIKE_PRICE"]
    dRISK_FREE_RATE = args["dRISK_FREE_RATE"]
    dDIVIDEND_YIELD = args["dDIVIDEND_YIELD"]
    dIMP_VOLATILITY = args["dIMP_VOLATILITY"]
    iQUANTITY = args["iQUANTITY"]
    dEXPIRY = args['dEXPIRY']

    sPAYOFF = args["sPAYOFF"]

    if sPAYOFF.lower() == "put":

        pricer_obj = BlackScholesMertonPut(
            und_price=dSPOT_PRICE,
            und_strike=dSTRIKE_PRICE,
            risk_free_rate=dRISK_FREE_RATE,
            dividend_yield=dDIVIDEND_YIELD,
            time_to_expiry=dEXPIRY
        )

    elif sPAYOFF.lower() == "call":

        pricer_obj = BlackScholesMertonCall(
            und_price=dSPOT_PRICE,
            und_strike=dSTRIKE_PRICE,
            risk_free_rate=dRISK_FREE_RATE,
            dividend_yield=dDIVIDEND_YIELD,
            time_to_expiry=dEXPIRY
        )

    else:
        raise()

    pv = pricer_obj.price(dIMP_VOLATILITY) * iQUANTITY

    output = {
        'sVALUATION_DATE': args["sVALUATION_DATE"],
        'sID': args["sID"],
        'sTYPOLOGY': args["sTYPOLOGY"],
        'sSTRATEGY': args["sSTRATEGY"],
        'sPAYOFF': sPAYOFF,
        'sSCENARIO': args["sSCENARIO"],
        'iSCENARIO_NO': args["iSCENARIO_NO"],
        "sRETURN_ASSUMPTION": args["sRETURN_ASSUMPTION"],
        'sEXPIRY_DATE': args["sEXPIRY_DATE"],
        'iEXPIRY': args["iEXPIRY"],
        'iRISK_HORIZON': args["iRISK_HORIZON"],
        'iQUANTITY': iQUANTITY,
        'dSPOT_PRICE': np.float64(dSPOT_PRICE),
        'dSPOT_SHOCK': np.float64(args["dSPOT_SHOCK"]),
        'dSTRIKE_PRICE': np.float64(dSTRIKE_PRICE),
        'dIMP_VOLATILITY': dIMP_VOLATILITY.squeeze(),
        'dPV': pv.squeeze()
    }

    return output


def _get_pv_exotic(args: dict) -> dict:

    dSPOT_PRICE = args["dSPOT_PRICE"]
    dSTRIKE_PRICE = args["dSTRIKE_PRICE"]
    dRISK_FREE_RATE = args["dRISK_FREE_RATE"]
    dDIVIDEND_YIELD = args["dDIVIDEND_YIELD"]
    dIMP_VOLATILITY = args["dIMP_VOLATILITY"]
    iQUANTITY = args["iQUANTITY"]
    dEXPIRY = args['dEXPIRY']

    sPAYOFF = args["sPAYOFF"]
    sTYPOLOGY = args["sTYPOLOGY"]

    oSTATS_GATHERER = args["oSTATS_GATHERER"]
    oRAND_GENERATOR = args["oRAND_GENERATOR"]

    sVALUATION_DATE = args["sVALUATION_DATE"]

    oPRODUCT = None

    if sTYPOLOGY.lower() == 'asian discrete':

        sFIXING_FREQUENCY = args["sFIXING_FREQUENCY"]
        sFIXING_DATE_BEG = args["sFIXING_DATE_BEG"]
        sFIXING_DATE_END = args["sFIXING_DATE_END"]

        dFIXING_TIMES = pd.bdate_range(start=sFIXING_DATE_BEG, end=sFIXING_DATE_END, freq=sFIXING_FREQUENCY)
        dFIXING_TIMES = dFIXING_TIMES[dFIXING_TIMES >= sVALUATION_DATE]
        dFIXING_TIMES = get_fixings_vec(np.uint64(dFIXING_TIMES.shape[0]), dEXPIRY)

        oPAYOFF = payoff_mc_factory(sPAYOFF.lower())(dSTRIKE_PRICE)
        oPRODUCT = PathDependentAsianDiscrete(
            fixing_times=dFIXING_TIMES,
            delivery_time=dEXPIRY,
            the_payoff=oPAYOFF,
            quantity_amount=iQUANTITY
        )

    ExoticEngineBlackScholesMerton(
        the_product=oPRODUCT,
        risk_free_rate=ParametersConstant(dRISK_FREE_RATE),
        dividend_yield=[ParametersConstant(dDIVIDEND_YIELD)],
        imp_volatility=[ParametersConstant(dIMP_VOLATILITY)],
        rand_generator=oRAND_GENERATOR,
        spot_price=dSPOT_PRICE,
        number_of_paths=args["iNUM_MC_PATHS"]
    ).do_simulation(oSTATS_GATHERER)

    pv = oSTATS_GATHERER.get_result_so_far().reshape((-1,))

    output = {
        'sVALUATION_DATE': args["sVALUATION_DATE"],
        'sID': args["sID"],
        'sTYPOLOGY': args["sTYPOLOGY"],
        'sSTRATEGY': args["sSTRATEGY"],
        'sPAYOFF': sPAYOFF,
        'sSCENARIO': args["sSCENARIO"],
        'iSCENARIO_NO': args["iSCENARIO_NO"],
        "sRETURN_ASSUMPTION": args["sRETURN_ASSUMPTION"],
        'sEXPIRY_DATE': args["sEXPIRY_DATE"],
        'iEXPIRY': args["iEXPIRY"],
        'iRISK_HORIZON': args["iRISK_HORIZON"],
        'iQUANTITY': iQUANTITY,
        'dSPOT_PRICE': np.float64(dSPOT_PRICE),
        'dSPOT_SHOCK': np.float64(args["dSPOT_SHOCK"]),
        'dSTRIKE_PRICE': np.float64(dSTRIKE_PRICE),
        'dIMP_VOLATILITY': dIMP_VOLATILITY.squeeze(),
        'dPV': pv.squeeze()
    }

    return output

# %% cell 16
logger.info("[cell %d/27] define compound_relative_returns()", 16)
def compound_relative_returns(returns):
    returns = 1. + returns
    for i in range(1, returns.shape[1]):
        returns[:, i] *= returns[:, i-1]
    return returns - 1.

# %% cell 17
logger.info("[cell %d/27] define prepare_data_for_pricing()", 17)
def prepare_data_for_pricing(ser: namedtuple, spot_shock_dict: dict = None) -> dict:

    sID = ser.sidiosyncratic_id
    sVALUATION_DATE = ser.svaluation_date
    dBASE_DAYS = ser.dbase_days
    sEXPIRY_DATE = ser.sexpiry_date
    iRISK_HORIZON = 0
    sRETURN_ASSUMPTION = None
    sSCENARIO = None

    if spot_shock_dict is None:
        dSPOT_SHOCK = np.array(0)
        iSCENARIO_NO = -1

    else:
        dSPOT_SHOCK = spot_shock_dict['dSPOT_SHOCK']
        iRISK_HORIZON = spot_shock_dict['iRISK_HORIZON']
        sRETURN_ASSUMPTION = spot_shock_dict['sRETURN_ASSUMPTION']
        iSCENARIO_NO = spot_shock_dict['iSCENARIO_NO']
        sSCENARIO = spot_shock_dict['sSCENARIO']

    iEXPIRY = pd.bdate_range(start=sVALUATION_DATE, end=sEXPIRY_DATE).shape[0] - 1

    args = {}

    if iEXPIRY > 0:

        dEXPIRY = year_frac(dBASE_DAYS)(time1=np.array(0), time2=np.array(iEXPIRY))

        dSPOT_PRICE = np.array(price_ts.xs(sVALUATION_DATE)[sID] * (1. + dSPOT_SHOCK))
        dSTRIKE_PRICE = np.array(ser.dstrike_price)

        rates_data = rates_data_df.xs(sVALUATION_DATE)
        tenors = np.array([UST_TENOR_MAP[tenor] for tenor in rates_data.index])
        rates = rates_data.to_numpy().squeeze() / 100
        curve_fit, _ = calibrate_nss_ols(tenors, rates)

        dRISK_FREE_RATE = np.array(curve_fit(dEXPIRY))
        dDIVIDEND_YIELD = np.array(DIVIDEND_YIELDS[sID] / 100)

        dFORWARD_PRICE = dSPOT_PRICE * np.exp((dRISK_FREE_RATE - dDIVIDEND_YIELD) * dEXPIRY)
        dMONEYNESS = dSTRIKE_PRICE / dFORWARD_PRICE

        vol_dict = vol_surf_df[f"{sID}-{sVALUATION_DATE}"]
        vol_iexpiry = vol_dict["iEXPIRY"]
        vol_moneyness = vol_dict["dMONEYNESS"]
        vol_surf = vol_dict["dVOL"]
        dIMP_VOLATILITY = np.array(pchip_interpolator2d(x=vol_moneyness, y=vol_iexpiry, z=vol_surf.T, x1=dMONEYNESS, x2=np.array(iEXPIRY)).squeeze()) / 100.

        oRAND_GENERATOR = None
        oSTATS_GATHERER = None
        iNUM_MC_PATHS = None

        if ser.stypology.lower() != "vanilla":
            oRAND_GENERATOR = RandomMT19937(seed=seed_number)
            oSTATS_GATHERER = StatisticsMCMean()
            iNUM_MC_PATHS = np.uint64(10_000)

        args = {
            "sVALUATION_DATE": sVALUATION_DATE,
            "sID": sID,
            "sTYPOLOGY": ser.stypology,
            "sSTRATEGY": ser.sstrategy,
            "sPAYOFF": ser.spayoff_type,
            "iSCENARIO_NO": iSCENARIO_NO,
            "sSCENARIO": sSCENARIO,
            "sEXPIRY_DATE": sEXPIRY_DATE,
            "dEXPIRY": dEXPIRY,
            "iEXPIRY": iEXPIRY,
            "iRISK_HORIZON": iRISK_HORIZON,
            "dSTRIKE_PRICE": dSTRIKE_PRICE,
            "dSPOT_PRICE": dSPOT_PRICE,
            "dSPOT_SHOCK": dSPOT_SHOCK,
            "dFORWARD_PRICE": dFORWARD_PRICE,
            "dMONEYNESS": dMONEYNESS,
            "iQUANTITY": int(ser.dposition),
            "sFIXING_FREQUENCY": ser.sfixing_frequency,
            "sFIXING_DATE_BEG": ser.sfixing_date_beg,
            "sFIXING_DATE_END": ser.sfixing_date_end,
            "dBASE_DAYS": dBASE_DAYS,
            "oRAND_GENERATOR": oRAND_GENERATOR,
            "oSTATS_GATHERER": oSTATS_GATHERER,
            "iNUM_MC_PATHS": iNUM_MC_PATHS,
            "sRETURN_ASSUMPTION": sRETURN_ASSUMPTION,
            "dRISK_FREE_RATE": dRISK_FREE_RATE,
            "dDIVIDEND_YIELD": dDIVIDEND_YIELD,
            "dIMP_VOLATILITY": dIMP_VOLATILITY
        }

    return args

# %% cell 18
logger.info("[cell %d/27] compute args_list", 18)
args_list = []
args_list_delta_up = []
args_list_delta_dn = []
what_if_scenarios_keys = sorted(list(what_if_scenarios_dict.keys()))

# Optional scenario filter for fast reruns. Set LIQUIDITY_VAR_SCENARIOS=
#   "SCEN_0"                   -> only base scenario (Figure 7 only, minutes)
#   "SCEN_0,SCEN_10,SCEN_50"   -> specific subset
#   unset / empty              -> full lattice (Figure 3 + Figure 7, ~hours)
_env_scen_filter = os.environ.get("LIQUIDITY_VAR_SCENARIOS", "").strip()
if _env_scen_filter:
    _wanted = {s.strip() for s in _env_scen_filter.split(",") if s.strip()}
    _kept = [k for k in what_if_scenarios_keys if k in _wanted]
    _skipped = len(what_if_scenarios_keys) - len(_kept)
    what_if_scenarios_keys = _kept
    logger.info(
        "LIQUIDITY_VAR_SCENARIOS=%r: keeping %d scenario(s), skipping %d.",
        _env_scen_filter, len(_kept), _skipped,
    )

spot_shock_dict_delta_up = {
    'sRETURN_ASSUMPTION': None,
    'sSCENARIO': None,
    'iSCENARIO_NO': -1,
    'iRISK_HORIZON': 0,
    'dSPOT_SHOCK': np.array(.01)
}

spot_shock_dict_delta_dn = {
    'sRETURN_ASSUMPTION': None,
    'sSCENARIO': None,
    'iSCENARIO_NO': -1,
    'iRISK_HORIZON': 0,
    'dSPOT_SHOCK': -np.array(.01)
}

mask = (pos_df.sidiosyncratic_id=="COIN") & (pos_df.svaluation_date==point_in_time_dt)
pos_tmp_df = pos_df.loc[mask].copy()

for row in pos_tmp_df.itertuples(index=False):
    args_list.append(prepare_data_for_pricing(row))
    args_list_delta_up.append(prepare_data_for_pricing(row, spot_shock_dict_delta_up))
    args_list_delta_dn.append(prepare_data_for_pricing(row, spot_shock_dict_delta_dn))

# %% cell 19
logger.info("[cell %d/27] compute pv_df", 19)
pv_df = []
pv_up_df = []
pv_dn_df = []

if __name__=='__main__':
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    multiprocessing.set_start_method("fork", force=True)

    with ProcessPoolExecutor() as executor:

        results1 = executor.map(get_pv, args_list)
        results2 = executor.map(get_pv, args_list_delta_up)
        results3 = executor.map(get_pv, args_list_delta_dn)

        for result1 in results1:
            pv_df.append(result1)

        for result2 in results2:
            pv_up_df.append(result2)

        for result3 in results3:
            pv_dn_df.append(result3)

path = _REPO_ROOT_STR
_current_pv_date_dir = _CURRENT_PV_DIR_TEMPLATE / point_in_time_dt
_current_pv_date_dir.mkdir(parents=True, exist_ok=True)


def _save_pv_parquet(pv_records, out_path):
    """Persist a list-of-dicts PV cache as parquet.

    Each record's ``dPV`` field is a numpy array (one MC-path vector). Pyarrow
    infers the column type as an object array unless we explicitly cast each
    cell to a Python list of floats. Falls back to pickle only if the record
    structure still resists parquet (e.g., contains unrelated nested objects).
    """
    df = pd.DataFrame(pv_records)
    # Convert numpy-array columns to lists so pyarrow can emit list<double>.
    for _col in df.columns:
        _sample = df[_col].iloc[0] if len(df) else None
        if isinstance(_sample, np.ndarray):
            df[_col] = df[_col].apply(lambda x: [float(v) for v in np.asarray(x).ravel()])
    try:
        df.to_parquet(out_path)
        logger.info("Saved %s (%d rows)", out_path, len(pv_records))
    except Exception as parquet_err:
        pickle_path = str(out_path).replace(".parquet", ".pkl")
        with open(pickle_path, "wb") as fh:
            pickle.dump(pv_records, fh)
        logger.warning(
            "Parquet write failed for %s (%s); wrote pickle fallback at %s",
            out_path, parquet_err, pickle_path,
        )


_save_pv_parquet(pv_df,    str(_current_pv_date_dir / f"{point_in_time_dt}_pv.parquet"))
_save_pv_parquet(pv_up_df, str(_current_pv_date_dir / f"{point_in_time_dt}_pv_up.parquet"))
_save_pv_parquet(pv_dn_df, str(_current_pv_date_dir / f"{point_in_time_dt}_pv_dn.parquet"))

# %% cell 20
logger.info("[cell %d/27] compute ScenarioInputs", 20)
ScenarioInputs = namedtuple(typename="ScenarioInputs", field_names=list(pos_tmp_df.columns))
scenario_inputs_list = []

for tup in pos_tmp_df.itertuples(index=False):
    scenario_inputs_list.append(ScenarioInputs(*list(tup)))

# %% cell 21
logger.info("[cell %d/27] define get_scenario_dict()", 21)
def get_scenario_dict(
        return_assumption: str,
        scenario: str,
        scenario_number: int,
        risk_horizon: int,
        spot_shock: float
) -> dict:
    spot_shock_dict = {
        'sRETURN_ASSUMPTION': return_assumption,
        'sSCENARIO': scenario,
        'iSCENARIO_NO': scenario_number,
        'iRISK_HORIZON': risk_horizon,
        'dSPOT_SHOCK': np.array(spot_shock)
    }
    return spot_shock_dict

# %% cell 22
logger.info("[cell %d/27] define save_scenario_positions()", 22)
def save_scenario_positions(args) -> None:

    scenario_inputs_list, risk_horizon, scenario_key, return_assumption = args

    if return_assumption == 'LA':
        compound_returns = compound_relative_returns(shock_returns_idi_dict[f"COIN-{point_in_time_dt}-{scenario_key}"])
    else:
        compound_returns = compound_relative_returns(shock_returns_idi_base_dict[f"COIN-{point_in_time_dt}-{scenario_key}"])

    # compound_returns = compound_relative_returns(shock_returns_idi_dict[f"COIN-{point_in_time_dt}-{scenario_key}"])
    #
    # pv_list = []
    #
    # for tup in scenario_inputs_list:
    #     iEXPIRY  = pd.bdate_range(start=tup.svaluation_date, end=tup.sexpiry_date).shape[0] - 1
    #     iEXPIRY_ = iEXPIRY - risk_horizon
    #     risk_horizon_ = np.minimum(np.maximum(iEXPIRY_, iEXPIRY), risk_horizon)
    #
    #     spot_shock_vector = compound_returns[:, risk_horizon_]
    #
    #     for scenario_number, spot_shock in enumerate(spot_shock_vector):
    #         spot_shock_dict = get_scenario_dict(
    #             return_assumption=return_assumption,
    #             scenario=scenario_key,
    #             scenario_number=scenario_number,
    #             risk_horizon=risk_horizon,
    #             spot_shock=spot_shock
    #         )
    #
    #         pv_list.append(get_pv(prepare_data_for_pricing(ser=tup, spot_shock_dict=spot_shock_dict)))
    #
    # file_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), rf"Study/Collar Asian/ScenarioPV/{point_in_time_dt}/{point_in_time_dt}_{scenario_key}_{risk_horizon}D_{return_assumption}.pkl")
    #
    # try:
    #     with open(file_path, "wb") as file:
    #         pickle.dump(pv_list, file)
    #     print(f"Dictionary successfully saved to {file_path}")
    # except Exception as e:
    #     print(f"An error occured: {e}")

    # Compute + persist the P&L for every what-if scenario. Figure 3 (VaR
    # surface) needs the full (gamma, beta, rho) lattice, so restricting this
    # to SCEN_0 would leave Figure 3 unrenderable.
    pv_list = []
    for tup in scenario_inputs_list:
        iEXPIRY  = pd.bdate_range(start=tup.svaluation_date, end=tup.sexpiry_date).shape[0] - 1
        iEXPIRY_ = iEXPIRY - risk_horizon
        risk_horizon_ = np.minimum(np.maximum(iEXPIRY_, iEXPIRY), risk_horizon)

        spot_shock_vector = compound_returns[:, risk_horizon_]

        for scenario_number, spot_shock in enumerate(spot_shock_vector):
            spot_shock_dict = get_scenario_dict(
                return_assumption=return_assumption,
                scenario=scenario_key,
                scenario_number=scenario_number,
                risk_horizon=risk_horizon,
                spot_shock=spot_shock
            )

            pv_list.append(get_pv(prepare_data_for_pricing(ser=tup, spot_shock_dict=spot_shock_dict)))

    _scenario_pv_date_dir = _SCENARIO_PV_DIR_TEMPLATE / point_in_time_dt
    _scenario_pv_date_dir.mkdir(parents=True, exist_ok=True)
    file_path = str(_scenario_pv_date_dir / f"{point_in_time_dt}_{scenario_key}_{risk_horizon}D_{return_assumption}.parquet")
    _save_pv_parquet(pv_list, file_path)
    return None

# %% cell 23
logger.info("[cell %d/27] compute return_assumption", 23)
return_assumption = ["LA"]
risk_horizons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

if __name__=='__main__':
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    multiprocessing.set_start_method("fork", force=True)

    with ProcessPoolExecutor() as executor:

        results = executor.map(save_scenario_positions, product([scenario_inputs_list], risk_horizons, what_if_scenarios_keys, return_assumption))

        for result in results:
            result

# %% cell 24
logger.info("[cell %d/27] aggregate scenario PVs into a VaR table", 24)
# Sum per-position PVs into portfolio PV per MC path per scenario, subtract
# the T0 portfolio value, and take the 1st percentile as 99% VaR. This
# produces the DataFrame that Figures 3 and 7 consume. It reads directly
# from Study/Collar Asian/ScenarioPV/{date}/*.parquet so it works with
# whatever dates have been computed (adaptive to partial runs).
from matplotlib import cm as _cm  # noqa: E402

def _compute_var_table(base_dir=_SCENARIO_PV_DIR_TEMPLATE,
                       t0_dir=_CURRENT_PV_DIR_TEMPLATE,
                       confidence: float = 0.99) -> pd.DataFrame:
    """Walk ScenarioPV/{date}/{date}_{scen}_{h}D_{ra}.parquet and produce a
    tidy table with columns
        [date, scenario, return_assumption, horizon_days,
         gamma, beta, rho, var]
    """
    if not base_dir.exists():
        logger.warning("ScenarioPV dir missing: %s", base_dir)
        return pd.DataFrame()
    scen_meta = what_if_scenario_df.reset_index().rename(
        columns={"sSCENARIO": "scenario"}
    )
    # SCEN_0 is the current calibration; add it with the model-current
    # (gamma, beta, rho) triple so it can appear on the surface too.
    _series0 = params_df.loc[
        (params_df.sidiosyncratic_id == "COIN")
        & (params_df.svalue_statistics_desc == "dMEAN")
        & (params_df.svaluation_date == point_in_time_dt),
        ["sparameter", "svalue_statistics"],
    ].set_index("sparameter").svalue_statistics
    scen_meta = pd.concat([
        pd.DataFrame([{
            "scenario": "SCEN_0",
            "dGAMMAI": float(_series0.get("dGAMMAI", np.nan)),
            "dBETAI":  float(_series0.get("dBETAI",  np.nan)),
            "dRHOIX":  float(_series0.get("dRHOIX",  np.nan)),
        }]),
        scen_meta,
    ], ignore_index=True).drop_duplicates("scenario", keep="first")

    rows = []
    for date_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        date_str = date_dir.name
        t0_path = t0_dir / date_str / f"{date_str}_pv.parquet"
        if not t0_path.exists():
            logger.warning("missing T0 PV parquet for %s; skipping date", date_str)
            continue
        t0_df = pd.read_parquet(t0_path)
        t0_portfolio = float(sum(np.asarray(x).sum() for x in t0_df["dPV"]))

        for parquet in sorted(date_dir.glob(f"{date_str}_*.parquet")):
            # Parse filename: {date}_{scen}_{h}D_{return_assumption}.parquet
            stem = parquet.stem[len(date_str) + 1:]  # strip "20250416_"
            parts = stem.rsplit("_", 2)              # ["SCEN_12", "10D", "LA"]
            if len(parts) != 3:
                continue
            scenario, horizon_str, ra = parts
            if not horizon_str.endswith("D"):
                continue
            horizon = int(horizon_str[:-1])
            try:
                df = pd.read_parquet(parquet)
            except Exception as exc:
                logger.warning("failed to read %s: %s", parquet, exc)
                continue
            # Portfolio PV per MC path = sum across positions per iSCENARIO_NUM.
            df["pv_sum"] = df["dPV"].apply(lambda x: float(np.asarray(x).sum()))
            portfolio_pv = df.groupby("iSCENARIO_NO")["pv_sum"].sum().values
            pnl = portfolio_pv - t0_portfolio
            var = float(-np.percentile(pnl, (1 - confidence) * 100))
            rows.append({
                "date": date_str,
                "scenario": scenario,
                "return_assumption": ra,
                "horizon_days": horizon,
                "var": var,
                "pnl": pnl,
            })
    var_df = pd.DataFrame(rows)
    if var_df.empty:
        logger.warning("No scenario PV parquets found under %s", base_dir)
        return var_df
    var_df = var_df.merge(scen_meta, on="scenario", how="left")
    var_df = var_df.rename(columns={"dGAMMAI": "gamma", "dBETAI": "beta", "dRHOIX": "rho"})
    logger.info("VaR table: %d rows across %d dates, %d scenarios, %d horizons",
                len(var_df),
                var_df["date"].nunique(),
                var_df["scenario"].nunique(),
                var_df["horizon_days"].nunique())
    return var_df


var_df = _compute_var_table()

# %% cell 25
logger.info("[cell %d/27] Figure 3: VaR surface across (gamma, beta) at rho slices", 25)
if var_df.empty:
    logger.warning("Figure 3 skipped: no VaR data available.")
else:
    # 2 x 3 grid: rows are h in {1, 10}, cols are rho in {-1, 0, 1}.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)
    rho_slices = [-1.0, 0.0, 1.0]
    horizons_fig3 = [1, 10]
    fig3, axes3 = plt.subplots(
        len(horizons_fig3), len(rho_slices),
        figsize=(18, 10), subplot_kw={"projection": "3d"},
    )
    # Filter to Figure 3's valuation date (default point_in_time_dt).
    fig3_var = var_df.loc[var_df["date"] == point_in_time_dt].copy()
    for r_i, h in enumerate(horizons_fig3):
        for c_i, rho_target in enumerate(rho_slices):
            ax = axes3[r_i, c_i]
            sub = fig3_var.loc[
                (fig3_var["horizon_days"] == h)
                & (np.isclose(fig3_var["rho"], rho_target, atol=1e-3))
            ]
            if sub.empty:
                ax.set_title(f"h={h}, rho={rho_target} (no data)", fontsize=9)
                continue
            grid = sub.pivot_table(index="beta", columns="gamma", values="var")
            G, B = np.meshgrid(grid.columns.values, grid.index.values)
            V = grid.values
            surf = ax.plot_surface(G, B, V, cmap=_cm.coolwarm, edgecolor="none")
            ax.set_title(rf"$h={h},\ \rho_{{i,X}}={rho_target}$", fontsize=10, fontweight="bold")
            ax.set_xlabel(r"$\gamma_i$")
            ax.set_ylabel(r"$\beta_i$")
            ax.view_init(elev=25, azim=-45)
    fig3.suptitle("99% VaR in ($)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _fig3_path = _FIGURES_DIR / f"Figure_3_VaR_surface_{point_in_time_dt}.pdf"
    plt.savefig(str(_fig3_path), dpi=300)
    logger.info("Saved %s", _fig3_path)
    plt.show()

# %% cell 26
logger.info("[cell %d/27] Figure 7: VaR term structure with 95%% CI", 26)
if var_df.empty:
    logger.warning("Figure 7 skipped: no VaR data available.")
else:
    def _bootstrap_var_ci(pnl: np.ndarray, confidence: float = 0.99,
                          n_boot: int = 10_000, ci: float = 0.95,
                          seed: int = 20250101):
        rng_ = np.random.default_rng(seed)
        n = len(pnl)
        pcts = np.empty(n_boot)
        for i in range(n_boot):
            sample = rng_.choice(pnl, size=n, replace=True)
            pcts[i] = -np.percentile(sample, (1 - confidence) * 100)
        alpha = (1 - ci) / 2
        return float(np.quantile(pcts, alpha)), float(np.quantile(pcts, 1 - alpha))

    fig7_var = var_df.loc[
        (var_df["scenario"] == "SCEN_0")
        & (var_df["return_assumption"] == "LA")
    ].copy()
    if fig7_var.empty:
        logger.warning("Figure 7 skipped: no SCEN_0 LA rows.")
    else:
        # Bootstrap CIs (once per (date, horizon)).
        boot = []
        for (dt, h), grp in fig7_var.groupby(["date", "horizon_days"]):
            lo, hi = _bootstrap_var_ci(grp["pnl"].iloc[0])
            boot.append({"date": dt, "horizon_days": h, "var_lo": lo, "var_hi": hi})
        fig7_var = fig7_var.merge(pd.DataFrame(boot), on=["date", "horizon_days"])

        fig7, ax7 = plt.subplots(figsize=(12, 6))
        _palette = plt.get_cmap("tab10")
        for i, (dt, grp) in enumerate(fig7_var.sort_values("horizon_days").groupby("date")):
            color = _palette(i)
            label = f"{pd.to_datetime(dt).strftime('%d-%b-%Y')} with 95% CI"
            ax7.plot(grp["horizon_days"], grp["var"], marker="o",
                     linestyle="-" if i == 0 else "--", color=color, label=label)
            ax7.fill_between(grp["horizon_days"], grp["var_lo"], grp["var_hi"],
                             color=color, alpha=0.2)
        ax7.set_xlabel("Risk Horizon (Days)", fontweight="bold")
        ax7.set_ylabel(r"$\bf VaR_{LA}$ ($)", fontweight="bold")
        ax7.set_title(r"Evolution of 99% Liquidity Adjusted Value-at-Risk (VaR$_{LA}$) "
                      r"Term Structure with 95% Confidence Interval (CI)",
                      fontweight="bold")
        ax7.legend(title="Portfolio Valuation Date", title_fontproperties={"weight": "bold"})
        ax7.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        _fig7_path = _FIGURES_DIR / "Figure_7_VaR_termstructure.pdf"
        plt.savefig(str(_fig7_path), dpi=300)
        logger.info("Saved %s", _fig7_path)
        plt.show()

# %% cell 27
logger.info("[cell %d/27] ", 27)

