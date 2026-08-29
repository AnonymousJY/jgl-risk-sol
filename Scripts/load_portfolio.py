# Ensure package imports resolve regardless of cwd.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from typing import List
from collections import namedtuple


Portfolio = namedtuple(
    typename='Portfolio',
    field_names=[
        'sVALUATION_DATE',
        'sIDIOSYNCRATIC_ID',
        'sSTRATEGY',
        'sTYPOLOGY',
        'sPAYOFF_TYPE',
        'sEXPIRY_DATE',
        'dSTRIKE_PRICE',
        'dPOSITION',
        'sFIXING_FREQUENCY',
        'sFIXING_DATE_BEG',
        'sFIXING_DATE_END',
        'dBASE_DAYS'
    ]
)


def get_idiosyncratic_ids() -> List[str]:
    """Names of the idiosyncratic underlyings the pipeline calibrates.

    Reads from Scripts/config_skew.py:IDIOSYNCRATIC_UNDERLYINGS so that
    adding a new name (e.g. "AAPL") is a one-place edit in config_skew.py;
    the seven downstream callers of this function -- run_pmle_kimyi2025.py,
    run_var_kimyi2025.ipynb, collar_asian.ipynb, skew_target_idi.py,
    export_snapshots.py, skew_calibrate_{systematic,idiosyncratic}.py, and
    this module's own __main__ -- pick it up automatically.

    Prior to the config-consolidation refactor this returned a hard-coded
    ``["COIN"]``; config_skew.py used to import this function to populate
    IDIOSYNCRATIC_TICKERS, and that arrow now runs the other way.

    The import is deferred to call time so this module can still be imported
    before Scripts/ is on sys.path (e.g. from a notebook whose CWD is a
    sibling directory).
    """
    import config_skew as cfg  # deferred import; see docstring.
    return list(cfg.IDIOSYNCRATIC_UNDERLYINGS.keys())


# ----------------------------------------------------------------------------
# Asian Collar portfolio specification
#
# The worked example in the paper: a discrete-Asian collar (short put / long
# call) hedged with vanilla options (long put / short call). These constants
# define that structure; build_portfolio() materialises it for a given
# valuation date so notebooks and scripts can obtain the portfolio without a
# database round-trip.
# ----------------------------------------------------------------------------
EXPIRY_DATE = '20250417'
BASE_DAYS = 252
FIXING_FREQUENCY = 'B'
FIXING_DATE_BEG = '20250331'
FIXING_DATE_END = '20250417'

COLLAR_STRIKE_PUT = 172.23
COLLAR_STRIKE_CALL = 198.06
VANILLA_STRIKE_PUT = 180.84
VANILLA_STRIKE_CALL = 206.67

COLLAR_POSITION_PUT = -100.
COLLAR_POSITION_CALL = 100.
VANILLA_POSITION_PUT = 85.
VANILLA_POSITION_CALL = -120.


def build_portfolio(valuation_date: str, idiosyncratic_id: str) -> List[Portfolio]:
    """Return the four Portfolio legs for one (valuation date, underlying).

    Parameters
    ----------
    valuation_date : str
        Valuation date in ``YYYYMMDD`` format.
    idiosyncratic_id : str
        Underlying ticker, e.g. ``"COIN"``.

    Returns
    -------
    list[Portfolio]
        The discrete-Asian collar put/call legs followed by the two vanilla
        hedge legs.
    """
    collar_put = Portfolio(
        sVALUATION_DATE=valuation_date,
        sIDIOSYNCRATIC_ID=idiosyncratic_id,
        sTYPOLOGY='asian discrete',
        sSTRATEGY='collar',
        sPAYOFF_TYPE='put',
        sEXPIRY_DATE=EXPIRY_DATE,
        dSTRIKE_PRICE=COLLAR_STRIKE_PUT,
        dPOSITION=COLLAR_POSITION_PUT,
        sFIXING_FREQUENCY=FIXING_FREQUENCY,
        sFIXING_DATE_BEG=FIXING_DATE_BEG,
        sFIXING_DATE_END=FIXING_DATE_END,
        dBASE_DAYS=BASE_DAYS
    )
    collar_call = Portfolio(
        sVALUATION_DATE=valuation_date,
        sIDIOSYNCRATIC_ID=idiosyncratic_id,
        sTYPOLOGY='asian discrete',
        sSTRATEGY='collar',
        sPAYOFF_TYPE='call',
        sEXPIRY_DATE=EXPIRY_DATE,
        dSTRIKE_PRICE=COLLAR_STRIKE_CALL,
        dPOSITION=COLLAR_POSITION_CALL,
        sFIXING_FREQUENCY=FIXING_FREQUENCY,
        sFIXING_DATE_BEG=FIXING_DATE_BEG,
        sFIXING_DATE_END=FIXING_DATE_END,
        dBASE_DAYS=BASE_DAYS
    )
    vanilla_put = Portfolio(
        sVALUATION_DATE=valuation_date,
        sIDIOSYNCRATIC_ID=idiosyncratic_id,
        sTYPOLOGY='vanilla',
        sSTRATEGY='vanilla',
        sPAYOFF_TYPE='put',
        sEXPIRY_DATE=EXPIRY_DATE,
        dSTRIKE_PRICE=VANILLA_STRIKE_PUT,
        dPOSITION=VANILLA_POSITION_PUT,
        sFIXING_FREQUENCY=None,
        sFIXING_DATE_BEG=None,
        sFIXING_DATE_END=None,
        dBASE_DAYS=BASE_DAYS
    )
    vanilla_call = Portfolio(
        sVALUATION_DATE=valuation_date,
        sIDIOSYNCRATIC_ID=idiosyncratic_id,
        sTYPOLOGY='vanilla',
        sSTRATEGY='vanilla',
        sPAYOFF_TYPE='call',
        sEXPIRY_DATE=EXPIRY_DATE,
        dSTRIKE_PRICE=VANILLA_STRIKE_CALL,
        dPOSITION=VANILLA_POSITION_CALL,
        sFIXING_FREQUENCY=None,
        sFIXING_DATE_BEG=None,
        sFIXING_DATE_END=None,
        dBASE_DAYS=BASE_DAYS
    )
    return [collar_put, collar_call, vanilla_put, vanilla_call]


if __name__ == "__main__":
    # Materialise the portfolio for the paper's valuation window and write it
    # to a committed CSV snapshot. There is no database dependency: notebooks
    # and scripts can either read data/snapshots/portfolio.csv or call
    # build_portfolio() directly.
    import pandas as pd

    from Library.DataAccess import write_snapshot

    valuation_beg_dt = '20250331'
    valuation_end_dt = '20250416'
    dt_format = '%Y%m%d'

    valuation_date_window = pd.bdate_range(
        start=pd.to_datetime(arg=valuation_beg_dt, format=dt_format),
        end=pd.to_datetime(arg=valuation_end_dt, format=dt_format)
    )

    portfolio_rows = []
    for val_dt in valuation_date_window:
        val_dt = val_dt.strftime(dt_format)
        for idiosyncratic_id in get_idiosyncratic_ids():
            portfolio_rows.extend(build_portfolio(val_dt, idiosyncratic_id))

    portfolio_df = pd.DataFrame(portfolio_rows, columns=Portfolio._fields)
    path = write_snapshot(portfolio_df, "portfolio.csv", index=False)
    print(f"{len(portfolio_df)} portfolio rows written to {path}")