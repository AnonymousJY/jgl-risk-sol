"""
DataAccess.py — unified data layer for the mkt-depth-n-resiliency project.

Purpose
-------
The original scripts read market data live from FinanceDataReader. That makes
the repository impossible to reproduce later: vendor history is revised over
time, so an exact rerun is not generally possible.

This module decouples the analysis code from that live source. By default it
runs in ``snapshot`` mode, reading committed CSV snapshots and the per-date
P-MLE parameter CSVs that ship in ``Study/Estimated Parameters PMLE/``. The
repository is therefore a self-contained replication package: clone, install
the environment, and every table and figure can be regenerated from committed
inputs.

``live`` mode restores the FinanceDataReader pull for users who want to
refresh the data or extend the sample.

Modes
-----
The active mode is read from the ``MKTDEPTH_DATA_MODE`` environment variable
(``snapshot`` or ``live``); it defaults to ``snapshot``. Any individual call
can override it with the ``mode=`` keyword.

    export MKTDEPTH_DATA_MODE=live      # refresh prices from FinanceDataReader
    export MKTDEPTH_DATA_MODE=snapshot  # default; committed inputs only

Snapshots are produced by ``Scripts/export_snapshots.py``. P-MLE parameter
estimates are produced and written directly to
``Study/Estimated Parameters PMLE/`` by ``Scripts/run_pmle_kimyi2025.py``;
no database is required.
"""

import os
import warnings

import pandas as pd

# --- repository layout -------------------------------------------------------
# Library/DataAccess.py  ->  repo root is one level up from this file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_DIR = os.path.join(_REPO_ROOT, "data", "snapshots")
PMLE_DIR = os.path.join(_REPO_ROOT, "Study", "Estimated Parameters PMLE")

# --- mode --------------------------------------------------------------------
_VALID_MODES = ("snapshot", "live")
DATA_MODE = os.environ.get("MKTDEPTH_DATA_MODE", "snapshot").lower()
if DATA_MODE not in _VALID_MODES:
    warnings.warn(
        f"Unknown MKTDEPTH_DATA_MODE={DATA_MODE!r}; falling back to 'snapshot'."
    )
    DATA_MODE = "snapshot"


def _resolve_mode(mode):
    """Return the effective mode, validating any per-call override."""
    if mode is None:
        return DATA_MODE
    mode = mode.lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    return mode


def _safe_symbol(symbol):
    """Make a symbol filesystem-safe (e.g. the '^SPX' index ticker)."""
    return symbol.replace("^", "").replace("/", "_").replace("\\", "_")


# ============================================================================
# Price series
# ============================================================================
def price_snapshot_path(symbol):
    """Path to the committed price snapshot for ``symbol``."""
    return os.path.join(SNAPSHOT_DIR, f"prices_{_safe_symbol(symbol)}.csv")


def get_price_series(symbol, mode=None):
    """Adjusted-close price series for ``symbol`` as a ``pd.Series``.

    Parameters
    ----------
    symbol : str
        Ticker, e.g. ``"^SPX"`` or ``"COIN"``.
    mode : {"snapshot", "live", None}
        Override the module-level data mode for this call. ``None`` uses
        the module default (``DATA_MODE``).

    Snapshot mode reads ``data/snapshots/prices_{symbol}.csv``. If that file
    is missing, it warns and falls back to a live FinanceDataReader pull so a
    fresh clone still works; run ``Scripts/export_snapshots.py`` to create the
    snapshot and make the result deterministic.
    """
    mode = _resolve_mode(mode)

    if mode == "snapshot":
        path = price_snapshot_path(symbol)
        if os.path.exists(path):
            series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            series.index.name = "sVALUATION_DATE"
            series.name = symbol
            return series
        warnings.warn(
            f"No price snapshot at {path}. Falling back to live "
            f"FinanceDataReader for {symbol!r}. Run Scripts/export_snapshots.py "
            f"to freeze a reproducible snapshot."
        )

    # live mode (or snapshot fallback)
    import FinanceDataReader as fdr

    series = fdr.DataReader(symbol)["Adj Close"]
    series.index.name = "sVALUATION_DATE"
    series.name = symbol
    return series


def get_price_panel(symbols, mode=None):
    """Adjusted-close prices for several symbols as a forward/back-filled
    ``pd.DataFrame`` (columns = symbols). Mirrors the concatenation logic
    previously inlined in ``run_pmle_kimyi2025.py``.

    .. warning::

       This function **back-fills**, and back-fill propagates the first
       observed price backwards into every date before a name was listed. For
       a 2021 IPO that fabricates a flat price series stretching to 2007 -
       precisely the history the backfill study exists to model. A flat series
       has zero return, zero volatility and zero systematic loading, so the
       fabrication is silent: nothing looks wrong, the estimates are simply
       answers about a constant.

       It is preserved unchanged because ``Scripts/`` reproduces the published
       paper and its inputs must not move. **For any new work use**
       :func:`get_aligned_price_panel`, which reindexes onto a reference
       trading calendar, forward-fills interior gaps only, and never
       back-fills. ``poc/backfill_poc.py`` already avoids this function by hand.
    """
    frames = [get_price_series(sym, mode=mode) for sym in symbols]
    panel = pd.concat(frames, axis=1)
    panel.index = pd.to_datetime(panel.index)
    return panel.ffill().bfill()


def get_aligned_price_panel(symbols, reference=None, mode=None,
                            ffill_limit=5, verbose=True):
    """Price panel aligned to one symbol's trading calendar.

    The reference symbol - normally the systematic factor, ``"^SPX"`` - defines
    which dates exist. Every other symbol is reindexed onto those dates. This
    is what makes a per-name return series comparable with the systematic
    series: both are then observed on exactly the same dates, so beta and rho
    are estimated from genuinely paired observations.

    Gap handling, and the reasoning:

    * **Interior gaps are forward-filled**, up to ``ffill_limit`` sessions. A
      US name missing a price on an NYSE session is a trading halt or a vendor
      gap. Forward-filling reports a zero return on the halt day and the full
      move on the reopen, which is what a holder actually experienced - they
      could not trade, and then the gap happened. For a model whose subject is
      gap risk that is the right representation, and it keeps the jump on the
      date it was realised. Halts are rare enough (well under 1% of sessions)
      that the resulting atom at zero is immaterial.

    * **Leading gaps are never filled.** Dates before a name's first
      observation stay ``NaN``. This is the pre-listing period, and inventing
      prices for it would fabricate exactly the history the study is meant to
      reconstruct and then score itself against.

    * **Gaps longer than** ``ffill_limit`` **are left as** ``NaN`` **and
      reported.** A twenty-session gap is not a halt; it is a suspension, a
      corporate action or a broken snapshot, and it should be looked at rather
      than smoothed over.

    Parameters
    ----------
    symbols : sequence of str
    reference : str, optional
        Symbol whose calendar governs. Defaults to ``symbols[0]``.
    ffill_limit : int or None
        Maximum consecutive sessions to carry a stale price forward. ``None``
        for unlimited (not advised).
    verbose : bool
        Print the fill report. The report is worth reading rather than
        skipping: a name needing many fills is a name that stops trading, which
        is a liquidity statement about the very thing being modelled.

    Returns
    -------
    panel : pd.DataFrame
        Columns in the order given, indexed by the reference calendar.
    report : dict
        Per symbol: ``filled`` (sessions carried forward), ``leading_na``
        (pre-listing sessions), ``unfilled_na`` (gaps exceeding the limit),
        ``first_obs``, ``longest_gap``.
    """
    if not symbols:
        raise ValueError("symbols is empty")
    reference = reference or symbols[0]
    if reference not in symbols:
        symbols = [reference] + list(symbols)

    frames = {sym: get_price_series(sym, mode=mode) for sym in symbols}
    for sym, ser in frames.items():
        ser.index = pd.to_datetime(ser.index)

    calendar = frames[reference].dropna().index.sort_values()
    panel = pd.DataFrame(index=calendar)
    report = {}

    for sym in symbols:
        raw = frames[sym].reindex(calendar)
        first = raw.first_valid_index()

        if first is None:
            report[sym] = {"filled": 0, "leading_na": len(calendar),
                           "unfilled_na": 0, "first_obs": None,
                           "longest_gap": 0}
            panel[sym] = raw
            continue

        body = raw.loc[first:]
        # longest run of consecutive NaN inside the observed period
        isna = body.isna().to_numpy()
        longest = 0
        run = 0
        for flag in isna:
            run = run + 1 if flag else 0
            longest = max(longest, run)

        filled_body = body.ffill(limit=ffill_limit)
        n_filled = int(body.isna().sum() - filled_body.isna().sum())

        out = raw.copy()
        out.loc[first:] = filled_body          # leading NaNs untouched
        panel[sym] = out

        report[sym] = {
            "filled": n_filled,
            "leading_na": int(raw.loc[:first].isna().sum()),
            "unfilled_na": int(filled_body.isna().sum()),
            "first_obs": first,
            "longest_gap": longest,
        }

    if verbose:
        print("price panel aligned to %s: %d sessions, %s -> %s"
              % (reference, len(calendar),
                 calendar[0].date(), calendar[-1].date()))
        print("  %-8s %10s %8s %10s %10s  %s"
              % ("symbol", "first obs", "filled", "pre-list", "unfilled",
                 "longest gap"))
        for sym in symbols:
            r = report[sym]
            flag = "  <-- CHECK" if r["unfilled_na"] else ""
            print("  %-8s %10s %8d %10d %10d  %11d%s"
                  % (sym,
                     r["first_obs"].date() if r["first_obs"] is not None else "never",
                     r["filled"], r["leading_na"], r["unfilled_na"],
                     r["longest_gap"], flag))

    return panel, report


# ============================================================================
# P-MLE parameter estimates
# ============================================================================
def pmle_params_path(valuation_date, underlying_name):
    """Path to the cached wide-format P-MLE parameter CSV for one
    (valuation date, underlying)."""
    date_str = pd.to_datetime(valuation_date).strftime("%Y%m%d")
    return os.path.join(
        PMLE_DIR,
        underlying_name,
        f"estimated_params_pmle_{underlying_name}_{date_str}.csv",
    )


def get_pmle_params(valuation_date, underlying_name):
    """Cached P-MLE parameter estimates for one (valuation date, underlying)
    as a wide ``pd.Series``.

    The returned Series exposes the model parameters as attributes
    (``.dALPHA``, ``.dSIGMA``, ``.dGAMMAI`` ...), matching the shape expected
    by ``RiskEngineKimYi2025.simulate_shock_returns`` and the Study notebooks.

    These CSVs already ship in ``Study/Estimated Parameters PMLE/`` and are the
    estimates underlying Table 1 of the paper.
    """
    path = pmle_params_path(valuation_date, underlying_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached P-MLE parameter file for underlying={underlying_name!r}, "
            f"date={valuation_date!r} at:\n  {path}\n"
            f"Available dates can be listed with available_pmle_dates()."
        )
    return pd.read_csv(path).iloc[0]


def get_pmle_params_dict(valuation_date, underlying_name, params=None):
    """Same data as :func:`get_pmle_params`, returned as the
    ``{'dALPHA': ..., 'dSIGMA': ...}`` long-form dict that
    ``pmle_kimyirisk_idiosyncratic`` consumes for the systematic parameter set.

    ``params`` optionally restricts the keys (e.g. the six systematic
    parameters ``['dALPHA', 'dSIGMA', 'dPPROB', 'dLAMB', 'dETA1', 'dETA2']``).
    """
    series = get_pmle_params(valuation_date, underlying_name)
    if params is None:
        # Numeric model parameters begin with 'd' (e.g., dALPHA, dSIGMA) but
        # exclude the date column 'dtVALUATION_DATE' and the credible-interval
        # bounds '*_CI_LOWER'/'*_CI_UPPER'.
        params = [
            c for c in series.index
            if c.startswith("d") and not c.startswith("dt") and "_CI_" not in c
        ]
    return {k: float(series[k]) for k in params}


def available_pmle_dates(underlying_name):
    """Sorted list of valuation dates (``YYYYMMDD`` strings) for which a cached
    P-MLE parameter CSV exists for ``underlying_name``."""
    folder = os.path.join(PMLE_DIR, underlying_name)
    if not os.path.isdir(folder):
        return []
    prefix = f"estimated_params_pmle_{underlying_name}_"
    dates = [
        fn[len(prefix):-len(".csv")]
        for fn in os.listdir(folder)
        if fn.startswith(prefix) and fn.endswith(".csv")
    ]
    return sorted(dates)


def pmle_params_exists(valuation_date, underlying_name):
    """True if a cached P-MLE parameter CSV already exists for this
    (valuation date, underlying). Used by ``run_pmle_kimyi2025.py`` to skip
    dates that have already been estimated — the file-based replacement for
    the old "is this row already in Postgres?" check."""
    return os.path.exists(pmle_params_path(valuation_date, underlying_name))


# Canonical column order of the wide-format P-MLE parameter CSVs.
PMLE_PARAM_ORDER = (
    "dMUI", "dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX",
    "dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2",
)


def save_pmle_params(valuation_date, underlying_name, params):
    """Write one wide-format P-MLE parameter CSV to
    ``Study/Estimated Parameters PMLE/{underlying}/``.

    Parameters
    ----------
    valuation_date : str
        Valuation date in ``YYYYMMDD`` format.
    underlying_name : str
        Underlying ticker, e.g. ``"^SPX"`` or ``"COIN"``.
    params : dict
        Maps each of the eleven parameter names in :data:`PMLE_PARAM_ORDER`
        to a ``(mean, ci_lower, ci_upper)`` triple.

    The column layout matches the CSVs already shipping under
    ``Study/Estimated Parameters PMLE/`` (date, underlying, eleven means,
    then the twenty-two CI bounds in parameter order), so files written here
    are read back transparently by :func:`get_pmle_params`.
    """
    missing = [p for p in PMLE_PARAM_ORDER if p not in params]
    if missing:
        raise KeyError(f"save_pmle_params: missing parameters {missing}")

    row = {
        "dtVALUATION_DATE": pd.to_datetime(valuation_date).strftime("%Y-%m-%d %H:%M:%S"),
        "sUNDERLYING_NAME": underlying_name,
    }
    for p in PMLE_PARAM_ORDER:
        row[p] = params[p][0]
    for p in PMLE_PARAM_ORDER:
        _, ci_lower, ci_upper = params[p]
        row[f"{p}_CI_LOWER"] = ci_lower
        row[f"{p}_CI_UPPER"] = ci_upper

    path = pmle_params_path(valuation_date, underlying_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


# ============================================================================
# Generic snapshot helpers
# ============================================================================
def read_snapshot(name, **read_csv_kwargs):
    """Read an arbitrary committed snapshot CSV from ``data/snapshots/``."""
    path = os.path.join(SNAPSHOT_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Snapshot {name!r} not found at {path}. "
            f"Run Scripts/export_snapshots.py to generate it."
        )
    return pd.read_csv(path, **read_csv_kwargs)


def write_snapshot(df, name, **to_csv_kwargs):
    """Write ``df`` to ``data/snapshots/{name}``, creating the directory if
    needed. Used by ``Scripts/export_snapshots.py``."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, name)
    df.to_csv(path, **to_csv_kwargs)
    return path
