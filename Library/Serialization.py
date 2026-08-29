"""
Library.Serialization - portable, journal-friendly serialization helpers.

Replaces the pickle-based caches under ``Study/Vol Surface From Model/`` and
``Study/Estimated Parameters QLSQ/`` with Apache Parquet files. Parquet is
typed, cross-language, and safe to load (pickle is not).

Two data shapes are supported:

1. *Vol surface* nested dicts of the form::

       {key: {'iEXPIRY': ndarray, 'dMONEYNESS': ndarray, 'dVOL': ndarray}}

   persisted as a long-form DataFrame ``[key, iEXPIRY, dMONEYNESS, dVOL]``.

2. *Calibration result* dicts of the form ``{key: OptimizeResult}`` where
   downstream code uses only ``result.x`` (the calibrated parameter vector).
   Persisted as a wide DataFrame ``[key, x0, x1, ...]``; on load, values are
   wrapped in a lightweight :class:`CalibrationResult` so downstream
   ``obj.x`` access keeps working.

Both loaders transparently fall back to pickle if the parquet file is not
present. This means the transition is gradual: legacy ``.pkl`` files keep
working until the pipeline is re-run and writes new ``.parquet`` outputs.
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Vol surface
# ----------------------------------------------------------------------------
def save_vol_surface(vol_surface_dict: Dict[str, Dict[str, np.ndarray]], out_path) -> None:
    """Persist ``{key: {iEXPIRY, dMONEYNESS, dVOL}}`` as a long-form parquet.

    The vol surface for each ``key`` is a grid with shape
    ``(len(iEXPIRY), len(dMONEYNESS))`` stored in ``dVOL``. The parquet
    representation is one row per ``(key, expiry_idx, moneyness_idx)`` cell,
    with the numeric expiry / moneyness / vol values inlined. Explicit
    indices preserve the original array ordering on round-trip.
    """
    rows = []
    for key, inner in vol_surface_dict.items():
        iexp = np.asarray(inner["iEXPIRY"])
        dmon = np.asarray(inner["dMONEYNESS"])
        dvol = np.asarray(inner["dVOL"])

        # Support two layouts:
        #   (a) dVOL grid shape (E, M) — matching iEXPIRY(E) x dMONEYNESS(M)
        #   (b) dVOL 1D of length E == M — legacy per-point layout
        if dvol.ndim == 2 and dvol.shape == (len(iexp), len(dmon)):
            for i in range(len(iexp)):
                for j in range(len(dmon)):
                    rows.append({
                        "key": key,
                        "expiry_idx": i,
                        "moneyness_idx": j,
                        "iEXPIRY": int(iexp[i]),
                        "dMONEYNESS": float(dmon[j]),
                        "dVOL": float(dvol[i, j]),
                    })
        elif dvol.ndim == 1 and len(iexp) == len(dmon) == len(dvol):
            for i in range(len(iexp)):
                rows.append({
                    "key": key,
                    "expiry_idx": i,
                    "moneyness_idx": i,
                    "iEXPIRY": int(iexp[i]),
                    "dMONEYNESS": float(dmon[i]),
                    "dVOL": float(dvol[i]),
                })
        else:
            raise ValueError(
                f"Unexpected vol surface shape for key={key!r}: "
                f"iEXPIRY={iexp.shape}, dMONEYNESS={dmon.shape}, dVOL={dvol.shape}"
            )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(str(out_path))


def load_vol_surface(path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load a vol surface from parquet (preferred) or pickle (fallback).

    Reconstructs the original ``{key: {iEXPIRY(E), dMONEYNESS(M), dVOL(E,M)}}``
    nested-dict form the pipeline expects, honouring the explicit
    ``expiry_idx`` / ``moneyness_idx`` ordering.
    """
    p = Path(str(path))
    parquet_path = p if p.suffix == ".parquet" else p.with_suffix(".parquet")
    pkl_path = p if p.suffix == ".pkl" else p.with_suffix(".pkl")

    if parquet_path.exists():
        df = pd.read_parquet(str(parquet_path))
        result: Dict[str, Dict[str, np.ndarray]] = {}
        for key, group in df.groupby("key", sort=False):
            group = group.sort_values(["expiry_idx", "moneyness_idx"])
            e_max = int(group["expiry_idx"].max()) + 1
            m_max = int(group["moneyness_idx"].max()) + 1

            # 2D grid case (typical).
            if len(group) == e_max * m_max:
                iexp = group.drop_duplicates("expiry_idx").sort_values("expiry_idx")["iEXPIRY"].to_numpy()
                dmon = group.drop_duplicates("moneyness_idx").sort_values("moneyness_idx")["dMONEYNESS"].to_numpy()
                pivot = group.pivot(index="expiry_idx", columns="moneyness_idx", values="dVOL")
                dvol = pivot.sort_index().reindex(sorted(pivot.columns), axis=1).to_numpy()
                result[str(key)] = {"iEXPIRY": iexp, "dMONEYNESS": dmon, "dVOL": dvol}
            # 1D per-point layout.
            else:
                result[str(key)] = {
                    "iEXPIRY": group["iEXPIRY"].to_numpy(),
                    "dMONEYNESS": group["dMONEYNESS"].to_numpy(),
                    "dVOL": group["dVOL"].to_numpy(),
                }
        return result

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(f"No vol surface at {p}[.parquet|.pkl]")


# ----------------------------------------------------------------------------
# Calibration results
# ----------------------------------------------------------------------------
class CalibrationResult:
    """Minimal replacement for ``scipy.optimize.OptimizeResult``.

    Only exposes ``.x`` (the calibrated parameter vector), which is what all
    downstream code actually reads from these cache files.
    """

    __slots__ = ("x",)

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def __repr__(self):
        return f"CalibrationResult(x={self.x!r})"


def save_calibration_results(calib_dict: Dict[str, Any], out_path) -> None:
    """Persist ``{key: OptimizeResult}`` as a wide parquet keyed by ``key``.

    Only ``result.x`` is written. Each parameter position becomes column
    ``x0``, ``x1``, ... Missing positions (variable-length ``x``) are stored
    as ``NaN``.
    """
    max_len = max(len(getattr(v, "x", v)) for v in calib_dict.values())
    rows = []
    for key, res in calib_dict.items():
        x = np.asarray(getattr(res, "x", res), dtype=float)
        row = {"key": key}
        for i in range(max_len):
            row[f"x{i}"] = float(x[i]) if i < len(x) else float("nan")
        rows.append(row)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(str(out_path))


def load_calibration_results(path) -> Dict[str, CalibrationResult]:
    """Load calibration results from parquet (preferred) or pickle (fallback).

    Returns ``{key: obj}`` where ``obj.x`` is the parameter vector, matching
    the shape downstream code expects.
    """
    p = Path(str(path))
    parquet_path = p if p.suffix == ".parquet" else p.with_suffix(".parquet")
    pkl_path = p if p.suffix == ".pkl" else p.with_suffix(".pkl")

    if parquet_path.exists():
        df = pd.read_parquet(str(parquet_path))
        result: Dict[str, CalibrationResult] = {}

        # Schema A: our own layout — [key, x0, x1, ...]
        if "key" in df.columns:
            x_cols = sorted(
                [c for c in df.columns if c.startswith("x") and c[1:].isdigit()],
                key=lambda c: int(c[1:]),
            )
            for _, row in df.iterrows():
                x_values = [row[c] for c in x_cols if not pd.isna(row[c])]
                result[str(row["key"])] = CalibrationResult(np.array(x_values))
            return result

        # Schema B: legacy layout from Scripts/migrate_qlsq_cache_to_parquet.py:
        # [sTICKER, sVALUATION_DATE, iEXPIRY, dEXPIRY, dKAPPAI, dGAMMAI, dBETAI,
        #  dRHOIX, dSIGMA, dPPROB, dLAMB, dETA1, dETA2, bCONVERGED, dOBJECTIVE, iN_OBS]
        # Each row holds both systematic and idiosyncratic parameter columns;
        # the "true" calibrated .x depends on whether the ticker is systematic:
        #   ^-prefixed ticker  -> [dSIGMA, dPPROB, dLAMB, dETA1, dETA2]  (5)
        #   otherwise          -> [dKAPPAI, dGAMMAI, dBETAI, dRHOIX]    (4)
        if "sTICKER" in df.columns and "sVALUATION_DATE" in df.columns:
            SYS_PARAMS = ["dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]
            IDI_PARAMS = ["dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX"]
            for _, row in df.iterrows():
                ticker = str(row["sTICKER"])
                key = f"{ticker}-{row['sVALUATION_DATE']}"
                cols = SYS_PARAMS if ticker.startswith("^") else IDI_PARAMS
                x_values = [float(row[c]) for c in cols]
                result[key] = CalibrationResult(np.array(x_values))
            return result

        raise ValueError(
            f"Unrecognised calibration parquet schema at {parquet_path}: "
            f"columns={list(df.columns)}"
        )

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(f"No calibration results at {p}[.parquet|.pkl]")
