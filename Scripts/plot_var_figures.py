"""
plot_var_figures.py -- render Figures 3 and 7 from the ScenarioPV / CurrentPV
parquet caches produced by run_var_kimyi2025.py.

No Monte Carlo, no calibration -- pure I/O + matplotlib. Runs in seconds.
Adaptive to whatever dates and scenarios are already on disk:

    * Figure 3 (VaR surface) uses point_in_time_dt (default = most recent
      date under Study/Collar Asian/ScenarioPV/). It needs the full 245-cell
      what-if lattice; if fewer scenarios are present the surface will be
      sparse and a warning is logged.

    * Figure 7 (VaR term structure) renders one curve per (date, SCEN_0)
      pair present under Study/Collar Asian/ScenarioPV/. Two dates is what
      the paper shows; the plot still renders if only one date is available.

Usage
-----
    python Scripts/plot_var_figures.py                         # auto-detect date
    python Scripts/plot_var_figures.py --date 20250416         # single-date Figure 3
    python Scripts/plot_var_figures.py --date 20250416 \
        --no-fig-7                                             # skip term structure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib import cm as _cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Library.Logging import setup_logging  # noqa: E402

logger = setup_logging(__name__)

_SCENARIO_PV_DIR = _REPO_ROOT / "Study" / "Collar Asian" / "ScenarioPV"
_CURRENT_PV_DIR  = _REPO_ROOT / "Study" / "Collar Asian" / "CurrentPV"
_SCENARIOS_CSV   = _REPO_ROOT / "Scripts" / "Scenarios" / "what_if_scenarios.csv"
_FIGURES_DIR     = _REPO_ROOT / "Figures"


# ---------------------------------------------------------------------------
# Scenario -> (gamma, beta, rho) lookup
# ---------------------------------------------------------------------------
def _load_scenario_meta() -> pd.DataFrame:
    if not _SCENARIOS_CSV.exists():
        raise FileNotFoundError(
            f"Scenarios CSV missing: {_SCENARIOS_CSV}. "
            "Run run_var_kimyi2025.py first to generate it."
        )
    scen = pd.read_csv(_SCENARIOS_CSV).rename(
        columns={"sSCENARIO": "scenario"}
    )
    return scen[["scenario", "dGAMMAI", "dBETAI", "dRHOIX"]]


# ---------------------------------------------------------------------------
# Compute VaR from disk
# ---------------------------------------------------------------------------
def compute_var_table(confidence: float = 0.99) -> pd.DataFrame:
    """Walk ScenarioPV/*/{date}_{scen}_{h}D_{ra}.parquet, subtract T0, and
    return a tidy table with columns:

        date  scenario  return_assumption  horizon_days  gamma beta rho
        var   pnl (numpy array)
    """
    if not _SCENARIO_PV_DIR.exists():
        logger.error("ScenarioPV dir missing: %s", _SCENARIO_PV_DIR)
        return pd.DataFrame()

    scen_meta = _load_scenario_meta()

    rows = []
    for date_dir in sorted(p for p in _SCENARIO_PV_DIR.iterdir() if p.is_dir()):
        date_str = date_dir.name
        t0_path = _CURRENT_PV_DIR / date_str / f"{date_str}_pv.parquet"
        if not t0_path.exists():
            logger.warning("missing T0 PV parquet for %s; skipping date", date_str)
            continue
        t0_df = pd.read_parquet(t0_path)
        t0_portfolio = float(sum(np.asarray(x).sum() for x in t0_df["dPV"]))

        for parquet in sorted(date_dir.glob(f"{date_str}_*.parquet")):
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
        logger.warning("no scenario PV parquets found under %s", _SCENARIO_PV_DIR)
        return var_df
    var_df = var_df.merge(scen_meta, on="scenario", how="left")
    var_df = var_df.rename(columns={"dGAMMAI": "gamma", "dBETAI": "beta", "dRHOIX": "rho"})
    logger.info(
        "VaR table: %d rows across %d date(s), %d scenario(s), %d horizon(s)",
        len(var_df),
        var_df["date"].nunique(),
        var_df["scenario"].nunique(),
        var_df["horizon_days"].nunique(),
    )
    return var_df


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------
def plot_figure_3(var_df: pd.DataFrame, point_in_time_dt: str) -> Path:
    fig3_var = var_df.loc[var_df["date"] == point_in_time_dt].copy()
    if fig3_var.empty:
        raise RuntimeError(f"No VaR rows for date {point_in_time_dt}.")

    n_scen = fig3_var["scenario"].nunique()
    if n_scen < 100:
        logger.warning(
            "Figure 3: only %d scenarios available for %s. Surface will be sparse; "
            "rerun run_var_kimyi2025.py (with the SCEN_0 guard removed) to fill "
            "the full what-if lattice.",
            n_scen, point_in_time_dt,
        )

    rho_slices = [-1.0, 0.0, 1.0]
    horizons = [1, 10]
    fig, axes = plt.subplots(
        len(horizons), len(rho_slices),
        figsize=(18, 10), subplot_kw={"projection": "3d"},
    )
    for r_i, h in enumerate(horizons):
        for c_i, rho_target in enumerate(rho_slices):
            ax = axes[r_i, c_i]
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
            ax.plot_surface(G, B, V, cmap=_cm.coolwarm, edgecolor="none")
            ax.set_title(rf"$h={h},\ \rho_{{i,X}}={rho_target}$",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel(r"$\gamma_i$")
            ax.set_ylabel(r"$\beta_i$")
            ax.view_init(elev=25, azim=-45)
    fig.suptitle("99% VaR in ($)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = _FIGURES_DIR / f"Figure_3_VaR_surface_{point_in_time_dt}.pdf"
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ---------------------------------------------------------------------------
# Figure 7
# ---------------------------------------------------------------------------
def _bootstrap_var_ci(pnl: np.ndarray, confidence: float = 0.99,
                      n_boot: int = 10_000, ci: float = 0.95,
                      seed: int = 20250101):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    pcts = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(pnl, size=n, replace=True)
        pcts[i] = -np.percentile(sample, (1 - confidence) * 100)
    alpha = (1 - ci) / 2
    return float(np.quantile(pcts, alpha)), float(np.quantile(pcts, 1 - alpha))


def plot_figure_7(var_df: pd.DataFrame) -> Path:
    fig7_var = var_df.loc[
        (var_df["scenario"] == "SCEN_0")
        & (var_df["return_assumption"] == "LA")
    ].copy()
    if fig7_var.empty:
        raise RuntimeError("No SCEN_0 LA rows to plot.")

    # Bootstrap CIs once per (date, horizon).
    boot = []
    for (dt, h), grp in fig7_var.groupby(["date", "horizon_days"]):
        lo, hi = _bootstrap_var_ci(grp["pnl"].iloc[0])
        boot.append({"date": dt, "horizon_days": h, "var_lo": lo, "var_hi": hi})
    fig7_var = fig7_var.merge(pd.DataFrame(boot), on=["date", "horizon_days"])

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = plt.get_cmap("tab10")
    for i, (dt, grp) in enumerate(fig7_var.sort_values("horizon_days").groupby("date")):
        color = palette(i)
        label = f"{pd.to_datetime(dt).strftime('%d-%b-%Y')} with 95% CI"
        ax.plot(grp["horizon_days"], grp["var"], marker="o",
                linestyle="-" if i == 0 else "--", color=color, label=label)
        ax.fill_between(grp["horizon_days"], grp["var_lo"], grp["var_hi"],
                        color=color, alpha=0.2)
    ax.set_xlabel("Risk Horizon (Days)", fontweight="bold")
    ax.set_ylabel(r"$\bf VaR_{LA}$ ($)", fontweight="bold")
    ax.set_title(
        r"Evolution of 99% Liquidity Adjusted Value-at-Risk (VaR$_{LA}$) "
        r"Term Structure with 95% Confidence Interval (CI)",
        fontweight="bold",
    )
    ax.legend(title="Portfolio Valuation Date",
              title_fontproperties={"weight": "bold"})
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    plt.tight_layout()
    out = _FIGURES_DIR / "Figure_7_VaR_termstructure.pdf"
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _latest_date_on_disk() -> str:
    if not _SCENARIO_PV_DIR.exists():
        raise FileNotFoundError(f"No ScenarioPV/ dir at {_SCENARIO_PV_DIR}.")
    dates = sorted(p.name for p in _SCENARIO_PV_DIR.iterdir() if p.is_dir())
    if not dates:
        raise FileNotFoundError(f"No date subdirs under {_SCENARIO_PV_DIR}.")
    return dates[-1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", type=str, default=None,
                   help="Valuation date YYYYMMDD for Figure 3 (default: latest on disk).")
    p.add_argument("--no-fig-3", action="store_true", help="Skip Figure 3.")
    p.add_argument("--no-fig-7", action="store_true", help="Skip Figure 7.")
    args = p.parse_args(argv)

    var_df = compute_var_table()
    if var_df.empty:
        logger.error("No VaR data. Run run_var_kimyi2025.py first.")
        return 1

    point_in_time_dt = args.date or _latest_date_on_disk()
    logger.info("Using date %s for Figure 3.", point_in_time_dt)

    if not args.no_fig_3:
        try:
            plot_figure_3(var_df, point_in_time_dt)
        except Exception as exc:
            logger.error("Figure 3 failed: %s", exc)
    if not args.no_fig_7:
        try:
            plot_figure_7(var_df)
        except Exception as exc:
            logger.error("Figure 7 failed: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
