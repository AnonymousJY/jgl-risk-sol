"""
mcmc_diagnostics.py - MCMC convergence and identification diagnostics for a
single (valuation date, ticker) P-MLE calibration.

This script re-runs the same MCMC as run_pmle_kimyi2025.py for one case and
reports:
  1. Per-parameter ESS (bulk, tail) and R-hat (convergence diagnostics)
  2. Posterior correlation matrix in the natural (economic) parameter space
     (identification diagnostic: highly correlated pairs indicate trade-offs)
  3. Trace plots (saved to disk)
  4. Pair plots (saved to disk)
  5. Full InferenceData saved as NetCDF for later re-analysis

If ESS < 400 or R-hat > 1.01 for a parameter, that parameter's wide credible
interval is at least partly a chain-mixing artifact rather than a purely
structural feature; longer chains or a non-centered parameterization may help.

If a pair of parameters has |correlation| > 0.9 in the posterior, they are
trading off (i.e. the marginal credible interval is wide because the JOINT
posterior is a narrow diagonal ridge). Reparameterizing to an orthogonal basis
would shrink the marginals without changing the underlying information content.

Run from the repository root:

    PYTHONPATH=. python Scripts/mcmc_diagnostics.py \
        --valuation-date 20250409 --ticker COIN

Outputs are written to ``Study/Diagnostics/{YYYYMMDD}_{TICKER}/``.
"""

import argparse
import sys
import warnings
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402
import pytensor.tensor as pt  # noqa: E402

# Ensure package imports work regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Library.DataAccess import get_price_panel, get_pmle_params_dict  # noqa: E402
from Library.RiskEngineKimYi2025 import (  # noqa: E402
    _dist_loglike_systematic,
    _dist_loglike_idiosyncratic,
)


# ----------------------------------------------------------------------------
# Configuration (mirrors run_pmle_kimyi2025.py)
# ----------------------------------------------------------------------------
SYSTEMATIC_TICKER = "^SPX"
DEFAULT_LOOKBACK = 252
DELTA_T = np.array(1.0 / 252.0)
DEFAULT_SEED = 20240114
DEFAULT_N_DRAWS = 10_000

SYS_SAMPLE_VARS = ["alpha_rv", "sigma", "pprob_rv", "lamb", "eta1", "eta2"]
SYS_NATURAL_VARS = ["alpha", "sigma", "pprob", "lamb", "eta1", "eta2"]

IDI_SAMPLE_VARS = ["mui", "kappai_rv", "gamma_rv", "betai_rv", "rhoix_rv"]
IDI_NATURAL_VARS = ["mui", "kappai", "gammai", "betai", "rhoix"]

# Rule-of-thumb thresholds for warnings.
ESS_THRESHOLD = 400
RHAT_THRESHOLD = 1.01
CORR_THRESHOLD = 0.9


# ----------------------------------------------------------------------------
# Model builders (duplicated from Library.RiskEngineKimYi2025 to keep this
# script standalone and to avoid touching production functions)
# ----------------------------------------------------------------------------
def build_systematic_model(sys_returns, delta_t):
    with pm.Model() as model:
        sigma = pm.Gamma(name="sigma", alpha=1.0, beta=1.0)

        alpha_rv = pm.Beta(name="alpha_rv", alpha=5.0, beta=2.0)
        alpha = pm.Deterministic("alpha", pt.log(alpha_rv))

        pprob_rv = pm.Beta(name="pprob_rv", alpha=5.0, beta=2.0)
        pprob = pm.Deterministic("pprob", pt.log(pprob_rv))

        lamb = pm.Gamma(name="lamb", alpha=10.0, beta=0.5)
        eta1 = pm.Gamma(name="eta1", alpha=50.0, beta=1.0)
        eta2 = pm.Gamma(name="eta2", alpha=25.0, beta=1.0)

        observed_data = np.cumsum(sys_returns).reshape((-1, 1))

        pm.CustomDist(
            "likelihood",
            alpha,
            sigma,
            pprob,
            lamb,
            eta1,
            eta2,
            delta_t,
            observed=observed_data,
            logp=_dist_loglike_systematic,
        )
    return model


def build_idiosyncratic_model(idi_returns, params_sys, delta_t):
    with pm.Model() as model:
        mui = pm.Normal(name="mui")

        kappai_rv = pm.Gamma(name="kappai_rv", alpha=2.0, beta=1.0)
        kappai = pm.Deterministic("kappai", pt.log(kappai_rv))

        gammai_rv = pm.Gamma(name="gamma_rv", alpha=3.0, beta=1.0)
        gammai = pm.Deterministic("gammai", pt.log(gammai_rv))

        betai_rv = pm.Gamma(name="betai_rv", alpha=3.0, beta=1.0)
        betai = pm.Deterministic("betai", pt.log(betai_rv))

        rhoix_rv = pm.Beta(name="rhoix_rv", alpha=5.0, beta=2.0)
        loc, scale = -1.0, 2.0
        rhoix = pm.Deterministic("rhoix", pt.arctanh((scale * rhoix_rv) + loc))

        observed_data = np.cumsum(idi_returns).reshape((-1, 1))

        pm.CustomDist(
            "likelihood",
            mui,
            kappai,
            gammai,
            betai,
            rhoix,
            params_sys["dALPHA"],
            params_sys["dSIGMA"],
            params_sys["dPPROB"],
            params_sys["dLAMB"],
            params_sys["dETA1"],
            params_sys["dETA2"],
            delta_t,
            observed=observed_data,
            logp=_dist_loglike_idiosyncratic,
        )
    return model


# ----------------------------------------------------------------------------
# Natural-space extraction
# ----------------------------------------------------------------------------
def natural_sample_matrix(idata, is_systematic):
    """Return posterior samples in NATURAL (economic) parameter space.

    Returns a DataFrame with columns matching the paper's Table 1 notation, one
    row per (chain, draw). Used to compute the posterior correlation matrix.
    """
    post = idata.posterior
    if is_systematic:
        cols = {
            "alpha":  post["alpha_rv"].values,    # exp(log(alpha_rv)) = alpha_rv
            "sigma":  post["sigma"].values,
            "pprob":  post["pprob_rv"].values,
            "lamb":   post["lamb"].values,
            "eta1":   post["eta1"].values,
            "eta2":   post["eta2"].values,
        }
    else:
        cols = {
            "mui":    post["mui"].values,
            "kappai": post["kappai_rv"].values,
            "gammai": post["gamma_rv"].values,
            "betai":  post["betai_rv"].values,
            # rhoix natural = tanh(arctanh(2 * rhoix_rv - 1)) = 2 * rhoix_rv - 1
            "rhoix":  2.0 * post["rhoix_rv"].values - 1.0,
        }
    flat = {k: v.reshape(-1) for k, v in cols.items()}
    return pd.DataFrame(flat)


# ----------------------------------------------------------------------------
# Diagnostic reporting
# ----------------------------------------------------------------------------
def report_summary(idata, var_names, label, out_path):
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.95)
    cols = ["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "ess_tail", "r_hat"]
    cols = [c for c in cols if c in summary.columns]
    print(f"\n=== {label} ===")
    print(summary[cols].round(4).to_string())
    summary.to_csv(out_path)
    return summary


def flag_convergence_issues(summary, label):
    issues = []
    for pname, row in summary.iterrows():
        ess_bulk = row.get("ess_bulk", np.nan)
        ess_tail = row.get("ess_tail", np.nan)
        r_hat = row.get("r_hat", np.nan)
        if ess_bulk < ESS_THRESHOLD or ess_tail < ESS_THRESHOLD:
            issues.append(
                f"  {pname}: ESS below {ESS_THRESHOLD} "
                f"(bulk={ess_bulk:.0f}, tail={ess_tail:.0f})"
            )
        if r_hat > RHAT_THRESHOLD:
            issues.append(f"  {pname}: R-hat above {RHAT_THRESHOLD} ({r_hat:.4f})")
    if issues:
        print(f"\n[!] Convergence warnings ({label}):")
        for line in issues:
            print(line)
    else:
        print(f"\n[OK] All chains converged in {label} (ESS >= {ESS_THRESHOLD}, R-hat <= {RHAT_THRESHOLD}).")


def report_correlation(nat_df, out_path):
    corr = nat_df.corr()
    print("\n=== Posterior correlation matrix (natural parameter space) ===")
    print(corr.round(3).to_string())
    corr.to_csv(out_path)

    # Flag high-correlation pairs.
    high_corr = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if abs(c) >= CORR_THRESHOLD:
                high_corr.append((cols[i], cols[j], c))
    if high_corr:
        print(f"\n[!] Highly correlated posterior pairs (|corr| >= {CORR_THRESHOLD}):")
        for a, b, c in high_corr:
            print(f"  {a} <-> {b}: {c:+.3f}  (candidates for reparameterization)")
    else:
        print(f"\n[OK] No posterior pairs exceed |corr| >= {CORR_THRESHOLD}.")


def save_trace_plots(idata, var_names, out_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        az.plot_trace(idata, var_names=var_names, compact=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_dir / "trace.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_pair_plots(idata, var_names, out_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        az.plot_pair(idata, var_names=var_names, kind="scatter", marginals=True)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_dir / "pairs.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--valuation-date", required=True, help="YYYYMMDD")
    ap.add_argument("--ticker", required=True, help="e.g. COIN, ^SPX")
    ap.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help="Number of trailing business days of returns to use.",
    )
    ap.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--out-dir",
        default="Study/Diagnostics",
        help="Root directory for diagnostic outputs.",
    )
    args = ap.parse_args()

    date = args.valuation_date
    ticker = args.ticker
    is_sys = ticker == SYSTEMATIC_TICKER

    lookback = args.lookback
    tickers = [ticker] if is_sys else [SYSTEMATIC_TICKER, ticker]
    price_ts = get_price_panel(tickers)
    return_ts = price_ts.pct_change().dropna()
    return_vec = (
        return_ts.loc[return_ts.index <= date, ticker]
        .iloc[-lookback:]
        .to_numpy()
    )
    if len(return_vec) < lookback:
        raise SystemExit(
            f"Insufficient history for {ticker} at {date}: "
            f"got {len(return_vec)} observations, need {lookback}."
        )

    if is_sys:
        model = build_systematic_model(return_vec, DELTA_T)
        sample_vars = SYS_SAMPLE_VARS
        natural_vars = SYS_NATURAL_VARS
    else:
        params_sys = get_pmle_params_dict(
            date,
            SYSTEMATIC_TICKER,
            params=["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"],
        )
        model = build_idiosyncratic_model(return_vec, params_sys, DELTA_T)
        sample_vars = IDI_SAMPLE_VARS
        natural_vars = IDI_NATURAL_VARS

    out_dir = Path(args.out_dir) / f"{date}_{ticker.replace('^', '')}_L{lookback}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {ticker} {date} (lookback={lookback}): {args.n_draws} draws x 4 chains...")
    with model:
        rng = np.random.default_rng(np.uint64(args.seed))
        idata = pm.sample(
            args.n_draws,
            chains=4,
            tune=1000,
            cores=4,
            target_accept=0.95,
            progressbar=True,
            random_seed=rng,
            nuts_sampler="nutpie",
        )

    # 1. Convergence diagnostics (sample-space).
    summary_sample = report_summary(
        idata, sample_vars, "Sample-space summary", out_dir / "summary_sample_space.csv"
    )
    flag_convergence_issues(summary_sample, "sample-space")

    # 2. Convergence diagnostics (natural / deterministic-space).
    summary_natural = report_summary(
        idata, natural_vars, "Natural-space summary (paper Table 1 notation)",
        out_dir / "summary_natural_space.csv",
    )
    flag_convergence_issues(summary_natural, "natural-space")

    # 3. Posterior correlation matrix (natural space) - identification diagnostic.
    nat_df = natural_sample_matrix(idata, is_sys)
    report_correlation(nat_df, out_dir / "posterior_corr_natural.csv")

    # 4. Trace plots.
    save_trace_plots(idata, sample_vars, out_dir)
    print(f"\nTrace plots saved to {out_dir / 'trace.png'}")

    # 5. Pair plots.
    save_pair_plots(idata, sample_vars, out_dir)
    print(f"Pair plots saved to {out_dir / 'pairs.png'}")

    # 6. Save full InferenceData for later re-analysis.
    idata.to_netcdf(out_dir / "idata.nc")
    print(f"InferenceData saved to {out_dir / 'idata.nc'}")

    print(f"\nAll diagnostics for {ticker} {date} saved to {out_dir}/")


if __name__ == "__main__":
    main()
