"""
mcmc_empirical_bayes.py - P-3 identification via empirical-Bayes rolling priors.

For a single (valuation date, ticker) idiosyncratic P-MLE calibration, this
script builds informative priors from the previous N days' cached point
estimates and re-runs the MCMC. The historical rolling window's estimates
serve as prior information under an assumption of day-to-day parameter
continuity, which is exactly the "stability across rolling re-calibrations"
already documented in the paper (Section 5.1, page 27).

Concretely, for each idiosyncratic parameter we compute the mean and standard
deviation of the previous N days' posterior means (in the same transformed
space the sampler uses), then set the new prior as
Normal(historical mean, scale x historical sd). The scale factor controls
how much room the new day's data has to update against the prior;
scale = 1 uses the historical variability directly, scale = 2 doubles
it to be more permissive.

Compare the resulting posterior sds against the diagnostics from
Scripts/mcmc_diagnostics.py to see the tightening produced by P-3.

Run from the repository root:

    PYTHONPATH=. python Scripts/mcmc_empirical_bayes.py \
        --valuation-date 20250409 --ticker COIN --hist-days 5 --prior-scale 1.0

Outputs are written to ``Study/EmpiricalBayes/{YYYYMMDD}_{TICKER}/``.
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Library.DataAccess import (  # noqa: E402
    get_price_panel,
    get_pmle_params,
    get_pmle_params_dict,
)
from Library.RiskEngineKimYi2025 import _dist_loglike_idiosyncratic  # noqa: E402


SYSTEMATIC_TICKER = "^SPX"
DEFAULT_LOOKBACK = 252
DELTA_T = np.array(1.0 / 252.0)
DEFAULT_SEED = 20240114
DEFAULT_N_DRAWS = 10_000
DEFAULT_HIST_DAYS = 5
DEFAULT_PRIOR_SCALE = 1.0

IDI_SAMPLE_VARS = ["mui", "kappai_log", "gamma_log", "betai_log", "rhoix_atanh"]
IDI_NATURAL_VARS = ["mui", "kappai", "gammai", "betai", "rhoix"]


def collect_historical_estimates(ticker, valuation_date, hist_days):
    """Return a DataFrame of the previous ``hist_days`` cached point estimates.

    Reads ``estimated_params_pmle_{TICKER}_{YYYYMMDD}.csv`` files from
    ``Study/Estimated Parameters PMLE/{TICKER}/`` in reverse chronological
    order, stopping before ``valuation_date``.
    """
    cache_dir = _REPO_ROOT / "Study" / "Estimated Parameters PMLE" / ticker
    if not cache_dir.is_dir():
        raise SystemExit(f"No cache directory: {cache_dir}")

    all_dates = sorted(
        [p.stem.split("_")[-1] for p in cache_dir.glob("*.csv")]
    )
    prior_dates = [d for d in all_dates if d < valuation_date][-hist_days:]
    if len(prior_dates) < 2:
        raise SystemExit(
            f"Need at least 2 cached prior dates before {valuation_date}, "
            f"found {len(prior_dates)}: {prior_dates}"
        )

    rows = [get_pmle_params(d, ticker) for d in prior_dates]
    hist = pd.DataFrame(rows, index=prior_dates)
    return hist


def build_prior_stats(hist_df):
    """Compute prior mean and sd for each idiosyncratic parameter in the
    same transformed space the sampler uses.

    Returns a dict keyed by sample-variable name with (mean, sd) tuples.
    """
    stats = {}
    # mui: natural space (Normal support).
    stats["mui"] = (hist_df["dMUI"].mean(), hist_df["dMUI"].std(ddof=1))

    # kappai, gammai, betai: sampler uses log-space (via pt.log(*_rv)).
    for nat_col, key in [
        ("dKAPPAI", "kappai_log"),
        ("dGAMMAI", "gamma_log"),
        ("dBETAI", "betai_log"),
    ]:
        log_vals = np.log(hist_df[nat_col].clip(lower=1e-6))
        stats[key] = (log_vals.mean(), log_vals.std(ddof=1))

    # rhoix: sampler uses arctanh space.
    atanh_vals = np.arctanh(hist_df["dRHOIX"].clip(lower=-0.999, upper=0.999))
    stats["rhoix_atanh"] = (atanh_vals.mean(), atanh_vals.std(ddof=1))

    return stats


def build_idiosyncratic_model_eb(idi_returns, params_sys, delta_t, prior_stats, prior_scale):
    """Idiosyncratic MCMC model with empirical-Bayes priors on the sample-space
    variables. Preserves the deterministic transformations to the natural
    (economic) parameter space so the likelihood interface is unchanged.
    """
    mui_mean, mui_sd = prior_stats["mui"]
    kappai_log_mean, kappai_log_sd = prior_stats["kappai_log"]
    gamma_log_mean, gamma_log_sd = prior_stats["gamma_log"]
    betai_log_mean, betai_log_sd = prior_stats["betai_log"]
    rhoix_atanh_mean, rhoix_atanh_sd = prior_stats["rhoix_atanh"]

    # Prior scale factor: keep sd floor to avoid zero-width priors when
    # historical estimates are degenerate.
    sd_floor = 1e-3
    def scaled(sd):
        return max(sd * prior_scale, sd_floor)

    with pm.Model() as model:
        mui = pm.Normal(name="mui", mu=mui_mean, sigma=scaled(mui_sd))

        # Sample log-parameters directly on the real line, then exp inside
        # the likelihood via the existing interface.
        kappai = pm.Normal(name="kappai_log", mu=kappai_log_mean, sigma=scaled(kappai_log_sd))
        gammai = pm.Normal(name="gamma_log", mu=gamma_log_mean, sigma=scaled(gamma_log_sd))
        betai = pm.Normal(name="betai_log", mu=betai_log_mean, sigma=scaled(betai_log_sd))

        # rhoix in arctanh space (real line).
        rhoix = pm.Normal(name="rhoix_atanh", mu=rhoix_atanh_mean, sigma=scaled(rhoix_atanh_sd))

        # Deterministic natural-space quantities for reporting.
        pm.Deterministic("kappai", pt.exp(kappai))
        pm.Deterministic("gammai", pt.exp(gammai))
        pm.Deterministic("betai", pt.exp(betai))
        pm.Deterministic("rhoix", pt.tanh(rhoix))

        observed_data = np.cumsum(idi_returns).reshape((-1, 1))

        pm.CustomDist(
            "likelihood",
            mui,
            kappai,   # log-space, likelihood exps internally
            gammai,
            betai,
            rhoix,    # arctanh-space, likelihood tanhs internally
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


def report_summary(idata, var_names, label, out_path):
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.95)
    cols = ["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "ess_tail", "r_hat"]
    cols = [c for c in cols if c in summary.columns]
    print(f"\n=== {label} ===")
    print(summary[cols].round(4).to_string())
    summary.to_csv(out_path)
    return summary


def flag_convergence(summary, label, ess_threshold=400, rhat_threshold=1.01):
    issues = []
    for pname, row in summary.iterrows():
        ess_bulk = row.get("ess_bulk", np.nan)
        ess_tail = row.get("ess_tail", np.nan)
        r_hat = row.get("r_hat", np.nan)
        if ess_bulk < ess_threshold or ess_tail < ess_threshold:
            issues.append(
                f"  {pname}: ESS below {ess_threshold} (bulk={ess_bulk:.0f}, tail={ess_tail:.0f})"
            )
        if r_hat > rhat_threshold:
            issues.append(f"  {pname}: R-hat above {rhat_threshold} ({r_hat:.4f})")
    if issues:
        print(f"\n[!] Convergence warnings ({label}):")
        for line in issues:
            print(line)
    else:
        print(f"\n[OK] Chains converged in {label}.")


def natural_sample_matrix(idata):
    post = idata.posterior
    cols = {
        "mui":    post["mui"].values,
        "kappai": post["kappai"].values,   # deterministic exp(log-sample)
        "gammai": post["gammai"].values,
        "betai":  post["betai"].values,
        "rhoix":  post["rhoix"].values,    # deterministic tanh(atanh-sample)
    }
    flat = {k: v.reshape(-1) for k, v in cols.items()}
    return pd.DataFrame(flat)


def print_prior_vs_posterior(prior_stats, hist_df, summary_natural, out_path):
    """Side-by-side comparison of historical stats, empirical-Bayes prior,
    and posterior for each natural-space parameter.
    """
    rows = []
    # Historical (natural space)
    for nat_col, name in [
        ("dMUI", "mui"),
        ("dKAPPAI", "kappai"),
        ("dGAMMAI", "gammai"),
        ("dBETAI", "betai"),
        ("dRHOIX", "rhoix"),
    ]:
        hist_mean = hist_df[nat_col].mean()
        hist_sd = hist_df[nat_col].std(ddof=1)
        post_row = summary_natural.loc[name]
        rows.append({
            "param": name,
            "hist_mean": hist_mean,
            "hist_sd": hist_sd,
            "post_mean": post_row["mean"],
            "post_sd": post_row["sd"],
            "post_ci_lo": post_row["hdi_2.5%"],
            "post_ci_hi": post_row["hdi_97.5%"],
        })
    df = pd.DataFrame(rows).set_index("param")
    print("\n=== Historical stats vs empirical-Bayes posterior (natural space) ===")
    print(df.round(4).to_string())
    df.to_csv(out_path)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--valuation-date", required=True, help="YYYYMMDD")
    ap.add_argument("--ticker", required=True, help="Idiosyncratic ticker, e.g. COIN")
    ap.add_argument(
        "--hist-days",
        type=int,
        default=DEFAULT_HIST_DAYS,
        help="Number of trailing cached dates to use for building priors.",
    )
    ap.add_argument(
        "--prior-scale",
        type=float,
        default=DEFAULT_PRIOR_SCALE,
        help="Multiplier on historical sd when setting prior sd (1.0 = tight, "
             "2.0 = permissive, higher = more like the default weakly-informative prior).",
    )
    ap.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help="Number of trailing business days of returns for the likelihood.",
    )
    ap.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default="Study/EmpiricalBayes")
    args = ap.parse_args()

    date = args.valuation_date
    ticker = args.ticker
    if ticker == SYSTEMATIC_TICKER:
        raise SystemExit("This script targets idiosyncratic assets only.")

    # Historical stats
    hist_df = collect_historical_estimates(ticker, date, args.hist_days)
    print(f"Historical dates used for priors: {list(hist_df.index)}")
    print("Natural-space historical means:")
    print(hist_df[["dMUI", "dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX"]].round(4).to_string())

    prior_stats = build_prior_stats(hist_df)
    print("\nEmpirical-Bayes priors (sample-space):")
    for k, (m, s) in prior_stats.items():
        print(f"  {k}: Normal(mu={m:.4f}, sigma={s * args.prior_scale:.4f})")

    # Data
    price_ts = get_price_panel([SYSTEMATIC_TICKER, ticker])
    return_ts = price_ts.pct_change().dropna()
    return_vec = (
        return_ts.loc[return_ts.index <= date, ticker]
        .iloc[-args.lookback :]
        .to_numpy()
    )
    if len(return_vec) < args.lookback:
        raise SystemExit(
            f"Insufficient history for {ticker} at {date}: "
            f"got {len(return_vec)}, need {args.lookback}."
        )

    # Systematic conditioning (from existing cache; unchanged)
    params_sys = get_pmle_params_dict(
        date,
        SYSTEMATIC_TICKER,
        params=["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"],
    )

    model = build_idiosyncratic_model_eb(
        return_vec, params_sys, DELTA_T, prior_stats, args.prior_scale
    )

    out_dir = (
        Path(args.out_dir)
        / f"{date}_{ticker}_H{args.hist_days}_S{args.prior_scale:g}_L{args.lookback}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSampling {ticker} {date} (empirical-Bayes, H={args.hist_days}, "
          f"scale={args.prior_scale}, L={args.lookback}): {args.n_draws} draws x 4 chains...")
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

    # Diagnostics
    summary_sample = report_summary(
        idata, IDI_SAMPLE_VARS, "Sample-space summary (empirical-Bayes)",
        out_dir / "summary_sample_space.csv",
    )
    flag_convergence(summary_sample, "sample-space")

    summary_natural = report_summary(
        idata, IDI_NATURAL_VARS, "Natural-space summary (empirical-Bayes)",
        out_dir / "summary_natural_space.csv",
    )
    flag_convergence(summary_natural, "natural-space")

    # Prior vs posterior comparison
    print_prior_vs_posterior(
        prior_stats, hist_df, summary_natural,
        out_dir / "prior_vs_posterior.csv",
    )

    # Correlation matrix in natural space
    nat_df = natural_sample_matrix(idata)
    corr = nat_df.corr()
    print("\n=== Posterior correlation matrix (natural space, empirical-Bayes) ===")
    print(corr.round(3).to_string())
    corr.to_csv(out_dir / "posterior_corr_natural.csv")

    # Plots
    save_trace_plots(idata, IDI_SAMPLE_VARS, out_dir)
    save_pair_plots(idata, IDI_SAMPLE_VARS, out_dir)

    idata.to_netcdf(out_dir / "idata.nc")
    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
