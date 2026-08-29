"""
Spot-return backfill proof of concept.

Reconstructs crisis-period relative returns for names treated as if they had no
history before a chosen post-crisis date, and scores the reconstruction against
the actual series and against two incumbent proxies.

Model (Yi & Kim, Theorem 3.1), physical measure:

    dS_i,t / S_i,t- = m dt + kappa_i dW_i,t + sigma*beta_i dZ_t
                      + d( sum_j ( exp(gamma_i * Y_j) - 1 ) )

Reconstruction principle: the systematic diffusion (dZ) and the common jump
realisations (N_t, Y_j) are HISTORICAL FACT recovered from SPX and are held
fixed. Only the idiosyncratic diffusion (dW_i) is simulated. See spec §2.

Estimation here is a transparent two-regime regression, NOT the paper's full
likelihood. It is adequate for a PoC and must be labelled as such in any
write-up; substitute the PyMC estimator for production.

INPUT: the price snapshots written by poc/fetch_prices.py, loaded through
Library.DataAccess so the study uses the same source as the paper.

    python poc/fetch_prices.py      # once, to freeze the snapshots
    python poc/backfill_poc.py      # from the repo root
"""

import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.DataAccess import get_price_series  # noqa: E402

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

INDEX = "^SPX"

# Crisis window to reconstruct and score.
CRISIS = ("2007-01-01", "2009-12-31")

# Estimation windows. Variant D is the realistic 2021-IPO analogue (spec §3).
EST_WINDOWS = {
    "A_2011_2013": ("2011-01-01", "2013-12-31"),
    "B_2013_2015": ("2013-01-01", "2015-12-31"),
    "C_2019_2021": ("2019-01-01", "2021-12-31"),
    "D_2022_2024": ("2022-01-01", "2024-12-31"),
}

# Names to test, each mapped to its sector-ETF benchmark (spec §4, §5).
# Stratify across liquidity beta and include negative controls.
# Ticker -> sector ETF benchmark. Selection is ex-ante (sector and size), NOT
# on 2008 behaviour, which would be selection leakage. Bucket by post-crisis
# estimated liquidity beta after estimation.
#
# GE and BAC are the negative controls: both had large firm-specific events in
# 2008 (GE Capital; the Countrywide and Merrill acquisitions plus capital
# raises). The model is expected to do poorly on them. Report that.
NAMES = {
    # Technology
    "MSFT": "XLK", "ORCL": "XLK", "INTC": "XLK", "AAPL": "XLK", "NVDA": "XLK",
    # Financials
    "JPM":  "XLF", "GS":   "XLF", "BAC":  "XLF", "C":    "XLF", "WFC":  "XLF",
    # Consumer staples
    "PG":   "XLP", "KO":   "XLP", "WMT":  "XLP", "PEP":  "XLP", "CL":   "XLP",
    # Energy
    "XOM":  "XLE", "CVX":  "XLE", "COP":  "XLE", "SLB":  "XLE", "OXY":  "XLE",
    # Industrials
    "CAT":  "XLI", "GE":   "XLI", "BA":   "XLI", "HON":  "XLI", "UNP":  "XLI",
    # Health care
    "JNJ":  "XLV", "PFE":  "XLV", "MRK":  "XLV", "ABT":  "XLV", "UNH":  "XLV",
    # Consumer discretionary
    "HD":   "XLY", "SBUX": "XLY", "AMZN": "XLY", "MCD":  "XLY", "NKE":  "XLY",
    # Utilities
    "SO":   "XLU", "D":    "XLU", "AEP":  "XLU", "XEL":  "XLU", "ED":   "XLU",
}

N_PATHS = 10_000
JUMP_THRESHOLD_SD = 3.0   # sensitivity to this is a required robustness check
SEED = 20260829


# ----------------------------------------------------------------------------
# Step 1 - systematic factor from the index
# ----------------------------------------------------------------------------

def load_panel(symbols):
    """Adjusted closes for `symbols` as one DataFrame.

    Deliberately NOT Library.DataAccess.get_price_panel, which back-fills.
    Back-filling would invent prices before a security's first trade - exactly
    the fabricated history this study exists to avoid. Forward-fill only, so a
    gap is carried across a holiday but a pre-listing period stays NaN.
    """
    frames = []
    for sym in symbols:
        s = get_price_series(sym)
        s.index = pd.to_datetime(s.index)
        frames.append(s.rename(sym))
    panel = pd.concat(frames, axis=1).sort_index()
    return panel.ffill()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def filter_systematic(index_ret: pd.Series, threshold_sd: float = JUMP_THRESHOLD_SD):
    """Split index log returns into a diffusive part and common jump marks.

    Simple threshold identification. A bipower-variation or particle-filter
    approach is the production version; report sensitivity to the threshold
    either way (spec §7, and doc 03 known weaknesses).
    """
    r = index_ret.dropna()
    scale = r.rolling(250, min_periods=60).std().bfill()
    is_jump = r.abs() > threshold_sd * scale

    marks = r.where(is_jump, 0.0)              # Y_j on jump days, 0 otherwise
    diffusive = r.where(~is_jump, 0.0)         # dZ proxy on non-jump days

    sigma = diffusive[~is_jump].std()
    lam = is_jump.mean() * 252
    pos = marks[marks > 0]
    neg = marks[marks < 0]
    params = {
        "sigma_daily": float(sigma),
        "lambda_annual": float(lam),
        "p_up": float(len(pos) / max(len(pos) + len(neg), 1)),
        "eta1": float(1.0 / pos.mean()) if len(pos) else np.nan,
        "eta2": float(1.0 / abs(neg.mean())) if len(neg) else np.nan,
    }
    return diffusive, marks, is_jump, params


def sanity_check(marks: pd.Series) -> pd.DataFrame:
    """Face-validity gate (spec §7). Jump mass must cluster on known episodes."""
    episodes = {
        "GFC 2008Q4": ("2008-09-01", "2008-12-31"),
        "Euro 2011": ("2011-08-01", "2011-10-31"),
        "Aug 2015": ("2015-08-01", "2015-09-30"),
        "Feb 2018": ("2018-02-01", "2018-02-28"),
        "Covid 2020": ("2020-02-20", "2020-04-30"),
        "Apr 2025": ("2025-04-01", "2025-04-30"),
    }
    total = marks.abs().sum()
    rows = []
    for label, (a, b) in episodes.items():
        w = marks.loc[a:b]
        rows.append({
            "episode": label,
            "jump_days": int((w != 0).sum()),
            "abs_mark_share_pct": round(100 * w.abs().sum() / total, 2) if total else np.nan,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Step 2 - per-name parameters, estimation window only (no leakage)
# ----------------------------------------------------------------------------

def estimate_name(name_ret, diffusive, marks, is_jump, window):
    """Two-regime regression. Returns sigma*beta, kappa, gamma.

    Only data inside `window` is touched - this is the leakage control.
    """
    a, b = window
    r = name_ret.loc[a:b].dropna()
    d = diffusive.reindex(r.index)
    m = marks.reindex(r.index)
    j = is_jump.reindex(r.index).fillna(False)

    calm = ~j
    if calm.sum() < 60:
        raise ValueError("insufficient non-jump observations in estimation window")

    # Non-jump days: r ~ sigma_beta * dZ + kappa * eps
    X = d[calm].values
    Y = r[calm].values
    sigma_beta = float(X @ Y / (X @ X))
    resid = Y - sigma_beta * X
    kappa = float(resid.std(ddof=1))
    rho_proxy = float(np.corrcoef(resid, X)[0, 1]) if X.std() > 0 else np.nan

    # Jump days: r ~ gamma * Y_j  (falls back to 1.0 if too few observations)
    if j.sum() >= 5:
        Xj = m[j].values
        Yj = r[j].values
        gamma = float(Xj @ Yj / (Xj @ Xj))
    else:
        gamma = 1.0

    return {
        "sigma_beta": sigma_beta,
        "kappa": kappa,
        "gamma": gamma,
        "rho_resid_dZ": rho_proxy,
        "n_calm": int(calm.sum()),
        "n_jump": int(j.sum()),
    }


# ----------------------------------------------------------------------------
# Step 3 - reconstruct the crisis window
# ----------------------------------------------------------------------------

def reconstruct(params, diffusive, marks, crisis, n_paths=N_PATHS, seed=SEED):
    """Ensemble of crisis-period log-return paths.

    Systematic diffusion and jump marks are held at their realised historical
    values; only the idiosyncratic component is drawn.
    """
    a, b = crisis
    dz = diffusive.loc[a:b].values
    yj = marks.loc[a:b].values
    T = len(dz)

    systematic = params["sigma_beta"] * dz + params["gamma"] * yj   # shape (T,)
    rng = np.random.default_rng(seed)
    idio = rng.normal(0.0, params["kappa"], size=(n_paths, T))
    return systematic[None, :] + idio                                # (n_paths, T)


# ----------------------------------------------------------------------------
# Step 4 - scoring
# ----------------------------------------------------------------------------

def var_es(returns, q=0.99):
    """Loss-side VaR and ES at level q, from a 1-D array of log returns."""
    losses = -np.asarray(returns).ravel()
    var = np.quantile(losses, q)
    tail = losses[losses >= var]
    es = tail.mean() if tail.size else var
    return float(var), float(es)


def score(actual, ensemble, benchmarks: dict):
    """Tier 1 and Tier 2 metrics (spec §6)."""
    actual = np.asarray(actual)
    pooled = ensemble.ravel()

    # Tier 1 - PIT ranks of the realised path within the ensemble, per day
    pit = (ensemble < actual[None, :]).mean(axis=0)

    out = {
        "pit_mean": float(pit.mean()),          # ~0.5 if unbiased
        "pit_std": float(pit.std()),            # ~0.289 if uniform
        "actual_vol_ann": float(actual.std() * np.sqrt(252)),
        "recon_vol_ann": float(pooled.std() * np.sqrt(252)),
        "actual_skew": float(pd.Series(actual).skew()),
        "recon_skew": float(pd.Series(pooled).skew()),
        "actual_kurt": float(pd.Series(actual).kurtosis()),
        "recon_kurt": float(pd.Series(pooled).kurtosis()),
        "actual_maxdd": float(np.exp(np.minimum.accumulate(
            np.cumsum(actual) - np.maximum.accumulate(np.cumsum(actual)))).min() - 1),
        "exceed_4sd": int((np.abs(actual) > 4 * pooled.std()).sum()),
    }

    # Tier 2 - the headline. Error in the risk measure vs each construction.
    a_var, a_es = var_es(actual)
    r_var, r_es = var_es(pooled)
    out["actual_var99"] = a_var
    out["actual_es975"] = var_es(actual, 0.975)[1]
    out["recon_var99_err_pct"] = 100 * (r_var - a_var) / a_var
    out["recon_es975_err_pct"] = 100 * (var_es(pooled, 0.975)[1] - out["actual_es975"]) / out["actual_es975"]

    for label, series in benchmarks.items():
        b_var, _ = var_es(series)
        b_es = var_es(series, 0.975)[1]
        out[f"{label}_var99_err_pct"] = 100 * (b_var - a_var) / a_var
        out[f"{label}_es975_err_pct"] = 100 * (b_es - out["actual_es975"]) / out["actual_es975"]

    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main():
    symbols = [INDEX] + sorted(set(NAMES) | set(NAMES.values()))
    prices = load_panel(symbols)
    rets = log_returns(prices)

    diffusive, marks, is_jump, sys_params = filter_systematic(rets[INDEX])
    print("\nSystematic parameters (from %s):" % INDEX)
    for k, v in sys_params.items():
        print("  %-16s %.6f" % (k, v))
    print("\nFace-validity gate - jump mass by episode:")
    print(sanity_check(marks).to_string(index=False))
    print("\nIf the GFC and Covid rows are not dominant, stop here. The filter is wrong.\n")

    if not NAMES:
        print("Populate NAMES with {ticker: sector_etf} and re-run.")
        return

    rows = []
    for wlabel, window in EST_WINDOWS.items():
        for ticker, etf in NAMES.items():
            try:
                p = estimate_name(rets[ticker], diffusive, marks, is_jump, window)
                ens = reconstruct(p, diffusive, marks, CRISIS)
                actual = rets[ticker].loc[CRISIS[0]:CRISIS[1]].dropna()
                ens = ens[:, :len(actual)]

                # Benchmark 1: sector ETF proxy (the regulatory fallback).
                # Benchmark 2: beta-scaled index, beta from the same window.
                idx_c = rets[INDEX].loc[CRISIS[0]:CRISIS[1]].reindex(actual.index)
                nm_w = rets[ticker].loc[window[0]:window[1]]
                ix_w = rets[INDEX].reindex(nm_w.index)
                ok = nm_w.notna() & ix_w.notna()
                beta = float(np.polyfit(ix_w[ok], nm_w[ok], 1)[0])
                bench = {
                    "etf": rets[etf].loc[CRISIS[0]:CRISIS[1]].reindex(actual.index).dropna().values,
                    "betaidx": (beta * idx_c).dropna().values,
                }

                s = score(actual.values, ens, bench)
                s.update({"window": wlabel, "ticker": ticker, **p})
                rows.append(s)
            except Exception as e:                      # noqa: BLE001
                print("skip %s / %s: %s" % (ticker, wlabel, e))

    res = pd.DataFrame(rows)
    res.to_csv("poc_results.csv", index=False)

    print("\nTier 2 headline - ES(97.5%) error vs actual, by construction:")
    cols = ["window", "ticker", "gamma", "sigma_beta",
            "recon_es975_err_pct", "etf_es975_err_pct", "betaidx_es975_err_pct"]
    print(res[cols].round(2).to_string(index=False))
    print("\nThe claim is |recon| < |etf| and |recon| < |betaidx|, "
          "with the margin widening in gamma. Written to poc_results.csv")


if __name__ == "__main__":
    main()
