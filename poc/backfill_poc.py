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

# Systematic liquidity proxy. Fixed to SPX, matching the paper.
#
# The proxy stands for aggregate LIQUIDITY, not the equity market's return. SPX
# is chosen because it is liquid enough that its own moves are not distorted by
# its own illiquidity, broad enough that no issuer's idiosyncratic risk leaks
# in, and it is the instrument participants actually reach for when de-risking -
# which is where liquidity demand concentrates in a crisis.
#
# Alternatives (NDX, RUT, RSP, and SPY vs the index) are a deferred robustness
# test, not an open question. The selection criterion is already defined: the
# best proxy is the one leaving idiosyncratic residuals uncorrelated across
# names and free of regime-dependent heteroscedasticity - the same test as the
# FRTB MAR33.16(2) idiosyncrasy demonstration. Note this panel is all S&P 500
# large caps, so RUT is a mismatch by construction and would lose for reasons
# that do not generalise.
INDEX = "^SPX"

# Crisis windows to reconstruct and score, each paired with the estimation
# windows that may be used for it.
#
# RULE: an estimation window must never overlap the crisis it reconstructs.
# That is the leakage control, and it is asserted at run time below.
#
# Each window brackets the crisis - calm run-up, stress, partial recovery - so
# the reconstruction is scored across regimes rather than on the crash alone.
#
# GFC is primary: under FRTB the stress period is searched back to 2007, and
# for most equity portfolios 2008-09 still binds. COVID is the second window
# because it is a different KIND of liquidity event (violent, exogenous,
# policy-truncated, V-shaped) and because its estimation gaps are short enough
# to match the real production case - a 2021 listing backfilled to 2020.
CRISES = {
    "GFC":   ("2007-01-01", "2009-12-31"),
    "COVID": ("2019-06-01", "2021-06-30"),
}

# Estimation windows per crisis. The gap to the crisis is the experiment: it is
# the regime-transportability question (spec §3), answered as a by-product.
#
# Do NOT label these "calm" or "stressed". There is no genuinely calm three-year
# window in the last fifteen years - 2011 euro, 2013 taper, 2015 China, 2018 Q1
# and Q4, 2020 COVID, 2022 bear, 2023 SVB, 2025 tariffs. Every window below
# contains stress of some kind.
#
# Stress content is therefore MEASURED, not asserted: window_stress() below
# reports jump-day count and jump mass per window from the Stage 1 filter, and
# those land in the results as covariates. Reconstruction quality can then be
# related to (gap length, estimation-window stress) quantitatively instead of
# resting on hand-labels.
EST_WINDOWS = {
    "GFC": {
        "A_2011_2013": ("2011-01-01", "2013-12-31"),   # ~2y gap, has 2011 euro stress
        "B_2013_2015": ("2013-01-01", "2015-12-31"),   # ~4y gap, calm
        "C_2019_2021": ("2019-01-01", "2021-12-31"),   # ~11y gap, contains COVID
        "D_2022_2024": ("2022-01-01", "2024-12-31"),   # ~14y gap, the 2021-IPO analogue
    },
    "COVID": {
        "E_2022_2024": ("2022-01-01", "2024-12-31"),   # ~2y after: the real production case
        "F_2016_2018": ("2016-01-01", "2018-12-31"),   # ~2y before: tests symmetry
        "G_2013_2015": ("2013-01-01", "2015-12-31"),   # ~5y before
    },
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

# Rolling estimation. A single 1-year window often has too few jump days to
# identify gamma. Rolling a 1-year window through the estimation period keeps
# the regulatory 1-year lookback convention while pooling information across
# many window positions.
ROLL_WINDOW_DAYS = 252          # the regulatory lookback
ROLL_STEP_DAYS = 21             # monthly steps

# Block bootstrap over the rolling estimates.
#
# CRITICAL: rolling 1-year windows stepped monthly share 11/12 of their data.
# The sequence of estimates is therefore heavily autocorrelated and an i.i.d.
# bootstrap would understate the standard error by roughly sqrt(12), producing
# a falsely tight parameter distribution. Blocks must span at least one full
# estimation window so that resampled blocks are approximately independent.
BOOT_BLOCK = ROLL_WINDOW_DAYS // ROLL_STEP_DAYS      # 12 positions = 1 window
N_BOOT = 200                    # parameter vector draws
N_PATHS_PER_DRAW = 100          # idiosyncratic paths per draw

# Minimum jump days in a rolling window for gamma to be identified there.
MIN_JUMP_DAYS_FOR_GAMMA = 5
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

    # Jump days: r ~ gamma * Y_j
    #
    # gamma is identified ONLY off jump days, and jump days are rare - at a 3-sd
    # threshold, perhaps 5-15 a year. A short or calm estimation window can
    # leave too few to identify it. The fallback below sets gamma = 1.0, which
    # silently converts the central parameter of this study into an assumption,
    # so it is flagged rather than hidden: gamma_identified is carried into the
    # results and any row with gamma_identified == False must be excluded from
    # the headline or reported separately.
    #
    # This constraint, not kappa or beta, is what drives estimation window
    # length. A one-year window matching the paper's regulatory lookback will
    # often fail to identify gamma.
    n_jump = int(j.sum())
    if n_jump >= MIN_JUMP_DAYS_FOR_GAMMA:
        Xj = m[j].values
        Yj = r[j].values
        gamma = float(Xj @ Yj / (Xj @ Xj))
        gamma_identified = True
    else:
        gamma = 1.0
        gamma_identified = False

    return {
        "sigma_beta": sigma_beta,
        "kappa": kappa,
        "gamma": gamma,
        "rho_resid_dZ": rho_proxy,
        "gamma_identified": gamma_identified,
        "n_calm": int(calm.sum()),
        "n_jump": int(j.sum()),
    }


# ----------------------------------------------------------------------------
# Step 3 - reconstruct the crisis window
# ----------------------------------------------------------------------------

def reconstruct(draws, diffusive, marks, crisis, seed=SEED):
    """Crisis-period log-return ensemble under BOTH sources of uncertainty.

    Systematic diffusion and jump realisations are held at their observed
    historical values - only the name's loadings and its idiosyncratic
    innovations vary.

    Returns (pooled_returns, es_by_draw):
      pooled_returns : all simulated returns, for distributional scoring
      es_by_draw     : one 97.5% ES per parameter draw, so the spread across
                       draws IS the parameter-uncertainty assessment that FRTB
                       Principle seven and SR 26-2 both ask for.
    """
    a, b = crisis
    dz = diffusive.loc[a:b].values
    yj = marks.loc[a:b].values
    T = len(dz)
    rng = np.random.default_rng(seed)

    pooled = []
    es_by_draw = np.empty(len(draws))
    for i, (sigma_beta, kappa, gamma) in enumerate(draws):
        systematic = sigma_beta * dz + gamma * yj
        idio = rng.normal(0.0, kappa, size=(N_PATHS_PER_DRAW, T))
        paths = systematic[None, :] + idio
        es_by_draw[i] = var_es(paths.ravel(), 0.975)[1]
        pooled.append(paths)
    return np.concatenate(pooled, axis=0), es_by_draw



# ----------------------------------------------------------------------------
# Step 4 - scoring
# ----------------------------------------------------------------------------

def window_stress(marks, is_jump, window):
    """Measured stress content of an estimation window, from the filter output.

    Reported rather than assumed, because no recent three-year window is calm
    and a hand-label would smuggle the analyst's prior into the result.
    """
    a, b = window
    m = marks.loc[a:b]
    j = is_jump.loc[a:b]
    n_days = len(m)
    return {
        "win_jump_days": int(j.sum()),
        "win_jump_per_yr": round(float(j.sum()) / max(n_days / 252.0, 1e-9), 1),
        "win_jump_mass": round(float(m.abs().sum()), 4),
        "win_max_jump_pct": round(float(m.abs().max()) * 100, 1) if j.any() else 0.0,
    }


def estimate_rolling(name_ret, diffusive, marks, is_jump, window):
    """Parameter vectors from a 1-year window rolled through `window`.

    Returns a DataFrame with one row per window position. Only data inside
    `window` is touched - the leakage control is unchanged.

    Rolling solves the identification problem: any single calm year may hold
    too few jump days to pin gamma, but the union of positions across several
    years will contain enough, while each individual estimate still respects
    the 1-year regulatory lookback.
    """
    a, b = window
    r = name_ret.loc[a:b].dropna()
    if len(r) < ROLL_WINDOW_DAYS + ROLL_STEP_DAYS:
        raise ValueError("estimation window shorter than one rolling window")

    rows = []
    for end in range(ROLL_WINDOW_DAYS, len(r) + 1, ROLL_STEP_DAYS):
        sub = r.iloc[end - ROLL_WINDOW_DAYS:end]
        d = diffusive.reindex(sub.index)
        m = marks.reindex(sub.index)
        j = is_jump.reindex(sub.index).fillna(False)
        calm = ~j
        if calm.sum() < 60:
            continue

        X = d[calm].values
        Y = sub[calm].values
        if X @ X <= 0:
            continue
        sigma_beta = float(X @ Y / (X @ X))
        resid = Y - sigma_beta * X
        kappa = float(resid.std(ddof=1))

        n_jump = int(j.sum())
        if n_jump >= MIN_JUMP_DAYS_FOR_GAMMA and float(m[j].values @ m[j].values) > 0:
            Xj, Yj = m[j].values, sub[j].values
            gamma = float(Xj @ Yj / (Xj @ Xj))
            gid = True
        else:
            gamma = np.nan          # left missing, not defaulted to 1.0
            gid = False

        rows.append({"asof": sub.index[-1], "sigma_beta": sigma_beta,
                     "kappa": kappa, "gamma": gamma, "gamma_identified": gid,
                     "n_jump": n_jump})

    if not rows:
        raise ValueError("no usable rolling positions")
    return pd.DataFrame(rows).set_index("asof")


def block_bootstrap_params(est, n_boot=N_BOOT, block=BOOT_BLOCK, seed=SEED):
    """Moving-block bootstrap over the sequence of rolling estimates.

    Resamples the parameter VECTOR jointly, never each parameter separately:
    sigma_beta and kappa come from the same regression and are negatively
    correlated (more variance explained systematically leaves less residual),
    so independent resampling would break that dependence and double-count.

    Blocks of `block` consecutive positions preserve the autocorrelation that
    overlapping windows induce. Returns an array of shape (n_boot, 3) holding
    (sigma_beta, kappa, gamma) draws.
    """
    cols = ["sigma_beta", "kappa", "gamma"]
    arr = est[cols].to_numpy(dtype=float)
    n = len(arr)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)

    draws = np.empty((n_boot, len(cols)))
    n_blocks = int(np.ceil(n / block))
    for b in range(n_boot):
        starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
        idx = np.concatenate([np.arange(st, min(st + block, n)) for st in starts])[:n]
        sample = arr[idx]
        draws[b] = np.nanmean(sample, axis=0)     # gamma may be NaN in some rows
    return draws


def _assert_no_overlap(window, crisis, wlabel, clabel):
    """Leakage control: an estimation window may not touch the crisis it
    reconstructs. Cheap to check, fatal if violated, and easy to break when
    editing the windows above."""
    w0, w1 = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    c0, c1 = pd.Timestamp(crisis[0]), pd.Timestamp(crisis[1])
    if w0 <= c1 and c0 <= w1:
        raise ValueError(
            "LEAKAGE: estimation window %s (%s..%s) overlaps crisis %s (%s..%s)"
            % (wlabel, window[0], window[1], clabel, crisis[0], crisis[1]))


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

    print("Measured stress content of each estimation window")
    print("(no window is calm - this is why it is measured, not labelled)\n")
    print("  %-6s %-12s %-24s %5s %8s %9s %8s"
          % ("crisis", "window", "range", "jumps", "per_yr", "mass", "max%"))
    stress = {}
    for crisis_label, CRISIS in CRISES.items():
        for wlabel, window in EST_WINDOWS[crisis_label].items():
            _assert_no_overlap(window, CRISIS, wlabel, crisis_label)
            st = window_stress(marks, is_jump, window)
            stress[(crisis_label, wlabel)] = st
            print("  %-6s %-12s %s..%s %5d %8.1f %9.4f %8.1f"
                  % (crisis_label, wlabel, window[0], window[1],
                     st["win_jump_days"], st["win_jump_per_yr"],
                     st["win_jump_mass"], st["win_max_jump_pct"]))
    print()

    rows = []
    for crisis_label, CRISIS in CRISES.items():
        for wlabel, window in EST_WINDOWS[crisis_label].items():
          for ticker, etf in NAMES.items():
            try:
                est = estimate_rolling(rets[ticker], diffusive, marks,
                                       is_jump, window)
                draws = block_bootstrap_params(est)
                ens, es_draws = reconstruct(draws, diffusive, marks, CRISIS)
                actual = rets[ticker].loc[CRISIS[0]:CRISIS[1]].dropna()
                ens = ens[:, :len(actual)]

                # Point estimates and their bootstrap spread. Conservatism is
                # defined on the OUTPUT (ES), never by taking an upper quantile
                # of each input: the paper shows VaR rises with rho when
                # rho >= 0 but FALLS with beta when rho < 0, so "upper quantile
                # of every parameter" is not reliably conservative.
                p = {
                    "sigma_beta": float(np.nanmean(draws[:, 0])),
                    "kappa": float(np.nanmean(draws[:, 1])),
                    "gamma": float(np.nanmean(draws[:, 2])),
                    "gamma_sd_boot": float(np.nanstd(draws[:, 2])),
                    "n_roll_positions": int(len(est)),
                    "pct_roll_gamma_ident": round(
                        100.0 * float(est["gamma_identified"].mean()), 1),
                    "gamma_identified": bool(est["gamma_identified"].any()),
                    "es975_recon_mean": float(np.mean(es_draws)),
                    "es975_recon_p025": float(np.quantile(es_draws, 0.025)),
                    "es975_recon_p975": float(np.quantile(es_draws, 0.975)),
                }

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
                s.update({"crisis": crisis_label, "window": wlabel,
                          "ticker": ticker, "etf": etf,
                          **stress[(crisis_label, wlabel)], **p})
                rows.append(s)
            except Exception as e:                      # noqa: BLE001
                print("skip %s / %s / %s: %s" % (crisis_label, ticker, wlabel, e))

    res = pd.DataFrame(rows)
    res.to_csv("poc_results.csv", index=False)

    print("\nTier 2 headline - mean |ES(97.5%) error| by crisis and estimation window:")
    for c in ("recon", "etf", "betaidx"):
        res[c + "_abs"] = res[c + "_es975_err_pct"].abs()
    summary = (res.groupby(["crisis", "window"])[["recon_abs", "etf_abs", "betaidx_abs"]]
                  .mean().round(1))
    print(summary.to_string())
    print("\nSame, excluding financials (XLF) - the sector whose 2008 moves were")
    print("driven by solvency rather than market-wide liquidity:")
    print(res[res.etf != "XLF"].groupby(["crisis", "window"])
             [["recon_abs", "etf_abs", "betaidx_abs"]].mean().round(1).to_string())
    print("\nThe claim is recon_abs < etf_abs and recon_abs < betaidx_abs, with the")
    print("margin widening in gamma, and holding across BOTH crises. Degradation")
    print("across estimation windows is the transportability result.")
    n_unident = int((res["pct_roll_gamma_ident"] < 50).sum())
    if n_unident:
        print("\nWARNING: gamma identified in fewer than half the rolling positions")
        print("for %d of %d runs. Those estimates rest on very few jump days -"
              % (n_unident, len(res)))
        print("report them separately from the headline.")
        print(res[res["pct_roll_gamma_ident"] < 50]
              .groupby(["crisis", "window"]).size().to_string())
    else:
        print("\ngamma identified in a majority of rolling positions for all %d runs."
              % len(res))

    res["es975_ci_width_pct"] = (100.0 * (res.es975_recon_p975 - res.es975_recon_p025)
                                 / res.es975_recon_mean)
    print("\nParameter uncertainty - width of the bootstrap 95% interval on the")
    print("reconstructed ES, as % of the mean. This is the Principle seven /")
    print("SR 26-2 'assessment of uncertainty in the final outcome':")
    print(res.groupby(["crisis", "window"])["es975_ci_width_pct"]
             .describe()[["mean", "50%", "max"]].round(1).to_string())

    print("\nFull detail written to poc_results.csv")


if __name__ == "__main__":
    main()
