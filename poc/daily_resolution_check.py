"""Is a DAILY parameter series resolved, or is it mostly sampler noise?

A 252-day rolling window stepped one day at a time reuses 251 of its 252
observations, so the posterior barely moves between consecutive dates. The
question this answers is whether what DOES move is signal or Monte-Carlo error,
because a backfilled series sold as continuous cannot jitter for a reason that
has nothing to do with the market.

The comparison is:

    median |Delta|        day-over-day change in the posterior mean
    ------------------
    posterior sd          how wide the posterior is at a single date

against the sampler's noise floor, MCSE / sd ~= 1 / sqrt(ESS). At the
benchmark in estimate_systematic.py (1,000 draws over 4 chains, min tail ess
2,153) that floor is about 0.022 - roughly 2% of a posterior sd. If the
day-over-day ratio does not clear it by a comfortable margin, the daily series
is re-drawing the same posterior and needs more draws, not more dates.

Posterior sd is inferred from the stored 95% equal-tailed interval as
(upper - lower) / (2 * 1.96). That is a normal approximation and these
posteriors are visibly skewed - dLAMB especially - so read the ratio as an
order of magnitude, not a p-value. It is a screen: a comfortable pass means no
seed study is needed, a marginal result means run one.

MARKET HOLIDAYS. valuation_dates() uses pd.bdate_range with no exchange
calendar, so New Year's Day and every other weekday holiday is a valuation
date whose trailing window is IDENTICAL to the previous trading day's. SEED is
a fixed constant, so those fits are bit-identical and contribute exact zeros.
They are dropped here and counted; at --step 1 they are about one date in
twenty and would otherwise drag the median down.

    python poc/daily_resolution_check.py
    python poc/daily_resolution_check.py --beg 20090101 --end 20091231
    python poc/daily_resolution_check.py --drawer '^SPX__skewtight_913f43b7' --ess 26032
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import PMLE_DIR                        # noqa: E402

SYS_PARAMS = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]
IDIO_PARAMS = ["dMUI", "dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX"]
Z95 = 1.959963984540054


def load_drawer(drawer, beg=None, end=None):
    """Every per-date CSV in one drawer, as a frame indexed by valuation date."""
    folder = os.path.join(PMLE_DIR, drawer)
    if not os.path.isdir(folder):
        raise SystemExit("no such drawer: %s\n  (looked in %s)" % (drawer, PMLE_DIR))
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit("drawer %s holds no CSV files" % drawer)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dtVALUATION_DATE"] = pd.to_datetime(df["dtVALUATION_DATE"])
    df = df.sort_values("dtVALUATION_DATE").set_index("dtVALUATION_DATE")
    if beg:
        df = df[df.index >= pd.to_datetime(beg, format="%Y%m%d")]
    if end:
        df = df[df.index <= pd.to_datetime(end, format="%Y%m%d")]
    return df


def which_params(df):
    """Systematic drawers park the idiosyncratic columns at 0/0/1/1/0."""
    return [p for p in SYS_PARAMS + IDIO_PARAMS
            if p in df.columns and df[p].nunique() > 1]


def drop_frozen_windows(df, params):
    """Consecutive rows identical across every parameter are holiday repeats.

    Same 252-day window, same fixed SEED, so the fit is bit-identical. Keeping
    them would put exact zeros into the day-over-day distribution.
    """
    if len(df) < 2:
        return df, 0
    same = np.ones(len(df), dtype=bool)
    same[0] = False
    for p in params:
        v = df[p].to_numpy()
        same[1:] &= (v[1:] == v[:-1])
    return df[~same], int(same.sum())


def report(df, params, ess, gap_days):
    floor = 1.0 / np.sqrt(ess)
    print("  %-9s %12s %12s %10s %10s   %s"
          % ("param", "med |delta|", "post sd", "ratio", "floor", "verdict"))
    print("  " + "-" * 76)
    step = df.index.to_series().diff().dt.days.to_numpy()[1:]
    for p in params:
        v = df[p].to_numpy(dtype=float)
        d = np.abs(np.diff(v))[step <= gap_days]
        lo, hi = df.get(p + "_CI_LOWER"), df.get(p + "_CI_UPPER")
        if lo is None or hi is None:
            continue
        sd = float(np.median((hi.to_numpy(dtype=float)
                              - lo.to_numpy(dtype=float)) / (2 * Z95)))
        med = float(np.median(d)) if d.size else float("nan")
        ratio = med / sd if sd > 0 else float("nan")
        if ratio > 4 * floor:
            verdict = "resolved"
        elif ratio > 2 * floor:
            verdict = "marginal - run a seed study"
        else:
            verdict = "NOISE DOMINATED - more draws"
        print("  %-9s %12.6f %12.6f %10.4f %10.4f   %s"
              % (p, med, sd, ratio, floor, verdict))


def decompose(df, params, gap_days):
    """Split increment variance into sampler noise and real movement.

    Write x_t = mu_t + eps_t, with mu the true slowly-moving parameter and eps
    the sampler error at that date. Then

        Delta x_t = Delta mu_t + eps_t - eps_{t-1}

    and if the eps are independent across dates the lag-1 autocovariance of the
    increments is exactly -sigma_eps^2. That identifies the noise WITHOUT a
    seed study:

        sigma_eps^2  = -gamma_1
        var(Delta mu) = gamma_0 + 2 gamma_1        (<= 0 means none detectable)
        rho_1 = -0.5  is pure noise on a flat path; rho_1 -> 0 or positive
                      means the window is genuinely moving the posterior.

    The caveat is the independence assumption. SEED is one fixed constant and
    consecutive windows share 251 of 252 observations, so adjacent fits follow
    nearly the same sampler trajectory and their errors are positively
    correlated - which makes sigma_eps^2 here an estimate of the noise that
    SURVIVES differencing. That is the right quantity for this question: it is
    exactly the jitter a client would see in the series.
    """
    print("  %-9s %9s %11s %11s %11s %9s   %s"
          % ("param", "rho_1", "noise sd", "signal sd", "post sd",
             "noise/sd", "reading"))
    print("  " + "-" * 84)
    step = df.index.to_series().diff().dt.days.to_numpy()[1:]
    ok = step <= gap_days
    for p in params:
        v = df[p].to_numpy(dtype=float)
        d = np.diff(v)[ok]
        if d.size < 30:
            continue
        d = d - d.mean()
        g0 = float(np.mean(d * d))
        g1 = float(np.mean(d[1:] * d[:-1]))
        rho1 = g1 / g0 if g0 > 0 else float("nan")
        noise_var = max(-g1, 0.0)
        sig_var = g0 + 2 * g1
        lo, hi = df.get(p + "_CI_LOWER"), df.get(p + "_CI_UPPER")
        post_sd = float(np.median((hi.to_numpy(dtype=float)
                                   - lo.to_numpy(dtype=float)) / (2 * Z95)))
        noise_sd = np.sqrt(noise_var)
        sig_sd = np.sqrt(sig_var) if sig_var > 0 else 0.0
        if rho1 < -0.40:
            reading = "noise - flat path"
        elif rho1 < -0.20:
            reading = "mixed"
        else:
            reading = "signal dominates"
        print("  %-9s %9.3f %11.6f %11.6f %11.6f %9.4f   %s"
              % (p, rho1, noise_sd, sig_sd, post_sd,
                 noise_sd / post_sd if post_sd > 0 else float("nan"), reading))


def spans(df, params):
    """How far each parameter actually travelled over the sample."""
    print("  %-9s %12s %12s %12s   %s"
          % ("param", "min", "max", "range", "range / post sd"))
    print("  " + "-" * 70)
    for p in params:
        v = df[p].to_numpy(dtype=float)
        lo, hi = df.get(p + "_CI_LOWER"), df.get(p + "_CI_UPPER")
        post_sd = float(np.median((hi.to_numpy(dtype=float)
                                   - lo.to_numpy(dtype=float)) / (2 * Z95)))
        rng = float(v.max() - v.min())
        print("  %-9s %12.6f %12.6f %12.6f   %.2f"
              % (p, v.min(), v.max(), rng, rng / post_sd if post_sd else 0))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drawer", default="^SPX__skewtight_913f43b7")
    ap.add_argument("--beg", default=None, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--ess", type=float, default=2153.0,
                    help="min tail ess for the draw count used (1000 draws -> "
                         "2153, 10000 -> 26032; see estimate_systematic.py)")
    ap.add_argument("--max-gap", type=int, default=4,
                    help="ignore differences across gaps longer than this many "
                         "calendar days, so a monthly drawer is not mistaken "
                         "for a daily one")
    a = ap.parse_args()

    df = load_drawer(a.drawer, a.beg, a.end)
    params = which_params(df)
    if not params:
        raise SystemExit("no varying parameter columns in %s" % a.drawer)

    df, frozen = drop_frozen_windows(df, params)
    gaps = df.index.to_series().diff().dt.days.dropna()
    consecutive = int((gaps <= a.max_gap).sum())

    print()
    print("=" * 78)
    print("daily resolution check :: %s" % a.drawer)
    print("=" * 78)
    print("  dates            : %d   %s -> %s"
          % (len(df), df.index[0].date(), df.index[-1].date()))
    print("  median step      : %.0f calendar day(s)" % gaps.median())
    print("  consecutive pairs: %d (steps <= %d days)" % (consecutive, a.max_gap))
    print("  frozen windows   : %d dropped (holiday repeats - identical window"
          % frozen)
    print("                     and fixed SEED, so the fit is bit-identical)")
    print("  assumed tail ess : %.0f  ->  noise floor %.4f sd"
          % (a.ess, 1.0 / np.sqrt(a.ess)))
    print()

    if consecutive < 20:
        print("  This drawer is not daily - only %d consecutive pair(s). The"
              % consecutive)
        print("  comparison below is between dates a month apart, which is not")
        print("  the question. Point --drawer at a --step 1 run.")
        print()

    report(df, params, a.ess, a.max_gap)
    print()
    print("  ratio = median day-over-day move / posterior sd at one date.")
    print("  floor = MCSE / sd ~= 1/sqrt(ess): movement the sampler invents.")
    print("  Posterior sd is inferred from the 95%% interval assuming normality;")
    print("  dLAMB and the etas are skewed, so treat this as a screen.")
    print()
    print("  -- increment decomposition (no seed study needed) --")
    print()
    decompose(df, params, a.max_gap)
    print()
    print("  -- how far each parameter actually travelled --")
    print()
    spans(df, params)
    print()
    print("  A one-step increment near the noise floor does NOT mean the series")
    print("  is noise: a slow signal accumulates while independent noise does")
    print("  not. Read rho_1 and the range together, not med |delta| alone.")
    print()


if __name__ == "__main__":
    main()
