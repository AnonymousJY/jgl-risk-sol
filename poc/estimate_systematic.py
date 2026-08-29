"""
Step 1 :: robust SPX P-measure parameters, 2007-01-01 -> 2026-08-31.

Uses the repository's own P-MLE estimator - Library.RiskEngineKimYi2025's
pmle_kimyirisk_systematic, the MCMC estimator behind the paper - NOT a
threshold heuristic. It returns all six common parameters with credible
intervals:

    dALPHA  OU mean reversion of the latent liquidity process
    dSIGMA  diffusive volatility of that process
    dLAMB   jump intensity
    dPPROB  probability a jump is upward
    dETA1   upward jump decay (mean up jump = 1/eta1)
    dETA2   downward jump decay

Each valuation date uses a 252-day lookback, matching the paper and the
regulatory convention. Running across many dates yields a TIME SERIES of
parameter estimates - which is exactly the rolling input the block bootstrap
in backfill_poc.py consumes.

    python poc/estimate_systematic.py                 # default monthly step
    python poc/estimate_systematic.py --step 5        # weekly
    python poc/estimate_systematic.py --verify        # April 2025 vs the paper

Incremental: dates already written to Study/Estimated Parameters PMLE/ are
skipped, so the run can be interrupted and resumed.

COST WARNING. This is MCMC, not a closed form. One date takes seconds to
minutes. Daily over 2007-2026 is ~4,900 dates and is not sensible as a first
run. Start with --step 21 (monthly, ~235 dates), confirm the series is stable,
then decide whether finer stepping changes anything.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from concurrent.futures import ProcessPoolExecutor          # noqa: E402

from Library.PosteriorSummary import (                       # noqa: E402
    CI_WIDTH_TO_SD, CI_CONVENTION, CI_PROB,
)
from Library.DataAccess import (                            # noqa: E402
    get_price_panel, get_pmle_params, pmle_params_exists,
    save_pmle_params, available_pmle_dates,
)
from Scripts.run_pmle_kimyi2025 import (                    # noqa: E402
    pmle_kimyirisk_systematic_helper, assemble_systematic_params,
    SYSTEMATIC_PARAMS,
)

SYSTEMATIC_ID = "^SPX"
DATE_FMT = "%Y%m%d"

BEG = "20070101"
END = "20260831"
LOOKBACK = 252
BASE_DAYS = 252
SEED = np.uint64(20240114)
N_MC_PATHS = 10_000

# The paper's reported April 2025 ranges. Reproducing these end to end is the
# strongest available check that this pipeline is wired correctly.
PAPER_APRIL_2025 = {
    "dALPHA": (0.67, 0.70),
    "dSIGMA": (0.13, 0.17),
    "dLAMB":  (9.7, 12.1),
    "dETA1":  (45.0, 55.0),      # paper says "near 50"
    "dETA2":  (25.0, 28.0),      # paper says "26-27"
}


def run_full_sample(beg=BEG, end=END):
    """ONE fit on the entire sample, not a 252-day window.

    This is the cheap test that should be run before anything clever. A 252-day
    window holds ~12 jumps; 2007-2026 holds ~230. If that is enough to move
    eta1 and eta2 off their priors, the identification problem is solved with
    returns alone - no options, no Q-to-P mapping, no tenor question.

    Writes under the pseudo-date FULLSAMPLE so it does not collide with the
    rolling estimates.
    """
    price_ts = get_price_panel([SYSTEMATIC_ID])
    return_ts = price_ts.pct_change().dropna()
    rv = return_ts.loc[
        (return_ts.index >= pd.to_datetime(beg, format=DATE_FMT))
        & (return_ts.index <= pd.to_datetime(end, format=DATE_FMT)),
        SYSTEMATIC_ID].to_numpy()

    print("  full-sample fit on %d daily returns (%s to %s)"
          % (len(rv), beg, end))
    print("  this is one MCMC fit - expect minutes, not seconds\n")

    t0 = time.perf_counter()
    _, _, results = pmle_kimyirisk_systematic_helper(
        ("FULLSAMPLE", rv, np.array(1 / BASE_DAYS), SEED, N_MC_PATHS,
         SYSTEMATIC_ID))
    print("  done in %.0fs\n" % (time.perf_counter() - t0))

    params = assemble_systematic_params(results)
    save_pmle_params("FULLSAMPLE", SYSTEMATIC_ID, params)

    # the only question that matters: did the posterior move off the prior?
    from poc.prior_diagnostics import PRIORS, HDI_TO_SD          # noqa: E402
    print("  %-8s %-14s %9s %9s %9s %7s  %s"
          % ("param", "prior", "pri_mean", "post_mean", "post_sd", "ratio",
             "verdict"))
    print("  " + "-" * 74)
    for k, (label, pmean, psd) in PRIORS.items():
        if k not in params:
            continue
        m, lo, hi = params[k]
        post_sd = (hi - lo) * HDI_TO_SD
        ratio = post_sd / psd
        v = ("PRIOR-DRIVEN" if ratio > 0.90 else
             "weak" if ratio > 0.70 else
             "partial" if ratio > 0.35 else "DATA-DRIVEN")
        print("  %-8s %-14s %9.3f %9.3f %9.3f %7.2f  %s"
              % (k, label, pmean, m, post_sd, ratio, v))
    print("\n  If dETA1 and dETA2 are now DATA-DRIVEN, the identification")
    print("  problem is solved from returns alone and nothing further is")
    print("  needed. If they are still PRIOR-DRIVEN with ~230 jumps, then and")
    print("  only then is the option-implied route worth the complexity.")


def valuation_dates(beg, end, step):
    days = pd.bdate_range(pd.to_datetime(beg, format=DATE_FMT),
                          pd.to_datetime(end, format=DATE_FMT))
    return [d.strftime(DATE_FMT) for d in days[::step]]


def run(dates):
    """Estimate the systematic parameters for each date not already on disk."""
    price_ts = get_price_panel([SYSTEMATIC_ID])
    return_ts = price_ts.pct_change().dropna()

    first_ok = return_ts.index[LOOKBACK - 1]
    usable, too_early = [], []
    for dt in dates:
        (usable if pd.to_datetime(dt, format=DATE_FMT) >= first_ok
         else too_early).append(dt)
    if too_early:
        print("  %d dates dropped: fewer than %d prior returns available."
              % (len(too_early), LOOKBACK))
        print("  Earliest estimable date is %s. To reach back to %s, the price"
              % (first_ok.date(), BEG))
        print("  snapshot must start ~%d business days earlier." % LOOKBACK)

    todo = [d for d in usable if not pmle_params_exists(d, SYSTEMATIC_ID)]
    print("  %d dates requested, %d usable, %d already on disk, %d to estimate."
          % (len(dates), len(usable), len(usable) - len(todo), len(todo)))
    if not todo:
        return

    args = []
    for dt in todo:
        rv = (return_ts.loc[return_ts.index <= dt, SYSTEMATIC_ID]
              .iloc[-LOOKBACK:].to_numpy())
        args.append((dt, rv, np.array(1 / BASE_DAYS), SEED, N_MC_PATHS,
                     SYSTEMATIC_ID))

    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor() as ex:
        for dt, sid, results in ex.map(pmle_kimyirisk_systematic_helper, args):
            save_pmle_params(dt, sid, assemble_systematic_params(results))
            done += 1
            if done % 10 == 0 or done == len(todo):
                el = time.perf_counter() - t0
                print("    %4d/%d  %.1fs elapsed, ~%.1fs remaining"
                      % (done, len(todo), el, el / done * (len(todo) - done)))


def load_series():
    """All estimated systematic parameters as a DataFrame indexed by date."""
    dates = available_pmle_dates(SYSTEMATIC_ID)
    rows = []
    for dt in dates:
        s = get_pmle_params(dt, SYSTEMATIC_ID)
        row = {"date": pd.to_datetime(dt, format=DATE_FMT)}
        for k in SYSTEMATIC_PARAMS:
            if k in s:
                row[k] = float(s[k])
            if k + "_CI_LOWER" in s and k + "_CI_UPPER" in s:
                row[k + "_W"] = float(s[k + "_CI_UPPER"]) - float(s[k + "_CI_LOWER"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("date").sort_index()


def report(df):
    print("\n" + "=" * 72)
    print("SPX P-measure parameters :: %s to %s   (%d valuation dates)"
          % (df.index.min().date(), df.index.max().date(), len(df)))
    print("=" * 72)
    print("\nDistribution across valuation dates:")
    print(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
            .round(4).to_string())

    print("\nStability - coefficient of variation (sd / |mean|):")
    cv = (df.std() / df.mean().abs()).sort_values()
    for k, v in cv.items():
        note = "" if v < 0.25 else "   <-- unstable"
        print("   %-8s %6.3f%s" % (k, v, note))
    print("\n   A well-identified parameter should not swing wildly across")
    print("   adjacent 1-year windows. High CV means the estimate is absorbing")
    print("   sample-specific noise, not structure.")

    print("\nBy year (median):")
    print(df.groupby(df.index.year).median().round(4).to_string())

    # prior sds, to judge identification date by date
    try:
        from poc.prior_diagnostics import PRIORS
        prior_sd = {k: v[2] for k, v in PRIORS.items()}
    except Exception:                                            # noqa: BLE001
        prior_sd = {}

    if prior_sd:
        print("\nIdentification over time - share of valuation dates where the")
        print("posterior is narrower than the prior (ratio < 0.70):")
        for k in SYSTEMATIC_PARAMS:
            w = k + "_W"
            if k not in prior_sd or w not in df:
                continue
            ratio = (df[w] * CI_WIDTH_TO_SD) / prior_sd[k]
            share = 100.0 * float((ratio < 0.70).mean())
            print("   %-8s %5.1f%%   median ratio %.2f   (min %.2f, max %.2f)"
                  % (k, share, ratio.median(), ratio.min(), ratio.max()))
        print("\n   0%% means the parameter is never identified at any date in")
        print("   the sample - the rolling series is a series of priors.")

    print("\nJump intensity dLAMB at known stress episodes (should spike):")
    for label, (a, b) in {
        "GFC 2008H2":  ("2008-07-01", "2008-12-31"),
        "Euro 2011":   ("2011-07-01", "2011-12-31"),
        "Covid 2020":  ("2020-02-01", "2020-06-30"),
        "SVB 2023":    ("2023-03-01", "2023-06-30"),
        "Tariffs 2025": ("2025-04-01", "2025-07-31"),
    }.items():
        w = df.loc[a:b, "dLAMB"] if "dLAMB" in df else pd.Series(dtype=float)
        if len(w):
            print("   %-14s median %.2f   (full-sample median %.2f)"
                  % (label, w.median(), df["dLAMB"].median()))


def verify(df):
    """Check the April 2025 estimates against the values reported in the paper."""
    w = df.loc["2025-04-01":"2025-04-30"]
    print("\n" + "=" * 72)
    print("VERIFICATION :: April 2025 against the paper's reported ranges")
    print("=" * 72)
    if not len(w):
        print("  No April 2025 valuation dates estimated yet. Run with a step")
        print("  that lands in that window before relying on this check.")
        return
    print("  %d valuation dates in April 2025\n" % len(w))
    for k, (lo, hi) in PAPER_APRIL_2025.items():
        if k not in w:
            continue
        obs_lo, obs_hi = w[k].min(), w[k].max()
        ok = (obs_hi >= lo) and (obs_lo <= hi)
        print("   %-8s paper [%6.2f, %6.2f]   here [%6.2f, %6.2f]   %s"
              % (k, lo, hi, obs_lo, obs_hi, "OK" if ok else "*** MISMATCH ***"))
    print("\n  A mismatch means this pipeline is not reproducing the published")
    print("  estimator. Fix that before trusting any parameter in this run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=21,
                    help="business days between valuation dates (21 = monthly)")
    ap.add_argument("--beg", default=BEG)
    ap.add_argument("--end", default=END)
    ap.add_argument("--verify", action="store_true",
                    help="also check April 2025 against the paper")
    ap.add_argument("--report-only", action="store_true",
                    help="skip estimation, just summarise what is on disk")
    ap.add_argument("--full-sample", action="store_true",
                    help="ONE fit on the whole sample. Run this first - it is "
                         "the cheap test of whether more jumps fixes eta1/eta2.")
    a = ap.parse_args()

    print("=" * 72)
    print("Step 1 :: SPX P-measure parameters via the repository's P-MLE")
    print("=" * 72)
    print("  %s -> %s, every %d business days, %d-day lookback"
          % (a.beg, a.end, a.step, LOOKBACK))
    print("  credible intervals: %.0f%% equal-tailed (%s)" % (100 * CI_PROB, CI_CONVENTION))

    if a.full_sample:
        run_full_sample(a.beg, a.end)
        return

    if not a.report_only:
        run(valuation_dates(a.beg, a.end, a.step))

    df = load_series()
    if not len(df):
        print("\nNothing estimated yet.")
        return
    report(df)
    if a.verify:
        verify(df)

    out = os.path.join(_REPO_ROOT, "poc", "systematic_params.csv")
    df.to_csv(out)
    print("\nWritten to %s" % out)
    print("This series is the rolling systematic input to backfill_poc.py.")


if __name__ == "__main__":
    main()
