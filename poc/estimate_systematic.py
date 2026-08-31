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
# forkserver, not fork. fork() in a process that has already started
# threads is unsafe; Python 3.12+ warns and 3.14 changes the Linux
# default for this reason. PyMC with a numba backend does start
# threads, and the failure mode is a hang that looks like slow
# sampling. Set JGL_MP_START to override.
multiprocessing.set_start_method(
    os.environ.get("JGL_MP_START", "forkserver"), force=True)
from concurrent.futures import (                           # noqa: E402
    ProcessPoolExecutor, as_completed, BrokenProcessPool,
)

from Library.PosteriorSummary import (                       # noqa: E402
    CI_WIDTH_TO_SD, CI_CONVENTION, CI_PROB,
)
from Library.TableHeatmap import (                          # noqa: E402
    render as heat, legend as heat_legend,
)

# Heat shading: on for a terminal, off when piped or NO_COLOR is set.
# --no-color / --color override.
COLOR = None
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


def _init_child():
    """Stop each worker's native libraries from spawning a full thread pool.

    nutpie samples 4 chains as THREADS inside one worker process, and numpy /
    BLAS / numba will each independently try to claim every core on top of
    that. With W workers the machine sees W x 4 chain threads x C BLAS threads.
    Pinning the inner libraries to one thread leaves the chains as the only
    source of parallelism, which is what the worker count was sized against.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


def default_workers():
    """Outer-pool width.

    Each fit calls pm.sample(chains=4, cores=4), so it already claims 4 cores.
    ProcessPoolExecutor() with no max_workers defaults to os.cpu_count(), which
    on a 16-core box means 16 workers x 4 chains = 64 sampling processes
    competing for 16 cores. That oversubscription is usually the single largest
    avoidable cost in this run - far larger than anything a GPU would recover.

    Default to cpu_count // 4 so total demand matches the machine.
    """
    return max(1, (os.cpu_count() or 4) // 4)


def run(dates, workers=None):
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

    workers = workers or default_workers()
    print("  %d workers x 4 chains = %d concurrent samplers on %d cores"
          % (workers, workers * 4, os.cpu_count() or 0))

    t0 = time.perf_counter()
    done = failed = 0
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_child) as ex:
            futures = {ex.submit(pmle_kimyirisk_systematic_helper, a): a[0]
                       for a in args}
            for fut in as_completed(futures):
                requested = futures[fut]
                try:
                    dt, sid, results = fut.result()
                except BrokenProcessPool:
                    raise
                except Exception as exc:                      # noqa: BLE001
                    # One bad date must not end a run of thousands. Record it
                    # and carry on; the date simply stays absent from disk and
                    # a later rerun will retry it.
                    failed += 1
                    print("    FAILED  %s  %s: %s"
                          % (requested, type(exc).__name__, exc))
                    continue
                save_pmle_params(dt, sid, assemble_systematic_params(results))
                done += 1
                if done % 10 == 0 or done == len(todo):
                    el = time.perf_counter() - t0
                    print("    %4d/%d  %.1fs elapsed, ~%.1fs remaining"
                          % (done, len(todo), el,
                             el / done * (len(todo) - done)))
    except BrokenProcessPool:
        # A worker died without raising - it was killed by a signal, not by a
        # Python exception. Overwhelmingly this is the kernel OOM killer:
        # every worker holds a compiled model, N_MC_PATHS paths and four
        # chains of draws, so peak memory scales with the worker count while
        # the estimate of "how many cores" does not.
        el = time.perf_counter() - t0
        print("\n" + "=" * 72)
        print("  WORKER KILLED - the pool is dead and the run stopped early.")
        print("=" * 72)
        print("  %d of %d dates completed and ARE SAFELY ON DISK (%.0fs)."
              % (done, len(todo), el))
        print("  Nothing is lost: rerunning skips what is already written.")
        print()
        print("  A worker terminated without a Python exception, which means")
        print("  it was killed by a signal rather than failing in Python.")
        print("  Confirm the cause before rerunning:")
        print()
        print("      dmesg -T | grep -i -E 'killed process|out of memory' | tail")
        print()
        print("  If that shows an OOM kill, rerun with fewer workers - memory")
        print("  scales with worker count, cores do not:")
        print()
        print("      ./run_daily.sh %d" % max(1, workers // 2))
        print()
        print("  If it shows nothing, suspect a native crash in the sampler.")
        print("  Reproduce one date in the foreground to get a real traceback:")
        print()
        print("      python poc/estimate_systematic.py --full-sample")
        print("=" * 72)
        return

    if failed:
        print("\n  %d date(s) failed and were skipped; rerun to retry them."
              % failed)


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
    _p = [c for c in df.columns if not c.endswith("_W")]
    print(df[_p].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
            .round(4).to_string())

    params_only = [c for c in df.columns if not c.endswith("_W")]
    if len(df) < 2:
        print("\nStability - needs at least 2 valuation dates; %d estimated."
              % len(df))
    else:
        print("\nStability - coefficient of variation (sd / |mean|):")
        cv = (df[params_only].std() / df[params_only].mean().abs()).sort_values()
        for k, v in cv.items():
            note = "" if v < 0.25 else "   <-- varies a lot across windows"
            print("   %-8s %6.3f%s" % (k, v, note))
        print("\n   Read this WITH the identification table below, not alone. A")
        print("   low CV means the estimate barely moves - which is evidence of")
        print("   identification only if the parameter is actually identified.")
        print("   A prior-driven parameter is stable because its prior is.")

    # By year, mean and median, same layout. Both are shown because they answer
    # different questions on ~12 overlapping 252-day windows per year: the mean
    # gives the level, the median resists a single divergent fit. Where they
    # diverge the year is skewed - in crisis years that is expected, since the
    # crisis moves in and out of the trailing window across the twelve fits.
    for stat in ("mean", "median"):
        t = getattr(df.groupby(df.index.year), stat)().round(4)
        t.index = [str(i) for i in t.index]
        print("\nBy year (%s):" % stat)
        print(heat(t, decimals=4, color=COLOR))
    print(heat_legend(color=COLOR))


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
        print("\n   0% means the parameter is never identified at any date in the")
        print("   sample - that rolling series is a series of priors, not estimates.")
        print("   A ratio at or above 1.00 means the posterior is no narrower than")
        print("   the prior: the likelihood is flat in that direction.")

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
    ap.add_argument("--color", dest="color", action="store_true", default=None,
                    help="force heat shading on (default: on for a terminal)")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="plain numbers, no shading")
    ap.add_argument("--workers", type=int, default=None,
                    help="outer pool width. Default cpu_count//4, because each "
                         "fit already uses 4 chains on 4 cores.")
    ap.add_argument("--full-sample", action="store_true",
                    help="ONE fit on the whole sample. Run this first - it is "
                         "the cheap test of whether more jumps fixes eta1/eta2.")
    a = ap.parse_args()

    global COLOR
    COLOR = a.color

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
        run(valuation_dates(a.beg, a.end, a.step), workers=a.workers)

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
