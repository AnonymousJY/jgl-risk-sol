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
import hashlib
import json
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
    ProcessPoolExecutor, as_completed,
)
# BrokenProcessPool is NOT re-exported by concurrent.futures - its __all__
# carries the generic BrokenExecutor only. It lives in the .process submodule.
from concurrent.futures.process import BrokenProcessPool     # noqa: E402

from Library.RiskEngineKimYi2025 import (                     # noqa: E402
    SYSTEMATIC_PRIORS_RECENTRED, SYSTEMATIC_PRIORS_CAPPED,
    SYSTEMATIC_PRIORS_CAPPED_BETA, SYSTEMATIC_PRIORS_GAPS,
    SYSTEMATIC_PRIORS_ASYM, SYSTEMATIC_PRIORS_SKEW, FULL_SAMPLE,
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
    save_pmle_params, available_pmle_dates, PMLE_DIR,
)
from Scripts.run_pmle_kimyi2025 import (                    # noqa: E402
    pmle_kimyirisk_systematic_helper, assemble_systematic_params,
    SYSTEMATIC_PARAMS,
)

SYSTEMATIC_ID = "^SPX"
# Full-sample fits are stored under their own underlying id so they cannot
# collide with, or be mistaken for, a rolling estimate.
FULL_SAMPLE_ID = "^SPX_FULLSAMPLE"

# Where the ESTIMATES are stored, as opposed to where the PRICES come from.
#
# These have to be different, and the reason is a trap that has already cost a
# run. run() skips any date already on disk, which is what makes a 5,000-date
# job interruptible. But "already on disk" was keyed on the underlying alone,
# so changing a prior and rerunning skipped every date and reprinted the OLD
# posteriors under the NEW prior's header - a silent, entirely plausible-looking
# wrong answer. Prices are the same ^SPX series whatever the prior; estimates
# are not, and must not share a drawer.
#
# "paper" keeps the bare id so the committed replication files under
# Study/Estimated Parameters PMLE/^SPX/ stay exactly where Scripts/ expects.
STORE_SUFFIX = {"paper": "", "recentred": "__recentred", "capped": "__capped",
                "capped-beta": "__cappedbeta", "gaps": "__gaps",
                "asym": "__asym", "skew": "__skew"}

# Set by main() from --priors, alongside PRIORS_IN_FORCE.
STORE_ID = SYSTEMATIC_ID
PRIORS_TAG = "paper"


def priors_digest(priors):
    """An 8-hex fingerprint of a RESOLVED prior specification.

    Naming the drawer after --priors is not enough, and the reason is the
    mistake this function exists to prevent. The tag "gaps" names a variable,
    not a value: recentring alpha inside SYSTEMATIC_PRIORS_RECENTRED changes
    every spec that inherits from it - gaps, capped, capped-beta - while every
    tag stays the same. A rerun then finds 245 dates "already on disk", skips
    all of them, and reprints posteriors fitted under the OLD alpha prior under
    a header naming the NEW one.

    Hashing the numbers themselves removes the judgement call. Any edit to any
    prior in the spec produces a different drawer automatically, so a stale
    result cannot be silently reused no matter how the spec was reached. The
    converse matters just as much: an unchanged spec keeps its digest, so an
    interrupted run still resumes.
    """
    if priors is None:
        payload = "paper-defaults"           # the engine's published literals
    else:
        payload = json.dumps(
            {k: [kind, {kk: float(vv) for kk, vv in sorted(kw.items())}]
             for k, (kind, kw) in sorted(priors.items())},
            sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def store_id(tag, priors):
    """Where estimates fitted under (tag, priors) are stored.

    "paper" keeps the bare id: those literals are frozen by publication and the
    committed replication files under Study/Estimated Parameters PMLE/^SPX/ are
    addressed by it. Everything else is tag + digest, so the name says which
    arm it is and the digest says which VERSION of that arm.
    """
    if tag == "paper":
        return SYSTEMATIC_ID
    return "%s%s_%s" % (SYSTEMATIC_ID, STORE_SUFFIX[tag], priors_digest(priors))


def artifact_suffix(tag, priors):
    """Filename suffix for the loose poc/ artefacts, matching the drawer.

    systematic_params.csv and full_sample_params.json have to carry the digest
    too. Tagging them by name alone would let an edited spec overwrite the
    assembled series of the run it was compared against - the same failure as
    the drawer, on the file you actually read.
    """
    if tag == "paper":
        return ""
    return "%s_%s" % (STORE_SUFFIX[tag], priors_digest(priors))


def write_manifest(drawer_id, tag, priors):
    """Drop a _priors.json beside the estimates so the drawer is self-describing.

    A digest tells you two runs differ. It does not tell you how. Without this
    file, ^SPX__gaps_3f9a1c2b six months from now is an unreadable hash over a
    spec that has since been edited, and the estimates in it are unusable
    because nobody can say what produced them.
    """
    folder = os.path.join(PMLE_DIR, drawer_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "_priors.json")
    spec = ({k: [kind, kw] for k, (kind, kw) in sorted(priors.items())}
            if priors is not None else "engine defaults (SYSTEMATIC_PRIORS)")
    with open(path, "w") as fh:
        json.dump({"tag": tag,
                   "digest": priors_digest(priors),
                   "lookback": LOOKBACK,
                   "base_days": BASE_DAYS,
                   "seed": int(SEED),
                   "n_mc_paths": N_MC_PATHS,
                   "priors": spec}, fh, indent=2)
    return path
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
         SYSTEMATIC_ID, PRIORS_IN_FORCE))
    elapsed = time.perf_counter() - t0
    print("  done in %.0fs\n" % elapsed)

    params = assemble_systematic_params(results)

    # Dump to a plain file FIRST, before anything that can fail. This fit costs
    # ~8 minutes and an earlier version lost one entirely to a serialisation
    # error raised after the sampler had finished. Nothing expensive should be
    # destroyed by a failure in how it gets written down.
    raw = os.path.join(
        _REPO_ROOT, "poc",
        "full_sample_params%s.json" % artifact_suffix(PRIORS_TAG, PRIORS_IN_FORCE))
    with open(raw, "w") as fh:
        json.dump({"beg": beg, "end": end, "n_returns": int(len(rv)),
                   "seconds": round(elapsed, 1),
                   "params": {k: [float(x) for x in v]
                              for k, v in params.items()}}, fh, indent=2)
    print("  raw results written to %s" % raw)

    # The parameter store keys on a parseable date, so the pseudo-date
    # "FULLSAMPLE" raised DateParseError in pd.to_datetime. Key on the real end
    # date instead, under a DISTINCT underlying id so the row cannot collide
    # with the rolling estimate for that date, and so load_series() - which
    # reads SYSTEMATIC_ID - never picks a full-sample fit up as a rolling one.
    try:
        fs_id = (FULL_SAMPLE_ID if PRIORS_TAG == "paper"
                 else "%s%s_%s" % (FULL_SAMPLE_ID, STORE_SUFFIX[PRIORS_TAG],
                                   priors_digest(PRIORS_IN_FORCE)))
        write_manifest(fs_id, PRIORS_TAG, PRIORS_IN_FORCE)
        out = save_pmle_params(end, fs_id, params)
        print("  saved to %s" % out)
    except Exception as exc:                                      # noqa: BLE001
        print("  WARNING could not write to the parameter store: %s: %s"
              % (type(exc).__name__, exc))
        print("  The fit itself is safe in %s" % raw)

    # the only question that matters: did the posterior move off the prior?
    from poc.prior_diagnostics import PRIORS, HDI_TO_SD          # noqa: E402
    print("  %-8s %-14s %9s %9s %9s %7s %8s  %s"
          % ("param", "prior", "pri_mean", "post_mean", "post_sd", "ratio",
             "shift", "verdict"))
    print("  " + "-" * 84)
    for k, (label, pmean, psd) in PRIORS.items():
        if k not in params:
            continue
        m, lo, hi = params[k]
        post_sd = (hi - lo) * HDI_TO_SD
        ratio = post_sd / psd
        shift = (m - pmean) / psd

        # The verdict needs BOTH width and location. Width alone mislabels a
        # parameter whose posterior has moved a long way from the prior mean
        # into a region where its own scale is larger: dLAMB came back at 77.0
        # against a prior mean of 20.0 - nine prior standard deviations - with
        # a ratio of 1.42, and a width-only rule called that PRIOR-DRIVEN. A
        # prior cannot drag a posterior nine sd away from itself. Any large
        # shift is decisive evidence of data dominance regardless of width.
        if ratio < 0.35 or abs(shift) > 2.0:
            v = "DATA-DRIVEN"
        elif ratio < 0.70:
            v = "partial"
        elif ratio < 0.90:
            v = "weak"
        else:
            v = "PRIOR-DRIVEN"
        print("  %-8s %-14s %9.3f %9.3f %9.3f %7.2f %+8.2f  %s"
              % (k, label, pmean, m, post_sd, ratio, shift, v))
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


# Set by main() from --priors. Module-level so the forkserver workers inherit
# it rather than needing it threaded through every call signature.
PRIORS_IN_FORCE = None


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

    todo = [d for d in usable if not pmle_params_exists(d, STORE_ID)]
    print("  %d dates requested, %d usable, %d already on disk, %d to estimate."
          % (len(dates), len(usable), len(usable) - len(todo), len(todo)))
    if not todo:
        return

    # Before any fitting, so an interrupted run still leaves a drawer that
    # says what it holds.
    print("  priors recorded in %s"
          % os.path.relpath(write_manifest(STORE_ID, PRIORS_TAG, PRIORS_IN_FORCE),
                            _REPO_ROOT))

    args = []
    for dt in todo:
        rv = (return_ts.loc[return_ts.index <= dt, SYSTEMATIC_ID]
              .iloc[-LOOKBACK:].to_numpy())
        args.append((dt, rv, np.array(1 / BASE_DAYS), SEED, N_MC_PATHS,
                     STORE_ID, PRIORS_IN_FORCE))

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
    dates = available_pmle_dates(STORE_ID)
    rows = []
    for dt in dates:
        s = get_pmle_params(dt, STORE_ID)
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
        from poc.prior_diagnostics import PRIORS as PAPER_PRIORS
        from Library.RiskEngineKimYi2025 import prior_moments

        # Measure against the priors ACTUALLY IN FORCE. Reporting a recentred
        # fit against the paper priors made dLAMB, dETA1 and dETA2 look "never
        # identified" with ratios of 2.5-3.6, when against their own priors the
        # medians are 0.78-0.92 - partial, not absent. Wrong denominator.
        _MAP = {"dSIGMA": "sigma", "dALPHA": "alpha_rv", "dPPROB": "pprob_rv",
                "dLAMB": "lamb", "dETA1": "eta1", "dETA2": "eta2"}
        if PRIORS_IN_FORCE is None:
            PRIORS = PAPER_PRIORS
        else:
            PRIORS = {k: (str(PRIORS_IN_FORCE[v]),
                          *prior_moments(PRIORS_IN_FORCE[v]))
                      for k, v in _MAP.items()}
            PRIORS = {k: (str(PRIORS_IN_FORCE[v]),
                          *prior_moments(PRIORS_IN_FORCE[v]))
                      for k, v in _map.items()}
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
    ap.add_argument("--priors",
                    choices=("paper", "recentred", "capped", "capped-beta",
                             "gaps", "asym", "skew"),
                    default="paper",
                    help="paper: the published priors; reproduces prior work "
                         "and stays the default so Scripts/ is untouched. "
                         "recentred: all six still free, centres moved onto "
                         "the full-sample calibration with wide sd, and eta1 "
                         "and eta2 sharing ONE prior so the jump-size "
                         "asymmetry is not asserted - the data separated them "
                         "unaided from an identical start. Recommended. "
                         "capped: recentred but pprob confined to [0, 0.6] by "
                         "a FLAT prior, which excludes and asserts nothing "
                         "about location inside. capped-beta: same cap via a "
                         "truncated Beta, keeping a central tendency at the "
                         "cost of a much tighter prior. gaps: shared "
                         "eta prior at mean 20 (5% mean jump) with lambda "
                         "brought down to mean 6 for coherence - asserts that "
                         "a jump is a GAP, against a full sample that prefers "
                         "many small ones. asym: gaps with the two etas pulled "
                         "apart, eta1 mean 25 and eta2 mean 50 - a 4% mean UP "
                         "jump against a 2% mean DOWN one, which is the "
                         "OPPOSITE sign to equity skew and to the full sample. "
                         "A deliberately wrong-signed prior, to see whether "
                         "the data drags it back. skew: the exact mirror of "
                         "asym - eta1 mean 50 and eta2 mean 25, a 2% up jump "
                         "against a 4% down one, which is the equity direction "
                         "and the paper's own assertion at the gaps scale. Run "
                         "it against asym: the pair have identical jump "
                         "variance, so any difference between them is the "
                         "window expressing a preference on SIGN.")
    ap.add_argument("--full-sample", action="store_true",
                    help="ONE fit on the whole sample. Run this first - it is "
                         "the cheap test of whether more jumps fixes eta1/eta2.")
    a = ap.parse_args()

    global COLOR, PRIORS_IN_FORCE, STORE_ID, PRIORS_TAG
    COLOR = a.color
    PRIORS_TAG = a.priors
    PRIORS_IN_FORCE = {"paper": None,
                       "recentred": SYSTEMATIC_PRIORS_RECENTRED,
                       "capped": SYSTEMATIC_PRIORS_CAPPED,
                       "capped-beta": SYSTEMATIC_PRIORS_CAPPED_BETA,
                       "gaps": SYSTEMATIC_PRIORS_GAPS,
                       "asym": SYSTEMATIC_PRIORS_ASYM,
                       "skew": SYSTEMATIC_PRIORS_SKEW}[a.priors]
    STORE_ID = store_id(a.priors, PRIORS_IN_FORCE)

    print("=" * 72)
    print("Step 1 :: SPX P-measure parameters via the repository's P-MLE")
    print("=" * 72)
    print("  %s -> %s, every %d business days, %d-day lookback"
          % (a.beg, a.end, a.step, LOOKBACK))
    print("  credible intervals: %.0f%% equal-tailed (%s)" % (100 * CI_PROB, CI_CONVENTION))
    print("  priors: %s" % a.priors)
    print("  estimates stored under: %s" % STORE_ID)
    print("    the trailing digest fingerprints the prior VALUES, so editing"
          "\n    any prior opens a new drawer instead of silently reusing the"
          "\n    old one. _priors.json in the drawer records the full spec.")
    if a.priors == "gaps":
        print("    eta1 and eta2 share Gamma(4,0.2): mean 20, i.e. a 5% jump")
        print("    lambda Gamma(3,0.5): mean 6, kept coherent with that size")
        print("    NOTE this asserts a jump scale the full sample argues")
        print("    against - watch whether the posterior is dragged back up")
    if a.priors == "asym":
        print("    eta1 Gamma(4,0.16): mean 25 -> mean UP   jump 4.0%")
        print("    eta2 Gamma(4,0.08): mean 50 -> mean DOWN jump 2.0%")
        print("    NOTE this asserts POSITIVE jump skew - bigger up moves than")
        print("    down. It is the opposite of the paper's own defaults and of")
        print("    the full sample (eta1 78.6 / eta2 60.7). Watch whether the")
        print("    posterior pulls eta2 back BELOW eta1; if it stays put, the")
        print("    separation seen elsewhere was never the data's doing.")
    if a.priors == "skew":
        print("    eta1 Gamma(4,0.08): mean 50 -> mean UP   jump 2.0%")
        print("    eta2 Gamma(4,0.16): mean 25 -> mean DOWN jump 4.0%")
        print("    NEGATIVE jump skew - the equity direction, and the exact")
        print("    mirror of --priors asym. E[Y^2] is identical under both at")
        print("    p=0.5, so if the window is skew-blind this should return")
        print("    sigma, lambda and total vol within noise of asym, an eta")
        print("    separation near -19.8 against asym's +19.8, and pprob near")
        print("    0.519 against asym's 0.481. A DIFFERENCE between the two is")
        print("    the only result that would overturn that reading.")
    if a.priors.startswith("capped"):
        print("    pprob CAPPED at 0.6 - watch for the posterior piling up")
        print("    against the cap in 2007, 2013, 2017, 2018, 2021, 2024,")
        print("    which currently sit above it")
    if a.priors == "recentred":
        print("    all six free; eta1 and eta2 share one prior "
              "(mean %.0f) so no asymmetry is asserted" % 70.0)

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

    out = os.path.join(
        _REPO_ROOT, "poc",
        "systematic_params%s.csv" % artifact_suffix(PRIORS_TAG, PRIORS_IN_FORCE))
    df.to_csv(out)
    print("\nWritten to %s" % out)
    print("This series is the rolling systematic input to backfill_poc.py.")


if __name__ == "__main__":
    main()
