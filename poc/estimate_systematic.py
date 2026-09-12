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
import signal
import sys
import time

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import multiprocessing
# forkserver where it exists, spawn where it does not. fork() in a process
# that has already started threads is unsafe - Python 3.12+ warns and 3.14
# changes the Linux default - and PyMC with a numba backend does start
# threads; the failure mode is a hang that looks like slow sampling.
#
# The old form asked for "forkserver" unconditionally, which raises
# ValueError on Windows, where get_all_start_methods() is ["spawn"] alone.
# Library.Parallel picks what the platform actually has. JGL_MP_START still
# overrides, and is now warned about rather than obeyed if unavailable.
from Library.Parallel import set_start_method as _set_start_method  # noqa: E402
MP_START = _set_start_method()
from concurrent.futures import (                           # noqa: E402
    ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED,
)
# BrokenProcessPool is NOT re-exported by concurrent.futures - its __all__
# carries the generic BrokenExecutor only. It lives in the .process submodule.
from concurrent.futures.process import BrokenProcessPool     # noqa: E402

from Library.RiskEngineKimYi2025 import (                     # noqa: E402
    SYSTEMATIC_PRIOR_SETS, STORE_SUFFIX, FULL_SAMPLE,
    JGL_CHAINS, JGL_CORES, JGL_DRAWS,
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

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)

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

# Set by main() from --priors, alongside PRIORS_IN_FORCE.
STORE_ID = SYSTEMATIC_ID
PRIORS_TAG = "paper"


def priors_digest(priors):
    """An 8-hex fingerprint of a RESOLVED prior specification.

    Naming the drawer after --priors is not enough, and the reason is the
    mistake this function exists to prevent. The tag "gaps" names a variable,
    not a value. When the specs still inherited from one another, recentring
    alpha inside the shared base changed every spec derived from it while
    every tag stayed the same. A rerun then finds 245 dates "already on disk", skips
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


def lookback_suffix(lookback=None):
    """Empty at the study default, "__lb756" otherwise.

    Window length is not in the prior digest and cannot be: the digest hashes
    the prior SPEC, and the spec does not mention the window. So a 3-year run
    under an unchanged spec produces an identical digest and would land in the
    252-day drawer, where run() skips every date as "already on disk" and the
    two window lengths interleave inside one series. The suffix is what keeps
    them apart. It is empty at 252 so every existing drawer keeps its name.
    """
    lb = LOOKBACK if lookback is None else lookback
    return "" if int(lb) == BASE_LOOKBACK else "__lb%d" % int(lb)


def store_id(tag, priors, lookback=None):
    """Where estimates fitted under (tag, priors, lookback) are stored.

    "paper" keeps the bare id: those literals are frozen by publication and the
    committed replication files under Study/Estimated Parameters PMLE/^SPX/ are
    addressed by it. Everything else is tag + digest, so the name says which
    arm it is and the digest says which VERSION of that arm.

    The window suffix comes LAST, after the digest, so the existing name stays
    a readable prefix and the arms sort together.
    """
    lb = lookback_suffix(lookback)
    if tag == "paper":
        return SYSTEMATIC_ID + lb
    return "%s%s_%s%s" % (SYSTEMATIC_ID, STORE_SUFFIX[tag],
                          priors_digest(priors), lb)


def artifact_suffix(tag, priors, lookback=None):
    """Filename suffix for the loose poc/ artefacts, matching the drawer.

    systematic_params.csv and full_sample_params.json have to carry the digest
    too. Tagging them by name alone would let an edited spec overwrite the
    assembled series of the run it was compared against - the same failure as
    the drawer, on the file you actually read.
    """
    if tag == "paper":
        return lookback_suffix(lookback)
    return "%s_%s%s" % (STORE_SUFFIX[tag], priors_digest(priors),
                        lookback_suffix(lookback))


def sampler_settings():
    """What the sampler was actually configured to do for this run.

    The drawer digest covers PRIOR VALUES only, so two runs with different
    sampler settings land in the SAME drawer and become indistinguishable once
    written. Draw count moves the credible-interval endpoints - 1,000 draws
    gives ~2,150 min tail ess against 10,000's ~26,000 - so a series that
    silently mixes them has intervals that are not comparable across dates.
    Recording it does not prevent the mixing; it makes it detectable.
    """
    return {"chains": JGL_CHAINS, "cores": JGL_CORES,
            "draws": JGL_DRAWS or N_MC_PATHS, "tune": 1000,
            "nuts_sampler": "nutpie"}

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
                   "base_lookback": BASE_LOOKBACK,
                   "base_days": BASE_DAYS,
                   "seed": int(SEED),
                   "n_mc_paths": N_MC_PATHS,
                   "sampler": sampler_settings(),
                   "priors": spec}, fh, indent=2)
    return path
DATE_FMT = "%Y%m%d"

BEG = "20070101"
END = "20260831"
# BASE_LOOKBACK is the study default and never moves; LOOKBACK is what THIS
# run uses and main() may set it from --lookback. The two are separate because
# the drawer name has to say when they differ - see lookback_suffix.
BASE_LOOKBACK = 252
LOOKBACK = BASE_LOOKBACK
BASE_DAYS = 252
SEED = np.uint64(20240114)
# Draws per chain. Was 10,000; the benchmark (poc/bench_sampler.py, ^SPX
# 2025-06-30, 24 cpus) says that is ~26x more than the reported intervals use:
#
#   draws   sec/fit   min tail ess
#    1000       6.8           2153
#    2000       7.3           4843
#   10000      13.4          26032
#
# A 95% equal-tailed interval needs enough TAIL ess to pin the 2.5%/97.5%
# quantiles - order 1,000. 1,000 draws over 4 chains delivers 2,153, and a fit
# costs half as long. Most of what remains is fixed cost: seconds = 6.13 +
# 0.00073 x draws, so compilation plus 1,000 tuning steps is ~90% of a 1,000-
# draw fit and cutting further buys almost nothing.
#
# This was measured on ONE date and the systematic model. Sampling efficiency
# varies by date and the idiosyncratic geometry differs, so the engine now
# checks min tail ess on every fit and warns when it falls below
# JGL_MIN_TAIL_ESS - the assumption is monitored rather than trusted.
# JGL_DRAWS overrides this for a whole run.
N_MC_PATHS = 1_000

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

    _LOG.info("  full-sample fit on %d daily returns (%s to %s)"
          % (len(rv), beg, end))
    _LOG.info("  this is one MCMC fit - expect minutes, not seconds\n")

    t0 = time.perf_counter()
    _, _, results = pmle_kimyirisk_systematic_helper(
        ("FULLSAMPLE", rv, np.array(1 / BASE_DAYS), SEED, N_MC_PATHS,
         SYSTEMATIC_ID, PRIORS_IN_FORCE))
    elapsed = time.perf_counter() - t0
    _LOG.info("  done in %.0fs\n" % elapsed)

    params = assemble_systematic_params(results)

    # Dump to a plain file FIRST, before anything that can fail. This fit costs
    # ~8 minutes and an earlier version lost one entirely to a serialisation
    # error raised after the sampler had finished. Nothing expensive should be
    # destroyed by a failure in how it gets written down.
    raw = os.path.join(
        _REPO_ROOT, "poc",
        # BASE_LOOKBACK, not the run's --lookback: a full-sample fit uses the
        # whole series and has no window, so tagging it with one would split
        # an identical fit across two filenames.
        "full_sample_params%s.json"
        % artifact_suffix(PRIORS_TAG, PRIORS_IN_FORCE, BASE_LOOKBACK))
    with open(raw, "w") as fh:
        json.dump({"beg": beg, "end": end, "n_returns": int(len(rv)),
                   "seconds": round(elapsed, 1),
                   "params": {k: [float(x) for x in v]
                              for k, v in params.items()}}, fh, indent=2)
    _LOG.info("  raw results written to %s" % raw)

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
        _LOG.info("  saved to %s" % out)
    except Exception as exc:                                      # noqa: BLE001
        _LOG.info("  WARNING could not write to the parameter store: %s: %s"
              % (type(exc).__name__, exc))
        _LOG.info("  The fit itself is safe in %s" % raw)

    # the only question that matters: did the posterior move off the prior?
    from poc.prior_diagnostics import PRIORS, HDI_TO_SD          # noqa: E402
    _LOG.info("  %-8s %-14s %9s %9s %9s %7s %8s  %s"
          % ("param", "prior", "pri_mean", "post_mean", "post_sd", "ratio",
             "shift", "verdict"))
    _LOG.info("  " + "-" * 84)
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
        _LOG.info("  %-8s %-14s %9.3f %9.3f %9.3f %7.2f %+8.2f  %s"
              % (k, label, pmean, m, post_sd, ratio, shift, v))
    _LOG.info("\n  If dETA1 and dETA2 are now DATA-DRIVEN, the identification")
    _LOG.info("  problem is solved from returns alone and nothing further is")
    _LOG.info("  needed. If they are still PRIOR-DRIVEN with ~230 jumps, then and")
    _LOG.info("  only then is the option-implied route worth the complexity.")


# Worker recycling: explicit pool teardown, NOT max_tasks_per_child.
#
# ProcessPoolExecutor keeps each worker for the life of the pool, so whatever
# PyMC/nutpie fail to release accumulates over a long run. Two ways to fix it,
# and the history here matters:
#
#   max_tasks_per_child (3.11+) respawns a worker after N tasks. It was set to
#   25 and it is the prime suspect for the hang that followed: a run stopped at
#   120/435 and sat there SIX HOURS, process alive, no output. Under forkserver
#   every respawn re-imports the whole PyMC/nutpie stack, and a wedge in that
#   path deadlocks the executor rather than raising. DEFAULT OFF for that
#   reason - set JGL_MAX_TASKS_PER_CHILD to re-enable it deliberately.
#
#   POOL_CHUNK tears the whole pool down and builds a new one every N fits.
#   Same memory effect, but it happens at a synchronisation point where every
#   worker has already exited, so there is no respawn racing live tasks. Cost
#   is one stack re-import per chunk, ~10s against N fits of 10-60s each.
MAX_TASKS_PER_CHILD = int(os.environ.get("JGL_MAX_TASKS_PER_CHILD", "0"))
POOL_CHUNK = int(os.environ.get("JGL_POOL_CHUNK", "50"))


def _pool_kwargs(workers):
    kw = {"max_workers": workers, "initializer": _init_child}
    if MAX_TASKS_PER_CHILD > 0:
        import inspect
        if "max_tasks_per_child" in inspect.signature(ProcessPoolExecutor).parameters:
            kw["max_tasks_per_child"] = MAX_TASKS_PER_CHILD
            _LOG.info("  NOTE max_tasks_per_child=%d requested. This has hung a run"
                  % MAX_TASKS_PER_CHILD)
            _LOG.info("       before; POOL_CHUNK is the safer recycler.")
        else:
            _LOG.info("  NOTE max_tasks_per_child needs Python 3.11+; ignoring.")
    return kw


# Stall watchdog.
#
# A run sat at 120/435 for SIX HOURS: process alive, no output, no exception.
# as_completed() blocks forever by construction, so a deadlocked executor is
# indistinguishable from a slow one - and under forkserver a worker that dies
# while the pool is respawning another can wedge it without ever raising
# BrokenProcessPool.
#
# So wait in bounded slices instead. If NOTHING completes within STALL_TIMEOUT
# the pool is wedged: say so, name the pending count, exit non-zero. Everything
# finished is already on disk and a rerun resumes, so a false positive costs
# one restart while a missed hang costs a night.
STALL_TIMEOUT = float(os.environ.get("JGL_STALL_TIMEOUT", "900"))


def _drain(pending, on_done, label="tasks"):
    """Consume futures, aborting if the pool stops making progress."""
    while pending:
        finished, pending = wait(pending, timeout=STALL_TIMEOUT,
                                 return_when=FIRST_COMPLETED)
        if not finished:
            _LOG.info("\n" + "=" * 72)
            _LOG.info("  STALLED - no %s completed in %.0fs. The pool is wedged."
                  % (label, STALL_TIMEOUT))
            _LOG.info("=" * 72)
            _LOG.info("  %d still pending. Everything finished is on disk; rerun"
                  % len(pending))
            _LOG.info("  to resume. To cut the risk, reduce memory pressure:")
            _LOG.info("    JGL_POOL_CHUNK=20 <same command>")
            _LOG.info("    <same command> --workers 2")
            _LOG.info("")
            _LOG.info("  Check whether the kernel killed a worker:")
            _LOG.info("    sudo dmesg -T | grep -i 'killed process' | tail")
            _LOG.info("    free -h        # is the machine in swap?")
            for f in pending:
                f.cancel()
            raise SystemExit(3)
        for fut in finished:
            on_done(fut)


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


# Set by main() from --priors, and read HERE, in the parent, when the task
# tuples are built - run() puts the value into every tuple. The comment that
# used to sit here said these globals existed "so the forkserver workers
# inherit" them, which was never how a worker got the value and is actively
# dangerous now that spawn is reachable: a spawn child inherits nothing, so
# anyone who trusted that line and dropped the tuple threading would get a
# silent fall back to the module default.
PRIORS_IN_FORCE = None


def run(dates, workers=None, force=False):
    """Estimate the systematic parameters for each date not already on disk.

    force=True re-estimates every usable date and overwrites what is there.
    The drawer is content-addressed on the prior values, so a forced rerun
    under an unchanged spec is a REPRODUCIBILITY CHECK - same seed, same
    priors, same data should give the same numbers. It is a destructive one:
    the old estimates are replaced in place, so if the point is to COMPARE
    old against new, archive first (poc/archive_run.py --drawer ... --label ...)
    rather than forcing over them.
    """
    price_ts = get_price_panel([SYSTEMATIC_ID])
    return_ts = price_ts.pct_change().dropna()

    first_ok = return_ts.index[LOOKBACK - 1]
    usable, too_early = [], []
    for dt in dates:
        (usable if pd.to_datetime(dt, format=DATE_FMT) >= first_ok
         else too_early).append(dt)
    if too_early:
        _LOG.info("  %d dates dropped: fewer than %d prior returns available."
              % (len(too_early), LOOKBACK))
        _LOG.info("  Earliest estimable date is %s. To reach back to %s, the price"
              % (first_ok.date(), BEG))
        _LOG.info("  snapshot must start ~%d business days earlier." % LOOKBACK)

    on_disk = [d for d in usable if pmle_params_exists(d, STORE_ID)]
    todo = usable if force else [d for d in usable if d not in set(on_disk)]
    _LOG.info("  %d dates requested, %d usable, %d already on disk, %d to estimate."
          % (len(dates), len(usable), len(on_disk), len(todo)))
    if force and on_disk:
        _LOG.info("")
        _LOG.info("  --force: OVERWRITING %d existing estimate(s) in %s"
              % (len(on_disk), STORE_ID))
        _LOG.info("  They are replaced in place. To keep them, stop now and run")
        _LOG.info("    python poc/archive_run.py --drawer '%s' --label <name> --apply"
              % STORE_ID)
        _LOG.info("")
    if not todo:
        return

    # Before any fitting, so an interrupted run still leaves a drawer that
    # says what it holds.
    _LOG.info("  priors recorded in %s"
          % os.path.relpath(write_manifest(STORE_ID, PRIORS_TAG, PRIORS_IN_FORCE),
                            _REPO_ROOT))

    args = []
    for dt in todo:
        rv = (return_ts.loc[return_ts.index <= dt, SYSTEMATIC_ID]
              .iloc[-LOOKBACK:].to_numpy())
        args.append((dt, rv, np.array(1 / BASE_DAYS), SEED, N_MC_PATHS,
                     STORE_ID, PRIORS_IN_FORCE))

    workers = workers or default_workers()
    _LOG.info("  %d workers x 4 chains = %d concurrent samplers on %d cores"
          % (workers, workers * 4, os.cpu_count() or 0))

    t0 = time.perf_counter()
    done = failed = 0
    chunk = POOL_CHUNK if POOL_CHUNK > 0 else len(args)
    try:
        # One pool per CHUNK fits. Tearing it down is the only way this
        # environment's workers ever release memory - see POOL_CHUNK.
        for start in range(0, len(args), chunk):
            batch = args[start:start + chunk]
            with ProcessPoolExecutor(**_pool_kwargs(workers)) as ex:
                futures = {ex.submit(pmle_kimyirisk_systematic_helper, a): a[0]
                           for a in batch}

                def _on_done(fut, _f=futures):
                    nonlocal done, failed
                    requested = _f[fut]
                    try:
                        dt, sid, results = fut.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:                  # noqa: BLE001
                        # One bad date must not end a run of thousands.
                        failed += 1
                        _LOG.info("    FAILED  %s  %s: %s"
                              % (requested, type(exc).__name__, exc))
                        return
                    save_pmle_params(dt, sid, assemble_systematic_params(results))
                    done += 1
                    if done % 10 == 0 or done == len(todo):
                        el = time.perf_counter() - t0
                        _LOG.info("    %4d/%d  %.1fs elapsed, ~%.1fs remaining"
                              % (done, len(todo), el,
                                 el / done * (len(todo) - done)))

                _drain(set(futures), _on_done, "dates")
            if start + chunk < len(args):
                _LOG.info("    -- pool recycled after %d dates --" % done)
    except BrokenProcessPool:
        # A worker died without raising - it was killed by a signal, not by a
        # Python exception. Overwhelmingly this is the kernel OOM killer:
        # every worker holds a compiled model, N_MC_PATHS paths and four
        # chains of draws, so peak memory scales with the worker count while
        # the estimate of "how many cores" does not.
        el = time.perf_counter() - t0
        _LOG.info("\n" + "=" * 72)
        _LOG.info("  WORKER KILLED - the pool is dead and the run stopped early.")
        _LOG.info("=" * 72)
        _LOG.info("  %d of %d dates completed and ARE SAFELY ON DISK (%.0fs)."
              % (done, len(todo), el))
        _LOG.info("  Nothing is lost: rerunning skips what is already written.")
        _LOG.info("")
        _LOG.info("  A worker terminated without a Python exception, which means")
        _LOG.info("  it was killed by a signal rather than failing in Python.")
        _LOG.info("  Confirm the cause before rerunning:")
        _LOG.info("")
        _LOG.info("      sudo dmesg -T | grep -i -E 'killed process|out of memory' | tail")
        _LOG.info("      journalctl -k --since '3 hours ago' | grep -i 'killed process'")
        _LOG.info("")
        _LOG.info("  If that shows an OOM kill, rerun with fewer workers - memory")
        _LOG.info("  scales with worker count, cores do not:")
        _LOG.info("")
        _LOG.info("      JGL_POOL_CHUNK=20 <same command>            (recycle sooner)")
        _LOG.info("      ./run_daily.sh %d                            (fewer workers)"
              % max(1, workers // 2))
        _LOG.info("")
        _LOG.info("  If it shows nothing, suspect a native crash in the sampler.")
        _LOG.info("  Reproduce one date in the foreground to get a real traceback:")
        _LOG.info("")
        _LOG.info("      python poc/estimate_systematic.py --full-sample")
        _LOG.info("=" * 72)
        return

    if failed:
        _LOG.info("\n  %d date(s) failed and were skipped; rerun to retry them."
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
    _LOG.info("\n" + "=" * 72)
    _LOG.info("SPX P-measure parameters :: %s to %s   (%d valuation dates)"
          % (df.index.min().date(), df.index.max().date(), len(df)))
    _LOG.info("=" * 72)
    _LOG.info("\nDistribution across valuation dates:")
    _p = [c for c in df.columns if not c.endswith("_W")]
    _LOG.info(df[_p].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
            .round(4).to_string())

    params_only = [c for c in df.columns if not c.endswith("_W")]
    if len(df) < 2:
        _LOG.info("\nStability - needs at least 2 valuation dates; %d estimated."
              % len(df))
    else:
        _LOG.info("\nStability - coefficient of variation (sd / |mean|):")
        cv = (df[params_only].std() / df[params_only].mean().abs()).sort_values()
        for k, v in cv.items():
            note = "" if v < 0.25 else "   <-- varies a lot across windows"
            _LOG.info("   %-8s %6.3f%s" % (k, v, note))
        _LOG.info("\n   Read this WITH the identification table below, not alone. A")
        _LOG.info("   low CV means the estimate barely moves - which is evidence of")
        _LOG.info("   identification only if the parameter is actually identified.")
        _LOG.info("   A prior-driven parameter is stable because its prior is.")

    # By year, mean and median, same layout. Both are shown because they answer
    # different questions on ~12 overlapping 252-day windows per year: the mean
    # gives the level, the median resists a single divergent fit. Where they
    # diverge the year is skewed - in crisis years that is expected, since the
    # crisis moves in and out of the trailing window across the twelve fits.
    for stat in ("mean", "median"):
        t = getattr(df.groupby(df.index.year), stat)().round(4)
        t.index = [str(i) for i in t.index]
        _LOG.info("\nBy year (%s):" % stat)
        _LOG.info(heat(t, decimals=4, color=COLOR))
    _LOG.info(heat_legend(color=COLOR))


    # prior sds, to judge identification date by date
    try:
        from poc.prior_diagnostics import PRIORS as PAPER_PRIORS
        from Library.RiskEngineKimYi2025 import prior_moments, prior_ci_width

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
        prior_sd = {k: v[2] for k, v in PRIORS.items()}
        # The denominator is the prior's own 95% WIDTH, not its sd. See
        # prior_ci_width: dividing a width by an sd assumes the prior is
        # normal, and under Uniform(0, L) an untouched posterior then reads
        # 0.84 instead of 1.00 - a 16% narrowing that never happened. Where
        # the spec is not available (the paper-prior fallback carries moments
        # only) fall back to the normal-equivalent width, which is what the
        # old ratio assumed throughout.
        if PRIORS_IN_FORCE is None:
            prior_w = {k: v / CI_WIDTH_TO_SD for k, v in prior_sd.items()}
        else:
            prior_w = {k: prior_ci_width(PRIORS_IN_FORCE[v], CI_PROB)
                       for k, v in _MAP.items()}
    except Exception as exc:                                     # noqa: BLE001
        # Say so. A silent {} here drops the entire identification table, which
        # is the one part of this report the conclusions rest on - a bug in this
        # block once removed it from every run without a word.
        _LOG.info("\n  WARNING identification table skipped: %s: %s"
              % (type(exc).__name__, exc))
        prior_sd, prior_w = {}, {}

    if prior_w:
        _LOG.info("\nIdentification over time - share of valuation dates where the")
        _LOG.info("posterior is narrower than the prior (ratio < 0.70):")
        _LOG.info("  ratio = posterior %.0f%% width / PRIOR %.0f%% width, both "
              "equal-tailed." % (100 * CI_PROB, 100 * CI_PROB))
        for k in SYSTEMATIC_PARAMS:
            w = k + "_W"
            if k not in prior_sd or w not in df:
                continue
            if not prior_w.get(k):
                continue
            ratio = df[w] / prior_w[k]
            share = 100.0 * float((ratio < 0.70).mean())
            _LOG.info("   %-8s %5.1f%%   median ratio %.2f   (min %.2f, max %.2f)"
                  % (k, share, ratio.median(), ratio.min(), ratio.max()))
        _LOG.info("\n   Width against width on purpose: the earlier form divided the")
        _LOG.info("   posterior width by the prior SD, which assumes a normal prior and")
        _LOG.info("   put the no-information floor at 0.84 under a flat one. Here an")
        _LOG.info("   unmoved posterior reads exactly 1.00 whatever the prior's shape.")
        _LOG.info("\n   0% means the parameter is never identified at any date in the")
        _LOG.info("   sample - that rolling series is a series of priors, not estimates.")
        _LOG.info("   A ratio at or above 1.00 means the posterior is no narrower than")
        _LOG.info("   the prior: the likelihood is flat in that direction.")

    _LOG.info("\nJump intensity dLAMB at known stress episodes (should spike):")
    for label, (a, b) in {
        "GFC 2008H2":  ("2008-07-01", "2008-12-31"),
        "Euro 2011":   ("2011-07-01", "2011-12-31"),
        "Covid 2020":  ("2020-02-01", "2020-06-30"),
        "SVB 2023":    ("2023-03-01", "2023-06-30"),
        "Tariffs 2025": ("2025-04-01", "2025-07-31"),
    }.items():
        w = df.loc[a:b, "dLAMB"] if "dLAMB" in df else pd.Series(dtype=float)
        if len(w):
            _LOG.info("   %-14s median %.2f   (full-sample median %.2f)"
                  % (label, w.median(), df["dLAMB"].median()))


def verify(df):
    """Check the April 2025 estimates against the values reported in the paper."""
    w = df.loc["2025-04-01":"2025-04-30"]
    _LOG.info("\n" + "=" * 72)
    _LOG.info("VERIFICATION :: April 2025 against the paper's reported ranges")
    _LOG.info("=" * 72)
    if not len(w):
        _LOG.info("  No April 2025 valuation dates estimated yet. Run with a step")
        _LOG.info("  that lands in that window before relying on this check.")
        return
    _LOG.info("  %d valuation dates in April 2025\n" % len(w))
    for k, (lo, hi) in PAPER_APRIL_2025.items():
        if k not in w:
            continue
        obs_lo, obs_hi = w[k].min(), w[k].max()
        ok = (obs_hi >= lo) and (obs_lo <= hi)
        _LOG.info("   %-8s paper [%6.2f, %6.2f]   here [%6.2f, %6.2f]   %s"
              % (k, lo, hi, obs_lo, obs_hi, "OK" if ok else "*** MISMATCH ***"))
    _LOG.info("\n  A mismatch means this pipeline is not reproducing the published")
    _LOG.info("  estimator. Fix that before trusting any parameter in this run.")


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
    ap.add_argument("--force", "--overwrite", dest="force", action="store_true",
                    help="re-estimate every date even if already on disk, "
                         "overwriting in place. Under an unchanged spec this "
                         "is a reproducibility check; to compare old against "
                         "new, archive_run.py the drawer first instead.")
    ap.add_argument("--color", dest="color", action="store_true", default=None,
                    help="force heat shading on (default: on for a terminal)")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="plain numbers, no shading")
    ap.add_argument("--workers", type=int, default=None,
                    help="outer pool width. Default cpu_count//4, because each "
                         "fit already uses 4 chains on 4 cores.")
    ap.add_argument("--priors",
                    choices=tuple(SYSTEMATIC_PRIOR_SETS),
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
                         "window expressing a preference on SIGN. "
                         "skew-tight: skew with both eta prior sds HALVED "
                         "(25->12.5, 12.5->6.25), means unchanged. Narrows the "
                         "eta columns without adding evidence - run it to see "
                         "the width fall while the prior/posterior ratio gets "
                         "WORSE.")
    ap.add_argument("--lookback", type=int, default=BASE_LOOKBACK,
                    help="trailing returns per fit. Default %d (one regulatory "
                         "year). 756 is three years. Anything other than the "
                         "default opens its OWN drawer (suffix __lb<n>), so a "
                         "long-window run cannot interleave with the one-year "
                         "series." % BASE_LOOKBACK)
    ap.add_argument("--full-sample", action="store_true",
                    help="ONE fit on the whole sample. Run this first - it is "
                         "the cheap test of whether more jumps fixes eta1/eta2.")
    a = ap.parse_args()

    global COLOR, PRIORS_IN_FORCE, STORE_ID, PRIORS_TAG, LOOKBACK
    COLOR = a.color
    PRIORS_TAG = a.priors
    PRIORS_IN_FORCE = SYSTEMATIC_PRIOR_SETS[a.priors]
    if a.lookback < 60:
        raise SystemExit("--lookback %d is too short to fit six systematic "
                         "parameters; the study default is %d."
                         % (a.lookback, BASE_LOOKBACK))
    if a.lookback != BASE_LOOKBACK and a.full_sample:
        raise SystemExit("--lookback has no meaning with --full-sample: that "
                         "fit uses the whole series, not a trailing window.")
    LOOKBACK = a.lookback
    STORE_ID = store_id(a.priors, PRIORS_IN_FORCE)

    _LOG.info("=" * 72)
    _LOG.info("Step 1 :: SPX P-measure parameters via the repository's P-MLE")
    _LOG.info("=" * 72)
    _LOG.info("  %s -> %s, every %d business days, %d-day lookback"
          % (a.beg, a.end, a.step, LOOKBACK))
    _LOG.info("  credible intervals: %.0f%% equal-tailed (%s)" % (100 * CI_PROB, CI_CONVENTION))
    _LOG.info("  priors: %s" % a.priors)
    _LOG.info("  estimates stored under: %s" % STORE_ID)
    if LOOKBACK != BASE_LOOKBACK:
        _LOG.info("    NON-DEFAULT WINDOW: %d returns, not %d. The __lb%d suffix"
              % (LOOKBACK, BASE_LOOKBACK, LOOKBACK))
        _LOG.info("    keeps this out of the one-year drawer; nothing here is")
        _LOG.info("    comparable to a %d-day estimate at the same date except by"
              % BASE_LOOKBACK)
        _LOG.info("    reading both drawers side by side.")
    _LOG.info("    the trailing digest fingerprints the prior VALUES, so editing"
          "\n    any prior opens a new drawer instead of silently reusing the"
          "\n    old one. _priors.json in the drawer records the full spec.")
    if a.priors == "gaps":
        _LOG.info("    eta1 and eta2 share Gamma(4,0.2): mean 20, i.e. a 5% jump")
        _LOG.info("    lambda Gamma(3,0.5): mean 6, kept coherent with that size")
        _LOG.info("    NOTE this asserts a jump scale the full sample argues")
        _LOG.info("    against - watch whether the posterior is dragged back up")
    if a.priors == "asym":
        _LOG.info("    eta1 Gamma(4,0.16): mean 25 -> mean UP   jump 4.0%")
        _LOG.info("    eta2 Gamma(4,0.08): mean 50 -> mean DOWN jump 2.0%")
        _LOG.info("    NOTE this asserts POSITIVE jump skew - bigger up moves than")
        _LOG.info("    down. It is the opposite of the paper's own defaults and of")
        _LOG.info("    the full sample (eta1 78.6 / eta2 60.7). Watch whether the")
        _LOG.info("    posterior pulls eta2 back BELOW eta1; if it stays put, the")
        _LOG.info("    separation seen elsewhere was never the data's doing.")
    if a.priors == "skew":
        _LOG.info("    eta1 Gamma(4,0.08): mean 50 -> mean UP   jump 2.0%")
        _LOG.info("    eta2 Gamma(4,0.16): mean 25 -> mean DOWN jump 4.0%")
        _LOG.info("    NEGATIVE jump skew - the equity direction, and the exact")
        _LOG.info("    mirror of --priors asym. E[Y^2] is identical under both at")
        _LOG.info("    p=0.5, so if the window is skew-blind this should return")
        _LOG.info("    sigma, lambda and total vol within noise of asym, an eta")
        _LOG.info("    separation near -19.8 against asym's +19.8, and pprob near")
        _LOG.info("    0.519 against asym's 0.481. A DIFFERENCE between the two is")
        _LOG.info("    the only result that would overturn that reading.")
    if a.priors == "skew-tight":
        _LOG.info("    every wide prior sd HALVED, all centres unchanged:")
        _LOG.info("      eta1     Gamma(16, 0.32)      mean 50.000  sd 12.500")
        _LOG.info("      eta2     Gamma(16, 0.64)      mean 25.000  sd  6.250")
        _LOG.info("      alpha    Beta(9.5, 9.5)       mean  0.500  sd  0.112")
        _LOG.info("      pprob    Beta(10.925, 8.075)  mean  0.575  sd  0.111")
        _LOG.info("      sigma and lambda unchanged - already data-driven")
        _LOG.info("    Predicted:")
        _LOG.info("      dETA1   width 84.24 -> 43.5    ratio 0.70 -> 0.89")
        _LOG.info("      dETA2   width 46.73 -> 22.7    ratio 0.77 -> 0.92")
        _LOG.info("      dALPHA  width 0.795 -> 0.427   ratio 0.91 -> 0.97")
        _LOG.info("      dPPROB  width 0.688 -> 0.404   ratio 0.79 -> 0.93")
        _LOG.info("    Every interval halves and every ratio moves TOWARD 1 - the")
        _LOG.info("    columns look tighter and score as LESS identified, because")
        _LOG.info("    the prior shrinks faster than the posterior. A narrow")
        _LOG.info("    interval here is not information. Do NOT read these columns")
        _LOG.info("    as better estimated than skew's; they are the same evidence")
        _LOG.info("    reported against a stronger assertion.")

    if a.full_sample:
        run_full_sample(a.beg, a.end)
        return

    if not a.report_only:
        run(valuation_dates(a.beg, a.end, a.step), workers=a.workers,
            force=a.force)

    df = load_series()
    if not len(df):
        _LOG.info("\nNothing estimated yet.")
        return
    report(df)
    if a.verify:
        verify(df)

    out = os.path.join(
        _REPO_ROOT, "poc",
        "systematic_params%s.csv" % artifact_suffix(PRIORS_TAG, PRIORS_IN_FORCE))
    df.to_csv(out)
    _LOG.info("\nWritten to %s" % out)
    _LOG.info("This series is the rolling systematic input to backfill_poc.py.")


def _reap_forkserver():
    """Kill the forkserver helper before we go.

    os._exit skips atexit, which is the point - but multiprocessing's atexit
    handler is also what shuts the forkserver helper down. Skipping it orphans
    that helper, and an orphan holds the terminal's pty open, so the shell
    never gets its prompt back even though this process is gone. That looked
    exactly like the original hang and was the fix trading one symptom for
    another. Kill it explicitly instead; it holds no state we need.
    """
    # Library.Parallel.reap_forkserver returns immediately when the start
    # method is not forkserver, so this is a no-op under spawn and on
    # Windows - where signal.SIGKILL does not exist and the old bare
    # `except Exception` was swallowing an AttributeError to get the same
    # result by accident.
    from Library.Parallel import reap_forkserver
    reap_forkserver()


def _exit_now(code=0):
    """Leave the process immediately, skipping interpreter teardown.

    A completed run printed its last line - the summary CSV was written -
    and then never returned the shell. Nothing was still computing: PyMC and
    nutpie start native (Rust) threads, and multiprocessing's forkserver
    leaves a helper process behind, and CPython's shutdown path joins those
    before exiting. If one does not come back, the process sits there forever
    looking exactly like a hang.

    Every result this script produces is already durably on disk when main()
    returns - per-date CSVs through save_pmle_params, the summary through
    to_csv - so an orderly teardown buys nothing. Flush explicitly, because
    os._exit does not.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:                                            # noqa: BLE001
        pass
    _reap_forkserver()
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
        _code = 0
    except SystemExit as _e:                # the stall watchdog exits 3
        _code = _e.code if isinstance(_e.code, int) else (0 if _e.code is None else 1)
    _exit_now(_code)
