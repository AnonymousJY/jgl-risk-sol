"""Step 1b :: idiosyncratic parameters for one or more names, conditional on
the systematic fit.

    python poc/estimate_idiosyncratic.py --names AAPL,MSFT --priors gaps
    python poc/estimate_idiosyncratic.py --names @poc/names.txt --anchor full
    python poc/estimate_idiosyncratic.py --names AAPL --report-only

Mirrors poc/estimate_systematic.py: resumable (dates already on disk are
skipped), parallel over (name, date) pairs, and stored in a drawer that records
WHICH systematic fit it was conditioned on.

--------------------------------------------------------------------------
The one modelling decision this script forces you to make
--------------------------------------------------------------------------
pmle_kimyirisk_idiosyncratic takes the six systematic parameters as plain
floats. They enter the likelihood as known constants with zero uncertainty, so
whatever is wrong or arbitrary upstream passes through silently. That matters
here more than it usually would, because of what the rolling fits showed:

  dSIGMA        identified in a 252-day window (252 observations)
  dLAMB         identified by location (2020: 26.66 vs 2024: 4.15)
  dPPROB        partial
  dETA1/dETA2   NOT identified in a window - a year holds ~4 jumps per side,
                and three prior settings (20/20, 25/50, 50/25) each came back
                as they were set
  dALPHA        NOT identified in a window - half-life 19 years

gamma_i is the parameter that suffers. In the likelihood it appears ONLY as
eta1/gamma_i and eta2/gamma_i, so it is the name's jump scale RELATIVE to the
systematic one. Anchor eta on a prior and gamma_i measures the name against
that prior rather than against a measured quantity.

--anchor chooses what to condition on:

  rolling  the systematic estimate for the SAME valuation date. Reproduces the
           paper's construction and lets the systematic state move with the
           regime, at the cost of carrying the window's prior-driven alpha and
           etas into every name.
  full     the full-sample calibration (Library.RiskEngineKimYi2025.FULL_SAMPLE)
           held fixed across all dates. alpha and the etas are then measured
           quantities, but the systematic state no longer varies, so a name
           fitted in 2009 sees the same systematic volatility as one fitted in
           2017.
  hybrid   sigma, lambda and pprob from the rolling fit (the ones a window can
           see); alpha, eta1 and eta2 from the full sample (the ones it cannot).
           Defensible on the evidence and the default, but it is a composite -
           say so wherever the numbers are reported.
"""
import argparse
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
multiprocessing.set_start_method(
    os.environ.get("JGL_MP_START", "forkserver"), force=True)
from concurrent.futures import (                            # noqa: E402
    ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED,
)
from concurrent.futures.process import BrokenProcessPool     # noqa: E402

from Library.RiskEngineKimYi2025 import FULL_SAMPLE, STORE_SUFFIX  # noqa: E402
from Library.PosteriorSummary import (                       # noqa: E402
    CI_WIDTH_TO_SD, CI_CONVENTION, CI_PROB,
)
from Library.TableHeatmap import (                           # noqa: E402
    render as heat, legend as heat_legend,
)
from Library.DataAccess import (                             # noqa: E402
    get_aligned_price_panel, get_pmle_params, pmle_params_exists,
    save_pmle_params, available_pmle_dates, PMLE_DIR,
)
from Scripts.run_pmle_kimyi2025 import (                     # noqa: E402
    pmle_kimyirisk_idiosyncratic_helper, assemble_idiosyncratic_params,
    SYSTEMATIC_PARAMS,
)
# Reused rather than re-declared so the two drivers cannot drift apart on the
# things that must match: how a drawer is named, and how work is sized.
from poc.estimate_systematic import (                        # noqa: E402
    SYSTEMATIC_ID, DATE_FMT, LOOKBACK, BASE_DAYS, SEED, N_MC_PATHS,
    priors_digest, store_id, valuation_dates,
    _init_child, default_workers, _pool_kwargs, POOL_CHUNK, _drain,
)
from Library.RiskEngineKimYi2025 import (                    # noqa: E402
    SYSTEMATIC_PRIOR_SETS as PRIOR_SETS,
)

COLOR = None

IDIO_PARAMS = ["dBETAI", "dKAPPAI", "dGAMMAI", "dRHOIX", "dMUI"]

# The priors the idiosyncratic block is actually sampled under
# (RiskEngineKimYi2025.pmle_kimyirisk_idiosyncratic, lines ~457-468).
# dRHOIX is reported as 2*Beta(5,2) - 1, so its prior moments are the Beta's
# scaled by 2 and shifted - NOT the Beta's own.
IDIO_PRIORS = {
    "dBETAI":  ("Gamma(3,1)",        3.0000, 1.7321),
    "dKAPPAI": ("Gamma(2,1)",        2.0000, 1.4142),
    "dGAMMAI": ("Gamma(3,1)",        3.0000, 1.7321),
    "dRHOIX":  ("2*Beta(5,2)-1",     0.4286, 0.3196),
    "dMUI":    ("Normal(0,1)",       0.0000, 1.0000),
}

BEG = "20070101"
END = "20260831"


# ---------------------------------------------------------------------------
# where results go
# ---------------------------------------------------------------------------
def name_store_id(name, anchor, tag, priors):
    """Drawer for one name's estimates.

    The systematic conditioning is part of the ESTIMATE, not context: the same
    returns fitted against a different systematic anchor give different
    loadings. Encoding anchor and prior digest in the drawer means a rerun
    after changing either one cannot be mistaken for a resume of the old one -
    the same trap that cost a systematic run.
    """
    if anchor == "full":
        return "%s__full" % name
    suffix = "" if tag == "paper" else "%s_%s" % (STORE_SUFFIX[tag],
                                                  priors_digest(priors))
    return "%s__%s%s" % (name, anchor, suffix)


def full_sample_series():
    """FULL_SAMPLE as the pd.Series assemble_idiosyncratic_params expects.

    The intervals are zero-width on purpose. A held constant has no posterior,
    and reporting a fabricated interval for one would be worse than reporting
    none - it would make the anchor look estimated at this date when it was not.
    """
    row = {}
    key = {"dALPHA": "alpha_rv", "dSIGMA": "sigma", "dPPROB": "pprob_rv",
           "dLAMB": "lamb", "dETA1": "eta1", "dETA2": "eta2"}
    for p in SYSTEMATIC_PARAMS:
        v = float(FULL_SAMPLE[key[p]])
        row[p] = v
        row[p + "_CI_LOWER"] = v
        row[p + "_CI_UPPER"] = v
    return pd.Series(row)


def systematic_for(dt, sys_store, anchor):
    """The six systematic values to condition on at this date, plus the series
    that will be written into the name's CSV row."""
    if anchor == "full":
        s = full_sample_series()
        return {p: float(s[p]) for p in SYSTEMATIC_PARAMS}, s

    s = get_pmle_params(dt, sys_store)          # raises if not estimated
    if anchor == "rolling":
        return {p: float(s[p]) for p in SYSTEMATIC_PARAMS}, s

    # hybrid: window-identified from the window, the rest from the full sample
    fs = full_sample_series()
    s = s.copy()
    for p in ("dALPHA", "dETA1", "dETA2"):
        s[p] = fs[p]
        s[p + "_CI_LOWER"] = fs[p]
        s[p + "_CI_UPPER"] = fs[p]
    return {p: float(s[p]) for p in SYSTEMATIC_PARAMS}, s


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(names, dates, sys_store, anchor, tag, priors, workers=None,
        force=False):
    panel, report = get_aligned_price_panel([SYSTEMATIC_ID] + names,
                                            reference=SYSTEMATIC_ID)
    rets = panel.pct_change().dropna()

    sys_dates = set(available_pmle_dates(sys_store)) if anchor != "full" else None

    args, skipped = [], []
    for name in names:
        drawer = name_store_id(name, anchor, tag, priors)
        # A name's first usable date is its own, not the panel's: a 2021 listing
        # has no 2009 window however much SPX history sits beside it.
        have = rets[name].dropna()
        if len(have) < LOOKBACK:
            skipped.append((name, "only %d returns in total" % len(have)))
            continue
        first_ok = have.index[LOOKBACK - 1]

        for dt in dates:
            d = pd.to_datetime(dt, format=DATE_FMT)
            if d < first_ok:
                continue
            if sys_dates is not None and dt not in sys_dates:
                continue                      # no systematic fit to condition on
            if not force and pmle_params_exists(dt, drawer):
                continue
            rv = rets.loc[rets.index <= d, name].dropna()
            if len(rv) < LOOKBACK:
                continue
            args.append((name, drawer, dt,
                         rv.iloc[-LOOKBACK:].to_numpy()))

    for name, why in skipped:
        print("  SKIP %-8s %s (need %d)" % (name, why, LOOKBACK))
    print("  %d name(s) x %d date(s) requested -> %d fit(s) to run."
          % (len(names), len(dates), len(args)))
    if not args:
        return

    # Resolve the systematic anchor in the PARENT, once per date. Doing it in
    # the worker would re-read the same CSV for every name.
    cache, tasks = {}, []
    for name, drawer, dt, rv in args:
        if dt not in cache:
            try:
                cache[dt] = systematic_for(dt, sys_store, anchor)
            except Exception as exc:                          # noqa: BLE001
                cache[dt] = None
                print("    no systematic fit for %s (%s) - dates skipped"
                      % (dt, type(exc).__name__))
        if cache[dt] is None:
            continue
        params_sys, sys_series = cache[dt]
        tasks.append(((dt, params_sys, rv, np.array(1 / BASE_DAYS), SEED,
                       N_MC_PATHS, drawer), sys_series, name))

    manifest_written = set()
    for _, _, name in tasks:
        drawer = name_store_id(name, anchor, tag, priors)
        if drawer in manifest_written:
            continue
        write_manifest(drawer, name, anchor, tag, priors)
        manifest_written.add(drawer)

    workers = workers or default_workers()
    print("  %d workers x 4 chains = %d concurrent samplers on %d cores"
          % (workers, workers * 4, os.cpu_count() or 0))

    t0 = time.perf_counter()
    done = failed = 0
    chunk = POOL_CHUNK if POOL_CHUNK > 0 else len(tasks)
    try:
        # One pool per CHUNK fits, not one pool for the whole run. Tearing the
        # pool down returns every worker's memory to the OS, which is what
        # max_tasks_per_child would do on Python 3.11+ and does not do here
        # (this environment is 3.10, where the argument is silently absent).
        for start in range(0, len(tasks), chunk):
            batch = tasks[start:start + chunk]
            with ProcessPoolExecutor(**_pool_kwargs(workers)) as ex:
                futures = {ex.submit(pmle_kimyirisk_idiosyncratic_helper, t[0]): t
                           for t in batch}

                def _on_done(fut, _f=futures):
                    nonlocal done, failed
                    _, sys_series, name = _f[fut]
                    try:
                        dt, drawer, results = fut.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:                  # noqa: BLE001
                        failed += 1
                        print("    FAILED  %-8s %s  %s: %s"
                              % (name, _f[fut][0][0],
                                 type(exc).__name__, exc), flush=True)
                        return
                    save_pmle_params(
                        dt, drawer,
                        assemble_idiosyncratic_params(results, sys_series))
                    done += 1
                    if done % 10 == 0 or done == len(tasks):
                        el = time.perf_counter() - t0
                        print("    %4d/%d  %.1fs elapsed, ~%.1fs remaining"
                              % (done, len(tasks), el,
                                 el / done * (len(tasks) - done)), flush=True)

                _drain(set(futures), _on_done, "fits")
            if start + chunk < len(tasks):
                print("    -- pool recycled after %d fits --" % done, flush=True)
    except BrokenProcessPool:
        print("\n  A worker died. Late in a long run this is usually memory")
        print("  accumulating in a reused worker rather than a bug in the model.")
        print("  %d fit(s) are safely on disk and a rerun skips them." % done)
        print()
        print("  Rerun the same command - it resumes. If it dies again, recycle")
        print("  the pool more often or use fewer workers:")
        print("    JGL_POOL_CHUNK=20 <same command>")
        print("    <same command> --workers %d" % max(1, (workers or 2) // 2))
        print()
        print("  To confirm the OOM killer (dmesg needs root on Ubuntu):")
        print("    sudo dmesg -T | grep -i 'killed process' | tail")
        print("    journalctl -k --since '3 hours ago' | grep -i 'killed process'")
        print("    grep -i 'killed process' /var/log/kern.log | tail")
        raise SystemExit(1)

    if failed:
        print("\n  %d fit(s) failed and were skipped; rerun to retry them."
              % failed)


def write_manifest(drawer, name, anchor, tag, priors):
    folder = os.path.join(PMLE_DIR, drawer)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "_conditioning.json"), "w") as fh:
        json.dump({"name": name, "anchor": anchor,
                   "systematic_priors": tag,
                   "systematic_digest": priors_digest(priors),
                   "lookback": LOOKBACK, "seed": int(SEED),
                   "note": "idiosyncratic parameters are fitted with the six "
                           "systematic values held as constants; gamma_i is a "
                           "jump scale RELATIVE to the systematic eta"},
                  fh, indent=2)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def load_series(name, anchor, tag, priors):
    drawer = name_store_id(name, anchor, tag, priors)
    rows = []
    for dt in available_pmle_dates(drawer):
        s = get_pmle_params(dt, drawer)
        row = {"date": pd.to_datetime(dt, format=DATE_FMT)}
        for k in IDIO_PARAMS + SYSTEMATIC_PARAMS:
            if k in s:
                row[k] = float(s[k])
            lo, hi = k + "_CI_LOWER", k + "_CI_UPPER"
            if lo in s and hi in s:
                row[k + "_W"] = float(s[hi]) - float(s[lo])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def report(name, df):
    """Same treatment the systematic report gives SPX, per name."""
    print("\n" + "=" * 72)
    print("%s :: %s to %s   (%d valuation dates)"
          % (name, df.index.min().date(), df.index.max().date(), len(df)))
    print("=" * 72)
    print("  credible intervals: %.0f%% equal-tailed (%s)"
          % (100 * CI_PROB, CI_CONVENTION))

    # b_i is a FUNCTION of four parameters, so it belongs in the tables as a
    # column of its own - reading beta_i alone understates the systematic
    # loading whenever kappa_i rho_iX is material.
    if all(c in df for c in ("dBETAI", "dKAPPAI", "dRHOIX", "dSIGMA")):
        df = df.copy()
        df["b_i"] = df["dBETAI"] + df["dKAPPAI"] * df["dRHOIX"] / df["dSIGMA"]

    cols = [c for c in IDIO_PARAMS + ["b_i"] if c in df]
    print("\nDistribution across valuation dates:")
    print(df[cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
            .round(4).to_string())

    if len(df) < 2:
        print("\nStability - needs at least 2 valuation dates.")
    else:
        print("\nStability - coefficient of variation (sd / |mean|):")
        cv = (df[cols].std() / df[cols].mean().abs()).sort_values()
        for k, v in cv.items():
            note = "" if v < 0.25 else "   <-- varies a lot across windows"
            print("   %-9s %6.3f%s" % (k, v, note))
        print("\n   A low CV is evidence of identification only if the")
        print("   parameter is identified. A prior-driven one is stable")
        print("   because its prior is - read this with the table below.")

    for stat in ("mean", "median"):
        t = getattr(df[cols].groupby(df.index.year), stat)().round(4)
        t.index = [str(i) for i in t.index]
        print("\nBy year (%s):" % stat)
        print(heat(t, decimals=4, color=COLOR))
    print(heat_legend(color=COLOR))

    print("\nIdentification over time - share of valuation dates where the")
    print("posterior is narrower than the prior (ratio < 0.70):")
    for k in IDIO_PARAMS:
        w = k + "_W"
        if k not in IDIO_PRIORS or w not in df:
            continue
        label, pmean, psd = IDIO_PRIORS[k]
        ratio = (df[w] * CI_WIDTH_TO_SD) / psd
        shift = (df[k].median() - pmean) / psd
        share = 100.0 * float((ratio < 0.70).mean())
        print("   %-8s %-14s %5.1f%%   median ratio %.2f   shift %+.2f sd"
              % (k, label, share, ratio.median(), shift))
    print("\n   0% means never identified at any date - that column is a")
    print("   series of priors. A large |shift| is decisive evidence of data")
    print("   dominance regardless of width: a prior cannot drag a posterior")
    print("   several sd away from itself.")

    if "dGAMMAI" in df:
        print("\nGap loading dGAMMAI at known stress episodes:")
        for lab, (a, b_) in {
            "GFC 2008H2":   ("2008-07-01", "2008-12-31"),
            "Euro 2011":    ("2011-07-01", "2011-12-31"),
            "Covid 2020":   ("2020-02-01", "2020-06-30"),
            "SVB 2023":     ("2023-03-01", "2023-06-30"),
            "Tariffs 2025": ("2025-04-01", "2025-07-31"),
        }.items():
            wnd = df.loc[a:b_, "dGAMMAI"]
            if len(wnd):
                print("   %-14s median %6.3f   (full-sample median %6.3f)"
                      % (lab, wnd.median(), df["dGAMMAI"].median()))
        print("\n   gamma_i enters the likelihood only as eta1/gamma_i and")
        print("   eta2/gamma_i, so it is a jump scale RELATIVE to the")
        print("   systematic eta, not an absolute one.")

    if "b_i" in df and "dGAMMAI" in df:
        r = (df["dGAMMAI"] / df["b_i"]).median()
        print("\n  gamma_i / b_i median %.3f - %s"
              % (r, "gaps HARDER than ordinary days imply" if r > 1
                 else "gaps SOFTER than ordinary days imply"))
        print("  This ratio crossing 1.0 across names is what a regression")
        print("  beta cannot reproduce; see poc/product_curve.py.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True,
                    help="comma-separated tickers, or @file with one per line")
    ap.add_argument("--beg", default=BEG)
    ap.add_argument("--end", default=END)
    ap.add_argument("--step", type=int, default=21,
                    help="business days between valuation dates (21 = monthly)")
    ap.add_argument("--anchor", choices=("hybrid", "rolling", "full"),
                    default="hybrid",
                    help="which systematic values to hold constant. See the "
                         "module docstring - this is a modelling choice, not a "
                         "detail.")
    ap.add_argument("--priors", choices=tuple(PRIOR_SETS), default="gaps",
                    help="which systematic run to condition on (ignored for "
                         "--anchor full)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--color", dest="color", action="store_true", default=None,
                    help="force heat shading on (default: on for a terminal)")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="plain numbers, no shading")
    ap.add_argument("--force", "--overwrite", dest="force", action="store_true",
                    help="re-fit every (name, date) even if already on disk, "
                         "overwriting in place")
    a = ap.parse_args()

    global COLOR
    COLOR = a.color

    if a.names.startswith("@"):
        with open(a.names[1:]) as fh:
            names = [ln.strip().upper() for ln in fh if ln.strip()
                     and not ln.startswith("#")]
    else:
        names = [n.strip().upper() for n in a.names.split(",") if n.strip()]

    priors = PRIOR_SETS[a.priors]
    sys_store = store_id(a.priors, priors)

    print("=" * 72)
    print("Step 1b :: idiosyncratic parameters, conditional on the systematic fit")
    print("=" * 72)
    print("  names   : %s" % ", ".join(names))
    print("  window  : %s -> %s every %d business days, %d-day lookback"
          % (a.beg, a.end, a.step, LOOKBACK))
    print("  anchor  : %s" % a.anchor)
    if a.anchor == "hybrid":
        print("    dSIGMA/dLAMB/dPPROB from the rolling fit in %s" % sys_store)
        print("    dALPHA/dETA1/dETA2 from FULL_SAMPLE - a window cannot")
        print("    identify them, so taking them from one asserts a prior")
        print("    NOTE this is a composite; report it as such")
    elif a.anchor == "rolling":
        print("    all six from %s" % sys_store)
        print("    NOTE dALPHA, dETA1 and dETA2 there are prior-driven and")
        print("    pass into every name as if known exactly")
    else:
        print("    all six held at FULL_SAMPLE for every date - identified,")
        print("    but the systematic state no longer moves with the regime")

    if not a.report_only:
        run(names, valuation_dates(a.beg, a.end, a.step),
            sys_store, a.anchor, a.priors, priors, workers=a.workers,
            force=a.force)

    for name in names:
        df = load_series(name, a.anchor, a.priors, priors)
        if not len(df):
            # "nothing estimated" on its own sends you looking for a bug that
            # is not there. Say WHICH drawer was empty, whether the systematic
            # fit it would need exists, and the command that fills it.
            drawer = name_store_id(name, a.anchor, a.priors, priors)
            print("\n%s: nothing on disk in drawer %s" % (name, drawer))
            if a.anchor != "full":
                have = len(available_pmle_dates(sys_store))
                print("   systematic %s: %d date(s) %s"
                      % (sys_store, have,
                         "- run estimate_systematic.py first" if not have else "ready"))
            print("   this run was --report-only, which never fits anything.")
            print("   to create it:")
            print("     ./run_bg.sh poc/estimate_idiosyncratic.py --names %s"
                  " --anchor %s --priors %s" % (name, a.anchor, a.priors))
            if a.anchor == "hybrid":
                print("   NOTE --anchor hybrid takes dALPHA/dETA1/dETA2 from")
                print("   FULL_SAMPLE, so --priors %s only selects sigma, lambda"
                      % a.priors)
                print("   and pprob. For all six from that run, use --anchor rolling.")
            continue
        report(name, df)
        out = os.path.join(_REPO_ROOT, "poc",
                           "idio_params_%s.csv"
                           % name_store_id(name, a.anchor, a.priors, priors))
        df.to_csv(out)
        print("\n  written to %s" % os.path.relpath(out, _REPO_ROOT))


def _exit_now(code=0):
    """Leave without waiting for interpreter shutdown.

    A completed run hung for an hour AFTER main() returned: the last report
    printed, the CSV was written, and the process would not exit. Nothing was
    still computing. Interpreter shutdown has to join the forkserver process
    and whatever native threads nutpie/numba left behind, and after dozens of
    pool create/destroy cycles that can wedge - a hang with the work already
    finished and safely on disk.

    Every output is written and flushed before this point, so there is nothing
    for a clean shutdown to do that matters. os._exit skips atexit handlers and
    the GC entirely and returns control immediately.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


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
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
        _code = 0
    except SystemExit as _e:                # the stall watchdog exits 3
        _code = _e.code if isinstance(_e.code, int) else (0 if _e.code is None else 1)
    _exit_now(_code)
