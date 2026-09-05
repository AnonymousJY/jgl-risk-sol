"""Where the estimation time actually goes, and whether it is buying anything.

    python poc/bench_sampler.py --date 20250630
    python poc/bench_sampler.py --date 20250630 --only draws
    python poc/bench_sampler.py --date 20250630 --draws 500,1000,2000,10000

Three questions, measured rather than argued:

  DRAWS   N_MC_PATHS is 10,000 per chain, so 40,000 post-tuning draws for a
          5-6 parameter posterior. The 95% equal-tailed intervals this repo
          reports need enough TAIL ess to pin the 2.5%/97.5% quantiles - order
          1,000 total is comfortable. If tail ess at 2,000 draws is already in
          the thousands, the other 8,000 draws are wall time bought for nothing.

  LAYOUT  Each fit runs chains=4, cores=4, so default_workers() is cpu_count//4.
          But a 252-point gradient is latency-bound, not compute-bound, and four
          threads may not saturate four cores. The alternative is cores=1 with
          workers=cpu_count: chains sequential inside each worker, parallelism
          from the outer pool. What matters is not seconds per fit - cores=1 is
          obviously slower per fit - but FITS PER HOUR ACROSS THE MACHINE:

              throughput(c) = (cpu_count // c) / fit_seconds(c)

  BACKEND ~1ms per gradient evaluation on 252 points is slow, which points at
          the CustomDist logp not compiling well. Compares whatever NUTS
          backends are installed.

Nothing here writes to the parameter store; it is pure measurement.
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

import pymc as pm                                              # noqa: E402
from Library.DataAccess import get_price_panel                 # noqa: E402
from Library.RiskEngineKimYi2025 import (                      # noqa: E402
    pmle_kimyirisk_systematic, SYSTEMATIC_PRIOR_SETS,
)

SYSTEMATIC_ID = "^SPX"
LOOKBACK = 252
BASE_DAYS = 252
SEED = np.uint64(20240114)

_REAL_SAMPLE = pm.sample
_CAPTURED = []
_OVERRIDE = {}


def _patched_sample(draws=None, **kw):
    """Intercept the single pm.sample call inside the estimator.

    Monkeypatching rather than duplicating the model definition: the point is
    to time THE MODEL THIS REPO FITS, and a copy would drift from it silently.
    """
    if "draws" in _OVERRIDE:
        draws = _OVERRIDE["draws"]
    for k in ("chains", "cores", "tune", "nuts_sampler"):
        if k in _OVERRIDE:
            kw[k] = _OVERRIDE[k]
    idata = _REAL_SAMPLE(draws, **kw)
    _CAPTURED.append(idata)
    return idata


pm.sample = _patched_sample


def returns_for(date):
    px = get_price_panel([SYSTEMATIC_ID])
    r = px.pct_change().dropna()
    r = r.loc[r.index <= pd.to_datetime(date, format="%Y%m%d"), SYSTEMATIC_ID]
    if len(r) < LOOKBACK:
        raise SystemExit("only %d returns before %s" % (len(r), date))
    return r.iloc[-LOOKBACK:].to_numpy()


def one_fit(rv, priors, **override):
    _OVERRIDE.clear(); _OVERRIDE.update(override)
    _CAPTURED.clear()
    t0 = time.perf_counter()
    pmle_kimyirisk_systematic(
        priors=priors, sys_returns=rv, delta_t=np.array(1 / BASE_DAYS),
        seed_number=SEED, n_mc_paths=override.get("draws", 10_000))
    el = time.perf_counter() - t0
    return el, (_CAPTURED[0] if _CAPTURED else None)


def tail_ess(idata):
    try:
        import arviz as az
    except ImportError:
        return None
    v = ["sigma", "alpha_rv", "pprob_rv", "lamb", "eta1", "eta2"]
    v = [x for x in v if x in idata.posterior]
    e = az.ess(idata, var_names=v, method="tail")
    return {k: float(e[k].values) for k in v}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="20250630")
    ap.add_argument("--priors", choices=tuple(SYSTEMATIC_PRIOR_SETS), default="skew")
    ap.add_argument("--draws", default="500,1000,2000,5000,10000")
    ap.add_argument("--cores", default="1,2,4")
    ap.add_argument("--backends", default="nutpie")
    ap.add_argument("--only", choices=("draws", "layout", "backend"), default=None)
    a = ap.parse_args()

    rv = returns_for(a.date)
    priors = SYSTEMATIC_PRIOR_SETS[a.priors]
    ncpu = os.cpu_count() or 1
    print("=" * 76)
    print("sampler benchmark  |  %s  %s  priors=%s  %d cpus"
          % (SYSTEMATIC_ID, a.date, a.priors, ncpu))
    print("=" * 76)

    if a.only in (None, "draws"):
        print("\nDRAWS  (chains=4, cores=4)   min tail ess across the six parameters")
        print("  draws    total    seconds    min tail ess    ess/sec   ess per 1000 draws")
        print("  " + "-" * 72)
        base = None
        for d in [int(x) for x in a.draws.split(",")]:
            el, idata = one_fit(rv, priors, draws=d, chains=4, cores=4)
            ess = tail_ess(idata)
            m = min(ess.values()) if ess else float("nan")
            if base is None:
                base = (el, m)
            print("  %6d  %7d  %9.1f  %14.0f  %9.1f  %14.0f"
                  % (d, 4 * d, el, m, m / el if el else 0, 1000 * m / (4 * d)))
        print("\n  Read: pick the smallest draws whose min tail ess clears ~1000.")
        print("  Everything above that is wall time buying precision you discard")
        print("  when you report a 95%% interval.")

    if a.only in (None, "layout"):
        print("\nLAYOUT  what matters is fits/hour ACROSS THE MACHINE, not per fit")
        print("  chains cores  workers  sec/fit   fits/hour   vs cores=4")
        print("  " + "-" * 60)
        ref = None
        for c in [int(x) for x in a.cores.split(",")]:
            el, _ = one_fit(rv, priors, draws=2000, chains=4, cores=c)
            w = max(1, ncpu // c)
            thr = 3600.0 * w / el
            if c == 4:
                ref = thr
            print("  %6d %5d  %7d  %7.1f  %10.1f   %s"
                  % (4, c, w, el, thr,
                     "reference" if c == 4 else
                     ("%+.0f%%" % (100 * (thr / ref - 1)) if ref else "-")))
        print("\n  cores=1 is slower per fit and can still win on throughput,")
        print("  because the outer pool then runs %d fits at once instead of %d."
              % (ncpu, max(1, ncpu // 4)))

    if a.only in (None, "backend"):
        print("\nBACKEND  (draws=2000, chains=4, cores=4)")
        print("  backend     seconds   grad evals   us/grad")
        print("  " + "-" * 48)
        for b in a.backends.split(","):
            try:
                el, idata = one_fit(rv, priors, draws=2000, chains=4,
                                    cores=4, nuts_sampler=b)
            except Exception as exc:                             # noqa: BLE001
                print("  %-10s  unavailable: %s" % (b, type(exc).__name__))
                continue
            n = 4 * (2000 + 1000)
            print("  %-10s  %7.1f  %11d  %8.1f" % (b, el, n, 1e6 * el / n))
        print("\n  ~1000 us/grad on a 252-point likelihood means the logp is not")
        print("  compiling to anything tight. Under ~100 us/grad it is fine and")
        print("  the only remaining lever is draws.")


def _exit_now(code=0):
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:                                            # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
        _c = 0
    except SystemExit as _e:
        _c = _e.code if isinstance(_e.code, int) else (0 if _e.code is None else 1)
    _exit_now(_c)
