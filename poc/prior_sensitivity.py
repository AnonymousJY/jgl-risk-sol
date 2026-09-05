"""
Does the data know about jump asymmetry, or is it asserted by the priors?

    python poc/prior_sensitivity.py --dates 20090630,20170630,20200630

THE QUESTION. The systematic block has TWO independent sources of jump
asymmetry that compete to explain the same feature of the data:

    pprob        how OFTEN a jump is upward
    eta1 / eta2  how BIG up jumps are relative to down jumps

Left-skewed returns can be fitted as "mostly down jumps" (small pprob) or as
"fewer but larger down jumps" (eta2 < eta1). When both are weakly identified,
the priors silently pick between them - and the defaults pick decisively:

    eta1 ~ Gamma(50,1)  mean up jump   1/50 = 2%
    eta2 ~ Gamma(25,1)  mean down jump 1/25 = 4%

That asserts down jumps are twice the size of up jumps before any data is seen.
Because both come back with posterior sd within a few percent of prior sd, the
assertion survives into the posterior untouched. So the model's negative jump
skew - the feature that makes it equity-plausible and that every gap-risk claim
rests on - may be an assumption rather than an estimate.

THE ARMS.

    A  baseline          eta1 G(50,1)  eta2 G(25,1)   pprob Beta(5,2)
    B  symmetric eta     eta1 G(35,1)  eta2 G(35,1)   pprob Beta(5,2)
    C  mirrored pprob    eta1 G(50,1)  eta2 G(25,1)   pprob Beta(2,5)
    D  both              eta1 G(35,1)  eta2 G(35,1)   pprob Beta(2,5)

Run B is the one that matters. A shared PRIOR is not a shared PARAMETER: eta1
and eta2 remain free, so the model keeps every ability to find asymmetry. It
has merely stopped being told to expect it.

    eta1 -> ~50 and eta2 -> ~25 on their own   the data knows. The defaults
                                               happen to be right, and the jump
                                               skew is an estimate.
    both stay near 35                          the asymmetry was entirely prior.
                                               Every skew-dependent claim in the
                                               business documents needs restating.

Run C is the companion test for pprob and is sharper than the sd ratio, because
Beta(5,2) and Beta(2,5) have IDENTICAL sd (0.160) - only their location differs.
A width-based diagnostic cannot move; where the posterior lands can.

    posterior -> ~0.29    it followed the prior. Prior-driven.
    posterior -> ~0.45+   pulled UP from the new prior just as it was pulled
                          DOWN from the old one, i.e. the data is overruling
                          both toward the same place.

That last case is live: the daily estimates put pprob 1.1 to 2.2 prior sd BELOW
its prior mean of 0.714, and moving with the regime. The sd ratio calls pprob
prior-driven; its location says otherwise. Beta support is bounded, so the
likelihood can shift the mass without much room to narrow it - which is a real
limitation of a width-only diagnostic, and the reason this script exists.

Run the arms on several dates spanning regimes. If the verdict is regime
dependent - the data separating the etas in 2009 but not in 2017 - that is
itself the finding, and it would say identification comes from crisis
observations rather than from sample length.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.DataAccess import get_price_panel                     # noqa: E402
from Library.PosteriorSummary import CI_WIDTH_TO_SD                # noqa: E402
from Library.RiskEngineKimYi2025 import (                          # noqa: E402
    pmle_kimyirisk_systematic, SYSTEMATIC_PRIORS, prior_moments,
)

SYSTEMATIC_ID = "^SPX"
LOOKBACK = 252
BASE_DAYS = 252
DATE_FMT = "%Y%m%d"

G = lambda a, b: ("Gamma", {"alpha": a, "beta": b})                # noqa: E731
Bt = lambda a, b: ("Beta", {"alpha": a, "beta": b})                # noqa: E731
U = lambda lo, hi: ("Uniform", {"lower": lo, "upper": hi})         # noqa: E731

ARMS = {
    "A baseline":       {},
    "B symmetric eta":  {"eta1": G(35.0, 1.0), "eta2": G(35.0, 1.0)},
    "C mirrored pprob": {"pprob_rv": Bt(2.0, 5.0)},
    "D both":           {"eta1": G(35.0, 1.0), "eta2": G(35.0, 1.0),
                         "pprob_rv": Bt(2.0, 5.0)},
    # E and F answer "what if we simply FORBID pprob above 0.5".
    #
    # F is the reference: flat on the whole interval, so the posterior is the
    # likelihood alone with no prior shape at all. Wherever it lands is what
    # the data actually wants.
    #
    # E imposes the constraint with a flat prior on (0, 0.5). If the likelihood
    # prefers pprob above 0.5, the posterior cannot follow it and instead PILES
    # UP AGAINST THE BOUNDARY - posterior mean pushed toward 0.5 and the upper
    # credible bound sitting on it. That pile-up is the diagnostic: it means the
    # constraint is binding and fighting the data rather than encoding
    # knowledge. A constraint that is NOT binding leaves the posterior interior
    # and costs nothing.
    "E pprob < 0.5":    {"pprob_rv": U(0.0, 0.5)},
    "F pprob flat":     {"pprob_rv": U(0.0, 1.0)},
    # G asserts NO asymmetry anywhere. The etas share a prior so nothing tells
    # the model up jumps and down jumps differ in size, and pprob is flat so
    # nothing tells it they differ in frequency. Jump asymmetry can then only
    # come from the data, and only through one of two channels:
    #
    #   eta1 pulling away from eta2   -> asymmetry is in jump SIZE
    #   pprob moving away from 0.5    -> asymmetry is in jump DIRECTION
    #
    # If neither moves, the model finds no jump asymmetry at all and the
    # negative skew was entirely prior. Run this on the FULL SAMPLE
    # (--full-sample): a 252-day window holds ~15 jumps, roughly 7 per side,
    # which cannot separate two exponential rates. The full sample holds ~193.
    "G no asymmetry":   {"eta1": G(35.0, 1.0), "eta2": G(35.0, 1.0),
                         "pprob_rv": U(0.0, 1.0)},
}

# reported name -> prior key
KEYS = [("dSIGMA", "sigma"), ("dALPHA", "alpha_rv"), ("dPPROB", "pprob_rv"),
        ("dLAMB", "lamb"), ("dETA1", "eta1"), ("dETA2", "eta2")]


def returns_for(date, full_sample=False, beg="20070101"):
    """The 252-day window ending at `date`, or the whole sample.

    Jump-size asymmetry is limited by how many jumps are observed, not by how
    long the series is in calendar terms. A window holds ~15; the full sample
    holds ~193. Any question about eta1 versus eta2 has to be asked of the
    latter.
    """
    panel = get_price_panel([SYSTEMATIC_ID])
    r = panel.pct_change().dropna()
    if full_sample:
        sel = r.loc[(r.index >= pd.to_datetime(beg, format=DATE_FMT))
                    & (r.index <= pd.to_datetime(date, format=DATE_FMT))]
        return sel[SYSTEMATIC_ID].to_numpy()
    upto = r.loc[r.index <= pd.to_datetime(date, format=DATE_FMT)]
    if len(upto) < LOOKBACK:
        raise SystemExit("only %d returns before %s, need %d"
                         % (len(upto), date, LOOKBACK))
    return upto[SYSTEMATIC_ID].iloc[-LOOKBACK:].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="20090630,20170630,20200630",
                    help="comma-separated YYYYMMDD, ideally spanning regimes")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated arm names to run")
    ap.add_argument("--full-sample", action="store_true",
                    help="fit each arm on the WHOLE series ending at each date "
                         "rather than a 252-day window. Required for any "
                         "question about eta1 vs eta2, which is limited by "
                         "jump count (~15 in a window, ~193 in the sample).")
    a = ap.parse_args()

    dates = a.dates.split(",")
    arms = [k for k in ARMS if k in a.arms.split(",")] or list(ARMS)

    print("=" * 78)
    print("Prior sensitivity of the systematic block")
    print("=" * 78)
    for nm in arms:
        pr = dict(SYSTEMATIC_PRIORS); pr.update(ARMS[nm])
        bits = []
        for lbl, k in (("eta1", "eta1"), ("eta2", "eta2"), ("pprob", "pprob_rv")):
            m, s = prior_moments(pr[k])
            bits.append("%s~%.1f+-%.2f" % (lbl, m, s))
        print("  %-18s %s" % (nm, "  ".join(bits)))
    print()

    rows = []
    if a.full_sample:
        print("  FULL-SAMPLE mode: each arm fits the whole series to that date")
        print()
    for dt in dates:
        rv = returns_for(dt, full_sample=a.full_sample)
        print("  %s: %d returns" % (dt, len(rv)))
        for nm in arms:
            t0 = time.perf_counter()
            res = pmle_kimyirisk_systematic(
                sys_returns=rv, delta_t=np.array(1 / BASE_DAYS),
                n_mc_paths=10_000, priors=ARMS[nm] or None)
            pr = dict(SYSTEMATIC_PRIORS); pr.update(ARMS[nm])
            row = {"date": dt, "arm": nm, "secs": time.perf_counter() - t0}
            for out_k, pk in KEYS:
                r = res[out_k]
                post_sd = (r.dCI_UPPER - r.dCI_LOWER) * CI_WIDTH_TO_SD
                pm_, ps_ = prior_moments(pr[pk])
                row[out_k] = r.dMEAN
                row[out_k + "_ratio"] = post_sd / ps_
                row[out_k + "_shift"] = (r.dMEAN - pm_) / ps_
            rows.append(row)
            print("  fitted %s  %-18s %.0fs" % (dt, nm, row["secs"]))

    df = pd.DataFrame(rows)
    out = os.path.join(_REPO_ROOT, "poc", "prior_sensitivity.csv")
    df.to_csv(out, index=False)

    print()
    for dt in dates:
        d = df[df.date == dt]
        print("-" * 78)
        print("%s   posterior mean  [ratio = post sd / prior sd, "
              "shift = (post mean - prior mean) / prior sd ]" % dt)
        print("  %-18s %18s %18s %14s" % ("arm", "dETA1", "dETA2", "dPPROB"))
        for _, r in d.iterrows():
            print("  %-18s %7.2f r%.2f s%+.1f %7.2f r%.2f s%+.1f %5.3f r%.2f s%+.1f"
                  % (r["arm"],
                     r["dETA1"], r["dETA1_ratio"], r["dETA1_shift"],
                     r["dETA2"], r["dETA2_ratio"], r["dETA2_shift"],
                     r["dPPROB"], r["dPPROB_ratio"], r["dPPROB_shift"]))
        for arm, cap in (("E pprob < 0.5", 0.5),):
            e = d[d.arm == arm]
            if len(e):
                m = float(e["dPPROB"].iloc[0])
                f = d[d.arm == "F pprob flat"]
                ref = float(f["dPPROB"].iloc[0]) if len(f) else float("nan")
                room = (cap - m) / cap
                # What matters is how much unconstrained posterior MASS sits
                # above the cap, not whether the unconstrained MEAN exceeds it.
                # A posterior centred just below the cap can still have 40-50%
                # of its mass above it, and truncating removes all of that.
                # Comparing means alone reported "not binding" for exactly such
                # a case.
                fsd = float(f["dPPROB_ratio"].iloc[0]) * (1.0 / math.sqrt(12))
                mass = 0.5 * math.erfc((cap - ref) / (fsd * math.sqrt(2))) \
                    if fsd > 0 else float("nan")
                print("  -> arm F (flat, pure likelihood) wants %.4f, "
                      "with %.0f%% of its mass above %.2f" % (ref, 100 * mass, cap))
                print("     arm E (capped) gives %.4f -> the cap moves it %+.3f  %s"
                      % (m, m - ref,
                         "BINDING" if abs(m - ref) > 0.05 else "little effect"))

        b = d[d.arm == "B symmetric eta"]
        if len(b):
            e1, e2 = float(b["dETA1"].iloc[0]), float(b["dETA2"].iloc[0])
            gap = abs(e1 - e2)
            print("  -> arm B separation |eta1 - eta2| = %.2f   %s"
                  % (gap, "DATA SEPARATES THEM" if gap > 8 else
                     "no separation - the asymmetry was prior"))
    print("-" * 78)
    print("\nWritten to %s" % out)


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
