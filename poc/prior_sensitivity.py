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

ARMS = {
    "A baseline":       {},
    "B symmetric eta":  {"eta1": G(35.0, 1.0), "eta2": G(35.0, 1.0)},
    "C mirrored pprob": {"pprob_rv": Bt(2.0, 5.0)},
    "D both":           {"eta1": G(35.0, 1.0), "eta2": G(35.0, 1.0),
                         "pprob_rv": Bt(2.0, 5.0)},
}

# reported name -> prior key
KEYS = [("dSIGMA", "sigma"), ("dALPHA", "alpha_rv"), ("dPPROB", "pprob_rv"),
        ("dLAMB", "lamb"), ("dETA1", "eta1"), ("dETA2", "eta2")]


def returns_for(date):
    panel = get_price_panel([SYSTEMATIC_ID])
    r = panel.pct_change().dropna()
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
    for dt in dates:
        rv = returns_for(dt)
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
        b = d[d.arm == "B symmetric eta"]
        if len(b):
            e1, e2 = float(b["dETA1"].iloc[0]), float(b["dETA2"].iloc[0])
            gap = abs(e1 - e2)
            print("  -> arm B separation |eta1 - eta2| = %.2f   %s"
                  % (gap, "DATA SEPARATES THEM" if gap > 8 else
                     "no separation - the asymmetry was prior"))
    print("-" * 78)
    print("\nWritten to %s" % out)


if __name__ == "__main__":
    main()
