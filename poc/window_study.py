"""How long a window do the systematic parameters need?

Runs the systematic stage across several window lengths on simulated data and
reports, per parameter: bias, the ACROSS-SEED spread, the spread THEORY
predicts, and interval coverage.

    SEEDS=$(seq -s, 1 40) WINDOWS=252,504,1008,2016 python poc/window_study.py

WHY THE THEORY COLUMN MATTERS. Each systematic parameter is limited by a
different thing, and they scale differently:

    param   limited by                        252d   504d  1008d  2016d
    alpha   CALENDAR SPAN  sqrt(2a/T_yr)       37%    26%    18%    13%
    mu_i    calendar span (it is a drift)      37%    26%    18%    13%
    lamb    jump count   1/sqrt(lam*T)         11%     8%     6%     4%
    eta1    UP jumps     1/sqrt(p*lam*T)       15%    11%     8%     5%
    eta2    DOWN jumps   1/sqrt(q*lam*T)       17%    12%     9%     6%
    sigma   observations 1/sqrt(2n)             4%     3%     2%     2%

alpha is a RATE, so only elapsed time informs it and no amount of intraday
sampling helps - see identification_alpha.html. sigma is a quadratic-variation
object and is pinned by observation count, so 252 days is already ample. The
jump parameters need JUMPS, and at the SIMULATED lambda ~ 77 even 252 days
carries 77 of them.

WHAT TO LOOK FOR, and it is not the bias column.

  1. MEASURED SPREAD vs PREDICTED. If measured is much wider, something beyond
     sampling noise is moving the estimate.
  2. COVERAGE AS THE WINDOW GROWS. Under correct specification it should hold
     near nominal at every window. If it DEGRADES with more data, that is the
     signature of a fixed proportional bias: the interval shrinks as 1/sqrt(T)
     while the bias stays put, so the truth falls outside. A preliminary
     5-seed run showed exactly that for lamb, eta1 and eta2 (5/5, 5/5, 3/5,
     3/5) with spreads 2-3x the prediction - consistent with the <=1-jump
     truncation in Appendix B. It was far too underpowered to conclude
     anything, which is why this script exists.

USE AT LEAST 30 SEEDS. With 5 the across-seed sd has ~35% relative error and
nothing is resolvable. 40 windows x seeds fits comfortably in an hour.
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recover_two_stage import DT, simulate                      # noqa: E402
from Library.RiskEngineKimYi2025 import pmle_kimyirisk_systematic  # noqa: E402

# TRUTH IS A CHOICE. It is NOT the paper's calibration - Table 1 reports
# alpha ~ 0.68, lamb ~ 10-12, eta1 ~ 50.5, eta2 ~ 26-27, pprob ~ 0.35-0.44,
# sigma ~ 0.13-0.17. These values are FULL_SAMPLE (itself a cross-arm
# calibration, not one fit) with alpha set to 15 by hand, chosen so that no
# truth sits at its prior mean: a truth AT the prior mean scores perfect
# coverage even when the likelihood carries no information at all, which
# would make this whole script vacuous.
#
# The choice is not neutral for what is measured here:
#   alpha  rel sd is sqrt(2/(alpha*T_yr)): 37% at alpha 15 over one year, but
#          171% at alpha 0.68, and still 61% over eight years. A SMALL alpha
#          is unmeasurable at every window, prior-driven by construction.
#   lamb   Appendix B drops P(N>=2) per step, order (lam*dt)^2/2: 3.8% at
#          lamb 77, 0.1% at lamb 11. The ~-12% specification bias driving
#          the coverage decay is measured at 77, UNMEASURED at 11.
# So "252 days" is conditional on this scale. Re-run at the paper's own scale
# before quoting the conclusion there.
TRUTH = dict(alpha=15.0, sigma=0.105, lamb=76.99, pprob=0.575, eta1=78.59,
             eta2=60.68, betai=1.5, kappai=0.15, gammai=2.5, mui=0.05)
KEYS = [("dALPHA", "alpha"), ("dSIGMA", "sigma"), ("dPPROB", "pprob"),
        ("dLAMB", "lamb"), ("dETA1", "eta1"), ("dETA2", "eta2")]


def truth_from_env():
    """TRUTH, with any entry overridden by an upper-case env var.

    To run at the scale the paper's own Table 1 reports:

        ALPHA=0.68 LAMB=11 ETA1=50.5 ETA2=26.5 PPROB=0.40 SIGMA=0.15 \
        SEEDS=$(seq -s, 1 40) python poc/window_study.py
    """
    t = dict(TRUTH)
    for k in t:
        v = os.environ.get(k.upper())
        if v is not None:
            t[k] = float(v)
    return t


def predicted_rel_sd(key, n_steps, truth):
    """Relative sd each parameter's own information source implies."""
    years = n_steps / 252.0
    lam, p = truth["lamb"], truth["pprob"]
    if key in ("dALPHA",):
        return np.sqrt(2 * truth["alpha"] / years) / truth["alpha"]
    if key == "dLAMB":
        return 1.0 / np.sqrt(lam * years)
    if key == "dETA1":
        return 1.0 / np.sqrt(p * lam * years)
    if key == "dETA2":
        return 1.0 / np.sqrt((1 - p) * lam * years)
    if key == "dSIGMA":
        return 1.0 / np.sqrt(2 * n_steps)
    return None


def main():
    seeds = [int(x) for x in os.environ.get("SEEDS", "1,2,3,4,5").split(",")]
    windows = [int(x) for x in os.environ.get("WINDOWS", "252,504,1008,2016").split(",")]
    draws = int(os.environ.get("NMC", 500))
    out = os.environ.get("OUTFILE", "window_study.jsonl")
    if len(seeds) < 30:
        print("WARNING: %d seeds. The across-seed sd has ~%.0f%% relative error "
              "at this count; window comparisons will not resolve. Use 30+.\n"
              % (len(seeds), 100 / np.sqrt(2 * (len(seeds) - 1))))

    truth = truth_from_env()
    print("truth: %s\n" % ", ".join(
        "%s %g" % (k, truth[k]) for _, k in KEYS))

    rows = []
    for n in windows:
        for sd in seeds:
            sys_r, _ = simulate(truth, n, sd)
            t0 = time.time()
            r = pmle_kimyirisk_systematic(
                sys_returns=sys_r, delta_t=np.array(DT),
                seed_number=np.uint64(sd), n_mc_paths=draws)
            rec = dict(n=n, seed=sd, secs=round(time.time() - t0, 1),
                       **{k: [float(r[k].dMEAN), float(r[k].dCI_LOWER),
                              float(r[k].dCI_UPPER)] for k, _ in KEYS})
            rows.append(rec)
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print("  n=%-5d seed %-6d alpha %6.2f  lam %6.1f  (%.0fs)"
                  % (n, sd, rec["dALPHA"][0], rec["dLAMB"][0], rec["secs"]),
                  flush=True)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["n"]].append(r)
    print("\n%d seeds per window. TRUTH IS A CHOICE, not the paper's "
          "Table 1 - see the module docstring." % len(seeds))
    for key, tname in KEYS:
        true_v = truth[tname]
        print("\n%s   (true %.4g)" % (key, true_v))
        print("   %6s %10s %8s %11s %10s %9s %8s"
              % ("window", "mean", "bias", "across-sd", "rel meas", "rel pred", "cover"))
        for n in sorted(by):
            e = np.array([r[key][0] for r in by[n]])
            lo = np.array([r[key][1] for r in by[n]])
            hi = np.array([r[key][2] for r in by[n]])
            pr = predicted_rel_sd(key, n, truth)
            print("   %5dd %10.4f %7.1f%% %11.4f %9.0f%% %8s %5d/%d"
                  % (n, e.mean(), 100 * (e.mean() - true_v) / true_v,
                     e.std(ddof=1), 100 * e.std(ddof=1) / abs(true_v),
                     "%.0f%%" % (100 * pr) if pr else "-",
                     int(np.sum((lo <= true_v) & (true_v <= hi))), len(e)))


if __name__ == "__main__":
    main()
