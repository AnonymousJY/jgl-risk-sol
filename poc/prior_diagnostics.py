"""
Prior-versus-posterior diagnostics for the systematic P-MLE.

Answers one question: for each systematic parameter, how much of the posterior
is coming from the DATA and how much from the PRIOR?

The test is the ratio of posterior sd to prior sd. If a posterior is as wide as
its prior and centred on the prior mean, the data has told us nothing and the
"estimate" is a restatement of the modelling choice.

This matters beyond statistics. FRTB MAR31.25 Principle seven requires that
model coefficients "must be empirically based and must not be determined based
on judgment", and says coefficients set by judgment "generally should be
considered as NMRFs". A prior-driven posterior is judgment wearing a credible
interval. SR 26-2 likewise asks conceptual soundness validation to document
"key modeling choices, assumptions, qualitative judgments".

So a TIGHTER credible interval obtained by a more informative prior is not an
improvement - it is a worse regulatory position. The objective is demonstrable
data-drivenness, not narrow intervals.

    python poc/prior_diagnostics.py [path/to/Estimated Parameters PMLE/^SPX]
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

# Priors as declared in Library/RiskEngineKimYi2025.pmle_kimyirisk_systematic.
# PyMC Gamma(alpha, beta) is shape-rate: mean = a/b, var = a/b^2.
# Beta(a, b): mean = a/(a+b), var = ab / ((a+b)^2 (a+b+1)).
def _gamma(a, b):
    return a / b, np.sqrt(a / b**2)


def _beta(a, b):
    m = a / (a + b)
    v = a * b / ((a + b) ** 2 * (a + b + 1))
    return m, np.sqrt(v)


PRIORS = {
    "dSIGMA": ("Gamma(1, 1)",   *_gamma(1.0, 1.0)),
    "dALPHA": ("Beta(5, 2)",    *_beta(5.0, 2.0)),
    "dPPROB": ("Beta(5, 2)",    *_beta(5.0, 2.0)),
    "dLAMB":  ("Gamma(10, .5)", *_gamma(10.0, 0.5)),
    "dETA1":  ("Gamma(50, 1)",  *_gamma(50.0, 1.0)),
    "dETA2":  ("Gamma(25, 1)",  *_gamma(25.0, 1.0)),
}

# ArviZ reports a 94% HDI by default; sd ~ width / (2 * 1.8808).
HDI_TO_SD = 1.0 / (2 * 1.8808)


def load(dirpath):
    rows = []
    for f in sorted(glob.glob(os.path.join(dirpath, "*.csv"))):
        d = pd.read_csv(f).iloc[0].to_dict()
        d["date"] = os.path.basename(f).split("_")[-1].replace(".csv", "")
        rows.append(d)
    df = pd.DataFrame(rows).drop_duplicates(subset="date").set_index("date").sort_index()
    return df


def main(dirpath):
    df = load(dirpath)
    print("=" * 78)
    print("Prior vs posterior :: systematic parameters   (%d valuation dates)" % len(df))
    print("=" * 78)
    print("\n  %-8s %-14s %9s %9s %9s %9s %7s  %s"
          % ("param", "prior", "pri_mean", "pri_sd", "post_mean", "post_sd",
             "sd_post/", "verdict"))
    print("  %-8s %-14s %9s %9s %9s %9s %7s"
          % ("", "", "", "", "(median)", "(median)", "sd_pri"))
    print("  " + "-" * 76)

    verdicts = {}
    for k, (label, pmean, psd) in PRIORS.items():
        if k not in df:
            continue
        post_mean = float(df[k].median())
        lo, hi = df[k + "_CI_LOWER"], df[k + "_CI_UPPER"]
        post_sd = float(((hi - lo) * HDI_TO_SD).median())
        ratio = post_sd / psd

        if ratio > 0.90:
            v = "PRIOR-DRIVEN - data adds nothing"
        elif ratio > 0.70:
            v = "weakly identified"
        elif ratio > 0.35:
            v = "partially identified"
        else:
            v = "data-driven"
        verdicts[k] = (ratio, v)

        print("  %-8s %-14s %9.3f %9.3f %9.3f %9.3f %7.2f  %s"
              % (k, label, pmean, psd, post_mean, post_sd, ratio, v))

    print("\n  Ratio near 1.00 means the posterior is as wide as the prior: the")
    print("  likelihood is flat and the number reported is the modelling choice.")

    # how far did the posterior mean travel from the prior mean, in prior sds?
    print("\n  Distance of posterior mean from prior mean, in prior standard deviations:")
    for k, (label, pmean, psd) in PRIORS.items():
        if k not in df:
            continue
        travel = abs(float(df[k].median()) - pmean) / psd
        note = "   <-- pinned to the prior" if travel < 0.30 else ""
        print("     %-8s %5.2f sd%s" % (k, travel, note))

    print("\n  Variation of the posterior mean ACROSS valuation dates:")
    print("  (the paper cites stability here as evidence of identification -")
    print("   but a prior-driven parameter is stable for the wrong reason)")
    for k in PRIORS:
        if k not in df:
            continue
        cv = float(df[k].std() / abs(df[k].mean()))
        r = verdicts.get(k, (np.nan, ""))[0]
        flag = "   <-- stable BECAUSE prior-driven" if (cv < 0.02 and r > 0.9) else ""
        print("     %-8s cv %6.4f%s" % (k, cv, flag))

    bad = [k for k, (r, _) in verdicts.items() if r > 0.90]
    if bad:
        print("\n" + "!" * 78)
        print("  %s are not identified by a 252-day sample." % ", ".join(bad))
        print("  Reporting them as estimates is not defensible. Options, in order")
        print("  of preference:")
        print("    1. Lengthen the sample for these parameters only - they are")
        print("       jump-SHAPE parameters and need jump counts, not day counts.")
        print("    2. Empirical Bayes: estimate them once on the full 2007-2026")
        print("       sample and use that posterior as the prior for 252-day fits.")
        print("       The prior is then empirically derived from the same series")
        print("       rather than chosen, which is what Principle seven requires.")
        print("    3. Disclose them as fixed calibration constants, not estimates.")
        print("!" * 78)


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(_ROOT), "mkt-depth-n-resiliency",
                           "Study", "Estimated Parameters PMLE", "^SPX") \
        if (_ROOT := os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) else ""
    main(sys.argv[1] if len(sys.argv) > 1 else default)
