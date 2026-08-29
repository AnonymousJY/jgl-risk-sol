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

# Project convention: 95% EQUAL-TAILED interval (2.5% / 97.5% quantiles).
# Defined once in Library/PosteriorSummary.py; sd ~ width / 3.919928.
#
# WARNING. Estimates produced BEFORE that convention was fixed came from
# az.summary's old default, a 94% HDI (factor 3.7616). Applying the 95% ETI
# factor to those files understates the posterior sd by about 4%, which is
# immaterial for the identification verdicts here - a ratio of 1.00 does not
# become 0.70 - but do not pool old and new estimates in one series.
try:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from Library.PosteriorSummary import CI_WIDTH_TO_SD as HDI_TO_SD, CI_CONVENTION
except Exception:                                                # noqa: BLE001
    HDI_TO_SD = 1.0 / 3.919927969080108
    CI_CONVENTION = "eti_0.95"


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
    print("Interval convention: %s" % CI_CONVENTION)
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

        travel = abs(post_mean - pmean) / psd
        if ratio > 0.90 and travel < 0.30:
            v = "PRIOR-DRIVEN - neither location nor width"
        elif ratio > 0.90:
            v = "LOCATION ONLY - mean moves, precision is the prior"
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

    print("\n  WHY each unidentified parameter is unidentified, and what fixes it.")
    print("  These are NOT the same problem and do not have the same fix:\n")
    print("    dALPHA  Mean-reversion SPEED. With alpha ~0.69 annualised the")
    print("            half-life is ln2/0.69 ~ 1 year, so a 252-day window is")
    print("            ONE half-life. Mean reversion cannot be estimated from")
    print("            one half-life - the standard near-unit-root problem.")
    print("            FIX: longer sample. 20 years is ~20 half-lives. This one")
    print("            is genuinely solved by length.")
    print("            WHY IT MATTERS MOST: alpha is the speed at which liquidity")
    print("            shocks decay - it IS resiliency, the paper's named")
    print("            contribution. If alpha is the prior, so is the resiliency")
    print("            indicator.")
    print()
    print("    dETA1   Jump-size tail shape. Identified off jump MAGNITUDES, of")
    print("    dETA2   which a 252-day window holds ~5-7 per tail. FIX: more")
    print("            jumps helps, but Ait-Sahalia (2004) shows daily frequency")
    print("            is itself limiting - below ~3.5 sd a large return is more")
    print("            likely diffusion than a jump. Intraday data, or")
    print("            option-implied tails, are the real answer.")
    print()
    print("    dPPROB  Jump DIRECTION. Binomial with n ~ 12 gives sd(p_hat) ~")
    print("            0.14, which is the prior width - hence location moves but")
    print("            precision does not. FIX: more jumps. Length alone will not")
    print("            buy precision at the rate you need.")
    print()
    print("    dLAMB   Jump INTENSITY. Poisson count with mean ~12 gives relative")
    print("            sd ~ 1/sqrt(12) ~ 29%. Already partially identified.")
    print("            FIX: more jumps, straightforwardly.")
    print()
    print("    dSIGMA  Diffusion SCALE. 252 observations of it. Identified, and")
    print("            Ait-Sahalia proves it stays identified even with jumps")
    print("            present (asymptotic variance 2 sigma^4 delta).")

    bad = [k for k, (r, _) in verdicts.items() if r > 0.90]
    if bad:
        print("\n" + "!" * 78)
        print("  %s are not identified by a 252-day sample." % ", ".join(bad))
        print("  That is %d of 6 parameters. Only dSIGMA is cleanly identified."
              % len(bad))
        print("  Reporting the rest as estimates is not defensible.")
        print()
        print("  DO NOT reach for the obvious fix - pooling these over a long")
        print("  sample while keeping 252 days for sigma. Calling them 'slow,")
        print("  structural' parameters rests on their apparent constancy, and")
        print("  that constancy IS the prior showing through. Using the artefact")
        print("  as evidence for the assumption that explains it away is circular.")
        print("  Principle seven does not distinguish judgment in a prior from")
        print("  judgment in a decision about what to pool.")
        print()
        print("  Whether to pool must be an empirical finding:")
        print("    1. Refit on the full 2007-2026 sample under genuinely weak")
        print("       priors, then re-run this diagnostic. If ratios stay near")
        print("       1.00 with ~230 jumps, the parameter is not identified by")
        print("       this data at all and no window length rescues it.")
        print("    2. Test constancy: fit non-overlapping blocks of ~30 jumps")
        print("       each and ask whether the block posteriors are")
        print("       distinguishable. Overlap justifies pooling BY THE DATA.")
        print("       Test sigma the same way - do not assume it is fast.")
        print("       Hierarchical form: block parameters drawn from a")
        print("       population with unknown dispersion tau, data estimates tau.")
        print("    3. The estimation design follows from step 2, and only from it.")
        print()
        print("  POWER CAVEAT: eight blocks of ~30 jumps leaves ~15 observations")
        print("  per tail for eta1/eta2. Overlapping posteriors may mean the test")
        print("  cannot tell them apart, not that the parameters are equal. That")
        print("  is a null result, not evidence for pooling. Report it as such.")
        print("!" * 78)


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(_ROOT), "mkt-depth-n-resiliency",
                           "Study", "Estimated Parameters PMLE", "^SPX") \
        if (_ROOT := os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) else ""
    main(sys.argv[1] if len(sys.argv) > 1 else default)
