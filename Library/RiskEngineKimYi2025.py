import os
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import pytensor.tensor as pt

from typing import Tuple, List, Literal
from collections import namedtuple
from numpy.typing import NDArray
from Library.StatisticsMC import get_corr_mat
from Library.Random import RandomBase, RandomMT19937
from Library.Parameters import ParametersBase, ParametersConstant
from Library.PosteriorSummary import summarize


ParamsResults = namedtuple('ParamsResults', ["dMEAN", "dCI_LOWER", "dCI_UPPER"])


def simulate_shock_returns(
        params: pd.Series,
        rng: RandomBase,
        size: Tuple[int, int, int],
        delta_time: NDArray[np.float64]=np.array(1/252)
) -> NDArray[np.float64]:

    ret, _, _ = KimYiRiskEngine(
        mui=[ParametersConstant(params.dMUI)],
        kappai=[ParametersConstant(params.dKAPPAI)],
        gammai=[ParametersConstant(params.dGAMMAI)],
        betai=[ParametersConstant(params.dBETAI)],
        rhoix=[ParametersConstant(params.dRHOIX)],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=delta_time
    ).random(rng=rng, size=size)

    return ret[0]

def simulate_shock_returns_base(
        params: pd.Series,
        rng: RandomBase,
        size: Tuple[int, int, int],
        delta_time: NDArray[np.float64]=np.array(1/252)
) -> NDArray[np.float64]:

    ret, _, _ = KimYiRiskEngine(
        mui=[ParametersConstant(params.dMUI)],
        kappai=[ParametersConstant(params.dKAPPAI)],
        gammai=[ParametersConstant(np.array(.0))],
        betai=[ParametersConstant(np.array(.0))],
        rhoix=[ParametersConstant(np.array(.0))],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=delta_time
    ).random(rng=rng, size=size)

    return ret[0]

def est_liquidity_process(
        params: pd.Series,
        observed_data: NDArray[np.float64],
        delta_time: NDArray[np.float64]=np.array(1/252)
) -> NDArray[np.float64]:

    Psi = KimYiRiskEngine(
        mui=[ParametersConstant(params.dMUI)],
        kappai=[ParametersConstant(params.dKAPPAI)],
        gammai=[ParametersConstant(params.dGAMMAI)],
        betai=[ParametersConstant(params.dBETAI)],
        rhoix=[ParametersConstant(params.dRHOIX)],
        alpha=ParametersConstant(params.dALPHA),
        sigma=ParametersConstant(params.dSIGMA),
        pprob=ParametersConstant(params.dPPROB),
        lamb=ParametersConstant(params.dLAMB),
        eta1=ParametersConstant(params.dETA1),
        eta2=ParametersConstant(params.dETA2),
        end_dt=delta_time
    ).est_liquidity_process(observed_data)

    return Psi


def _dist_loglike_systematic(y, alpha, sigma, pprob, lamb, eta1, eta2, delta_t) -> pt.TensorVariable:
    return KimYiLogLike(
        mui=np.array(.0),
        kappai=np.array(.0),
        gammai=np.array(1.),
        betai=np.array(1.),
        rhoix=np.array(.0),
        alpha=np.exp(alpha),
        sigma=sigma,
        pprob=np.exp(pprob),
        lamb=lamb,
        eta1=eta1,
        eta2=eta2,
        dt=delta_t
    ).logp(y=y)


def _dist_loglike_idiosyncratic(y, mui, kappai, gammai, betai, rhoix, alpha, sigma, pprob, lamb, eta1, eta2, delta_t) -> pt.TensorVariable:
    return KimYiLogLike(
        mui=mui,
        kappai=np.exp(kappai),
        gammai=np.exp(gammai),
        betai=np.exp(betai),
        rhoix=np.tanh(rhoix),
        alpha=alpha,
        sigma=sigma,
        pprob=pprob,
        lamb=lamb,
        eta1=eta1,
        eta2=eta2,
        dt=delta_t
    ).logp(y=y)


# ---------------------------------------------------------------------------
# Priors on the systematic block
# ---------------------------------------------------------------------------
# EVERY SPEC BELOW LISTS ALL SIX PRIORS EXPLICITLY. None inherits from another.
#
# They used to. SKEW_TIGHT was dict(SKEW), SKEW was dict(GAPS), GAPS was
# dict(RECENTRED) - four levels, and reading any one of them told you nothing
# about what it actually contained. That cost real work twice:
#
#   - Recentring alpha inside RECENTRED silently changed GAPS, ASYM, SKEW and
#     both CAPPED variants. Since the drawer was named after the ARM and not
#     the values, a rerun found the dates already on disk, skipped them, and
#     reprinted posteriors fitted under the OLD alpha beneath the NEW header.
#   - "What is lambda's prior mean under skew-tight?" needed a four-hop trace
#     to answer, and the answer (6.0, from GAPS) was not visible anywhere near
#     the spec that used it.
#
# Explicit repetition is the cheaper mistake. If two specs should share a value
# and one is edited, the diff shows it; under inheritance the diff shows
# nothing and the change happens somewhere else.
#
# REMOVED, 4 September 2026: SYSTEMATIC_PRIORS_RECENTRED, _CAPPED and
# _CAPPED_BETA. RECENTRED existed only as GAPS's base; the two CAPPED arms
# tested a pprob <= 0.6 constraint that the identification work made moot -
# pprob is barely identified in a window, so capping it constrains the prior
# rather than the data. Estimates already on disk under those drawers are
# untouched and still readable; only the ability to launch new runs of those
# arms is gone. `python poc/archive_run.py --list` shows what is there.

# The published configuration. Passing priors=None reproduces it exactly.
#
# Note what these DEFAULTS assert. eta1 ~ Gamma(50,1) and eta2 ~ Gamma(25,1)
# say the mean down jump (1/eta2 = 4%) is twice the mean up jump (1/eta1 = 2%)
# BEFORE any data is seen - and the rolling work showed a 252-day window
# returns that assertion untouched, so the model's negative jump skew is an
# assumption at this horizon rather than an estimate. On the full sample it IS
# measured, and in the same direction.
# Sampler layout, overridable by environment so a whole run can be retimed
# without editing any call site.
#
# The defaults were chains=4, cores=4, draws=10000. A benchmark on an unloaded
# 24-core box (poc/bench_sampler.py, ^SPX 2025-06-30) says both are wrong:
#
#   cores  sec/fit  workers=24/cores  fits/hour
#       4      7.4                 6       2914
#       2      9.5                12       4527
#       1     13.9                24       6216      <- 2.13x
#
# Four threaded chains return only 1.88x over one, i.e. 47% threading
# efficiency, so giving each fit four cores wastes half of them. The outer
# process pool converts them at 100%.
#
#   draws   sec/fit   min tail ess
#     500      14.3           1025     <- includes ~7s of one-off compilation
#    1000       7.1           2153
#    2000       7.2           4843
#   10000      13.4          26032
#
# 10,000 draws buys 26,000 tail ess to report a 95% interval that needs about
# 1,000. Most of a fit is fixed cost - compile plus 1,000 tuning steps - so
# cutting draws is 1.86x, not the 5x a naive draws-are-everything reading gives.
JGL_CHAINS = int(os.environ.get("JGL_CHAINS", "4"))
JGL_CORES  = int(os.environ.get("JGL_CORES", "4"))
JGL_DRAWS  = int(os.environ.get("JGL_DRAWS", "0")) or None   # None = caller's value

# Warn when a fit's minimum tail ess falls below this. Draws were cut from
# 10,000 to 1,000 on a benchmark of ONE date; sampling efficiency varies by
# date and the idiosyncratic model has a different geometry, so the cut is
# checked on every fit rather than assumed to hold. 400 is the usual floor for
# a usable tail quantile; the benchmark date returned 2,153.
JGL_MIN_TAIL_ESS = float(os.environ.get("JGL_MIN_TAIL_ESS", "400"))


def _warn_low_ess(idata, names, label):
    """Print a warning if any parameter's tail ess is too low to support a
    95% equal-tailed interval. Never raises - a noisy fit is still a fit, and
    the run must not die on a diagnostic."""
    try:
        import arviz as az
        have = [v for v in names if v in idata.posterior]
        if not have:
            return
        e = az.ess(idata, var_names=have, method="tail")
        low = {v: float(e[v].values) for v in have
               if float(e[v].values) < JGL_MIN_TAIL_ESS}
        if low:
            print("    LOW TAIL ESS (%s): %s  - interval endpoints for these "
                  "are unreliable at this draw count"
                  % (label, ", ".join("%s %.0f" % (k, v) for k, v in low.items())),
                  flush=True)
    except Exception:                                            # noqa: BLE001
        pass


SYSTEMATIC_PRIORS = {
    "sigma":    ("Gamma", {"alpha":  1.0,  "beta": 1.0}),    # mean  1.000 sd 1.000
    "alpha_rv": ("Beta",  {"alpha":  5.0,  "beta": 2.0}),    # mean  0.714 sd 0.160
    "pprob_rv": ("Beta",  {"alpha":  5.0,  "beta": 2.0}),    # mean  0.714 sd 0.160
    "lamb":     ("Gamma", {"alpha": 10.0,  "beta": 0.5}),    # mean 20.000 sd 6.325
    "eta1":     ("Gamma", {"alpha": 50.0,  "beta": 1.0}),    # mean 50.000 sd 7.071
    "eta2":     ("Gamma", {"alpha": 25.0,  "beta": 1.0}),    # mean 25.000 sd 5.000
}

# ---------------------------------------------------------------------------
# Full-sample calibration, 2007-01-01 to 2026-08-31 (~4,950 returns, ~193 jumps)
# ---------------------------------------------------------------------------
# eta1, eta2 and pprob come from the arm that asserts NOTHING: the two decays
# share a Gamma(35,1) prior and pprob is Uniform(0,1). From zero asserted
# separation they finished 17.9 apart - three prior sd - in the equity
# direction. Convergence from an uninformative start is what makes this a
# measurement rather than an assumption.
#
# alpha is from the baseline full-sample fit. A 252-day window cannot estimate
# it at all: half-life 19 years against a one-year window.
#
# NOT one run. eta1/eta2/pprob and alpha/sigma/lamb come from different arms,
# so quote it as a calibration, never as a single re-estimation result.
FULL_SAMPLE = {
    "alpha_rv": 0.036,      # half-life 19.3 yr; Psi is near a random walk
    "sigma":    0.105,
    "pprob_rv": 0.575,      # +-0.026 under a flat prior, 2.9 sd above 0.5
    "lamb":     76.99,      # ~77 jumps/yr, one every 3.3 trading days
    "eta1":     78.59,      # mean up jump   1.27%
    "eta2":     60.68,      # mean down jump 1.65%  -> jump skew -0.55
}

# "A jump is a GAP": both etas at mean 20, a 5% mean jump, with lambda brought
# down to mean 6 so jump volatility stays coherent (sqrt(6 * 2/20^2) = 17.3%).
# Identical priors on the two etas assert no asymmetry, so any separation in
# the posterior has to come from the data. It produced +0.40 - 0.04 prior sd,
# i.e. none.
SYSTEMATIC_PRIORS_GAPS = {
    "sigma":    ("Gamma", {"alpha":  2.0,  "beta": 10.0}),   # mean  0.200 sd  0.141
    "alpha_rv": ("Beta",  {"alpha":  2.0,  "beta":  2.0}),   # mean  0.500 sd  0.224
    "pprob_rv": ("Beta",  {"alpha":  2.3,  "beta":  1.7}),   # mean  0.575 sd  0.221
    "lamb":     ("Gamma", {"alpha":  3.0,  "beta":  0.5}),   # mean  6.000 sd  3.464
    "eta1":     ("Gamma", {"alpha":  4.0,  "beta":  0.2}),   # mean 20.000 sd 10.000
    "eta2":     ("Gamma", {"alpha":  4.0,  "beta":  0.2}),   # mean 20.000 sd 10.000
}

# GAPS with the etas pulled apart the WRONG WAY: eta1 25 (4% mean UP jump)
# against eta2 50 (2% mean DOWN jump). Positive skew - the opposite sign to
# equities, to the paper's defaults, and to the full sample. A deliberately
# wrong-signed prior, to test whether the data drags it back. It did not:
# eta2 moved 0.00 prior sd.
SYSTEMATIC_PRIORS_ASYM = {
    "sigma":    ("Gamma", {"alpha":  2.0,  "beta": 10.0}),   # mean  0.200 sd  0.141
    "alpha_rv": ("Beta",  {"alpha":  2.0,  "beta":  2.0}),   # mean  0.500 sd  0.224
    "pprob_rv": ("Beta",  {"alpha":  2.3,  "beta":  1.7}),   # mean  0.575 sd  0.221
    "lamb":     ("Gamma", {"alpha":  3.0,  "beta":  0.5}),   # mean  6.000 sd  3.464
    "eta1":     ("Gamma", {"alpha":  4.0,  "beta":  0.16}),  # mean 25.000 sd 12.500
    "eta2":     ("Gamma", {"alpha":  4.0,  "beta":  0.08}),  # mean 50.000 sd 25.000
}

# The exact mirror of ASYM, in the equity direction: eta1 50 (2% up) against
# eta2 25 (4% down). At p = 0.5 the pair have identical jump variance, so any
# difference between their fits is the window expressing a preference on SIGN.
# It expressed none - both returned their orderings, and the fitted
# distributions came back mirror images (total vol 18.82% vs 18.72%, jump third
# moment +6.76e-04 vs -6.74e-04).
SYSTEMATIC_PRIORS_SKEW = {
    "sigma":    ("Gamma", {"alpha":  2.0,  "beta": 10.0}),   # mean  0.200 sd  0.141
    "alpha_rv": ("Beta",  {"alpha":  2.0,  "beta":  2.0}),   # mean  0.500 sd  0.224
    "pprob_rv": ("Beta",  {"alpha":  2.3,  "beta":  1.7}),   # mean  0.575 sd  0.221
    "lamb":     ("Gamma", {"alpha":  3.0,  "beta":  0.5}),   # mean  6.000 sd  3.464
    "eta1":     ("Gamma", {"alpha":  4.0,  "beta":  0.08}),  # mean 50.000 sd 25.000
    "eta2":     ("Gamma", {"alpha":  4.0,  "beta":  0.16}),  # mean 25.000 sd 12.500
}

# SKEW with every wide prior sd HALVED and every centre unchanged.
#   Gamma: halving sd at fixed mean is (a, b) -> (4a, 4b)
#   Beta:  halving sd at fixed mean is k = a+b -> 4k+3, here 4 -> 19
# sigma and lamb are left alone - sigma has 252 observations behind it, and
# lamb is identified by LOCATION (2020 fits 27.9 against 2024's 4.0), which a
# tighter prior would not improve.
#
# RESULT, and it is the point of the arm. Widths landed within 1-6% of the
# predicted halving; every ratio moved TOWARD 1, i.e. every parameter scored as
# LESS identified; and every parameter whose prior was tightened slid to its
# prior centre while sigma and lamb, untouched, did not move at all
# (0.1514 -> 0.1512 and 8.7824 -> 8.7657). The reported CVs collapsed - dALPHA
# 0.124 -> 0.032, dPPROB 0.166 -> 0.053 - so the table LOOKS far more precise
# while containing strictly less information. That is how a published table of
# tight, stable estimates on unidentifiable parameters comes about without
# anyone doing anything careless.
SYSTEMATIC_PRIORS_SKEW_TIGHT = {
    "sigma":    ("Gamma", {"alpha":  2.0,   "beta": 10.0}),  # mean  0.200 sd  0.141
    "alpha_rv": ("Beta",  {"alpha":  9.5,   "beta":  9.5}),  # mean  0.500 sd  0.112
    "pprob_rv": ("Beta",  {"alpha": 10.925, "beta":  8.075}),# mean  0.575 sd  0.111
    "lamb":     ("Gamma", {"alpha":  3.0,   "beta":  0.5}),  # mean  6.000 sd  3.464
    "eta1":     ("Gamma", {"alpha": 16.0,   "beta":  0.32}), # mean 50.000 sd 12.500
    "eta2":     ("Gamma", {"alpha": 16.0,   "beta":  0.64}), # mean 25.000 sd  6.250
}

# ---------------------------------------------------------------------------
# Identification arms for alpha. Both are SKEW_TIGHT with one prior changed.
# ---------------------------------------------------------------------------
# The question these answer: is alpha estimated, or asserted? Every arm above
# except "paper" centres alpha_rv at 0.500, so they cannot separate "the data
# says 0.5" from "the prior says 0.5". Displacing the prior mean can.
#
# The evidence so far, reading the drawers on 2026-09-10:
#
#   arm          prior            prior mean   posterior mean
#   paper        Beta(5, 2)            0.714           0.6892
#   gaps/asym/   Beta(2, 2)            0.500           0.4592-0.4620
#     skew
#   skew-tight   Beta(9.5, 9.5)        0.500           0.4898
#
# Move the prior mean 0.214 and the posterior mean moves 0.228 - a ratio of
# 1.06. Across all three the posterior sits at roughly
#
#       prior mean  -  0.15 x prior sd
#
# so the likelihood's whole contribution is a small downward nudge scaled by
# whatever room the prior leaves it. These two arms test that relationship
# where it should break if alpha carries any information:
#
#   alpha-flat  Beta(1,1) = Uniform(0,1). Widest prior, same centre. The rule
#               predicts 0.500 - 0.15*0.2887 = 0.457, NOT 0.500. A posterior at
#               0.500 would mean the nudge is an artifact of the Beta shape; a
#               posterior at 0.457 confirms it is the likelihood.
#   alpha-low   Beta(6,14), mean 0.300 sd 0.100 - as tight as skew-tight but
#               displaced. Predicts 0.285. If a tight prior at 0.30 returns
#               0.29 while a tight prior at 0.50 returns 0.49, alpha is set by
#               assumption at this horizon and nothing else.
SYSTEMATIC_PRIORS_ALPHA_FLAT = dict(SYSTEMATIC_PRIORS_SKEW_TIGHT)
SYSTEMATIC_PRIORS_ALPHA_FLAT["alpha_rv"] = (
    "Beta", {"alpha": 1.0, "beta": 1.0})                     # mean 0.500 sd 0.289

#   alpha-tiny  Uniform(0, 0.02), mean 0.010 sd 0.0058. The only arm whose
#               SUPPORT EXCLUDES the full-sample value of 0.036, so it can
#               fail in a way the others cannot: if the likelihood pulls at
#               all toward 0.036 the posterior piles against the 0.02
#               boundary instead of sitting at the prior mean. A posterior at
#               0.010 means the likelihood is flat even down here, where the
#               AR coefficient is 1 - 0.01/252 and Psi is a random walk in all
#               but name.
SYSTEMATIC_PRIORS_ALPHA_TINY = dict(SYSTEMATIC_PRIORS_SKEW_TIGHT)
SYSTEMATIC_PRIORS_ALPHA_TINY["alpha_rv"] = (
    "Uniform", {"lower": 0.0, "upper": 0.02})                # mean 0.010 sd 0.0058


SYSTEMATIC_PRIORS_ALPHA_LOW = dict(SYSTEMATIC_PRIORS_SKEW_TIGHT)
SYSTEMATIC_PRIORS_ALPHA_LOW["alpha_rv"] = (
    "Beta", {"alpha": 6.0, "beta": 14.0})                    # mean 0.300 sd 0.100


# The registry the drivers select from. One source of truth: poc/ scripts used
# to keep their own copies of this mapping and of the drawer suffixes, which is
# how estimate_systematic and estimate_idiosyncratic came to offer different
# arms from the same library.
# skew-tight with lambda's prior replaced by a FLAT one, and nothing else
# touched. This exists to settle one question the skew-tight runs cannot
# answer about themselves.
#
# skew-tight asserts lambda ~ Gamma(3, 0.5): mean 6, a GAP prior - few large
# jumps. The 252-day posterior comes back at 8.8, near the prior. The 756-day
# posterior comes back at 19.6, more than three prior means out, sitting deep
# in the right tail of that Gamma, and 1.47x the prior's own width. A posterior
# dragged that far into a prior's tail widens for a reason that has nothing to
# do with the data: it is fighting the prior.
#
# How far out, as Gamma(3, 0.5) density relative to its own mode:
#
#     lambda  6.0   0.83      the prior mean
#     lambda  8.8   0.44      the 252-day posterior mean - still on the prior
#     lambda 19.6   0.0098    the 756-day posterior mean - 1% of peak
#     lambda 40.0   2.1e-07
#     lambda 56.5   1.1e-10   the 756 run's LARGEST posterior mean
#
# The three-year posterior sits where this prior is a wall, not a preference,
# and a crisis window at 40+ is somewhere the prior has declared impossible.
# That distorts the mean downward and the width in a direction no reading of
# the identification ratio can undo. It is a confound in the 252-day series
# too - crisis windows there reach the same region - which is why this arm
# should be run at BOTH lengths rather than only at 756.
#
# Under Uniform(0, 200) the prior is flat everywhere lambda has ever gone
# (the 756 run's maximum is 56.5), so the posterior width IS the likelihood's
# curvature, at both window lengths, with the prior taken out of the argument.
# Run 252 and 756 under this arm and compare the ABSOLUTE widths - the
# ratio-to-prior is meaningless against a support of 200 and the diagnostic
# prints it only because it prints it for everything.
#
# It came back at 63.5, which is neither. The prior was not shifting lambda,
# it was doing essentially all the work of pinning it: freeing it gives an
# interval as wide as the level itself. lambda, eta1, eta2 and sigma are
# identified only JOINTLY, up to a ridge along which many small jumps and a
# smaller diffusion trade against few large jumps and a larger one.
#
# WHICH IS WHY THIS ARM IS A DIAGNOSTIC AND NOT A CANDIDATE. A point on that
# ridge is chosen by what a jump is supposed to MEAN, and the flat arm chooses
# one where the word stops meaning anything:
#
#   cell             lambda  days/jump  up size  dn size  jump var%  ann sd
#   252 skew-tight     8.76       28.8    1.99%    3.75%      39.1%   19.4%
#   252 lamflat       31.01        8.1    1.76%    3.35%      69.7%   24.1%
#   756 skew-tight    19.57       12.9    1.85%    3.08%      56.7%   20.6%
#   756 lamflat       69.98        3.6    1.27%    2.24%      78.1%   23.0%
#
# lambda = 70 is a "jump" every 3.6 trading days of 1.3% up and 2.2% down,
# carrying 78% of total variance - the jump component has eaten the diffusion
# and the Kou decomposition no longer names two different things. The implied
# rate of jumps at or beyond 5% is 1.41/yr under skew-tight against 3.98/yr
# under the flat arm; SPX has had roughly 1.8 such days a year since 2007.
#
# So Gamma(3, 0.5) is an IDENTIFYING RESTRICTION, not a nuisance prior. The
# likelihood gives the ridge, economics picks the point, and the data then
# moves along what is left - which it demonstrably does: the skew-tight lambda
# prior has sd 3.46 while the fitted series has cross-date sd 6.28. The prior
# is the same at every valuation date and cannot generate dispersion 1.8x its
# own width. The LEVEL is asserted; the MOVEMENT is measured, and the movement
# is what the economic claim rests on (poc/stress_response.py).
SYSTEMATIC_PRIORS_SKEW_TIGHT_LAMFLAT = dict(SYSTEMATIC_PRIORS_SKEW_TIGHT)
SYSTEMATIC_PRIORS_SKEW_TIGHT_LAMFLAT["lamb"] = (
    "Uniform", {"lower": 0.0, "upper": 200.0})           # mean 100.0 sd 57.735


# ---------------------------------------------------------------------------
# Hard negative skew on the jump SIGN: P(up) centred at 0.25, confined to
# (0, 0.5]. Two versions, because "mean 0.25 on (0, 0.5]" has two readings and
# they answer different questions.
# ---------------------------------------------------------------------------
# READ pprob CAREFULLY. In KimYiLogLike.logp, pprob multiplies the UP branch
# and qprob = 1 - pprob the down branch, so pprob is P(UP JUMP). A prior mean
# of 0.25 therefore asserts P(down) = 0.75 - three down jumps for every up one
# - and the cap at 0.5 asserts that down jumps are never less likely than up
# ones at any date. That is a sign restriction, not a preference.
#
# WHAT THE DATA CURRENTLY SAYS ABOUT THAT RESTRICTION. Under skew-tight, whose
# pprob prior is centred at 0.575, the fitted by-year means run 0.484 to 0.602
# with a median of 0.565, and 85% of years sit ABOVE 0.5. So the cap binds for
# most of the sample and the posterior will pile against it rather than
# distribute inside it. That is not a malfunction - it is the same test the
# alpha-tiny arm ran, where a prior whose support excluded the full-sample
# value returned its own mean to four decimals and said so. But it changes
# what to read: pprob itself will be censored, so the informative output is
# what eta1 and eta2 do to ABSORB the constraint, and whether the fit to the
# window degrades (poc/variance_check.py).
#
# pdown-tight holds INFORMATIVENESS constant and moves only the centre. Its
# truncated sd is 0.1110 against skew-tight's 0.1105, so a change in the
# posterior is the centre moving and not the prior loosening - the confound
# that made the alpha arms hard to read until the floors were worked out.
SYSTEMATIC_PRIORS_PDOWN_TIGHT = dict(SYSTEMATIC_PRIORS_SKEW_TIGHT)
SYSTEMATIC_PRIORS_PDOWN_TIGHT["pprob_rv"] = (
    "Beta", {"alpha": 2.8385, "beta": 7.8121,           # truncated: mean 0.250
             "lower": 0.0, "upper": 0.5})               #            sd   0.111

# pdown-flat asserts the sign restriction and NOTHING about degree inside it.
# Mean 0.25 and support (0, 0.5] both hold exactly and by construction. It is
# the looser prior - sd 0.1443, 95% width 0.4750 against 0.4094 - so read its
# identification ratio against its own width, which the diagnostic now does.
SYSTEMATIC_PRIORS_PDOWN_FLAT = dict(SYSTEMATIC_PRIORS_SKEW_TIGHT)
SYSTEMATIC_PRIORS_PDOWN_FLAT["pprob_rv"] = (
    "Uniform", {"lower": 0.0, "upper": 0.5})            # mean 0.250 sd 0.1443


# ---------------------------------------------------------------------------
# pprob free. The arm for MEASURING the jump sign rather than asserting it.
# ---------------------------------------------------------------------------
# Why a cap cannot do this job, whatever value it takes. The economic claim is
# that P(down) RISES in stress, which is a statement about the gap between
# regimes. Under skew-tight the fitted by-year pprob means are 0.589 in calm
# years and 0.503 in stress ones - a gap of 0.086 - and the censoring a cap
# imposes falls on the CALM side, which is where the high values are:
#
#     cap    binds in        calm    stress     gap    share of gap surviving
#     0.50   17/20 (85%)    0.5000   0.4934   0.0066            8%
#     0.60    3/20 (15%)    0.5879   0.5028   0.0851           99%
#     0.65    0/20  (0%)      -        -        -             100%
#
# (Clipping is a LOWER bound on the damage: a truncated posterior piles AT the
# boundary, it does not merely stop there.)
#
# So 0.5 is the only cap with an economic meaning - down jumps are never the
# rarer kind - and it costs 92% of the very contrast the restriction is
# motivated by. 0.6 keeps the contrast and asserts nothing anyone would defend;
# "up jumps can be as likely as 60%" is not a claim. That arm has been here
# before: a293b64 added a pprob <= 0.6 cap and it was removed on 4 September
# 2026 because capping a parameter the window cannot identify constrains the
# prior rather than the data.
#
# That removal reasoning is now HALF out of date, which is why the question is
# worth reopening rather than re-closing. Under the corrected width-against-
# width ratio pprob sits at 0.937-0.945 on a 252-day window - still barely
# identified, the removal stands there - but at 756 days under skewtight-
# lamflat it reaches 0.562, with a width ratio of 0.600 against the sqrt(3) =
# 0.577 a purely statistical gain would give. pprob at three years is the one
# parameter in this study that gets the full theoretical benefit of the longer
# window.
#
# So: assert with pdown-tight / pdown-flat, MEASURE with this, and measure it
# at 756 where there is something to measure. Beta(1,1) rather than
# Uniform(0,1) to match how alpha-flat states the same thing, and to keep the
# sampler off a hard boundary in the transform.
SYSTEMATIC_PRIORS_PPROB_FLAT = dict(SYSTEMATIC_PRIORS_SKEW_TIGHT)
SYSTEMATIC_PRIORS_PPROB_FLAT["pprob_rv"] = (
    "Beta", {"alpha": 1.0, "beta": 1.0})                # mean 0.500 sd 0.2887


# ---------------------------------------------------------------------------
# Flat on BOTH alpha and pprob: assert neither the liquidity decay nor the
# jump sign, and let the window say what it can about each.
# ---------------------------------------------------------------------------
# The two parameters this study keeps discovering it does not want to assert.
# pprob is P(up jump), so Beta(1,1) says nothing about equity skew and lets a
# P(down) above one half, if it appears, be the data's doing. alpha is the
# liquidity decay; Beta(1,1) says nothing about its speed.
#
# WHAT TO EXPECT FROM EACH, BECAUSE THEY ARE NOT ALIKE.
#
# pprob is a real measurement at the longer windows - width ratio 0.895 at 252
# days but 0.462 at 756, better than the sqrt(3) = 0.577 that pure sample-size
# scaling would give. A flat prior there buys an estimate.
#
# alpha is not, at any window length. Under a flat prior it came back at 0.4396
# from a prior mean of 0.500 with a width ratio of essentially 1.00 - a mild
# tilt and no narrowing. It is identified only on the FULL sample, where it is
# 0.036, a half-life of 19.3 years against a one-year window that cannot see
# past its own end. So a flat prior on alpha is the honest choice precisely
# because it will report "we do not know" instead of reporting the prior's
# centre as an estimate - but the paper then has to take alpha from the
# full-sample fit wherever alpha actually ENTERS a calculation (the OU decay in
# the shock translation, the ES ladder), not from the rolling series.
#
# That costs nothing elsewhere. alpha is inert for the other five: pinning it
# at 0.037 against 0.46 - a factor of 12.7 - moves every other parameter by
# less than 0.7%, and 0.01 against 0.49, a factor of 49, moves sigma by 0.2%.
SYSTEMATIC_PRIORS_ALPHA_PPROB_FLAT = dict(SYSTEMATIC_PRIORS_SKEW_TIGHT)
SYSTEMATIC_PRIORS_ALPHA_PPROB_FLAT["alpha_rv"] = (
    "Beta", {"alpha": 1.0, "beta": 1.0})                # mean 0.500 sd 0.2887
SYSTEMATIC_PRIORS_ALPHA_PPROB_FLAT["pprob_rv"] = (
    "Beta", {"alpha": 1.0, "beta": 1.0})                # mean 0.500 sd 0.2887


SYSTEMATIC_PRIOR_SETS = {
    "paper":      None,                      # priors=None -> SYSTEMATIC_PRIORS
    "gaps":       SYSTEMATIC_PRIORS_GAPS,
    "asym":       SYSTEMATIC_PRIORS_ASYM,
    "skew":       SYSTEMATIC_PRIORS_SKEW,
    "skew-tight": SYSTEMATIC_PRIORS_SKEW_TIGHT,
    "alpha-flat": SYSTEMATIC_PRIORS_ALPHA_FLAT,
    "alpha-low":  SYSTEMATIC_PRIORS_ALPHA_LOW,
    "alpha-tiny": SYSTEMATIC_PRIORS_ALPHA_TINY,
    "skewtight-lamflat": SYSTEMATIC_PRIORS_SKEW_TIGHT_LAMFLAT,
    "pdown-tight": SYSTEMATIC_PRIORS_PDOWN_TIGHT,
    "pdown-flat":  SYSTEMATIC_PRIORS_PDOWN_FLAT,
    "pprob-flat":  SYSTEMATIC_PRIORS_PPROB_FLAT,
    "alpha-pprob-flat": SYSTEMATIC_PRIORS_ALPHA_PPROB_FLAT,
}

# Drawer suffix per arm. "paper" keeps the bare underlying id so the committed
# replication files under Study/Estimated Parameters PMLE/^SPX/ stay addressable.
STORE_SUFFIX = {"paper": "", "gaps": "__gaps", "asym": "__asym",
                "skew": "__skew", "skew-tight": "__skewtight",
                "alpha-flat": "__alphaflat", "alpha-low": "__alphalow",
                "alpha-tiny": "__alphatiny",
                "skewtight-lamflat": "__skewtightlamflat",
                "pdown-tight": "__pdowntight",
                "pdown-flat": "__pdownflat",
                "pprob-flat": "__pprobflat",
                "alpha-pprob-flat": "__alphapprobflat"}

# NOTE. An earlier design HELD alpha, eta1 and eta2 at their full-sample values
# on the grounds that a 252-day window cannot identify them. That was withdrawn.
# Holding a parameter reports a zero-width credible interval, which makes a
# rolling estimate look more certain than it is and bakes in a value that can
# no longer be questioned window by window. All six stay free. Where a window
# genuinely cannot identify a parameter the posterior sits on the prior, and
# that is visible in the ratio - which is information, not a defect to conceal.
#
# The ("Fixed", {"value": x}) prior spec remains available in _build_prior for
# cases where holding really is intended; nothing uses it by default.


def _build_prior(name, spec):
    """Instantiate one prior from a (distribution_name, kwargs) pair.

    ("Fixed", {"value": x}) pins the parameter to a constant instead of
    sampling it. Used for parameters a 252-day window cannot identify - alpha,
    whose half-life exceeds the window, and the jump-size decays, which see
    only ~7 jumps per side - so they are carried from a full-sample fit rather
    than re-estimated badly in every window.
    """
    dist, kw = spec
    if dist == "Fixed":
        return pt.as_tensor(np.float64(kw["value"]))
    kw = dict(kw)
    # Uniform owns lower/upper as its own parameters; for any other
    # distribution they mean truncation.
    if dist != "Uniform" and ("lower" in kw or "upper" in kw):
        lo, hi = kw.pop("lower", None), kw.pop("upper", None)
        return pm.Truncated(name, getattr(pm, dist).dist(**kw),
                            lower=lo, upper=hi)
    return getattr(pm, dist)(name=name, **kw)


def prior_moments(spec):
    """Analytic (mean, sd) of a prior spec, for the identification diagnostic.

    PyMC's Gamma is rate-parameterised: mean = alpha/beta, sd = sqrt(alpha)/beta.
    """
    dist, kw = spec
    if dist == "Fixed":
        return float(kw["value"]), 0.0
    if dist != "Uniform" and ("lower" in kw or "upper" in kw):
        # Truncated: integrate the base density over the retained interval.
        kw = dict(kw)
        lo = kw.pop("lower", None) or 1e-9
        hi = kw.pop("upper", None) or (1.0 - 1e-9)
        a, b = float(kw["alpha"]), float(kw["beta"])
        x = np.linspace(lo, hi, 200001)
        if dist == "Beta":
            f = x ** (a - 1) * (1 - x) ** (b - 1)
        else:
            f = x ** (a - 1) * np.exp(-b * x)
        Z = np.trapezoid(f, x)
        m = np.trapezoid(x * f, x) / Z
        v = np.trapezoid((x - m) ** 2 * f, x) / Z
        return float(m), float(np.sqrt(v))
    if dist == "Uniform":
        lo, hi = float(kw["lower"]), float(kw["upper"])
        return 0.5 * (lo + hi), (hi - lo) / np.sqrt(12.0)
    if dist == "Normal":
        return (float(kw.get("mu", 0.0)),
                float(kw.get("sigma", kw.get("sd", 1.0))))
    a, b = float(kw["alpha"]), float(kw["beta"])
    if dist == "Gamma":
        return a / b, np.sqrt(a) / b
    if dist == "Beta":
        return a / (a + b), np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    raise ValueError("no closed-form moments for %s" % dist)


def prior_ci_width(spec, prob=0.95):
    """Width of the prior's equal-tailed interval at `prob`.

    The denominator of the identification ratio. prior_moments gives the sd,
    which is the right summary of a prior but the WRONG denominator for a
    posterior width: dividing a width by an sd smuggles in a shape assumption,
    and the arms whose priors are flat are exactly the ones the assumption
    breaks on. Same quantity on both sides, so an unmoved posterior reads 1.00.

    Equal-tailed to match CI_CONVENTION, not highest-density: for a Uniform
    the HDI is the whole support and would read 1.00 for a prior that a
    genuinely informative likelihood should still be able to beat.

    HOW BIG THE ARTEFACT ACTUALLY WAS, per arm. The commit that introduced
    this function gave a table of old floors naming sigma, pprob, lamb and
    eta1 - those numbers are right for the distributions beside them but the
    distributions are the PAPER arm's, not skew-tight's, and skew-tight is
    what every rolling run since has used. Under skew-tight the priors are
    much tighter, a tight prior is closer to Gaussian in shape, and the
    normal approximation the old denominator made is correspondingly better:

        dSIGMA  Gamma(2, 10)          95% W  0.5329   old floor 0.961
        dALPHA  Beta(9.5, 9.5)               0.4329              0.988
        dPPROB  Beta(10.925, 8.075)          0.4278              0.987
        dLAMB   Gamma(3, 0.5)              13.2120              0.973
        dETA1   Gamma(16, 0.32)            48.7339              0.995
        dETA2   Gamma(16, 0.64)            24.3669              0.995

    So this fix moves a skew-tight verdict by at most 4%, not the 6-16% that
    table implied. Where it genuinely matters is the arm it was found on:
    alpha-flat (Beta(1,1)) and alpha-tiny (Uniform(0, 0.02)) really do have a
    floor of 0.840, and an alpha posterior sitting exactly on its prior really
    was being read there as a 16% narrowing. The fix is right; the claim that
    it rewrote the skew-tight numbers was not.
    """
    from scipy import stats

    lo_q, hi_q = 0.5 * (1.0 - prob), 0.5 * (1.0 + prob)
    dist, kw = spec
    if dist == "Fixed":
        return 0.0
    if dist != "Uniform" and ("lower" in kw or "upper" in kw):
        # Truncated: invert the CDF on the same grid prior_moments integrates.
        kw = dict(kw)
        lo = kw.pop("lower", None) or 1e-9
        hi = kw.pop("upper", None) or (1.0 - 1e-9)
        a, b = float(kw["alpha"]), float(kw["beta"])
        x = np.linspace(lo, hi, 200001)
        f = (x ** (a - 1) * (1 - x) ** (b - 1) if dist == "Beta"
             else x ** (a - 1) * np.exp(-b * x))
        cdf = np.cumsum(f)
        cdf = cdf / cdf[-1]
        return float(np.interp(hi_q, cdf, x) - np.interp(lo_q, cdf, x))
    if dist == "Uniform":
        return prob * (float(kw["upper"]) - float(kw["lower"]))
    if dist == "Normal":
        s = float(kw.get("sigma", kw.get("sd", 1.0)))
        return float(stats.norm.ppf(hi_q) - stats.norm.ppf(lo_q)) * s
    a, b = float(kw["alpha"]), float(kw["beta"])
    if dist == "Gamma":                      # PyMC Gamma is rate-parameterised
        return float(stats.gamma.ppf(hi_q, a, scale=1.0 / b)
                     - stats.gamma.ppf(lo_q, a, scale=1.0 / b))
    if dist == "Beta":
        return float(stats.beta.ppf(hi_q, a, b) - stats.beta.ppf(lo_q, a, b))
    raise ValueError("no interval for %s" % dist)


def pmle_kimyirisk_systematic(
        sys_returns: NDArray[np.float64],
        delta_t: NDArray[np.float64],
        seed_number: np.uint64 = np.uint64(20240114),
        n_mc_paths: int = 10_000,
        nuts_sampler: Literal["pymc", "nutpie", "jax", "numpyro", "blackjax"] = "nutpie",
        is_progress_bar: bool = False,
        priors: dict = None
) -> dict:
    SEED = np.uint64(seed_number)

    # priors=None reproduces the published configuration exactly.
    pr = dict(SYSTEMATIC_PRIORS)
    if priors:
        unknown = set(priors) - set(pr)
        if unknown:
            raise ValueError("unknown prior key(s): %s" % sorted(unknown))
        pr.update(priors)

    N_SIMS_MCMC = n_mc_paths
    # use PyMC to sampler from log-likelihood
    with pm.Model():
        sigma = _build_prior("sigma", pr["sigma"])

        alpha_rv = _build_prior("alpha_rv", pr["alpha_rv"])
        alpha = pm.Deterministic("alpha", pt.log(alpha_rv))

        pprob_rv = _build_prior("pprob_rv", pr["pprob_rv"])
        pprob = pm.Deterministic("pprob", pt.log(pprob_rv))

        lamb = _build_prior("lamb", pr["lamb"])
        eta1 = _build_prior("eta1", pr["eta1"])
        eta2 = _build_prior("eta2", pr["eta2"])

        observed_data = np.cumsum(sys_returns).reshape((-1, 1))

        pm.CustomDist(
            "likelihood",
            alpha,
            sigma,
            pprob,
            lamb,
            eta1,
            eta2,
            delta_t,
            observed=observed_data,
            logp=_dist_loglike_systematic,
        )

        rng_pymc = np.random.default_rng(SEED)
        idata_systematic = pm.sample(JGL_DRAWS or N_SIMS_MCMC, chains=JGL_CHAINS, tune=1000,
                                     cores=JGL_CORES, target_accept=0.95,
                                     progressbar=is_progress_bar, random_seed=rng_pymc, nuts_sampler=nuts_sampler)

    # Summaries come from Library.PosteriorSummary, not az.summary. The old code
    # read CI bounds by COLUMN POSITION, which ArviZ 1.0 breaks, and their
    # meaning was a library default that changed (94% HDI -> 89% ETI). The
    # convention is now fixed at 95% equal-tailed in one place.
    #
    # alpha and pprob are summarised on their CONSTRAINED variables (alpha_rv,
    # pprob_rv) rather than by exponentiating the summary of log(.). The old
    # np.exp(mean(log x)) reported a geometric mean and shifted the interval.
    def _summ(var, key):
        # A Fixed parameter is not a random variable, so it has no posterior to
        # summarise. Report the constant with a zero-width interval, which is
        # the honest representation: no uncertainty was estimated because none
        # was estimated here.
        if pr[key][0] == "Fixed":
            v = float(pr[key][1]["value"])
            return v, v, v
        return summarize(idata_systematic, var)

    _warn_low_ess(idata_systematic,
                  ["sigma", "alpha_rv", "pprob_rv", "lamb", "eta1", "eta2"],
                  "systematic")

    a_m, a_lo, a_hi = _summ("alpha_rv", "alpha_rv")
    s_m, s_lo, s_hi = _summ("sigma", "sigma")
    p_m, p_lo, p_hi = _summ("pprob_rv", "pprob_rv")
    l_m, l_lo, l_hi = _summ("lamb", "lamb")
    e1_m, e1_lo, e1_hi = _summ("eta1", "eta1")
    e2_m, e2_lo, e2_hi = _summ("eta2", "eta2")

    return {
        'dALPHA': ParamsResults(dMEAN=a_m, dCI_LOWER=a_lo, dCI_UPPER=a_hi),
        'dSIGMA': ParamsResults(dMEAN=s_m, dCI_LOWER=s_lo, dCI_UPPER=s_hi),
        'dPPROB': ParamsResults(dMEAN=p_m, dCI_LOWER=p_lo, dCI_UPPER=p_hi),
        'dLAMB': ParamsResults(dMEAN=l_m, dCI_LOWER=l_lo, dCI_UPPER=l_hi),
        'dETA1': ParamsResults(dMEAN=e1_m, dCI_LOWER=e1_lo, dCI_UPPER=e1_hi),
        'dETA2': ParamsResults(dMEAN=e2_m, dCI_LOWER=e2_lo, dCI_UPPER=e2_hi)
    }


# ---------------------------------------------------------------------------
# Priors on the idiosyncratic block
# ---------------------------------------------------------------------------
# These name the UNCONSTRAINED variable in each case, because the transform
# that follows is structural and is not a modelling choice: kappai, gammai and
# betai are exp() of their _rv, and rhoix is 2*rhoix_rv - 1 carried through
# arctanh/tanh. So a spec here changes the prior and never the parameterisation.
#
# WHICH OF THESE THE DATA CAN ACTUALLY SPEAK TO. In KimYiLogLike the five enter
# only through two scalars:
#
#     _variance() = (sigma*betai)^2 + 2*sigma*betai*kappai*rhoix + kappai^2
#     _drift()    = mui + 0.5*(sigma*betai)^2 - sigma*betai*kappai*rhoix
#
# Two equations, four unknowns - mui, betai, kappai, rhoix - and rhoix appears
# ONLY inside the product sigma*betai*kappai*rhoix. The idiosyncratic
# likelihood never sees the systematic SERIES either, only systematic
# constants, so there is no cross-moment to break the tie. gammai is the
# exception: it rescales the jump decays as eta/gammai and is separately
# identified.
#
# So a prior on rhoix is not a starting point the data refines - it PICKS A
# POINT on a two-dimensional flat manifold, and betai and kappai follow it
# there. That is why the flat arm below is a diagnostic and not an estimator.
IDIOSYNCRATIC_PRIORS = {
    "mui":       ("Normal", {"mu": 0.0, "sigma": 1.0}),   # mean 0.000 sd 1.000
    "kappai_rv": ("Gamma",  {"alpha": 2.0, "beta": 1.0}), # mean 2.000 sd 1.414
    "gamma_rv":  ("Gamma",  {"alpha": 3.0, "beta": 1.0}), # mean 3.000 sd 1.732
    "betai_rv":  ("Gamma",  {"alpha": 3.0, "beta": 1.0}), # mean 3.000 sd 1.732
    "rhoix_rv":  ("Beta",   {"alpha": 5.0, "beta": 2.0}), # rhoix mean 0.4286
}

# Flat on mui and rhoix, nothing else touched.
#
# mui is already close to flat and, unlike rhoix, IDENTIFIED: at 504 days the
# likelihood pins a drift to about sigma/sqrt(T) = 0.11 annualised, so
# Normal(0,1) is already nine times wider than the data's own resolution.
# Uniform(-5, 5) is literally flat and its edges sit ~45 likelihood sds away,
# so they cannot bind. Expect this half to change nothing, which is the point:
# it shows the drift prior was never doing the work.
#
# rhoix_rv Uniform gives rhoix = 2*rhoix_rv - 1 flat on (-1, 1) with mean 0 -
# no assertion that a name's idiosyncratic Brownian is positively correlated
# with the systematic one. BOUNDED JUST INSIDE: rhoix is carried through
# arctanh, which is infinite at +-1. Beta(5,2) has zero density there and is
# safe; a uniform is not. 0.0005/0.9995 puts rhoix in (-0.999, 0.999) and
# keeps arctanh finite, at the cost of 0.1% of the support at each end.
IDIOSYNCRATIC_PRIORS_FLAT_MU_RHOIX = dict(IDIOSYNCRATIC_PRIORS)
IDIOSYNCRATIC_PRIORS_FLAT_MU_RHOIX["mui"] = (
    "Uniform", {"lower": -5.0, "upper": 5.0})             # mean 0.000 sd 2.887
IDIOSYNCRATIC_PRIORS_FLAT_MU_RHOIX["rhoix_rv"] = (
    "Uniform", {"lower": 0.0005, "upper": 0.9995})        # rhoix mean 0.000

# rhoix flat, mui left at Normal(0, 1). THIS is the arm to run against
# "paper": the two differ in exactly one prior, so any movement in betai or
# kappai between them is attributable to rhoix alone. flat-mu-rhoix above
# moves two things at once and cannot support that reading - it stays only
# because an arm that has written a drawer should not be deleted out from
# under it.
#
# Leaving mui at Normal(0, 1) costs nothing. It is the one idiosyncratic
# parameter the window genuinely identifies - at 504 days the likelihood pins
# a drift to about sigma/sqrt(T) = 0.11 annualised, so a prior nine times
# wider than that is already not doing any work, and widening it further only
# makes mui's identification RATIO look better without changing its posterior.
# A ratio against an arbitrary support is a statement about the support.
IDIOSYNCRATIC_PRIORS_RHOIX_FLAT = dict(IDIOSYNCRATIC_PRIORS)
IDIOSYNCRATIC_PRIORS_RHOIX_FLAT["rhoix_rv"] = (
    "Uniform", {"lower": 0.0005, "upper": 0.9995})        # rhoix mean 0.000

IDIOSYNCRATIC_PRIOR_SETS = {
    "paper": None,                        # priors=None -> IDIOSYNCRATIC_PRIORS
    "rhoix-flat": IDIOSYNCRATIC_PRIORS_RHOIX_FLAT,
    "flat-mu-rhoix": IDIOSYNCRATIC_PRIORS_FLAT_MU_RHOIX,
}
IDIO_STORE_SUFFIX = {"paper": "", "rhoix-flat": "__rhoixflat",
                     "flat-mu-rhoix": "__flatmurhoix"}


def pmle_kimyirisk_idiosyncratic(
        idi_returns: NDArray[np.float64],
        params_sys: dict,
        delta_t: NDArray[np.float64],
        seed_number: np.uint64 = np.uint64(20240114),
        n_mc_paths: int = 10_000,
        nuts_sampler: Literal["pymc", "nutpie", "jax", "numpyro", "blackjax"] = "nutpie",
        is_progress_bar: bool = False,
        priors: dict = None
) -> dict:
    SEED = np.uint64(seed_number)
    Delta_t = delta_t

    N_SIMS_MCMC = n_mc_paths

    alpha = params_sys["dALPHA"]
    sigma = params_sys["dSIGMA"]
    pprob = params_sys["dPPROB"]
    lamb = params_sys["dLAMB"]
    eta1 = params_sys["dETA1"]
    eta2 = params_sys["dETA2"]

    # priors=None reproduces the published configuration exactly.
    pr = dict(IDIOSYNCRATIC_PRIORS)
    if priors:
        unknown = set(priors) - set(pr)
        if unknown:
            raise ValueError("unknown idiosyncratic prior key(s): %s"
                             % sorted(unknown))
        pr.update(priors)

    with pm.Model():
        # The pymc variable names are load-bearing: _warn_low_ess and every
        # summarize() call below address them by name, and "gamma_rv" is the
        # odd one out. Keep them exactly as they are.
        mui = _build_prior("mui", pr["mui"])

        kappai_rv = _build_prior("kappai_rv", pr["kappai_rv"])
        kappai = pm.Deterministic("kappai", pt.log(kappai_rv))

        gammai_rv = _build_prior("gamma_rv", pr["gamma_rv"])
        gammai = pm.Deterministic("gammai", pt.log(gammai_rv))

        betai_rv = _build_prior("betai_rv", pr["betai_rv"])
        betai = pm.Deterministic("betai", pt.log(betai_rv))

        rhoix_rv = _build_prior("rhoix_rv", pr["rhoix_rv"])
        loc, scale = -1., 2.
        rhoix = pm.Deterministic("rhoix", pt.arctanh((scale * rhoix_rv) + loc))

        observed_data = np.cumsum(idi_returns).reshape((-1, 1))

        pm.CustomDist(
            "likelihood",
            mui,
            kappai,
            gammai,
            betai,
            rhoix,
            alpha,
            sigma,
            pprob,
            lamb,
            eta1,
            eta2,
            Delta_t,
            observed=observed_data,
            logp=_dist_loglike_idiosyncratic
        )

        rng_pymc = np.random.default_rng(SEED)
        idata_idiosyncratic = pm.sample(JGL_DRAWS or N_SIMS_MCMC, chains=JGL_CHAINS, tune=1000,
                                     cores=JGL_CORES, target_accept=0.95,
                                        progressbar=is_progress_bar, random_seed=rng_pymc, nuts_sampler=nuts_sampler)

    # See the note in pmle_kimyirisk_systematic. Transforms are applied to the
    # DRAWS, not to the summary, so means are arithmetic and intervals are the
    # transformed quantiles rather than quantiles of the transform's input.
    _warn_low_ess(idata_idiosyncratic,
                  ["mui", "kappai_rv", "gamma_rv", "betai_rv", "rhoix_rv"],
                  "idiosyncratic")

    mu_m, mu_lo, mu_hi = summarize(idata_idiosyncratic, "mui")
    k_m, k_lo, k_hi = summarize(idata_idiosyncratic, "kappai", transform=np.exp)
    g_m, g_lo, g_hi = summarize(idata_idiosyncratic, "gammai", transform=np.exp)
    b_m, b_lo, b_hi = summarize(idata_idiosyncratic, "betai", transform=np.exp)
    r_m, r_lo, r_hi = summarize(idata_idiosyncratic, "rhoix", transform=np.tanh)

    return {
        'dMUI': ParamsResults(dMEAN=mu_m, dCI_LOWER=mu_lo, dCI_UPPER=mu_hi),
        'dKAPPAI': ParamsResults(dMEAN=k_m, dCI_LOWER=k_lo, dCI_UPPER=k_hi),
        'dGAMMAI': ParamsResults(dMEAN=g_m, dCI_LOWER=g_lo, dCI_UPPER=g_hi),
        'dBETAI': ParamsResults(dMEAN=b_m, dCI_LOWER=b_lo, dCI_UPPER=b_hi),
        'dRHOIX': ParamsResults(dMEAN=r_m, dCI_LOWER=r_lo, dCI_UPPER=r_hi)
    }


class KimYiRiskEngine:

    def __init__(
            self,
            mui: List[ParametersBase],
            kappai: List[ParametersBase],
            gammai: List[ParametersBase],
            betai: List[ParametersBase],
            rhoix: List[ParametersBase],
            alpha: ParametersBase,
            sigma: ParametersBase,
            pprob: ParametersBase,
            lamb: ParametersBase,
            eta1: ParametersBase,
            eta2: ParametersBase,
            end_dt: NDArray[np.float64],
            rhoij: List[ParametersBase]=None
    ):
        self.gammai = gammai
        self.pprob = pprob
        self.qprob = 1. - self.pprob
        self.eta1 = eta1
        self.eta2 = eta2

        self.alpha_dt = alpha, end_dt
        self.lamb_dt = lamb, end_dt

        self.drift_dt = mui, sigma, betai, kappai, rhoix, end_dt
        self.variance_dt = sigma, betai, kappai, rhoix, end_dt

        if rhoij is None:
            self.L = np.array(1.).reshape((-1, 1))
        else:
            correl = np.array([x.integral(np.array(0.), np.array(1.)) for x in rhoij]).reshape((-1, 1))
            self.L = np.linalg.cholesky(get_corr_mat(correl, len(mui)))

    def est_liquidity_process(self, observed_data: NDArray[np.float64]) -> NDArray[np.float64]:

        m, n = observed_data.shape

        r_cumsum = np.cumsum(observed_data, axis=0)

        liquidity_process = np.zeros(shape=(m + 1, n))

        drift_scaler = ((np.arange(m) + 1) * self.drift_dt).T
        liquidity_process[1:] = r_cumsum - drift_scaler

        return liquidity_process

    def random(self, rng: RandomBase=None, size: Tuple[int, int, int]=None) -> Tuple:

        if rng is None:
            rng = RandomMT19937(np.int64(20240114))

        if size is None:
            n_assets, n_sims, n_steps = 1, 1, 1
        else:
            n_assets, n_sims, n_steps = size

        Z = np.zeros(shape=(n_assets, n_sims, n_steps))
        N = np.zeros(shape=(n_sims, n_steps), dtype=np.int64)
        Y = np.zeros(shape=(n_sims, n_steps))

        rng.get_gaussian(variates=Z)

        rng.get_poisson(
            variates=N,
            lamb=self.lamb_dt
        )

        rng.get_aded(
            variates=Y,
            n=N,
            eta1=self.eta1,
            eta2=self.eta2,
            pprob=self.pprob
        )

        psi = np.zeros(shape=(n_assets, n_sims, n_steps + 1))
        dpsi = np.zeros(shape=(n_assets, n_sims, n_steps + 1))
        returns = np.zeros(shape=(n_assets, n_sims, n_steps + 1))

        # do simulations
        for t in range(n_steps):
            psi[:, :, t + 1]  = (1. - self.alpha_dt) * psi[:, :, t]
            psi[:, :, t + 1] += np.sqrt(self.variance_dt) * np.dot(self.L, Z[:, :, t])
            psi[:, :, t + 1] += self.gammai * Y[:, t]

        dpsi[:, :, 1:] = psi[:, :, 1:] - psi[:, :, :-1]

        drift_dt = np.tile(self.drift_dt[:, :, np.newaxis], (1, 1, n_sims))
        drift_dt = np.tile(drift_dt[:, :, :, np.newaxis], (1, 1, 1, n_steps)).squeeze()
        returns[:, :, 1:] = dpsi[:, :, 1:] + drift_dt

        return returns, psi, dpsi

    @property
    def alpha_dt(self) -> NDArray[np.float64]:
        return self._alpha_dt

    @alpha_dt.setter
    def alpha_dt(self, values_tuple: Tuple[ParametersBase, NDArray[np.float64]]) -> None:
        if not isinstance(values_tuple, tuple) or len(values_tuple) != 2:
            raise ValueError("Setter for 'alpha_dt' expects a tuple of two elements.")
        parameter, end_time = values_tuple
        self._alpha_dt = np.array(parameter.integral(time1=np.array(0.), time2=end_time)).reshape((-1, 1))

    @property
    def lamb_dt(self) -> NDArray[np.float64]:
        return self._lamb_dt

    @lamb_dt.setter
    def lamb_dt(self, values_tuple: Tuple[ParametersBase, NDArray[np.float64]]) -> None:
        if not isinstance(values_tuple, tuple) or len(values_tuple) != 2:
            raise ValueError("Setter for 'lamb_dt' expects a tuple of two elements.")
        parameter, end_time = values_tuple
        self._lamb_dt = np.array(parameter.integral(time1=np.array(0.), time2=end_time)).reshape((-1, 1))

    @property
    def drift_dt(self) -> NDArray[np.float64]:
        return self._drift_dt

    @drift_dt.setter
    def drift_dt(
            self,
            values_tuple: Tuple[
                List[ParametersBase],
                ParametersBase,
                List[ParametersBase],
                List[ParametersBase],
                List[ParametersBase],
                NDArray[np.float64]
            ]
    ) -> None:
        if not isinstance(values_tuple, tuple) or len(values_tuple) != 6:
            raise ValueError("Setter for 'drift_dt' expects a tuple of six elements.")
        mui, sigma, betai, kappai, rhoix, end_dt = values_tuple

        mui_dt = np.array([x.integral(np.array(0.), np.array(1.)) for x in mui]).reshape((-1, 1))
        sigma_dt = sigma.integral(np.array(0.), np.array(1.)).reshape((-1, 1))
        betai_dt = np.array([x.integral(np.array(0.), np.array(1.)) for x in betai]).reshape((-1, 1))
        rhoix_dt = np.array([x.integral(np.array(0.), np.array(1.)) for x in rhoix]).reshape((-1, 1))
        kappai_dt = np.array([x.integral(np.array(0.), np.array(1.)) for x in kappai]).reshape((-1, 1))
        self._drift_dt = (mui_dt + .5 * (sigma_dt * betai_dt)**2 - sigma_dt * betai_dt * kappai_dt * rhoix_dt) * end_dt

    @property
    def variance_dt(self) -> NDArray[np.float64]:
        return self._variance_dt

    @variance_dt.setter
    def variance_dt(
            self,
            values_tuple: Tuple[
                ParametersBase,
                List[ParametersBase],
                List[ParametersBase],
                List[ParametersBase],
                NDArray[np.float64]
            ]
    ) -> None:
        if not isinstance(values_tuple, tuple) or len(values_tuple) != 5:
            raise ValueError("Setter for 'variance_dt' expects a tuple of five elements.")
        sigma, betai, kappai, rhoix, end_dt = values_tuple
        sigma_dt = sigma.integral(np.array(0.), np.array(1.)).reshape((-1, 1))
        betai_dt = np.array([x.integral(np.array(0.), np.array(1.)) for x in betai]).reshape((-1, 1))
        rhoix_dt = np.array([x.integral(np.array(0.), np.array(1.)) for x in rhoix]).reshape((-1, 1))
        kappai_dt = np.array([x.integral(np.array(0.), np.array(1.)) for x in kappai]).reshape((-1, 1))
        self._variance_dt = (
                (sigma_dt * betai_dt)**2 + (2. * sigma_dt * betai_dt * kappai_dt * rhoix_dt) + kappai_dt**2
        ) * end_dt

    @property
    def gammai(self) -> NDArray[np.float64]:
        return self._gammai

    @gammai.setter
    def gammai(self, value: List[ParametersBase]) -> None:
        self._gammai = np.array([x.integral(time1=np.array(0), time2=np.array(1)) for x in value]).reshape((-1, 1))

    @property
    def pprob(self) -> NDArray[np.float64]:
        return self._pprob

    @pprob.setter
    def pprob(self, value: ParametersBase) -> None:
        self._pprob = np.array(value.integral(time1=np.array(0), time2=np.array(1))).reshape((-1, 1))

    @property
    def eta1(self) -> NDArray[np.float64]:
        return self._eta1

    @eta1.setter
    def eta1(self, value: ParametersBase) -> None:
        self._eta1 = np.array(value.integral(time1=np.array(0), time2=np.array(1))).reshape((-1, 1))

    @property
    def eta2(self) -> NDArray[np.float64]:
        return self._eta2

    @eta2.setter
    def eta2(self, value: ParametersBase) -> None:
        self._eta2 = np.array(value.integral(time1=np.array(0), time2=np.array(1))).reshape((-1, 1))


class KimYiLogLike:

    def __init__(
            self,
            mui: NDArray[np.float64],
            kappai: NDArray[np.float64],
            gammai: NDArray[np.float64],
            betai: NDArray[np.float64],
            rhoix: NDArray[np.float64],
            alpha: NDArray[np.float64],
            sigma: NDArray[np.float64],
            pprob: NDArray[np.float64],
            lamb: NDArray[np.float64],
            eta1: NDArray[np.float64],
            eta2: NDArray[np.float64],
            dt: NDArray[np.float64]
    ):
        self.mui = pt.as_tensor(mui)
        self.kappai = pt.as_tensor(kappai)
        self.gammai = pt.as_tensor(gammai)
        self.betai = pt.as_tensor(betai)
        self.rhoix = pt.as_tensor(rhoix)
        self.alpha = pt.as_tensor(alpha)
        self.sigma = pt.as_tensor(sigma)
        self.pprob = pt.as_tensor(pprob)
        self.qprob = pt.as_tensor(1. - pprob)
        self.lamb = pt.as_tensor(lamb)
        self.eta1 = pt.as_tensor(eta1)
        self.eta2 = pt.as_tensor(eta2)
        self.dt = pt.as_tensor(dt)

    def logp(self, y: pt.TensorVariable) -> pt.TensorVariable:

        sigma_squared = self._variance()
        sigma_root_dt = pt.sqrt(sigma_squared * self.dt)

        m, n = y.shape
        drift_scaler = pt.zeros(m).reshape((-1, 1))
        drift_scaler = drift_scaler[1:].set(pt.arange(m-1).reshape((-1, 1)) + 1)
        y = y - drift_scaler * self._drift() * self.dt
        # y = y[0].set(0.)
        # x = pytensor.clone_replace(y[:-1])
        x = y[:-1]
        y = y[1:]

        diff_y_x = y - (1. - self.alpha * self.dt) * x
        eta1 = self.eta1 / self.gammai
        eta2 = self.eta2 / self.gammai

        norm_dist = pm.Normal.dist()

        # Each jump branch carries its OWN normal CDF. The previous form
        # accumulated the down branch with += BEFORE applying the down Phi with
        # *=, so the down Phi multiplied the running sum and the up branch was
        # weighted by Phi_up * Phi_down. The result was not a density (it
        # integrated to 0.986) and was wrong by a factor of ~8 on positive
        # moves, while negative moves were almost unaffected - which is why it
        # went unnoticed. Systematically suppressing the up branch biases pprob
        # downward, since pprob multiplies exactly that branch.
        up_branch  = self.pprob * eta1 * pt.exp(0.5 * sigma_squared * eta1**2 * self.dt - diff_y_x * eta1)
        up_branch *= pt.exp(pm.logcdf(norm_dist, value=(diff_y_x - sigma_squared * eta1 * self.dt) / sigma_root_dt))

        down_branch  = self.qprob * eta2 * pt.exp(0.5 * sigma_squared * eta2**2 * self.dt + diff_y_x * eta2)
        down_branch *= pt.exp(pm.logcdf(norm_dist, value=(diff_y_x + sigma_squared * eta2 * self.dt) / sigma_root_dt * -1.))

        g_x  = (up_branch + down_branch) * self.lamb * self.dt
        g_x += (1. - self.lamb * self.dt) / sigma_root_dt * pt.exp(pm.logp(norm_dist, value=diff_y_x / sigma_root_dt))

        return pt.log(g_x)

    def _drift(self) -> pt.TensorVariable:
        drift  = self.mui
        drift += 0.5 * (self.sigma * self.betai)**2
        drift -= self.sigma * self.betai * self.kappai * self.rhoix
        return drift.reshape((-1, 1))

    def _variance(self) -> pt.TensorVariable:
        variance  = (self.sigma * self.betai)**2
        variance += 2. * self.sigma * self.betai * self.kappai * self.rhoix
        variance += self.kappai**2
        return variance.reshape((-1, 1))
