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
# These are the values the paper was estimated with, kept here as data rather
# than as literals inside the model so a sensitivity arm can override one
# without a second copy of the model definition. Passing priors=None
# reproduces the published configuration exactly.
#
# Note what the DEFAULTS assert. eta1 ~ Gamma(50,1) and eta2 ~ Gamma(25,1)
# say the mean down-jump (1/eta2 = 4%) is twice the mean up-jump (1/eta1 = 2%)
# BEFORE any data is seen. Since both parameters come back prior-driven, that
# assertion passes untouched into the posterior - so the model's negative jump
# skew, which is its most equity-plausible feature, is currently an assumption
# rather than an estimate. Giving the two the same prior and asking whether the
# data separates them is the test; see poc/prior_sensitivity.py.
SYSTEMATIC_PRIORS = {
    "sigma":    ("Gamma", {"alpha": 1.0,  "beta": 1.0}),
    "alpha_rv": ("Beta",  {"alpha": 5.0,  "beta": 2.0}),
    "pprob_rv": ("Beta",  {"alpha": 5.0,  "beta": 2.0}),
    "lamb":     ("Gamma", {"alpha": 10.0, "beta": 0.5}),
    "eta1":     ("Gamma", {"alpha": 50.0, "beta": 1.0}),
    "eta2":     ("Gamma", {"alpha": 25.0, "beta": 1.0}),
}

# ---------------------------------------------------------------------------
# Full-sample calibration, 2007-01-01 to 2026-08-31 (~4,950 returns, ~193 jumps)
# ---------------------------------------------------------------------------
# eta1, eta2 and pprob are taken from the arm that asserts NOTHING: the two
# decays share a Gamma(35,1) prior and pprob is Uniform(0,1). Starting from
# zero asserted separation they ended 17.9 apart - three prior sd - and landed
# within 1.7% of eta2 from the asymmetric-prior fit. Convergence from opposite
# priors is what makes this a measurement rather than an assumption.
#
# alpha is from the baseline full-sample fit. It cannot be estimated in a
# window at all: its half-life is 19 years against a one-year window.
FULL_SAMPLE = {
    "alpha_rv": 0.036,      # half-life 19.3 yr; Psi is near a random walk
    "sigma":    0.105,
    "pprob_rv": 0.575,      # +-0.026 under a flat prior, 2.9 sd above 0.5
    "lamb":     76.99,      # ~77 jumps/yr, one every 3.3 trading days
    "eta1":     78.59,      # mean up jump   1.27%
    "eta2":     60.68,      # mean down jump 1.65%  -> jump skew -0.55
}

# Priors recentred on the full-sample calibration, with deliberately generous
# widths. This is empirical Bayes: the same returns inform the prior and are
# then fitted again, so the widths must be wide enough that the prior guides
# rather than dictates. Compare the old centres - sigma at 1.0 against a fitted
# 0.105, lambda at 20 against 77, pprob at 0.714 against 0.575 - all of which
# dragged the rolling estimates.
# eta1 and eta2 get the SAME prior here. Arm G showed the data separates them
# unaided - from an identical Gamma(35,1) start they finished 17.9 apart, three
# prior sd - so there is no reason left to assert the asymmetry, and every
# reason not to. The shared centre sits at the midpoint of the two full-sample
# values, 70, wide enough that both 60.7 and 78.6 are inside one sd.
_ETA_SHARED = ("Gamma", {"alpha": 12.0, "beta": 0.1714})    # mean 70.0  sd 20.2

SYSTEMATIC_PRIORS_RECENTRED = {
    "sigma":    ("Gamma", {"alpha": 2.0,  "beta": 10.0}),   # mean 0.200 sd 0.141
    "alpha_rv": ("Beta",  {"alpha": 1.0,  "beta": 26.0}),   # mean 0.037 sd 0.036
    "pprob_rv": ("Beta",  {"alpha": 2.3,  "beta": 1.7}),    # mean 0.575 sd 0.221
    "lamb":     ("Gamma", {"alpha": 9.0,  "beta": 0.15}),   # mean 60.0  sd 20.0
    "eta1":     _ETA_SHARED,
    "eta2":     _ETA_SHARED,
}

# The two-stage specification. alpha and the two decays are HELD at their
# full-sample values because a 252-day window cannot identify them - alpha on
# span, the decays on jump count - and sigma, lambda and pprob stay free
# because it can. Every number is then either estimated from data that can
# identify it, or explicitly carried from a fit that could.
TWO_STAGE_PRIORS = dict(SYSTEMATIC_PRIORS_RECENTRED)
TWO_STAGE_PRIORS.update({
    "alpha_rv": ("Fixed", {"value": FULL_SAMPLE["alpha_rv"]}),
    "eta1":     ("Fixed", {"value": FULL_SAMPLE["eta1"]}),
    "eta2":     ("Fixed", {"value": FULL_SAMPLE["eta2"]}),
})


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
    return getattr(pm, dist)(name=name, **kw)


def prior_moments(spec):
    """Analytic (mean, sd) of a prior spec, for the identification diagnostic.

    PyMC's Gamma is rate-parameterised: mean = alpha/beta, sd = sqrt(alpha)/beta.
    """
    dist, kw = spec
    if dist == "Fixed":
        return float(kw["value"]), 0.0
    if dist == "Uniform":
        lo, hi = float(kw["lower"]), float(kw["upper"])
        return 0.5 * (lo + hi), (hi - lo) / np.sqrt(12.0)
    a, b = float(kw["alpha"]), float(kw["beta"])
    if dist == "Gamma":
        return a / b, np.sqrt(a) / b
    if dist == "Beta":
        return a / (a + b), np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    raise ValueError("no closed-form moments for %s" % dist)


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
        idata_systematic = pm.sample(N_SIMS_MCMC, chains=4, tune=1000, cores=4, target_accept=0.95,
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


def pmle_kimyirisk_idiosyncratic(
        idi_returns: NDArray[np.float64],
        params_sys: dict,
        delta_t: NDArray[np.float64],
        seed_number: np.uint64 = np.uint64(20240114),
        n_mc_paths: int = 10_000,
        nuts_sampler: Literal["pymc", "nutpie", "jax", "numpyro", "blackjax"] = "nutpie",
        is_progress_bar: bool = False
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

    with pm.Model():
        mui = pm.Normal(name="mui")

        kappai_rv = pm.Gamma(name="kappai_rv", alpha=2., beta=1.)
        kappai = pm.Deterministic("kappai", pt.log(kappai_rv))

        gammai_rv = pm.Gamma(name="gamma_rv", alpha=3., beta=1.)
        gammai = pm.Deterministic("gammai", pt.log(gammai_rv))

        betai_rv = pm.Gamma(name="betai_rv", alpha=3., beta=1.)
        betai = pm.Deterministic("betai", pt.log(betai_rv))

        rhoix_rv = pm.Beta(name="rhoix_rv", alpha=5, beta=2.)
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
        idata_idiosyncratic = pm.sample(N_SIMS_MCMC, chains=4, tune=1000, cores=4, target_accept=0.95,
                                        progressbar=is_progress_bar, random_seed=rng_pymc, nuts_sampler=nuts_sampler)

    # See the note in pmle_kimyirisk_systematic. Transforms are applied to the
    # DRAWS, not to the summary, so means are arithmetic and intervals are the
    # transformed quantiles rather than quantiles of the transform's input.
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
