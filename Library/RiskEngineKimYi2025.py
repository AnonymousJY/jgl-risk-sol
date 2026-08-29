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


def pmle_kimyirisk_systematic(
        sys_returns: NDArray[np.float64],
        delta_t: NDArray[np.float64],
        seed_number: np.uint64 = np.uint64(20240114),
        n_mc_paths: int = 10_000,
        nuts_sampler: Literal["pymc", "nutpie", "jax", "numpyro", "blackjax"] = "nutpie",
        is_progress_bar: bool = False
) -> dict:
    SEED = np.uint64(seed_number)

    N_SIMS_MCMC = n_mc_paths
    # use PyMC to sampler from log-likelihood
    with pm.Model():
        sigma = pm.Gamma(name="sigma", alpha=1., beta=1.)

        alpha_rv = pm.Beta(name="alpha_rv", alpha=5., beta=2.)
        alpha = pm.Deterministic("alpha", pt.log(alpha_rv))

        pprob_rv = pm.Beta(name="pprob_rv", alpha=5., beta=2.)
        pprob = pm.Deterministic("pprob", pt.log(pprob_rv))

        lamb = pm.Gamma(name="lamb", alpha=10., beta=.5)
        eta1 = pm.Gamma(name="eta1", alpha=50., beta=1.)
        eta2 = pm.Gamma(name="eta2", alpha=25., beta=1.)

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

    params_sys_df = az.summary(idata_systematic, stat_focus="mean")

    columns = params_sys_df.columns
    column_mean = columns[0]
    column_ci_lower = columns[2]
    column_ci_upper = columns[3]

    est_alpha = np.exp(params_sys_df.xs('alpha').xs(column_mean))
    est_sigma = params_sys_df.xs('sigma').xs(column_mean)
    est_pprob = np.exp(params_sys_df.xs('pprob').xs(column_mean))
    est_lamb = params_sys_df.xs('lamb').xs(column_mean)
    est_eta1 = params_sys_df.xs('eta1').xs(column_mean)
    est_eta2 = params_sys_df.xs('eta2').xs(column_mean)

    est_alpha_ci_lower = np.exp(params_sys_df.xs('alpha').xs(column_ci_lower))
    est_alpha_ci_upper = np.exp(params_sys_df.xs('alpha').xs(column_ci_upper))
    est_sigma_ci_lower = params_sys_df.xs('sigma').xs(column_ci_lower)
    est_sigma_ci_upper = params_sys_df.xs('sigma').xs(column_ci_upper)
    est_pprob_ci_lower = np.exp(params_sys_df.xs('pprob').xs(column_ci_lower))
    est_pprob_ci_upper = np.exp(params_sys_df.xs('pprob').xs(column_ci_upper))
    est_lamb_ci_lower = params_sys_df.xs('lamb').xs(column_ci_lower)
    est_lamb_ci_upper = params_sys_df.xs('lamb').xs(column_ci_upper)
    est_eta1_ci_lower = params_sys_df.xs('eta1').xs(column_ci_lower)
    est_eta1_ci_upper = params_sys_df.xs('eta1').xs(column_ci_upper)
    est_eta2_ci_lower = params_sys_df.xs('eta2').xs(column_ci_lower)
    est_eta2_ci_upper = params_sys_df.xs('eta2').xs(column_ci_upper)

    return {
        'dALPHA': ParamsResults(dMEAN=est_alpha, dCI_LOWER=est_alpha_ci_lower, dCI_UPPER=est_alpha_ci_upper),
        'dSIGMA': ParamsResults(dMEAN=est_sigma, dCI_LOWER=est_sigma_ci_lower, dCI_UPPER=est_sigma_ci_upper),
        'dPPROB': ParamsResults(dMEAN=est_pprob, dCI_LOWER=est_pprob_ci_lower, dCI_UPPER=est_pprob_ci_upper),
        'dLAMB': ParamsResults(dMEAN=est_lamb, dCI_LOWER=est_lamb_ci_lower, dCI_UPPER=est_lamb_ci_upper),
        'dETA1': ParamsResults(dMEAN=est_eta1, dCI_LOWER=est_eta1_ci_lower, dCI_UPPER=est_eta1_ci_upper),
        'dETA2': ParamsResults(dMEAN=est_eta2, dCI_LOWER=est_eta2_ci_lower, dCI_UPPER=est_eta2_ci_upper)
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

    params_idi_df = az.summary(idata_idiosyncratic, stat_focus="mean")

    columns = params_idi_df.columns
    column_mean = columns[0]
    column_ci_lower = columns[2]
    column_ci_upper = columns[3]

    est_mui = params_idi_df.xs('mui').xs(column_mean)
    est_kappai = np.exp(params_idi_df.xs('kappai').xs(column_mean))
    est_gammai = np.exp(params_idi_df.xs('gammai').xs(column_mean))
    est_betai = np.exp(params_idi_df.xs('betai').xs(column_mean))
    est_rhoix = np.tanh(params_idi_df.xs('rhoix').xs(column_mean))

    est_mui_ci_lower = params_idi_df.xs('mui').xs(column_ci_lower)
    est_mui_ci_upper = params_idi_df.xs('mui').xs(column_ci_upper)
    est_kappai_ci_lower = np.exp(params_idi_df.xs('kappai').xs(column_ci_lower))
    est_kappai_ci_upper = np.exp(params_idi_df.xs('kappai').xs(column_ci_upper))
    est_gammai_ci_lower = np.exp(params_idi_df.xs('gammai').xs(column_ci_lower))
    est_gammai_ci_upper = np.exp(params_idi_df.xs('gammai').xs(column_ci_upper))
    est_betai_ci_lower = np.exp(params_idi_df.xs('betai').xs(column_ci_lower))
    est_betai_ci_upper = np.exp(params_idi_df.xs('betai').xs(column_ci_upper))
    est_rhoix_ci_lower = np.tanh(params_idi_df.xs('rhoix').xs(column_ci_lower))
    est_rhoix_ci_upper = np.tanh(params_idi_df.xs('rhoix').xs(column_ci_upper))

    return {
        'dMUI': ParamsResults(dMEAN=est_mui, dCI_LOWER=est_mui_ci_lower, dCI_UPPER=est_mui_ci_upper),
        'dKAPPAI': ParamsResults(dMEAN=est_kappai, dCI_LOWER=est_kappai_ci_lower, dCI_UPPER=est_kappai_ci_upper),
        'dGAMMAI': ParamsResults(dMEAN=est_gammai, dCI_LOWER=est_gammai_ci_lower, dCI_UPPER=est_gammai_ci_upper),
        'dBETAI': ParamsResults(dMEAN=est_betai, dCI_LOWER=est_betai_ci_lower, dCI_UPPER=est_betai_ci_upper),
        'dRHOIX': ParamsResults(dMEAN=est_rhoix, dCI_LOWER=est_rhoix_ci_lower, dCI_UPPER=est_rhoix_ci_upper)
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

        g_x  = self.pprob * eta1 * pt.exp(0.5 * sigma_squared * eta1**2 * self.dt - diff_y_x * eta1)
        g_x *= pt.exp(pm.logcdf(norm_dist, value=(diff_y_x - sigma_squared * eta1 * self.dt) / sigma_root_dt))
        g_x += self.qprob * eta2 * pt.exp(0.5 * sigma_squared * eta2**2 * self.dt + diff_y_x * eta2)
        g_x *= pt.exp(pm.logcdf(norm_dist, value=(diff_y_x + sigma_squared * eta2 * self.dt) / sigma_root_dt * -1.))
        g_x *= self.lamb * self.dt
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
