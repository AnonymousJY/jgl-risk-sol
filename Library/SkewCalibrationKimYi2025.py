import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
# from Library.RootFinder import newton_raphson
from Library.SkewCalibrationBase import SkewCalibrationBase
from Library.OptionPricerBSM1973 import BlackScholesMertonCall, BlackScholesMertonPut
from Library.OptionPricerKimYi2025 import kimyi_call, kimyi_put, psi_vol


class KimYiSkewCalibrationSystematic(SkewCalibrationBase):
    """Calibrate the index's Q-measure JUMP parameters to its option surface.

    sigma IS NOT FITTED. It is a diffusion coefficient, and an equivalent
    change of measure cannot touch one: Girsanov shifts the drift of a
    Brownian motion and leaves its quadratic variation alone, and quadratic
    variation is a pathwise quantity. So sigma_Q = sigma_P, and sigma comes
    from the systematic P-MLE stage. The same argument covers beta_i and
    kappa_i, hence phi_i, on the idiosyncratic side.

    What a measure change CAN move is the jump compensator - the intensity
    lambda and the jump-size distribution (p, eta1, eta2). Those are the four
    this class fits, and they are where the index's jump risk premium lives.

    This is not a restriction imposed for parsimony; it is what the model
    already implies, and fitting sigma here was letting the surface overwrite
    a quantity the returns measure better. LIQUIDITY_SKEW_FIT_SIGMA=1 in the
    calibration script restores the five-parameter fit for comparison.
    """

    def __init__(
            self,
            mkt_imp_vol: NDArray[np.float64],
            und_price: NDArray[np.float64],
            und_strike: NDArray[np.float64],
            risk_free_rate: NDArray[np.float64],
            dividend_yield: NDArray[np.float64],
            time_to_expiry: NDArray[np.float64],
            is_call_option: NDArray[np.bool_],
            option_weights: NDArray[np.float64],
            sigma: NDArray[np.float64] = None,
    ):
        # Keyword and optional: a caller replaying a cached five-vector does
        # not need it.
        self.sigma_p = None if sigma is None else float(np.asarray(sigma))
        self.mkt_imp_vol = mkt_imp_vol
        self.und_price = und_price
        self.und_strike = und_strike
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.time_to_expiry = time_to_expiry
        self.is_call_option = is_call_option
        self.option_weights = option_weights
        self.penalty = np.array(0.)

    def _unpack(self, x):
        """(sigma, p, lambda, eta1, eta2) from a four- or five-element x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if x.size == 4:
            if self.sigma_p is None:
                raise ValueError(
                    "x is the four jump parameters, so sigma has to come from "
                    "the constructor; pass sigma= (the P-measure value), or a "
                    "five-element x to fit it.")
            return (self.sigma_p,) + tuple(x)
        if x.size == 5:
            return tuple(x)
        raise ValueError("x has %d elements; expected 4 or 5" % x.size)

    def target(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        mod_imp_vol = self.model_vol(x=x)

        # If Newton IV-inversion diverged for any strike inside model_vol(),
        # the resulting NaN(s) would poison the sum below. Return a large
        # finite penalty so SLSQP sees "very high objective" and moves away
        # from this parameter region, rather than crashing outright on a
        # pathological (sigma, pprob, lamb, eta1, eta2) trial.
        if not np.all(np.isfinite(mod_imp_vol)):
            return np.array(1e6)

        return 0.5 * np.sum((self.mkt_imp_vol - mod_imp_vol) ** 2 * self.option_weights) + self.penalty

    def model_vol(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        sigma, pprob, lamb, eta1, eta2 = self._unpack(x)

        mod_imp_vol_put = _kimyi_imp_vol_put(
            kappai=np.array(0.),
            gammai=np.array(1.),
            betai=np.array(1.),
            rhoix=np.array(0.),
            sigma=sigma,
            pprob=pprob,
            lamb=lamb,
            eta1=eta1,
            eta2=eta2,
            und_price=self.und_price[~self.is_call_option],
            und_strike=self.und_strike[~self.is_call_option],
            risk_free_rate=self.risk_free_rate[~self.is_call_option],
            dividend_yield=self.dividend_yield[~self.is_call_option],
            time_to_expiry=self.time_to_expiry[~self.is_call_option]
        )

        mod_imp_vol_call = _kimyi_imp_vol_call(
            kappai=np.array(0.),
            gammai=np.array(1.),
            betai=np.array(1.),
            rhoix=np.array(0.),
            sigma=sigma,
            pprob=pprob,
            lamb=lamb,
            eta1=eta1,
            eta2=eta2,
            und_price=self.und_price[self.is_call_option],
            und_strike=self.und_strike[self.is_call_option],
            risk_free_rate=self.risk_free_rate[self.is_call_option],
            dividend_yield=self.dividend_yield[self.is_call_option],
            time_to_expiry=self.time_to_expiry[self.is_call_option]
        )

        return np.vstack((mod_imp_vol_put, mod_imp_vol_call))

    @property
    def mkt_imp_vol(self) -> NDArray[np.float64]:
        return self._mkt_imp_vol

    @mkt_imp_vol.setter
    def mkt_imp_vol(self, value: NDArray[np.float64]):
        self._mkt_imp_vol = value.reshape((-1, 1))

    @property
    def und_price(self) -> NDArray[np.float64]:
        return self._und_price

    @und_price.setter
    def und_price(self, value: NDArray[np.float64]):
        self._und_price = value.reshape((-1, 1))

    @property
    def risk_free_rate(self) -> NDArray[np.float64]:
        return self._risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, value: NDArray[np.float64]):
        self._risk_free_rate = value.reshape((-1, 1))

    @property
    def dividend_yield(self) -> NDArray[np.float64]:
        return self._dividend_yield

    @dividend_yield.setter
    def dividend_yield(self, value: NDArray[np.float64]):
        self._dividend_yield = value.reshape((-1, 1))

    @property
    def time_to_expiry(self) -> NDArray[np.float64]:
        return self._time_to_expiry

    @time_to_expiry.setter
    def time_to_expiry(self, value: NDArray[np.float64]):
        self._time_to_expiry = value.reshape((-1, 1))

    @property
    def option_weights(self) -> NDArray[np.float64]:
        return self._option_weights

    @option_weights.setter
    def option_weights(self, value: NDArray[np.float64]):
        self._option_weights = value.reshape((-1, 1))


class KimYiSkewCalibrationIdiosyncratic(SkewCalibrationBase):
    """Calibrate the name's Q-measure parameters to its own option surface.

    ONLY TWO THINGS ABOUT THE NAME REACH THE PRICE. kimyi_call/kimyi_put
    collapse beta_i, kappa_i and rho_iX into the single diffusive volatility
    psi_vol(beta_i, kappa_i, rho_iX, sigma) = phi_i, and gamma_i enters only
    as eta/gamma_i on the two jump decays. So the surface identifies phi_i and
    gamma_i, and NOTHING ELSE: the four-parameter fit this class used to run
    was searching a two-dimensional flat manifold, where SLSQP stopped
    wherever its path happened to end and the reported beta_i, kappa_i and
    rho_iX were arbitrary points on it.

    This class therefore takes (gamma_i) or (gamma_i, phi_i) - never the four.

    phi_i IS THE P-MEASURE VALUE, unscaled. beta_i and kappa_i are diffusion
    coefficients and an equivalent change of measure cannot touch one -
    Girsanov shifts a Brownian motion's drift and leaves its quadratic
    variation, a pathwise quantity, alone - so phi_i_Q = phi_i_P is not an
    assumption but a consequence of the model. The same argument pins
    sigma_Q = sigma_P on the systematic side.

    That leaves gamma_i as the name's only free Q parameter, AND THAT IS
    WHERE THE PREMIUM BELONGS. The strict reading of the same Girsanov
    argument would pin gamma_i too - it is a pathwise loading, so the name
    moves gamma_i times as far as the index on any given jump under either
    measure. So gamma_i_Q != gamma_i_P is not a structural difference but a
    PRICING WEDGE: the name's own jump risk premium.

    Putting the wedge there is what the evidence says. Bollerslev and Todorov
    (2011) find that much of the equity and variance risk premium is
    compensation for jump TAIL risk rather than diffusive risk; Bakshi et al.
    (2003) find that the channel through which individual equity options
    price differently from index options is SKEW. In this model the level of
    the smile is phi_i and its skew is gamma_i, through eta1/gamma_i and
    eta2/gamma_i. Confining the measure change to the jump channel therefore
    puts the premium where it is documented to be, and gamma_i_Q / gamma_i_P
    becomes a measurable name-level jump risk premium rather than a fitting
    residual.

    LIQUIDITY_SKEW_FIT_PHI=1 in the calibration script frees phi_i as a
    second parameter. That is a misspecification diagnostic, not an
    alternative: if the one-parameter fit cannot reach the market's ATM
    level, the gap is telling you the single-factor jump structure is too
    thin, not that phi_i has a risk premium.
    """

    def __init__(
            self,
            sigma: NDArray[np.float64],
            pprob: NDArray[np.float64],
            lamb: NDArray[np.float64],
            eta1: NDArray[np.float64],
            eta2: NDArray[np.float64],
            mkt_imp_vol: NDArray[np.float64],
            und_price: NDArray[np.float64],
            und_strike: NDArray[np.float64],
            risk_free_rate: NDArray[np.float64],
            dividend_yield: NDArray[np.float64],
            time_to_expiry: NDArray[np.float64],
            is_call_option: NDArray[np.bool_],
            option_weights: NDArray[np.float64],
            phii: NDArray[np.float64] = None,
    ):
        # Keyword and optional: a caller replaying a cached four-vector does
        # not need it, and _unpack() says so precisely if a one-element x
        # arrives without it.
        # A PLAIN FLOAT, not a column. The pricer broadcasts sigma against
        # the strike vectors and phi_i takes the slot a scalar used to
        # occupy, so reshaping it here would change model_vol()'s output
        # shape and silently break the vstack below.
        self.phii = None if phii is None else float(np.asarray(phii))
        self.sigma = sigma
        self.pprob = pprob
        self.lamb = lamb
        self.eta1 = eta1
        self.eta2 = eta2
        self.mkt_imp_vol = mkt_imp_vol
        self.und_price = und_price
        self.und_strike = und_strike
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.time_to_expiry = time_to_expiry
        self.is_call_option = is_call_option
        self.option_weights = option_weights
        self.penalty = np.array(0.)

    def _unpack(self, x):
        """(gamma_i, phi_i) as plain floats, from a one-, two- or four-element x.

        phi_i is passed to the pricer as betai=0, kappai=phi_i, rhoix=0,
        because psi_vol(0, phi_i, 0, sigma) = phi_i EXACTLY. It looks like a
        trick and is not: phi_i is the only combination of the three that the
        price depends on, so any (beta_i, kappa_i, rho_iX) with the same
        psi_vol gives the same surface, and this is the one that needs no
        constraint to exist. The structural beta_i and kappa_i come from the
        P-measure joint likelihood, which can separate them; the option
        surface cannot, and pretending otherwise is what produced the old
        flat manifold.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if x.size == 1:
            if self.phii is None:
                raise ValueError(
                    "x is gamma_i alone, so phi_i has to come from the "
                    "constructor; pass phii= (the P-measure phi_i, scaled by "
                    "the index's sigma_Q/sigma_P), or a two-element x to fit "
                    "phi_i, or the legacy four-vector.")
            return float(x[0]), self.phii
        if x.size == 2:
            return float(x[0]), float(x[1])
        if x.size == 4:
            # LEGACY, for replaying cached four-vector fits and the stored
            # (dKAPPAI, dGAMMAI, dBETAI, dRHOIX) columns. Collapsed through
            # psi_vol exactly as the pricer used to collapse them, so plots
            # and comparisons off the old cache reproduce to the last digit -
            # while no NEW fit can wander the manifold those four spanned.
            kappai, gammai, betai, rhoix = x
            return float(gammai), float(np.asarray(
                psi_vol(betai=betai, kappai=kappai, rhoix=rhoix,
                        sigma=self.sigma)).ravel()[0])
        raise ValueError(
            "the name's surface identifies gamma_i and phi_i and nothing "
            "else; x has %d elements" % x.size)

    def target(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        mod_imp_vol = self.model_vol(x=x)

        # If Newton IV-inversion diverged for any strike inside model_vol(),
        # the resulting NaN(s) would poison the sum below. Return a large
        # finite penalty so SLSQP sees "very high objective" and moves away
        # from this parameter region, rather than crashing outright on a
        # pathological (sigma, pprob, lamb, eta1, eta2) trial.
        if not np.all(np.isfinite(mod_imp_vol)):
            return np.array(1e6)

        return 0.5 * np.sum((self.mkt_imp_vol - mod_imp_vol) ** 2 * self.option_weights) + self.penalty

    def model_vol(self, x: NDArray[np.float64]) -> NDArray[np.float64]:

        gammai, phii = self._unpack(x)
        kappai, betai, rhoix = phii, np.array(0.), np.array(0.)

        mod_imp_vol_put = _kimyi_imp_vol_put(
            kappai=kappai,
            gammai=gammai,
            betai=betai,
            rhoix=rhoix,
            sigma=self.sigma,
            pprob=self.pprob,
            lamb=self.lamb,
            eta1=self.eta1,
            eta2=self.eta2,
            und_price=self.und_price[~self.is_call_option],
            und_strike=self.und_strike[~self.is_call_option],
            risk_free_rate=self.risk_free_rate[~self.is_call_option],
            dividend_yield=self.dividend_yield[~self.is_call_option],
            time_to_expiry=self.time_to_expiry[~self.is_call_option]
        )

        mod_imp_vol_call = _kimyi_imp_vol_call(
            kappai=kappai,
            gammai=gammai,
            betai=betai,
            rhoix=rhoix,
            sigma=self.sigma,
            pprob=self.pprob,
            lamb=self.lamb,
            eta1=self.eta1,
            eta2=self.eta2,
            und_price=self.und_price[self.is_call_option],
            und_strike=self.und_strike[self.is_call_option],
            risk_free_rate=self.risk_free_rate[self.is_call_option],
            dividend_yield=self.dividend_yield[self.is_call_option],
            time_to_expiry=self.time_to_expiry[self.is_call_option]
        )

        return np.vstack((mod_imp_vol_put, mod_imp_vol_call))

    @property
    def sigma(self) -> NDArray[np.float64]:
        return self._sigma

    @sigma.setter
    def sigma(self, value: NDArray[np.float64]):
        self._sigma = np.array(value).reshape((-1, 1))

    @property
    def pprob(self) -> NDArray[np.float64]:
        return self._pprob

    @pprob.setter
    def pprob(self, value: NDArray[np.float64]):
        self._pprob = np.array(value).reshape((-1, 1))

    @property
    def lamb(self) -> NDArray[np.float64]:
        return self._lamb

    @lamb.setter
    def lamb(self, value: NDArray[np.float64]):
        self._lamb = np.array(value).reshape((-1, 1))

    @property
    def eta1(self) -> NDArray[np.float64]:
        return self._eta1

    @eta1.setter
    def eta1(self, value: NDArray[np.float64]):
        self._eta1 = np.array(value).reshape((-1, 1))

    @property
    def eta2(self) -> NDArray[np.float64]:
        return self._eta2

    @eta2.setter
    def eta2(self, value: NDArray[np.float64]):
        self._eta2 = np.array(value).reshape((-1, 1))

    @property
    def mkt_imp_vol(self) -> NDArray[np.float64]:
        return self._mkt_imp_vol

    @mkt_imp_vol.setter
    def mkt_imp_vol(self, value: NDArray[np.float64]):
        self._mkt_imp_vol = value.reshape((-1, 1))

    @property
    def und_price(self) -> NDArray[np.float64]:
        return self._und_price

    @und_price.setter
    def und_price(self, value: NDArray[np.float64]):
        self._und_price = value.reshape((-1, 1))

    @property
    def risk_free_rate(self) -> NDArray[np.float64]:
        return self._risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, value: NDArray[np.float64]):
        self._risk_free_rate = value.reshape((-1, 1))

    @property
    def dividend_yield(self) -> NDArray[np.float64]:
        return self._dividend_yield

    @dividend_yield.setter
    def dividend_yield(self, value: NDArray[np.float64]):
        self._dividend_yield = value.reshape((-1, 1))

    @property
    def time_to_expiry(self) -> NDArray[np.float64]:
        return self._time_to_expiry

    @time_to_expiry.setter
    def time_to_expiry(self, value: NDArray[np.float64]):
        self._time_to_expiry = value.reshape((-1, 1))

    @property
    def option_weights(self) -> NDArray[np.float64]:
        return self._option_weights

    @option_weights.setter
    def option_weights(self, value: NDArray[np.float64]):
        self._option_weights = value.reshape((-1, 1))


def _kimyi_imp_vol_call(
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    prices = kimyi_call(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        kappai=kappai,
        gammai=gammai,
        betai=betai,
        rhoix=rhoix,
        sigma=sigma,
        pprob=pprob,
        lamb=lamb,
        eta1=eta1,
        eta2=eta2,
        time_to_expiry=time_to_expiry
    )

    bsm_call_obj = BlackScholesMertonCall(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_expiry=time_to_expiry
    )

    obj_func = lambda x: bsm_call_obj.price(volatility=x) - prices

    # Initial guess x0=1.5 (150% vol) matches the paper's original code.
    # Starting high gives Newton non-negligible BSM vega even at deep OTM
    # strikes, whereas a strike-adaptive low x0 (e.g. Brenner-Subrahmanyam)
    # can leave Newton stuck at the floor for deep-OTM options where BSM
    # vega at low vol is ~zero. Wrapped in try/except because scipy.newton
    # raises RuntimeError when its 200 iterations don't converge -- happens
    # when the outer SLSQP tries a pathological (sigma, pprob, lamb, eta1,
    # eta2) combination that pushes model prices near the arbitrage bounds.
    x0 = np.full(prices.shape, 1.5).reshape((-1, 1))
    try:
        mod_iv_call = newton(
            func=obj_func,
            x0=x0,
            fprime=bsm_call_obj.vega,
            fprime2=bsm_call_obj.vomma,
            maxiter=200,
            tol=1e-8,
        )
    except RuntimeError:
        mod_iv_call = np.full(prices.shape, np.nan).reshape((-1, 1))

    # Newton can also silently return inf/NaN (via overflow warnings, not
    # exceptions) for individual strikes when the outer SLSQP tries a
    # pathological (sigma, pprob, lamb, eta1, eta2) combination. Mark only
    # those bad entries as NaN -- keep the good ones intact so plotting +
    # per-strike diagnostics still work. target() below checks isfinite on
    # the whole array and returns a 1e6 penalty if any NaN is present, so
    # SLSQP's avoidance behavior is preserved.
    mod_iv_call = np.where(np.isfinite(mod_iv_call), mod_iv_call, np.nan)

    # mod_iv_call = newton_raphson(
    #     func=bsm_call_obj.price,
    #     func_deriv=bsm_call_obj.vega,
    #     target_value=prices,
    #     initial_value=np.array([1.5] * prices.shape[0]).reshape((-1, 1))
    # )

    return mod_iv_call


def _kimyi_imp_vol_put(
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
    ) -> NDArray[np.float64]:

    prices = kimyi_put(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        kappai=kappai,
        gammai=gammai,
        betai=betai,
        rhoix=rhoix,
        sigma=sigma,
        pprob=pprob,
        lamb=lamb,
        eta1=eta1,
        eta2=eta2,
        time_to_expiry=time_to_expiry
    )

    bsm_put_obj = BlackScholesMertonPut(
        und_price=und_price,
        und_strike=und_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        time_to_expiry=time_to_expiry
    )

    obj_func = lambda x: bsm_put_obj.price(volatility=x) - prices

    # See _kimyi_imp_vol_call above for the rationale on x0=1.5, maxiter=200,
    # try/except, and per-strike NaN handling -- same treatment for symmetry.
    x0 = np.full(prices.shape, 1.5).reshape((-1, 1))
    try:
        mod_iv_put = newton(
            func=obj_func,
            x0=x0,
            fprime=bsm_put_obj.vega,
            fprime2=bsm_put_obj.vomma,
            maxiter=200,
            tol=1e-8,
        )
    except RuntimeError:
        mod_iv_put = np.full(prices.shape, np.nan).reshape((-1, 1))

    # See _kimyi_imp_vol_call above for the per-strike NaN rationale.
    mod_iv_put = np.where(np.isfinite(mod_iv_put), mod_iv_put, np.nan)

    # mod_iv_put = newton_raphson(
    #     func=bsm_put_obj.price,
    #     func_deriv=bsm_put_obj.vega,
    #     target_value=prices,
    #     initial_value=np.array([1.5] * prices.shape[0]).reshape((-1, 1))
    # )

    return mod_iv_put


def kimyi_vol_surface(
        kappai: NDArray[np.float64],
        gammai: NDArray[np.float64],
        betai: NDArray[np.float64],
        rhoix: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pprob: NDArray[np.float64],
        lamb: NDArray[np.float64],
        eta1: NDArray[np.float64],
        eta2: NDArray[np.float64],
        und_price: NDArray[np.float64],
        und_strike: NDArray[np.float64],
        risk_free_rate: NDArray[np.float64],
        dividend_yield: NDArray[np.float64],
        time_to_expiry: NDArray[np.float64]
) -> NDArray[np.float64]:

    mask_call = und_strike > 100.
    mask_put = und_strike <= 100.

    mod_iv_call = _kimyi_imp_vol_call(
        kappai=np.array(kappai).reshape((-1, 1)),
        gammai=np.array(gammai).reshape((-1, 1)),
        betai=np.array(betai).reshape((-1, 1)),
        rhoix=np.array(rhoix).reshape((-1, 1)),
        sigma=np.array(sigma).reshape((-1, 1)),
        pprob=np.array(pprob).reshape((-1, 1)),
        lamb=np.array(lamb).reshape((-1, 1)),
        eta1=np.array(eta1).reshape((-1, 1)),
        eta2=np.array(eta2).reshape((-1, 1)),
        und_price=np.array(und_price).reshape((-1, 1)),
        und_strike=np.array(und_strike[mask_call]).reshape((-1, 1)),
        risk_free_rate=np.array(risk_free_rate).reshape((-1, 1)),
        dividend_yield=np.array(dividend_yield).reshape((-1, 1)),
        time_to_expiry=np.array(time_to_expiry).reshape((-1, 1))
    )

    mod_iv_put = _kimyi_imp_vol_call(
        kappai=np.array(kappai).reshape((-1, 1)),
        gammai=np.array(gammai).reshape((-1, 1)),
        betai=np.array(betai).reshape((-1, 1)),
        rhoix=np.array(rhoix).reshape((-1, 1)),
        sigma=np.array(sigma).reshape((-1, 1)),
        pprob=np.array(pprob).reshape((-1, 1)),
        lamb=np.array(lamb).reshape((-1, 1)),
        eta1=np.array(eta1).reshape((-1, 1)),
        eta2=np.array(eta2).reshape((-1, 1)),
        und_price=np.array(und_price).reshape((-1, 1)),
        und_strike=np.array(und_strike[mask_put]).reshape((-1, 1)),
        risk_free_rate=np.array(risk_free_rate).reshape((-1, 1)),
        dividend_yield=np.array(dividend_yield).reshape((-1, 1)),
        time_to_expiry=np.array(time_to_expiry).reshape((-1, 1))
    )

    # results = {}
    # results['dMONEYNESS'] = np.array(und_strike) / np.array(und_price) * 100.
    # results['iEXPIRY'] = np.array(expiry_in_days, dtype=np.int64)
    # results['dVOL'] = np.vstack((mod_iv_put, mod_iv_call)) * 100.

    return np.vstack((mod_iv_put, mod_iv_call)) * 100.
