"""Systematic shock x -> name i's shock y.

    python poc/shock_to_name.py --name COIN --date 20250409 --shocks -5,-10,-20
    python poc/shock_to_name.py --name COIN --date 20250409 --shocks -5 --horizon 10

    from poc.shock_to_name import name_shock
    name_shock(-0.05, sys_params, idio_params)["y"]

THE ANSWER, in one line

    y = (m_i - alpha Psi_i) dt  +  b_diff (x - E[Y|x])  +  gamma_i E[Y|x]

A name loads on the systematic factor through TWO channels with different
coefficients, so the translation splits the observed move into its diffusive
and jump parts and applies the right one to each:

    b_diff = beta_i + kappa_i rho_iX / sigma      ordinary days
    gamma_i                                       gaps
    m_i    = mu_i + (sigma beta_i)^2/2 - sigma beta_i kappa_i rho_iX

E[Y|x] is the posterior mean of the systematic JUMP given the observed move.
It is what makes this different from a beta: for a small x almost none of the
move is a jump and the answer is b_diff * x; for a large x almost all of it is
and the answer is gamma_i * x. The crossover is sharp - for SPX-like
parameters essentially all of it happens between a 2% and an 8% one-day move.

WHAT THIS IS NOT
    Not a distribution. y is a conditional MEAN; sd(y|x) is returned beside it
    and is routinely a third of the answer. Never quote y alone.
    Not a probability statement. x and h are prescribed. The return period of
    the (x, h) pair is a separate question - a +40% one-day move is arithmetic,
    not a scenario.
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import norm, poisson
from scipy.signal import fftconvolve

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SYS_KEYS  = ("dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2")
IDIO_KEYS = ("dMUI", "dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX")
BASE_DAYS = 252


# ---------------------------------------------------------------------------
# E[Y | x] : posterior mean of the systematic jump given the observed move
# ---------------------------------------------------------------------------
def _jump_posterior_1d(x, sg, lam, p, e1, e2, dt):
    """Exact at one day, where at most one jump is the O(dt) form the engine's
    Kou transition density itself assumes. No quadrature, no simulation."""
    s = sg * np.sqrt(dt)
    s2 = s * s
    a1 = (x - e1 * s2) / s
    a2 = -(x + e2 * s2) / s
    Iu = p * e1 * np.exp(0.5 * e1**2 * s2 - e1 * x) * norm.cdf(a1)
    Id = (1 - p) * e2 * np.exp(0.5 * e2**2 * s2 + e2 * x) * norm.cdf(a2)
    # y-weighted branches, from -dA/deta of the same integrals
    Nu = Iu * (x - e1 * s2) + p * e1 * np.exp(0.5 * e1**2 * s2 - e1 * x) * s * norm.pdf(a1)
    Nd = Id * (x + e2 * s2) - (1 - p) * e2 * np.exp(0.5 * e2**2 * s2 + e2 * x) * s * norm.pdf(a2)
    f = (1 - lam * dt) * norm.pdf(x / s) / s + lam * dt * (Iu + Id)
    return lam * dt * (Nu + Nd) / f, lam * dt * (Iu + Id) / f, f


def _jump_mgf_posterior_1d(g, x, sg, lam, p, e1, e2, dt, f):
    """E[e^(g Y) - 1 | x] at one day, closed form, same mixture as above.

    E[Y|x] and E[e^(gY)-1|x] have to come from ONE posterior. Taking the first
    from the closed form and the second from the multi-day quadrature left the
    two disagreeing by 2.4e-5 in E[Y|x] - the quadrature carries the N >= 2
    branches the O(dt) form drops - which b_diff amplified to 4e-5 in y.
    The tilted integrals are the untilted ones with eta1 -> eta1 - g and
    eta2 -> eta2 + g, keeping the density's own eta prefactors. Needs g < eta1.
    """
    if g >= e1:
        raise ValueError("gamma_i = %.4f exceeds eta1 = %.4f; E[e^(gY)] diverges"
                         % (g, e1))
    s = sg * np.sqrt(dt)
    s2 = s * s

    def _A(q1, q2):
        return (p * e1 * np.exp(0.5 * q1**2 * s2 - q1 * x) * norm.cdf((x - q1 * s2) / s)
                + (1 - p) * e2 * np.exp(0.5 * q2**2 * s2 + q2 * x) * norm.cdf(-(x + q2 * s2) / s))

    return lam * dt * (_A(e1 - g, e2 + g) - _A(e1, e2)) / f


_GRID = None
_CONV_CACHE = {}


def _conv_stack(sg, lam, p, e1, e2, al, T):
    """n-fold jump-sum densities for one horizon, cached.

    A 10-shock x 8-horizon grid calls this 80 times for 8 distinct horizons.
    The FFT stack is identical within a horizon and costs ~40 convolutions of a
    30k-point array, so building it per shock made the grid ~10x slower than it
    needs to be."""
    global _GRID
    if _GRID is None:
        _GRID = np.arange(-3.0, 3.0 + 1e-9, 2e-4)
    key = (sg, lam, p, e1, e2, al, T)
    if key in _CONV_CACHE:
        return _CONV_CACHE[key]
    Y = _GRID
    sD = sg * np.sqrt((1 - np.exp(-2 * al * T)) / (2 * al)) if al > 0 else sg * np.sqrt(T)
    cbar = (1 - np.exp(-al * T)) / (al * T) if al > 0 else 1.0   # mean OU decay
    q1, q2 = e1 / cbar, e2 / cbar
    fY = np.where(Y >= 0, p * q1 * np.exp(-q1 * np.clip(Y, 0, None)),
                  (1 - p) * q2 * np.exp(q2 * np.clip(Y, None, 0)))
    fY /= fY.sum() * 2e-4
    lamT = lam * T
    # Cap raised from 40: the tail probabilities in systematic_tail run out
    # to multi-year horizons where lambda*T ~ 26, and truncating the jump
    # count at 40 there loses ~0.3% of the mass - which is three times the
    # tail probability being solved for.
    nmax = min(80, int(max(6, poisson.ppf(1 - 1e-12, lamT) + 3)))
    conv = fY.copy(); dens = {1: conv}
    for n in range(2, nmax + 1):
        conv = fftconvolve(conv, fY, mode="same") * 2e-4
        dens[n] = conv
    _CONV_CACHE[key] = (dens, nmax, lamT, sD)
    return _CONV_CACHE[key]


def _jump_posterior_h(x, sg, lam, p, e1, e2, al, T):
    """Multi-day: sum over jump COUNT. The one-jump form above is only valid
    for lambda*T << 1; over 20 days lambda*T ~ 1 and two-jump paths matter."""
    dens, nmax, lamT, sD = _conv_stack(sg, lam, p, e1, e2, al, T)
    Y = _GRID
    ker = norm.pdf((x - Y) / sD) / sD
    den = poisson.pmf(0, lamT) * norm.pdf(x / sD) / sD
    num = 0.0; num2 = 0.0; pj = 0.0
    for n in range(1, nmax + 1):
        w = poisson.pmf(n, lamT)
        if w < 1e-16:
            continue
        m = w * np.sum(dens[n] * ker) * 2e-4
        den += m; pj += m
        num += w * np.sum(Y * dens[n] * ker) * 2e-4
        num2 += w * np.sum(Y * Y * dens[n] * ker) * 2e-4
    # E[Y^2|x] carries the n=0 branch implicitly: Y = 0 there, so it adds
    # nothing to either moment but does enter the denominator.
    return num / den, pj / den, den, sD, num2 / den


def _posterior_mean(f, x, sg, lam, p, e1, e2, al, T):
    """E[f(Y) | x] over the same jump-count mixture, for any f with f(0) = 0.

    The n = 0 branch contributes nothing to the numerator because there is no
    jump there, but it does enter the denominator - a move can always have been
    pure diffusion, and at small x it usually was.
    """
    dens, nmax, lamT, sD = _conv_stack(sg, lam, p, e1, e2, al, T)
    Y = _GRID
    ker = norm.pdf((x - Y) / sD) / sD
    den = poisson.pmf(0, lamT) * norm.pdf(x / sD) / sD
    num = 0.0
    for n in range(1, nmax + 1):
        w = poisson.pmf(n, lamT)
        if w < 1e-16:
            continue
        d = dens[n] * ker
        den += w * np.sum(d) * 2e-4
        num += w * np.sum(f(Y) * d) * 2e-4
    return num / den if den > 0 else 0.0

# ---------------------------------------------------------------------------
# How long does a shock of size x need to be plausible?
# ---------------------------------------------------------------------------
def systematic_tail(x, sys_params, horizon_days):
    """One-sided tail probability of the systematic factor over h days.

    P(X_h <= x) for x < 0, P(X_h >= x) for x > 0, under the same OU-plus-Kou
    mixture the posterior uses, so the horizon a shock is quoted at and the
    channel split applied at that horizon come from one distribution.
    """
    sg, lam, p = sys_params["dSIGMA"], sys_params["dLAMB"], sys_params["dPPROB"]
    e1, e2, al = sys_params["dETA1"], sys_params["dETA2"], sys_params["dALPHA"]
    T = horizon_days / BASE_DAYS
    dens, nmax, lamT, sD = _conv_stack(sg, lam, p, e1, e2, al, T)
    Y = _GRID
    # P(diffusion <= x - Y) pointwise, then integrated against the jump-sum law
    cdf = norm.cdf((x - Y) / sD) if x < 0 else norm.sf((x - Y) / sD)
    tail = poisson.pmf(0, lamT) * (norm.cdf(x / sD) if x < 0 else norm.sf(x / sD))
    for n in range(1, nmax + 1):
        w = poisson.pmf(n, lamT)
        if w < 1e-18:
            continue
        tail += w * np.sum(dens[n] * cdf) * 2e-4
    return float(tail)


_ES_CACHE = {}


def _density_h(sg, lam, p, e1, e2, al, T):
    """Unconditional density of the h-day systematic move, on _GRID.

    The jump-sum stack convolved with the OU-integrated diffusion. _GRID is
    symmetric about zero, so mode="same" keeps the convolution centred.
    """
    dens, nmax, lamT, sD = _conv_stack(sg, lam, p, e1, e2, al, T)
    ker = norm.pdf(_GRID / sD) / sD
    f = poisson.pmf(0, lamT) * ker
    for n in range(1, nmax + 1):
        w = poisson.pmf(n, lamT)
        if w < 1e-18:
            continue
        f = f + w * fftconvolve(dens[n], ker, mode="same") * 2e-4
    return f


def systematic_es(sys_params, horizon_days, alpha=0.025, side="left"):
    """Expected shortfall of the systematic factor over h days.

    side="left"  : ES at 97.5% confidence, E[X_h | X_h <= VaR_alpha] - the mean
                   of the worst alpha of outcomes, which is FRTB's measure.
    side="right" : its mirror, E[X_h | X_h >= VaR_(1-alpha)].

    Returned as a signed return, so the left ES is negative and compares
    directly against a prescribed down-shock.
    """
    key = (tuple(sorted(sys_params.items())), horizon_days, alpha, side)
    if key in _ES_CACHE:
        return _ES_CACHE[key]
    sg, lam, p = sys_params["dSIGMA"], sys_params["dLAMB"], sys_params["dPPROB"]
    e1, e2, al = sys_params["dETA1"], sys_params["dETA2"], sys_params["dALPHA"]
    f = _density_h(sg, lam, p, e1, e2, al, horizon_days / BASE_DAYS)
    w = f * 2e-4
    w = w / w.sum()                      # renormalise off the grid truncation
    if side == "left":
        c = np.cumsum(w)
        i = int(np.searchsorted(c, alpha))
        es = float(np.sum(_GRID[:i + 1] * w[:i + 1]) / c[i])
    else:
        c = np.cumsum(w[::-1])
        i = int(np.searchsorted(c, alpha))
        es = float(np.sum(_GRID[::-1][:i + 1] * w[::-1][:i + 1]) / c[i])
    _ES_CACHE[key] = es
    return es


def es_ladder_compounded(sys_params, alpha=0.025, ladder=None,
                        n_paths=1_000_000, chunk=250_000, seed=20240114):
    """ES ladder with the horizon built by COMPOUNDING daily relative moves.

    systematic_es above aggregates by summing daily increments, which is what
    the FFT stack gives for free. But the shocks are relative returns, so an
    h-day move is the product

        1 + X_h = prod_{k=1..h} (1 + r_k)

    not the sum. The two agree to first order and separate fast: compounding is
    LESS severe down (cross terms are positive when both moves are negative)
    and MORE severe up, and it is bounded below by -100% by construction, which
    the sum is not.

    Compounding does not factor through the FFT, so this is Monte Carlo - but
    one pass to the longest rung snapshots every shorter rung on the way, so
    the whole ladder costs one simulation rather than eighteen.

    The OU pull is kept: psi_k = phi psi_{k-1} + u_k with phi = 1 - alpha dt,
    and the daily return is the INCREMENT r_k = psi_k - psi_{k-1}, so a
    liquidity shock still decays instead of compounding forever.
    """
    ladder = tuple(ladder or HORIZON_LADDER)
    sg, lam, p = sys_params["dSIGMA"], sys_params["dLAMB"], sys_params["dPPROB"]
    e1, e2, al = sys_params["dETA1"], sys_params["dETA2"], sys_params["dALPHA"]
    dt = 1.0 / BASE_DAYS
    sd_d, lam_dt, phi = sg * np.sqrt(dt), lam * dt, 1.0 - al * dt
    hmax = max(ladder)
    rng = np.random.default_rng(seed)
    snaps = {h: [] for h in ladder}

    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        psi = np.zeros(m)
        logp = np.zeros(m)
        for k in range(1, hmax + 1):
            nj = rng.poisson(lam_dt, m)
            u = rng.standard_normal(m) * sd_d
            top = int(nj.max())
            for c in range(1, top + 1):
                idx = nj >= c
                cnt = int(idx.sum())
                up = rng.random(cnt) < p
                u[idx] += np.where(up, rng.exponential(1 / e1, cnt),
                                   -rng.exponential(1 / e2, cnt))
            psi_new = phi * psi + u
            # guard only against the ~1e-11 tail where a single jump would take
            # the daily move through -100%; log1p is undefined there
            logp += np.log1p(np.maximum(psi_new - psi, -0.999))
            psi = psi_new
            if k in snaps:
                snaps[k].append(np.expm1(logp).copy())
        done += m

    out = {}
    for h in ladder:
        x = np.sort(np.concatenate(snaps[h]))
        kL = max(1, int(alpha * x.size))
        out[h] = (float(x[:kL].mean()), float(x[-kL:].mean()))
    return out


# Coarse ladder, not a fine search: each rung costs its own FFT stack, and the
# answer is only meaningful to the nearest bucket anyway.
HORIZON_LADDER = (1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 125, 189, 252,
                  378, 504, 756)


def model_horizon_from_ladder(x, es_by_h, ladder=None):
    """Smallest rung whose ES reaches x, given a precomputed ES ladder."""
    for h in (ladder or sorted(es_by_h)):
        esL, esR = es_by_h[h]
        if (esL <= x) if x < 0 else (esR >= x):
            return h
    return None


def model_horizon(x, sys_params, alpha=0.025, ladder=HORIZON_LADDER):
    """Smallest horizon at which the h-day expected shortfall reaches x.

    A down-shock is compared against the 97.5% ES and an up-shock against its
    2.5% mirror, so the horizon a shock is quoted at is set by the same measure
    the trading book is capitalised on rather than by a bare quantile.

    Returns None when even the longest rung has an ES short of x - the honest
    answer for a large shock, because the OU diffusion saturates at
    sigma/sqrt(2 alpha_OU) and everything past that has to be carried by the
    jump tail, which does not thicken fast with horizon.
    """
    side = "left" if x < 0 else "right"
    for h in ladder:
        es = systematic_es(sys_params, h, alpha, side)
        if (es <= x) if x < 0 else (es >= x):
            return h
    return None


# ---------------------------------------------------------------------------
def _safe_mgf(G, xs, sg, lam, p, e1, e2, dt, f1):
    """E[exp(G Y) - 1 | x], or None where the MGF does not exist.

    The tilted integral needs G < eta1. Under the linear response that
    restriction is irrelevant, so a name past it must still translate - it
    just cannot report the exponential column beside its answer.
    """
    try:
        return _jump_mgf_posterior_1d(G, xs, sg, lam, p, e1, e2, dt, f1)
    except Exception:                                         # noqa: BLE001
        return None


def name_shock(x, sys_params, idio_params, horizon_days=1, psi_i=0.0,
               aggregate="compound", jump_response="linear"):
    """Translate a systematic shock x into name i's expected shock.

    Everything is in RELATIVE returns, because that is what Appendix B is in:
    dS_i/S_i- is proxied by the relative return, and the Psi increment

        dPsi_i = -alpha Psi_i dt + phi_i dW~ + d(sum (exp(gamma_i Y_j) - 1))

    carries the jump as a relative jump.

    WHICH JUMP RESPONSE, AND WHY IT IS NOT THE SDE'S. The SDE above is not what
    the parameters were fitted under. Appendix B's transition density convolves
    the Gaussian with an ADDITIVE double exponential - it is Kou's log-price
    density, cited to his Footnote 7 - so what the systematic likelihood fits
    is the relative jump exp(Y)-1 carried under the label Y, and what the
    idiosyncratic likelihood fits, through its rate eta/gamma_i, is gamma_i
    TIMES that same quantity. Both densities linearise, consistently, and at
    gamma_i = 1 with phi_i = sigma they are the same expression - which is the
    statement the paper makes directly when it sets beta_inf = gamma_inf = 1.

    So gamma_i is a LINEAR ratio of relative jumps. Applying exp(gamma_i Y) - 1
    to it adds curvature the estimate never contained, and the tell is that the
    systematic factor fails to reproduce itself: feed gamma_i = 1, b_diff = 1,
    m_i = 0 through the exponential form and a -5% shock comes back -4.89%,
    short by exactly E[exp(Y)-1 | x] - E[Y | x], the Jensen term the likelihood
    dropped. Under the linear form it returns -5.00% by construction. Run
    --selfcheck to see both.

        y_d = (m_i - alpha Psi_i) dt + b_diff E[D | x_d] + gamma_i E[J | x_d]

    jump_response="exp" restores E[exp(gamma_i Y) - 1 | x_d] for comparison. It
    is the SDE's form and it is the wrong one to pair with these parameters;
    it also understates the downside, by 7% at -5% on C and more further out.

    A HORIZON move is not that formula with T in place of dt - Appendix B's
    increment is one period. An h-day move is h daily moves COMPOUNDED, the
    same aggregation used for the shock's own plausibility:

        1 + x   = (1 + x_d)^h        going in
        1 + y   = (1 + y_d)^h        coming out

    which is what keeps the price positive without logging anything: a daily
    relative move never approaches -100%, so neither does the product. Applying
    the one-period formula directly at a 90-day horizon is what drove y through
    -100% on the 2009 vintage, where b_diff reaches 2.54 - that was a misuse of
    the increment, not a defect in it.

    aggregate="horizon" applies the one-period formula at the full horizon
    instead, for comparison.
    """
    sg, lam, p = sys_params["dSIGMA"], sys_params["dLAMB"], sys_params["dPPROB"]
    e1, e2, al = sys_params["dETA1"], sys_params["dETA2"], sys_params["dALPHA"]
    B, K, R = idio_params["dBETAI"], idio_params["dKAPPAI"], idio_params["dRHOIX"]
    G, MU = idio_params["dGAMMAI"], idio_params["dMUI"]

    b_diff = B + K * R / sg
    m_i = MU + 0.5 * (sg * B)**2 - sg * B * K * R
    h = max(1, int(horizon_days))

    if aggregate == "compound":
        # per-day systematic move that compounds to x over h days
        xs = float(np.expm1(np.log1p(x) / h))
        T = 1.0 / BASE_DAYS
    else:
        xs, T = float(x), h / BASE_DAYS

    if aggregate == "compound" or h == 1:
        dt = 1.0 / BASE_DAYS
        EY, pj, f1 = _jump_posterior_1d(xs, sg, lam, p, e1, e2, dt)
        sD = sg * np.sqrt(dt)
        gy = np.arange(-1.0, 1.0, 2e-4)
        fY = np.where(gy >= 0, p * e1 * np.exp(-e1 * np.clip(gy, 0, None)),
                      (1 - p) * e2 * np.exp(e2 * np.clip(gy, None, 0)))
        kr = norm.pdf((xs - gy) / sD) / sD
        dj = lam * dt * np.sum(fY * kr) * 2e-4
        d0 = (1 - lam * dt) * norm.pdf(xs / sD) / sD
        EY2 = lam * dt * np.sum(gy * gy * fY * kr) * 2e-4 / (dj + d0)
        EJ_exp = _safe_mgf(G, xs, sg, lam, p, e1, e2, dt, f1)
    else:
        EY, pj, _, sD, EY2 = _jump_posterior_h(xs, sg, lam, p, e1, e2, al, T)
        try:
            EJ_exp = _posterior_mean(lambda yy: np.exp(G * yy) - 1.0,
                                     xs, sg, lam, p, e1, e2, al, T)
        except Exception:                                     # noqa: BLE001
            EJ_exp = None

    # gamma_i was estimated as a LINEAR ratio of relative jumps - see the
    # docstring. EJ_exp is kept beside it so the two can be quoted together.
    EJ_lin = G * EY
    if jump_response == "linear":
        EJ = EJ_lin
    elif jump_response == "exp":
        if EJ_exp is None:
            raise ValueError("exp response needs gamma_i < eta1 (gamma_i=%.4f, "
                             "eta1=%.4f)" % (G, e1))
        EJ = EJ_exp
    else:
        raise ValueError("jump_response must be 'linear' or 'exp', not %r"
                         % (jump_response,))

    ED = xs - EY
    drift = (m_i - al * psi_i) * T
    y_step = drift + b_diff * ED + EJ                 # one period, relative
    if aggregate == "compound":
        y = float(np.expm1(h * np.log1p(max(y_step, -0.999999))))
    else:
        y = y_step

    w = EY / xs if xs else 0.0
    b_eff = (b_diff * ED + EJ) / xs if xs else b_diff

    # Dispersion: idiosyncratic noise, plus the uncertainty about WHICH channel
    # produced the move - the term that peaks at the crossover. Quoted at the
    # full horizon.
    Th = h / BASE_DAYS
    vfac = (1 - np.exp(-2 * al * Th)) / (2 * al) if al > 0 else Th
    var = K**2 * (1 - R**2) * vfac
    var += (G - b_diff)**2 * max(0.0, EY2 - EY**2) * (h if aggregate == "compound" else 1)

    return dict(y=y, y_step=y_step, x_step=xs, b_diff=b_diff, gamma=G,
                b_eff=b_eff, EY=EY, EJ=EJ, ED=ED,
                EJ_lin=EJ_lin, EJ_exp=EJ_exp, jump_response=jump_response,
                p_jump=pj, drift=drift, sd=float(np.sqrt(var)),
                sigma_h=sD, m_i=m_i, w=w)


def _load(name, date, drawer=None):
    from Library.DataAccess import get_pmle_params, available_pmle_dates
    d = drawer or name
    dates = available_pmle_dates(d)
    if not dates:
        raise SystemExit("no fits on disk for drawer %r" % d)
    dt = date if (date and date in dates) else dates[-1]
    s = get_pmle_params(dt, d)
    return ({k: float(s[k]) for k in SYS_KEYS},
            {k: float(s[k]) for k in IDIO_KEYS}, dt)


def selfcheck(name, date, drawer, shocks):
    """Does the translation reproduce the systematic factor from itself?

    Appendix B sets beta_inf = gamma_inf = 1 and mu_inf = kappa_inf = 0 for the
    systematic component, so putting those through the name translation must
    return the systematic shock unchanged. It is the one property the
    translation has to have that does not depend on any estimate being right.

    b_diff = beta + kappa rho / sigma = 1 needs beta = 1 and kappa = 0, and
    m_i = mu + (sigma beta)^2/2 - sigma beta kappa rho = 0 then needs
    mu = -(sigma)^2/2. Psi is zero. Everything else is the fitted systematic
    row, so the posterior split is the real one at a real date.
    """
    sysp, _, used = _load(name, date, drawer)
    sg = sysp["dSIGMA"]
    idio = {"dBETAI": 1.0, "dKAPPAI": 0.0, "dRHOIX": 0.0,
            "dGAMMAI": 1.0, "dMUI": -0.5 * sg * sg}

    _LOG.info("=" * 74)
    _LOG.info("self-consistency :: the systematic factor through its own translation")
    _LOG.info("%s   %s   (drawer %s)" % (name, used, drawer or name))
    _LOG.info("=" * 74)
    _LOG.info("  beta=1 kappa=0 rho=0 gamma=1 mu=-sigma^2/2  ->  b_diff 1.0000, m_i 0.0000")
    _LOG.info("")
    _LOG.info("      x      linear        err        exp        err")
    _LOG.info("  " + "-" * 54)
    worst = 0.0
    for x in shocks:
        rl = name_shock(x, sysp, idio, 1, 0.0, jump_response="linear")
        try:
            re_ = name_shock(x, sysp, idio, 1, 0.0, jump_response="exp")
            ye, ee = re_["y"], re_["y"] - x
        except Exception:                                     # noqa: BLE001
            ye = ee = float("nan")
        el = rl["y"] - x
        worst = max(worst, abs(el))
        _LOG.info("  %6.1f%%  %8.3f%%  %+8.4f%%  %8.3f%%  %+8.4f%%"
                  % (100 * x, 100 * rl["y"], 100 * el, 100 * ye, 100 * ee))
    _LOG.info("")
    _LOG.info("  The linear column must be zero to rounding. It is by")
    _LOG.info("  construction: E[D|x] + E[J|x] = x is an identity, so with")
    _LOG.info("  b_diff = gamma = 1 the two channels re-assemble the shock.")
    _LOG.info("  The exp column is the Jensen term E[exp(Y)-1|x] - E[Y|x] that")
    _LOG.info("  Appendix B's transition density drops - always positive, so")
    _LOG.info("  the exponential response always understates a downside move.")
    _LOG.info("")
    _LOG.info("  worst linear residual: %.2e" % worst)
    if worst > 1e-9:
        _LOG.info("  FAIL - the linear response is not reproducing the factor.")
    else:
        _LOG.info("  PASS")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--drawer", default=None,
                    help="estimate drawer; defaults to --name")
    ap.add_argument("--date", default=None, help="YYYYMMDD; default the latest")
    ap.add_argument("--shocks", default="-1,-2,-5,-10,-20",
                    help="comma-separated percentages")
    ap.add_argument("--horizons", default="1",
                    help="comma-separated horizons in days. One horizon gives "
                         "the detailed per-shock breakdown; several give a "
                         "shock x horizon grid.")
    ap.add_argument("--psi", type=float, default=0.0,
                    help="the name's current filtered liquidity level")
    ap.add_argument("--jump-response", choices=("linear", "exp"),
                    default="linear",
                    help="how the name answers a systematic jump. linear is "
                         "gamma_i E[J|x], which is the convention the "
                         "parameters were ESTIMATED under - Appendix B's "
                         "transition densities are both additive in the jump. "
                         "exp is E[exp(gamma_i Y)-1|x], the SDE's form; it "
                         "adds curvature the estimate does not contain and "
                         "fails --selfcheck.")
    ap.add_argument("--selfcheck", action="store_true",
                    help="translate the SYSTEMATIC factor through its own "
                         "parameters (gamma_i = b_diff = 1, m_i = 0). The "
                         "answer must be the shock itself. Prints the residual "
                         "under both responses and exits.")
    a = ap.parse_args()

    if a.selfcheck:
        selfcheck(a.name, a.date, a.drawer,
                  [float(v) / 100.0 for v in a.shocks.split(",")])
        return

    sysp, idio, used = _load(a.name, a.date, a.drawer)
    xs = [float(v) / 100.0 for v in a.shocks.split(",")]
    hs = [int(v) for v in a.horizons.split(",")]

    _LOG.info("=" * 74)
    _LOG.info("%s   %s" % (a.name, used))
    _LOG.info("=" * 74)
    _LOG.info("  systematic  " + "  ".join("%s %.4f" % (k[1:].lower(), sysp[k]) for k in SYS_KEYS))
    _LOG.info("  name        " + "  ".join("%s %.4f" % (k[1:].lower(), idio[k]) for k in IDIO_KEYS))
    r0 = name_shock(xs[0], sysp, idio, hs[0], a.psi,
                    jump_response=a.jump_response)
    _LOG.info("  jump response: %s" % a.jump_response)
    _LOG.info("\n  b_diff = beta + kappa*rho/sigma = %.4f      gamma_i = %.4f"
          % (r0["b_diff"], r0["gamma"]))
    _LOG.info("  m_i = %.4f" % r0["m_i"])

    if len(hs) == 1:
        h = hs[0]
        _LOG.info("\n  horizon %d day(s), sigma_h = %.3f%%" % (h, 100 * r0["sigma_h"]))
        _LOG.info("\n      x     P(jump|x)    E[Y|x]     E[D|x]    b_eff        y_i    sd(y|x)")
        _LOG.info("  " + "-" * 70)
        for x in xs:
            r = name_shock(x, sysp, idio, h, a.psi,
                           jump_response=a.jump_response)
            _LOG.info("  %6.1f%%    %7.4f  %8.3f%%  %8.3f%%  %7.3f  %8.2f%%  %7.2f%%"
                  % (100 * x, r["p_jump"], 100 * r["EY"], 100 * r["ED"],
                     r["b_eff"], 100 * r["y"], 100 * r["sd"]))
        _LOG.info("\n  y_i is a conditional MEAN. Quote it with sd(y|x), never alone.")
        return

    grids = {k: np.zeros((len(hs), len(xs))) for k in ("y", "b_eff", "w", "sd")}
    sig = []
    for i, h in enumerate(hs):
        for j, x in enumerate(xs):
            r = name_shock(x, sysp, idio, h, a.psi,
                           jump_response=a.jump_response)
            for k in grids:
                grids[k][i, j] = r[k]
        sig.append(r["sigma_h"])

    hdr = "   h \\ x " + "".join("%9.0f%%" % (100 * x) for x in xs)
    for title, key, pct in (("IDIOSYNCRATIC SHOCK  y_i(x, h)", "y", True),
                            ("CONDITIONAL SD  sd(y_i | x, h)", "sd", True),
                            ("EFFECTIVE BETA  b_eff(x, h)", "b_eff", False),
                            ("JUMP SHARE  w(x, h)", "w", False)):
        _LOG.info("\n" + title)
        _LOG.info("=" * len(hdr)); _LOG.info(hdr); _LOG.info("-" * len(hdr))
        for i, h in enumerate(hs):
            cells = "".join(("%9.2f%%" % (100 * grids[key][i, j])) if pct
                            else ("%10.3f" % grids[key][i, j])
                            for j in range(len(xs)))
            _LOG.info("  %4dd " % h + cells)
    _LOG.info("\n  sigma_h: " + "  ".join("%dd=%.2f%%" % (h, 100 * s_)
                                      for h, s_ in zip(hs, sig)))
    _LOG.info("""
  b_eff runs between b_diff (all diffusion) and gamma_i (all jump). Down a
  column it converges to the same value for EVERY shock as h grows - w tends
  to the jump share of total variance - so the two loadings only separate at
  short horizons and large moves.

  y_i is a conditional MEAN and the sd grid is not decoration: it peaks at the
  crossover, where the move could plausibly be either channel.""")


if __name__ == "__main__":
    main()
