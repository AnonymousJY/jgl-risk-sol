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


_GRID = None
def _jump_posterior_h(x, sg, lam, p, e1, e2, al, T):
    """Multi-day: sum over jump COUNT. The one-jump form above is only valid
    for lambda*T << 1; over 20 days lambda*T ~ 1 and two-jump paths matter."""
    global _GRID
    if _GRID is None:
        _GRID = np.arange(-3.0, 3.0 + 1e-9, 2e-4)
    Y = _GRID
    sD = sg * np.sqrt((1 - np.exp(-2 * al * T)) / (2 * al)) if al > 0 else sg * np.sqrt(T)
    cbar = (1 - np.exp(-al * T)) / (al * T) if al > 0 else 1.0   # mean OU decay
    q1, q2 = e1 / cbar, e2 / cbar
    fY = np.where(Y >= 0, p * q1 * np.exp(-q1 * np.clip(Y, 0, None)),
                  (1 - p) * q2 * np.exp(q2 * np.clip(Y, None, 0)))
    fY /= fY.sum() * 2e-4
    lamT = lam * T
    nmax = min(40, int(max(6, poisson.ppf(1 - 1e-12, lamT) + 3)))
    conv = fY.copy(); dens = {1: conv}
    for n in range(2, nmax + 1):
        conv = fftconvolve(conv, fY, mode="same") * 2e-4
        dens[n] = conv
    ker = norm.pdf((x - Y) / sD) / sD
    den = poisson.pmf(0, lamT) * norm.pdf(x / sD) / sD
    num = 0.0; pj = 0.0
    for n in range(1, nmax + 1):
        w = poisson.pmf(n, lamT)
        if w < 1e-16:
            continue
        m = w * np.sum(dens[n] * ker) * 2e-4
        den += m; pj += m
        num += w * np.sum(Y * dens[n] * ker) * 2e-4
    return num / den, pj / den, den, sD


# ---------------------------------------------------------------------------
def name_shock(x, sys_params, idio_params, horizon_days=1, psi_i=0.0):
    """Translate a systematic shock x into name i's expected shock."""
    sg, lam, p = sys_params["dSIGMA"], sys_params["dLAMB"], sys_params["dPPROB"]
    e1, e2, al = sys_params["dETA1"], sys_params["dETA2"], sys_params["dALPHA"]
    B, K, R = idio_params["dBETAI"], idio_params["dKAPPAI"], idio_params["dRHOIX"]
    G, MU = idio_params["dGAMMAI"], idio_params["dMUI"]

    T = horizon_days / BASE_DAYS
    b_diff = B + K * R / sg
    m_i = MU + 0.5 * (sg * B)**2 - sg * B * K * R

    if horizon_days == 1:
        dt = 1.0 / BASE_DAYS
        EY, pj, _ = _jump_posterior_1d(x, sg, lam, p, e1, e2, dt)
        sD = sg * np.sqrt(dt)
    else:
        EY, pj, _, sD = _jump_posterior_h(x, sg, lam, p, e1, e2, al, T)

    ED = x - EY
    w = EY / x if x else 0.0
    b_eff = b_diff + (G - b_diff) * w
    drift = (m_i - al * psi_i) * T
    y = drift + b_diff * ED + G * EY

    # dispersion: orthogonal noise + uncertainty about WHICH channel acted
    vfac = (1 - np.exp(-2 * al * T)) / (2 * al) if al > 0 else T
    var = K**2 * (1 - R**2) * vfac
    if horizon_days == 1:
        dt = 1.0 / BASE_DAYS
        s = sg * np.sqrt(dt); s2 = s * s
        gy = np.arange(-1.0, 1.0, 2e-4)
        fY = np.where(gy >= 0, p * e1 * np.exp(-e1 * np.clip(gy, 0, None)),
                      (1 - p) * e2 * np.exp(e2 * np.clip(gy, None, 0)))
        ker = norm.pdf((x - gy) / s) / s
        dj = lam * dt * np.sum(fY * ker) * 2e-4
        d0 = (1 - lam * dt) * norm.pdf(x / s) / s
        EY2 = lam * dt * np.sum(gy * gy * fY * ker) * 2e-4 / (dj + d0)
        var += (G - b_diff)**2 * max(0.0, EY2 - EY**2)

    return dict(y=y, b_diff=b_diff, gamma=G, b_eff=b_eff, EY=EY, ED=ED,
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
    ap.add_argument("--horizon", type=int, default=1, help="days")
    ap.add_argument("--psi", type=float, default=0.0,
                    help="the name's current filtered liquidity level")
    a = ap.parse_args()

    sysp, idio, used = _load(a.name, a.date, a.drawer)
    xs = [float(v) / 100.0 for v in a.shocks.split(",")]

    print("=" * 74)
    print("%s   %s   horizon %d day(s)" % (a.name, used, a.horizon))
    print("=" * 74)
    print("  systematic  " + "  ".join("%s %.4f" % (k[1:].lower(), sysp[k]) for k in SYS_KEYS))
    print("  name        " + "  ".join("%s %.4f" % (k[1:].lower(), idio[k]) for k in IDIO_KEYS))
    r0 = name_shock(xs[0], sysp, idio, a.horizon, a.psi)
    print("\n  b_diff = beta + kappa*rho/sigma = %.4f      gamma_i = %.4f"
          % (r0["b_diff"], r0["gamma"]))
    print("  m_i = %.4f      sigma over %dd = %.3f%%"
          % (r0["m_i"], a.horizon, 100 * r0["sigma_h"]))
    print("\n      x     P(jump|x)    E[Y|x]     E[D|x]    b_eff        y_i    sd(y|x)")
    print("  " + "-" * 70)
    for x in xs:
        r = name_shock(x, sysp, idio, a.horizon, a.psi)
        print("  %6.1f%%    %7.4f  %8.3f%%  %8.3f%%  %7.3f  %8.2f%%  %7.2f%%"
              % (100 * x, r["p_jump"], 100 * r["EY"], 100 * r["ED"],
                 r["b_eff"], 100 * r["y"], 100 * r["sd"]))
    print("\n  y_i is a conditional MEAN. Quote it with sd(y|x), never alone.")


if __name__ == "__main__":
    main()
