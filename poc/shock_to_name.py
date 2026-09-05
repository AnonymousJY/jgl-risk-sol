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
    nmax = min(40, int(max(6, poisson.ppf(1 - 1e-12, lamT) + 3)))
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
        # second moment by the same quadrature the h>1 path uses
        gy = np.arange(-1.0, 1.0, 2e-4)
        fY = np.where(gy >= 0, p * e1 * np.exp(-e1 * np.clip(gy, 0, None)),
                      (1 - p) * e2 * np.exp(e2 * np.clip(gy, None, 0)))
        kr = norm.pdf((x - gy) / sD) / sD
        dj = lam * dt * np.sum(fY * kr) * 2e-4
        d0 = (1 - lam * dt) * norm.pdf(x / sD) / sD
        EY2 = lam * dt * np.sum(gy * gy * fY * kr) * 2e-4 / (dj + d0)
    else:
        EY, pj, _, sD, EY2 = _jump_posterior_h(x, sg, lam, p, e1, e2, al, T)

    ED = x - EY
    w = EY / x if x else 0.0
    b_eff = b_diff + (G - b_diff) * w
    drift = (m_i - al * psi_i) * T
    y = drift + b_diff * ED + G * EY

    # Dispersion has two parts and BOTH are needed at every horizon:
    #   orthogonal   kappa_i^2 (1 - rho^2) : idiosyncratic noise, no x dependence
    #   split        (gamma_i - b_diff)^2 Var(Y|x) : uncertainty about WHICH
    #                channel produced the move. This is the term that peaks at
    #                the crossover, where the move could plausibly be either.
    # An earlier version computed the split term only at h=1, which made the
    # multi-day sd constant across x - it lost exactly the feature it exists to
    # show.
    vfac = (1 - np.exp(-2 * al * T)) / (2 * al) if al > 0 else T
    var = K**2 * (1 - R**2) * vfac
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
    ap.add_argument("--horizons", default="1",
                    help="comma-separated horizons in days. One horizon gives "
                         "the detailed per-shock breakdown; several give a "
                         "shock x horizon grid.")
    ap.add_argument("--psi", type=float, default=0.0,
                    help="the name's current filtered liquidity level")
    a = ap.parse_args()

    sysp, idio, used = _load(a.name, a.date, a.drawer)
    xs = [float(v) / 100.0 for v in a.shocks.split(",")]
    hs = [int(v) for v in a.horizons.split(",")]

    print("=" * 74)
    print("%s   %s" % (a.name, used))
    print("=" * 74)
    print("  systematic  " + "  ".join("%s %.4f" % (k[1:].lower(), sysp[k]) for k in SYS_KEYS))
    print("  name        " + "  ".join("%s %.4f" % (k[1:].lower(), idio[k]) for k in IDIO_KEYS))
    r0 = name_shock(xs[0], sysp, idio, hs[0], a.psi)
    print("\n  b_diff = beta + kappa*rho/sigma = %.4f      gamma_i = %.4f"
          % (r0["b_diff"], r0["gamma"]))
    print("  m_i = %.4f" % r0["m_i"])

    if len(hs) == 1:
        h = hs[0]
        print("\n  horizon %d day(s), sigma_h = %.3f%%" % (h, 100 * r0["sigma_h"]))
        print("\n      x     P(jump|x)    E[Y|x]     E[D|x]    b_eff        y_i    sd(y|x)")
        print("  " + "-" * 70)
        for x in xs:
            r = name_shock(x, sysp, idio, h, a.psi)
            print("  %6.1f%%    %7.4f  %8.3f%%  %8.3f%%  %7.3f  %8.2f%%  %7.2f%%"
                  % (100 * x, r["p_jump"], 100 * r["EY"], 100 * r["ED"],
                     r["b_eff"], 100 * r["y"], 100 * r["sd"]))
        print("\n  y_i is a conditional MEAN. Quote it with sd(y|x), never alone.")
        return

    grids = {k: np.zeros((len(hs), len(xs))) for k in ("y", "b_eff", "w", "sd")}
    sig = []
    for i, h in enumerate(hs):
        for j, x in enumerate(xs):
            r = name_shock(x, sysp, idio, h, a.psi)
            for k in grids:
                grids[k][i, j] = r[k]
        sig.append(r["sigma_h"])

    hdr = "   h \\ x " + "".join("%9.0f%%" % (100 * x) for x in xs)
    for title, key, pct in (("IDIOSYNCRATIC SHOCK  y_i(x, h)", "y", True),
                            ("CONDITIONAL SD  sd(y_i | x, h)", "sd", True),
                            ("EFFECTIVE BETA  b_eff(x, h)", "b_eff", False),
                            ("JUMP SHARE  w(x, h)", "w", False)):
        print("\n" + title)
        print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
        for i, h in enumerate(hs):
            cells = "".join(("%9.2f%%" % (100 * grids[key][i, j])) if pct
                            else ("%10.3f" % grids[key][i, j])
                            for j in range(len(xs)))
            print("  %4dd " % h + cells)
    print("\n  sigma_h: " + "  ".join("%dd=%.2f%%" % (h, 100 * s_)
                                      for h, s_ in zip(hs, sig)))
    print("""
  b_eff runs between b_diff (all diffusion) and gamma_i (all jump). Down a
  column it converges to the same value for EVERY shock as h grows - w tends
  to the jump share of total variance - so the two loadings only separate at
  short horizons and large moves.

  y_i is a conditional MEAN and the sd grid is not decoration: it peaks at the
  crossover, where the move could plausibly be either channel.""")


if __name__ == "__main__":
    main()
