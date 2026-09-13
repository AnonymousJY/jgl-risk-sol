"""Does conditioning on the market actually identify b_i? A probe, not a fit.

The derivation says the idiosyncratic likelihood is blind to rho_iX because it
conditions on the name's own lagged state and not on the market's
contemporaneous increment, and that adding that one conditioning contributes
exactly one new moment - Cov(U,V) - which identifies b_i and leaves a
one-dimensional ridge. That is a claim about arithmetic, and arithmetic can be
checked before anything touches PyMC.

Everything here is simulated from known parameters, so there is a right answer
to compare against. No data, no drawer, no sampler. Five checks:

  A  the closed form for the jump integral equals brute-force quadrature
  B  the joint density integrates to 1 over (u, v)
  C  the MARGINAL likelihood is flat in b_i - the profile is the finding
  D  the JOINT likelihood recovers b_i, kappa_perp and gamma_i
  E  regressing b_i on 1/sigma across windows recovers rho_iX

If C and D come out as the derivation says, the PyMC work is worth doing. If
they do not, this cost an afternoon instead of a rewrite.

    python poc/joint_likelihood_probe.py
    python poc/joint_likelihood_probe.py --paths 40000 --windows 40
"""
import argparse

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

BASE_DAYS = 252
DT = 1.0 / BASE_DAYS


# ---------------------------------------------------------------------------
# simulation - page 42's convention: the ADED variate is ADDITIVE on Psi
# ---------------------------------------------------------------------------
def simulate(n, sigma, lamb, p, e1, e2, betai, kappai, rhoix, gammai, rng):
    """Return (U, V): the market and name innovations of the derivation.

    U_k = sigma sqrt(dt) Z_k + dJ_k
    V_k = sigma b_i sqrt(dt) Z_k + kappa_perp sqrt(dt) W_perp_k + gamma_i dJ_k

    built from the STRUCTURAL parameters, so b_i and kappa_perp are known
    targets rather than things the simulation was told.
    """
    b = betai + kappai * rhoix / sigma
    kperp = kappai * np.sqrt(1.0 - rhoix ** 2)

    z = rng.standard_normal(n)
    wp = rng.standard_normal(n)
    k = rng.poisson(lamb * DT, size=n)
    m = int(k.sum())
    j = np.zeros(n)
    if m:
        idx = np.repeat(np.arange(n), k)
        up = rng.random(m) < p
        mag = np.where(up, rng.exponential(1.0 / e1, m),
                       -rng.exponential(1.0 / e2, m))
        j = np.bincount(idx, weights=mag, minlength=n)

    s = sigma * np.sqrt(DT)
    U = s * z + j
    V = b * s * z + kperp * np.sqrt(DT) * wp + gammai * j
    return U, V, b, kperp


# ---------------------------------------------------------------------------
# the two likelihoods
# ---------------------------------------------------------------------------
def loglik_marginal(V, phii, gammai, sigma, lamb, p, e1, e2):
    """Page 42: the name alone. Gaussian(0, phi^2 dt) convolved with the ADED
    scaled by gamma_i, which is the eta/gamma substitution the code makes."""
    s = phii * np.sqrt(DT)
    s2 = s * s
    q1, q2 = e1 / gammai, e2 / gammai
    Iu = p * q1 * np.exp(0.5 * q1 ** 2 * s2 - q1 * V) * norm.cdf((V - q1 * s2) / s)
    Id = (1 - p) * q2 * np.exp(0.5 * q2 ** 2 * s2 + q2 * V) * norm.cdf(-(V + q2 * s2) / s)
    f = (1 - lamb * DT) * norm.pdf(V / s) / s + lamb * DT * (Iu + Id)
    return float(np.sum(np.log(np.maximum(f, 1e-300))))


def joint_logpdf(U, V, b, kperp, gammai, sigma, lamb, p, e1, e2):
    """log f(u,v) of the derivation, Section 5, evaluated in LOG SPACE.

    The algebra is Section 5.1 - complete the square in j, integrate the
    half-line, get Phi - but written as logs throughout, because
    exp(A B^2 / 2) overflows on its own: on a (u,v) grid wide enough to check
    normalisation, w = v - b u reaches order 1 and w c / t^2 reaches 3e4, so
    A B^2 / 2 reaches 1e4 and exp() returns inf. The product with exp(-Q) is
    finite - Cauchy-Schwarz gives B0^2 <= 2 Q / A, hence -Q + A B0^2/2 <= 0 -
    so the fix is to never form the two factors separately.

        log f = log(pre) + logsumexp[ T0, T1, T2 ]

        T0 = log(1 - lam dt) - Q
        T1 = log(lam dt sqrt(2 pi A) p e1)  - Q + A Bu^2/2 + logPhi( Bu sqrt A)
        T2 = log(lam dt sqrt(2 pi A) q e2)  - Q + A Bd^2/2 + logPhi(-Bd sqrt A)

    norm.logcdf is stable in the far tail where norm.cdf underflows to 0.
    """
    s2 = sigma ** 2 * DT
    t2 = kperp ** 2 * DT
    c = gammai - b
    w = V - b * U

    A = 1.0 / (1.0 / s2 + c * c / t2)
    Q = U * U / (2 * s2) + w * w / (2 * t2)
    base = U / s2 + w * c / t2
    rtA = np.sqrt(A)

    Bu = base - e1
    Bd = base + e2
    lj = np.log(lamb * DT) + 0.5 * np.log(2 * np.pi * A)

    T0 = np.log1p(-lamb * DT) - Q
    T1 = lj + np.log(p * e1) - Q + 0.5 * A * Bu ** 2 + norm.logcdf(Bu * rtA)
    T2 = lj + np.log((1 - p) * e2) - Q + 0.5 * A * Bd ** 2 + norm.logcdf(-Bd * rtA)

    lpre = -np.log(2 * np.pi) - 0.5 * np.log(s2) - 0.5 * np.log(t2)
    return lpre + logsumexp(np.stack([T0, T1, T2]), axis=0)


def joint_density(U, V, b, kperp, gammai, sigma, lamb, p, e1, e2):
    return np.exp(joint_logpdf(U, V, b, kperp, gammai, sigma, lamb, p, e1, e2))


def loglik_joint(U, V, b, kperp, gammai, sigma, lamb, p, e1, e2):
    return float(np.sum(joint_logpdf(U, V, b, kperp, gammai,
                                     sigma, lamb, p, e1, e2)))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_A(sigma, lamb, p, e1, e2, b, kperp, gammai):
    """Closed form against brute-force quadrature, one (u,v) at a time."""
    print("\nA. closed form vs quadrature")
    s2, t2 = sigma ** 2 * DT, kperp ** 2 * DT
    c = gammai - b
    grid = np.linspace(-0.5, 0.5, 400001)
    dj = grid[1] - grid[0]
    fJ = np.where(grid >= 0, p * e1 * np.exp(-e1 * np.clip(grid, 0, None)),
                  (1 - p) * e2 * np.exp(e2 * np.clip(grid, None, 0)))
    print("   %8s %8s %16s %16s %10s" % ("u", "v", "closed", "quadrature", "rel err"))
    worst = 0.0
    for u in (-0.03, -0.005, 0.0, 0.01):
        for v in (-0.08, 0.0, 0.05):
            w = v - b * u
            integ = (fJ * norm.pdf((u - grid) / np.sqrt(s2)) / np.sqrt(s2)
                     * norm.pdf((w - c * grid) / np.sqrt(t2)) / np.sqrt(t2))
            quad = ((1 - lamb * DT) * norm.pdf(u / np.sqrt(s2)) / np.sqrt(s2)
                    * norm.pdf(w / np.sqrt(t2)) / np.sqrt(t2)
                    + lamb * DT * np.sum(integ) * dj)
            cf = float(joint_density(np.array([u]), np.array([v]), b, kperp,
                                     gammai, sigma, lamb, p, e1, e2)[0])
            rel = abs(cf - quad) / max(abs(quad), 1e-300)
            worst = max(worst, rel)
            print("   %8.3f %8.3f %16.8e %16.8e %10.2e" % (u, v, cf, quad, rel))
    print("   worst relative error %.2e  -> %s" % (worst,
          "PASS" if worst < 2e-5 else "FAIL"))
    print("   (tolerance 2e-5: the QUADRATURE is the coarse side here -")
    print("    its grid is 2.5e-6 wide against a density peaking at 2e3)")
    return worst < 2e-5


def check_B(sigma, lamb, p, e1, e2, b, kperp, gammai):
    """Does f(u,v) integrate to 1?"""
    print("\nB. normalisation of f(u,v)")
    n = 1400
    u = np.linspace(-0.45, 0.45, n)
    v = np.linspace(-1.6, 1.6, n)
    UU, VV = np.meshgrid(u, v, indexing="ij")
    f = joint_density(UU.ravel(), VV.ravel(), b, kperp, gammai,
                      sigma, lamb, p, e1, e2).reshape(n, n)
    mass = float(f.sum() * (u[1] - u[0]) * (v[1] - v[0]))
    print("   grid mass %.6f  -> %s" % (mass, "PASS" if abs(mass - 1) < 2e-3 else "FAIL"))
    return abs(mass - 1) < 2e-3


def _fit_joint(U, V, sigma, lamb, p, e1, e2, x0=None):
    """MLE of (b, kappa_perp, gamma) under the joint density. L-BFGS-B on
    (b, log kappa_perp, log gamma) - Nelder-Mead needs thousands of evaluations
    and this needs about a hundred."""
    def nll(th):
        return -loglik_joint(U, V, th[0], np.exp(th[1]), np.exp(th[2]),
                             sigma, lamb, p, e1, e2)
    if x0 is None:
        x0 = np.array([1.0, np.log(0.2), np.log(2.0)])
    r = minimize(nll, x0, method="L-BFGS-B",
                 bounds=[(0.01, 10.0), (np.log(1e-3), np.log(3.0)),
                         (np.log(0.05), np.log(30.0))])
    return r.x[0], np.exp(r.x[1]), np.exp(r.x[2]), r


def check_CD(U, V, b_true, kperp_true, gammai, sigma, lamb, p, e1, e2, phii):
    """The heart of it. The marginal's only handle on (b, kappa_perp) is
    phi_i^2 = sigma^2 b^2 + kappa_perp^2, so every point on that CIRCLE is
    likelihood-indifferent. Walk the circle and evaluate both densities."""
    print("\nC. the indifference circle  sigma^2 b^2 + kappa_perp^2 = phi_i^2")
    print("   every row below has the SAME phi_i, so the marginal cannot tell")
    print("   them apart. The joint can.")
    print("   %8s %11s %16s %16s" % ("b_i", "kap_perp", "MARGINAL logL", "JOINT logL"))
    print("   " + "-" * 56)
    mrefs, jrefs = [], []
    for bb in (1.20, 1.50, b_true, 2.10, 2.40):
        kp2 = phii ** 2 - sigma ** 2 * bb ** 2
        if kp2 <= 0:
            continue
        kp = np.sqrt(kp2)
        lm = loglik_marginal(V, phii, gammai, sigma, lamb, p, e1, e2)
        lj = loglik_joint(U, V, bb, kp, gammai, sigma, lamb, p, e1, e2)
        mark = "   <- truth" if abs(bb - b_true) < 1e-9 else ""
        print("   %8.4f %11.4f %16.3f %16.3f%s" % (bb, kp, lm, lj, mark))
        mrefs.append(lm); jrefs.append(lj)
    mspread = max(mrefs) - min(mrefs)
    jspread = max(jrefs) - min(jrefs)
    print("   " + "-" * 56)
    print("   spread over the circle:  marginal %.2e   joint %.1f" % (mspread, jspread))
    flat = mspread < 1e-8 and jspread > 50
    print("   marginal flat AND joint curved -> %s" % ("PASS" if flat else "FAIL"))

    print("\nD. JOINT likelihood: recover b_i, kappa_perp, gamma_i")
    bh, kh, gh, r = _fit_joint(U, V, sigma, lamb, p, e1, e2)
    print("   %-14s %12s %12s %10s" % ("", "true", "estimate", "rel err"))
    ok = True
    for nm, tv, ev in (("b_i", b_true, bh), ("kappa_perp", kperp_true, kh),
                       ("gamma_i", gammai, gh)):
        rel = abs(ev - tv) / abs(tv)
        ok &= rel < 0.05
        print("   %-14s %12.4f %12.4f %9.2f%%" % (nm, tv, ev, 100 * rel))
    print("   all within 5%% -> %s   (%d likelihood evaluations)"
          % ("PASS" if ok else "FAIL", r.nfev))
    return flat and ok


def check_E(a, rng):
    """sigma moves across windows; b_i = beta_i + (kappa_i rho)/sigma is linear
    in 1/sigma, so the slope splits beta_i from kappa_i rho."""
    print("\nE. cross-window regression of b_i on 1/sigma")
    sigmas = np.linspace(0.085, 0.26, a.windows)
    bs, kps = [], []
    for sg in sigmas:
        U, V, _, _ = simulate(a.epaths, sg, a.lamb, a.p, a.eta1, a.eta2,
                              a.betai, a.kappai, a.rhoix, a.gammai, rng)
        bh, kh, _, _ = _fit_joint(U, V, sg, a.lamb, a.p, a.eta1, a.eta2)
        bs.append(bh); kps.append(kh)
    bs, kps = np.array(bs), np.array(kps)

    X = np.column_stack([np.ones_like(sigmas), 1.0 / sigmas])
    (beta_hat, pi_hat), *_ = np.linalg.lstsq(X, bs, rcond=None)
    kperp_hat = float(np.mean(kps))
    kappa_hat = np.sqrt(kperp_hat ** 2 + pi_hat ** 2)
    rho_hat = pi_hat / kappa_hat

    print("   %d windows, sigma %.3f to %.3f, %s steps each"
          % (a.windows, sigmas[0], sigmas[-1], "{:,}".format(a.epaths)))
    print("   %-20s %12s %12s %10s" % ("", "true", "estimate", "rel err"))
    ok = True
    for nm, tv, ev in (("beta_i (intercept)", a.betai, beta_hat),
                       ("kappa_i rho (slope)", a.kappai * a.rhoix, pi_hat),
                       ("kappa_i", a.kappai, kappa_hat),
                       ("rho_iX", a.rhoix, rho_hat)):
        rel = abs(ev - tv) / abs(tv)
        ok &= rel < 0.12
        print("   %-20s %12.4f %12.4f %9.2f%%" % (nm, tv, ev, 100 * rel))
    print("   all within 12%% -> %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=20000,
                    help="steps per window (default 20,000 ~ 79 years)")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--epaths", type=int, default=6000,
                    help="steps per window in check E")
    ap.add_argument("--seed", type=int, default=20240114)
    ap.add_argument("--sigma", type=float, default=0.1437)
    ap.add_argument("--lamb", type=float, default=12.64)
    ap.add_argument("--p", type=float, default=0.3046)
    ap.add_argument("--eta1", type=float, default=33.22)
    ap.add_argument("--eta2", type=float, default=37.36)
    ap.add_argument("--betai", type=float, default=1.2427)
    ap.add_argument("--kappai", type=float, default=0.1488)
    ap.add_argument("--rhoix", type=float, default=0.5198)
    ap.add_argument("--gammai", type=float, default=3.095)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    print("=" * 72)
    print("JOINT (CONDITIONAL) LIKELIHOOD PROBE - simulated, known answer")
    print("=" * 72)
    print("  systematic : sigma %.4f  lambda %.2f  p %.4f  eta1 %.2f  eta2 %.2f"
          % (a.sigma, a.lamb, a.p, a.eta1, a.eta2))
    print("  name       : beta_i %.4f  kappa_i %.4f  rho_iX %.4f  gamma_i %.4f"
          % (a.betai, a.kappai, a.rhoix, a.gammai))

    U, V, b_true, kperp_true = simulate(a.paths, a.sigma, a.lamb, a.p, a.eta1,
                                        a.eta2, a.betai, a.kappai, a.rhoix,
                                        a.gammai, rng)
    phii = np.sqrt(a.sigma ** 2 * b_true ** 2 + kperp_true ** 2)
    print("  implied    : b_i %.4f  kappa_perp %.4f  phi_i %.4f"
          % (b_true, kperp_true, phii))
    print("  sample     : %s steps" % "{:,}".format(a.paths))

    res = [check_A(a.sigma, a.lamb, a.p, a.eta1, a.eta2, b_true, kperp_true, a.gammai),
           check_B(a.sigma, a.lamb, a.p, a.eta1, a.eta2, b_true, kperp_true, a.gammai),
           check_CD(U, V, b_true, kperp_true, a.gammai, a.sigma, a.lamb, a.p,
                    a.eta1, a.eta2, phii),
           check_E(a, rng)]

    print("\n" + "=" * 72)
    print("VERDICT: %s" % ("all checks passed - the derivation holds, and the "
                           "PyMC work is worth doing"
                           if all(res) else
                           "at least one check FAILED - see above"))
    print("=" * 72)


if __name__ == "__main__":
    main()
