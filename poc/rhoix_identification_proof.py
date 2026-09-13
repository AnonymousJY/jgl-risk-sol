"""Is rho_iX identified? A numerical proof, from the manuscript's own density.

Everything here is self-contained: it needs no fitted parameters and no data.
It simulates Psi_i from known truth and interrogates the EXACT likelihood
printed in Appendix A of Yi & Kim (SSRN 5454874), page 37.

THE CLAIM BEING TESTED. The published density depends on the idiosyncratic
parameter vector theta = (mu_i, beta_i, kappa_i, rho_iX, gamma_i) only through

    m_i   = mu_i + (sigma beta_i)^2 / 2 - sigma beta_i kappa_i rho_iX
    phi_i = sqrt((sigma beta_i)^2 + 2 sigma beta_i kappa_i rho_iX + kappa_i^2)
    gamma_i

- three numbers from five parameters. If that is right the Fisher information
matrix must be singular with a TWO dimensional null space, and the likelihood
must be exactly constant along the corresponding surface. Both are checked.

The conditional (joint) likelihood adds Cov(U, V) and therefore one equation.
Its null space should be ONE dimensional, and the surviving null direction
should coincide with the analytic fibre

    kappa_i(rho) = kperp / sqrt(1 - rho^2),  beta_i(rho) = b_i - kappa_i rho / sigma

along which (b_i, kperp) - hence the whole joint law - is constant.

A flat likelihood is not a hard optimisation problem. It is the absence of an
answer, and no sampler, prior, penalty or reparametrisation changes it.
"""
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

# Systematic constants: the manuscript's own SPX estimates, page 27.
SIG, ALPHA, LAM, PPROB, ETA1, ETA2 = 0.17, 0.69, 10.23, 0.44, 50.50, 25.86
DT = 1.0 / 252.0
N_OBS = 5_000          # long, so any real curvature would show up

# COIN's reported idiosyncratic estimates, page 27. Any interior point works.
TRUE = dict(mui=0.16, betai=2.16, kappai=0.54, rhoix=0.43, gammai=1.91)


# --------------------------------------------------------------------------
# the two reduced-form quantities the density is built from
# --------------------------------------------------------------------------
def m_phi(betai, kappai, rhoix, mui):
    cross = SIG * betai * kappai * rhoix
    m = mui + 0.5 * (SIG * betai) ** 2 - cross
    phi2 = (SIG * betai) ** 2 + 2.0 * cross + kappai ** 2
    return m, np.sqrt(phi2)


def b_kperp(betai, kappai, rhoix):
    """What the CONDITIONAL likelihood sees instead."""
    return betai + kappai * rhoix / SIG, kappai * np.sqrt(1.0 - rhoix ** 2)


# --------------------------------------------------------------------------
# Appendix A, page 37: the published transition density of Psi_i.
# --------------------------------------------------------------------------
def loglike_marginal(psi, betai, kappai, rhoix, mui, gammai):
    m, phi = m_phi(betai, kappai, rhoix, mui)

    # Psi_i is the cumulated return with the drift m_i removed, exactly as
    # dPsi_i = dS_i/S_i - m_i dt in the appendix.
    p = psi - m * DT * np.arange(len(psi))
    d = p[1:] - (1.0 - ALPHA * DT) * p[:-1]

    s = phi * np.sqrt(DT)
    e1, e2 = ETA1 / gammai, ETA2 / gammai          # jump of size gamma_i * Y

    t0 = np.log1p(-LAM * DT) + norm.logpdf(d, scale=s)
    t1 = (np.log(LAM * DT * PPROB * e1) + 0.5 * phi ** 2 * DT * e1 ** 2 - d * e1
          + norm.logcdf((d - phi ** 2 * e1 * DT) / s))
    t2 = (np.log(LAM * DT * (1.0 - PPROB) * e2) + 0.5 * phi ** 2 * DT * e2 ** 2
          + d * e2 + norm.logcdf(-(d + phi ** 2 * e2 * DT) / s))
    return logsumexp(np.vstack([t0, t1, t2]), axis=0).sum()


# --------------------------------------------------------------------------
# the conditional likelihood: V_k = b_i U_k + (gamma_i - b_i) dJ_k + kperp W_k
# --------------------------------------------------------------------------
def loglike_joint(psi, u, betai, kappai, rhoix, mui, gammai):
    m, _ = m_phi(betai, kappai, rhoix, mui)
    bi, kperp = b_kperp(betai, kappai, rhoix)

    p = psi - m * DT * np.arange(len(psi))
    V = p[1:] - (1.0 - ALPHA * DT) * p[:-1]

    s2, t2_ = SIG ** 2 * DT, kperp ** 2 * DT
    c = gammai - bi
    w = V - bi * u

    A = 1.0 / (1.0 / s2 + c * c / t2_)
    Q = u * u / (2 * s2) + w * w / (2 * t2_)
    base = u / s2 + w * c / t2_
    rtA = np.sqrt(A)

    Bu, Bd = base - ETA1, base + ETA2
    lj = np.log(LAM * DT) + 0.5 * np.log(2 * np.pi * A)

    T0 = np.log1p(-LAM * DT) - Q
    T1 = (lj + np.log(PPROB * ETA1) - Q + 0.5 * A * Bu ** 2
          + norm.logcdf(Bu * rtA))
    T2 = (lj + np.log((1 - PPROB) * ETA2) - Q + 0.5 * A * Bd ** 2
          + norm.logcdf(-Bd * rtA))

    lpre = -np.log(2 * np.pi) - 0.5 * np.log(s2) - 0.5 * np.log(t2_)
    return (lpre + logsumexp(np.vstack([T0, T1, T2]), axis=0)).sum()


# --------------------------------------------------------------------------
def simulate(seed=20240114):
    rng = np.random.default_rng(seed)
    m, phi = m_phi(TRUE["betai"], TRUE["kappai"], TRUE["rhoix"], TRUE["mui"])
    bi, kperp = b_kperp(TRUE["betai"], TRUE["kappai"], TRUE["rhoix"])

    n = rng.poisson(LAM * DT, N_OBS)
    up = rng.random(N_OBS) < PPROB
    y = np.where(up, rng.exponential(1 / ETA1, N_OBS),
                 -rng.exponential(1 / ETA2, N_OBS)) * (n > 0)

    z = rng.standard_normal(N_OBS) * np.sqrt(DT)
    wp = rng.standard_normal(N_OBS) * np.sqrt(DT)

    u = SIG * z + y                                   # market innovation
    v = bi * SIG * z + TRUE["gammai"] * y + kperp * wp  # name innovation

    psi = np.zeros(N_OBS + 1)
    for t in range(N_OBS):
        psi[t + 1] = (1 - ALPHA * DT) * psi[t] + v[t]
    return psi + m * DT * np.arange(N_OBS + 1), u


def fisher(fn, theta, keys, step=1e-5):
    """Observed information by central differences on the log-likelihood."""
    g = []
    for k in keys:
        hi, lo = dict(theta), dict(theta)
        h = step * max(abs(theta[k]), 1.0)
        hi[k], lo[k] = theta[k] + h, theta[k] - h
        g.append((fn(**hi) - fn(**lo)) / (2 * h))
    return np.array(g)


def jacobians(betai, kappai, rhoix):
    """Exact d(sufficient quantities)/d(mu_i, beta_i, kappa_i, rho_iX, gamma_i).

    THIS, not a finite-differenced Hessian, is the right object. The density
    depends on theta only through g(theta), so the chain rule gives
    I_theta = J' I_g J exactly. I_g is nonsingular - m_i, phi_i and gamma_i ARE
    identified - so rank(I_theta) = rank(J) and null(I_theta) = null(J), with
    no differencing error anywhere. A numerically differentiated Hessian of a
    flat direction returns noise of either sign and cannot settle a rank.
    """
    b, k, r = betai, kappai, rhoix
    #                     mu_i  beta_i                 kappa_i         rho_iX      gamma_i
    dm = np.array([1.0, SIG ** 2 * b - SIG * k * r, -SIG * b * r, -SIG * b * k, 0.0])
    dphi2 = np.array([0.0, 2 * SIG ** 2 * b + 2 * SIG * k * r,
                      2 * SIG * b * r + 2 * k, 2 * SIG * b * k, 0.0])
    dgam = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    # conditional likelihood sees b_i and kperp^2 in place of phi_i^2
    dbi = np.array([0.0, 1.0, r / SIG, k / SIG, 0.0])
    dkp2 = np.array([0.0, 0.0, 2 * k * (1 - r ** 2), -2 * k ** 2 * r, 0.0])
    return (np.vstack([dm, dphi2, dgam]),      # marginal: 3 x 5
            np.vstack([dm, dbi, dkp2, dgam]))  # joint:    4 x 5


def null_space(J, tol=1e-10):
    _, s, vt = np.linalg.svd(J)
    rank = int((s > tol * s.max()).sum())
    return rank, vt[rank:]


def main():
    psi, u = simulate()
    keys = ["mui", "betai", "kappai", "rhoix", "gammai"]

    fm = lambda **t: loglike_marginal(psi, **t)
    fj = lambda **t: loglike_joint(psi, u, **t)

    print("=" * 74)
    print("IS rho_iX IDENTIFIED?  Fisher information of the published density")
    print("=" * 74)
    print("  %d observations, truth %s" % (N_OBS, TRUE))
    m, phi = m_phi(TRUE["betai"], TRUE["kappai"], TRUE["rhoix"], TRUE["mui"])
    bi, kperp = b_kperp(TRUE["betai"], TRUE["kappai"], TRUE["rhoix"])
    print("  m_i %.6f   phi_i %.6f   b_i %.6f   kperp %.6f" % (m, phi, bi, kperp))

    Jm, Jj = jacobians(TRUE["betai"], TRUE["kappai"], TRUE["rhoix"])
    for name, J, sees in (
            ("MARGINAL (published, Appendix A)", Jm, "m_i, phi_i, gamma_i"),
            ("CONDITIONAL (joint with the market)", Jj, "m_i, b_i, kperp, gamma_i")):
        rank, ns = null_space(J)
        print("\n" + "-" * 74)
        print(name)
        print("-" * 74)
        print("  the density sees: %s" % sees)
        print("  rank(J) = %d of 5 parameters  ->  null space dimension %d"
              % (rank, 5 - rank))
        for v in ns:
            v = v / np.abs(v).max()
            print("    flat direction  " + "  ".join(
                "%s %+7.4f" % (k, x) for k, x in zip(keys, v)))

    # the joint's single flat direction must be tangent to the analytic fibre
    _, nsj = null_space(Jj)
    eps = 1e-6
    r0 = TRUE["rhoix"]
    k0 = kperp / np.sqrt(1 - r0 ** 2)
    k1 = kperp / np.sqrt(1 - (r0 + eps) ** 2)
    tangent = np.array([
        0.0,                                   # mu_i handled separately
        ((bi - k1 * (r0 + eps) / SIG) - (bi - k0 * r0 / SIG)) / eps,
        (k1 - k0) / eps, 1.0, 0.0])
    m0, _ = m_phi(bi - k0 * r0 / SIG, k0, r0, 0.0)
    m1, _ = m_phi(bi - k1 * (r0 + eps) / SIG, k1, r0 + eps, 0.0)
    tangent[0] = -(m1 - m0) / eps            # mu_i moves to hold m_i fixed
    v = nsj[0] / nsj[0][3]                   # normalise both on the rho_iX slot
    t = tangent / tangent[3]
    print("\n  the joint's flat direction vs the analytic fibre tangent"
          "\n  (both normalised on rho_iX; agreement means the SVD null space"
          "\n   IS the curve kappa = kperp/sqrt(1-rho^2), beta = b_i - kappa rho/sigma)")
    print("    %-9s %12s %12s" % ("", "null(J)", "fibre"))
    for kk, a, b_ in zip(keys, v, t):
        print("    %-9s %12.6f %12.6f" % (kk, a, b_))
    print("    max abs difference %.2e" % np.abs(v - t).max())

    # --- the decisive test: walk the analytic fibre, watch the likelihood ---
    print("\n" + "=" * 74)
    print("WALKING THE FIBRE: kappa(rho) = kperp/sqrt(1-rho^2),"
          "  beta(rho) = b_i - kappa rho/sigma")
    print("=" * 74)
    print("  Every point below implies a DIFFERENT rho_iX but the same (b_i, kperp).")
    print("  mu_i is moved to hold m_i fixed, as the marginal likelihood allows.\n")
    base_m = fm(**TRUE)
    base_j = fj(**TRUE)
    print("  %7s %9s %9s %9s %16s %16s"
          % ("rho_iX", "kappa_i", "beta_i", "mu_i", "d logL marginal", "d logL joint"))
    for r in (0.10, 0.25, 0.43, 0.60, 0.75, 0.90):
        k = kperp / np.sqrt(1 - r ** 2)
        b = bi - k * r / SIG
        mu = m - 0.5 * (SIG * b) ** 2 + SIG * b * k * r
        t = dict(mui=mu, betai=b, kappai=k, rhoix=r, gammai=TRUE["gammai"])
        print("  %7.2f %9.4f %9.4f %9.4f %16.3e %16.3e"
              % (r, k, b, mu, fm(**t) - base_m, fj(**t) - base_j))

    print("\n  A difference of zero to machine precision is not a numerical")
    print("  coincidence - it is the statement that these parameter vectors")
    print("  generate the identical distribution. No estimator can separate them.")

    # --- and a fibre that does NOT hold (b_i, kperp) fixed, for contrast ---
    print("\n" + "=" * 74)
    print("CONTROL: vary rho_iX holding kappa_i and beta_i FIXED")
    print("=" * 74)
    print("  Now phi_i moves, so the marginal likelihood must respond.\n")
    print("  %7s %9s %16s %16s" % ("rho_iX", "phi_i", "d logL marginal", "d logL joint"))
    for r in (0.10, 0.25, 0.43, 0.60, 0.75, 0.90):
        t = dict(TRUE); t["rhoix"] = r
        _, ph = m_phi(TRUE["betai"], TRUE["kappai"], r, TRUE["mui"])
        print("  %7.2f %9.4f %16.3e %16.3e"
              % (r, ph, fm(**t) - base_m, fj(**t) - base_j))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# THE RESOLUTION: rho_iX is a DETERMINISTIC FUNCTION of beta_i given the data.
# ---------------------------------------------------------------------------
# The fibre is one dimensional, so ANY one of beta_i, kappa_i, rho_iX fixes the
# other two. Choosing which one to assume is therefore free - and beta_i is the
# only one of the three a reader can form an opinion about, being the name's
# direct loading on systematic liquidity. Given identified (b_i, kperp, sigma),
#
#     kappa_i rho_iX = sigma (b_i - beta_i)
#     kappa_i        = sqrt(kperp^2 + sigma^2 (b_i - beta_i)^2)
#     rho_iX         = sigma (b_i - beta_i) / kappa_i
#
# and the Theorem 3.1 remark's condition b_i > 1.5 beta_i contains NO rho_iX at
# all: it is simply beta_i < (2/3) b_i, a bound on an interpretable quantity.
def rho_from_beta(beta, bi, kperp, sigma):
    q = sigma * (bi - beta)
    return q / np.sqrt(kperp ** 2 + q ** 2)


def resolution():
    sigma = 0.17                      # manuscript's own SPX estimate, page 27
    fits = {"C": (1.2995, 0.2419), "BAC": (1.2714, 0.2390), "JPM": (1.2721, 0.1896)}
    print("\n" + "=" * 74)
    print("REPARAMETRISATION:  rho_iX AS A FUNCTION OF beta_i")
    print("=" * 74)
    print("  joint-fit means, sigma = %.2f (manuscript page 27)\n" % sigma)
    betas = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25]
    print("  %-5s %8s %8s | %s" % ("name", "b_i", "kperp",
                                   "  ".join("b=%4.2f" % b for b in betas)))
    for nm, (bi_, kp) in fits.items():
        row = "  %-5s %8.4f %8.4f | " % (nm, bi_, kp)
        row += "  ".join("%6.3f" % rho_from_beta(b, bi_, kp, sigma) for b in betas)
        print(row)
    print("\n  %-5s %10s %10s %10s" % ("name", "rho_bar", "beta<(2/3)b_i", "rho at that"))
    for nm, (bi_, kp) in fits.items():
        bstar = 2.0 / 3.0 * bi_
        print("  %-5s %10.4f %10.4f %10.4f"
              % (nm, rho_from_beta(0.0, bi_, kp, sigma), bstar,
                 rho_from_beta(bstar, bi_, kp, sigma)))
    print("\n  Reading: the identified set is rho_iX in [0, rho_bar], reached at")
    print("  beta_i = 0. The Theorem 3.1 remark holds iff beta_i < (2/3) b_i,")
    print("  equivalently rho_iX above the last column. No prior on rho_iX is")
    print("  used anywhere; the only judgement required is about beta_i.")


resolution()
