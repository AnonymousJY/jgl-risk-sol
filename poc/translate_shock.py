"""
Translate a prescribed systematic shock into a name-level shock.

The CAPM answer is one number: r_i = beta_i * r_market. This model says that
is only right for small shocks, because a name loads on the systematic factor
through TWO channels with DIFFERENT coefficients:

    diffusion    r_X carries sigma dZ ; name i carries sigma beta_i dZ + kappa_i dW_i
    jump         r_X carries Y        ; name i carries gamma_i Y

The jump Y is a single shared draw (RiskEngineKimYi2025 line 359: Y has no
asset axis, it is scaled per name by gamma_i). The diffusion is shared only
through the correlation rho_iX between dW_i and dZ.

So there are two betas, not one:

    b_diff  =  beta_i + kappa_i * rho_iX / sigma      (diffusive channel)
    b_jump  =  gamma_i                                (jump channel)

and the translation of an observed systematic move u splits u into its
diffusive and jump parts and applies the right beta to each:

    E[r_i | r_X = u]  =  b_diff * (u - E[Y|u])  +  b_jump * E[Y|u]

which rearranges to the whole answer in one line:

    EFFECTIVE BETA(u)  =  b_diff  +  (b_jump - b_diff) * w(u)

        where  w(u) = E[Y|u] / u   is the share of the move attributable
                                    to the systematic jump

w(u) is near 0 for a small move (all diffusion) and near 1 for a large one
(a 5-sigma day is far more likely a jump than a tail draw of the diffusion).
So the effective beta MIGRATES from b_diff toward gamma_i as the prescribed
shock gets larger. That is the model's answer to "which beta do I use": it
depends on the size of the shock, which is exactly what CAPM cannot express.

w(u) comes from the one-day mixture posterior against the fitted parameters:

    no jump   (1 - lamb*dt) * phi(u ; sigma*sqrt(dt))
    one jump  lamb*dt * INT f_Y(y) phi(u - y ; sigma*sqrt(dt)) dy

with Y asymmetric-double-exponential (Kou): +Exp(eta1) w.p. pprob,
-Exp(eta2) w.p. 1-pprob.

DERIVATION
----------
Engine, lines 356-365:

    Psi_{i,t+1} = (1 - alpha dt) Psi_{i,t} + sqrt(v_i dt)(LZ)_{i,t} + gamma_i Y_t
    r_{i,t+1}   = dPsi_{i,t+1} + mu_i dt

Subtracting Psi_{i,t}:

    r_{i,t+1} = [-alpha dt Psi_{i,t} + mu_i dt]  +  [sqrt(v_i dt)(LZ) + gamma_i Y]
                 \_______ known at t _______/       \______ innovation ______/

Define the innovation by moving the known part across ("stripping the pull"):

    eps_i = r_i - mu_i dt + alpha dt Psi_{i,t}                             (2)

The variance setter gives v_i = (sigma b_i)^2 + 2 sigma b_i k_i rho + k_i^2,
which is Var(sigma b_i dZ + k_i dW_i)/dt with corr(dW_i, dZ) = rho_iX. So

    eps_i = sigma beta_i sqrt(dt) Z + kappa_i sqrt(dt) W_i + gamma_i Y     (3)
    eps_X = sigma sqrt(dt) Z + Y  =  D + Y                                 (4)

(systematic has mu=0, kappa=0, beta=1, rho=0, gamma=1).

Step 1, condition on the latent pair. E[W_i | Z] = rho_iX Z and Z = D/(sigma
sqrt(dt)), so kappa_i sqrt(dt) rho_iX D/(sigma sqrt(dt)) = (kappa_i rho_iX/sigma) D:

    E[eps_i | D, Y] = (beta_i + kappa_i rho_iX/sigma) D + gamma_i Y
                    = b_diff D + gamma_i Y                                 (5)

Step 2, tower property, using D = e - Y since D + Y = eps_X = e:

    E[eps_i | eps_X = e] = b_diff (e - E[Y|e]) + gamma_i E[Y|e]
                         = e [ b_diff + (gamma_i - b_diff) w(e) ],
      w(e) = E[Y|e]/e                                                      (6)

Step 3, the split, truncating the Poisson at one jump:

    E[Y|e] = pi_1 INT y f_Y(y) phi(e-y) dy
             ---------------------------------------------                (7)
             pi_0 phi(e) + pi_1 INT f_Y(y) phi(e-y) dy

Step 4, back to observables:

    E[r_i | r_X = u] = b_eff(e) e + mu_i dt - alpha dt Psi_{i,t},
        where  e = u + alpha dt Psi_{X,t}                                  (8)

The two pull terms use DIFFERENT accumulated levels, Psi_X,t and Psi_i,t, so
they do not cancel. If the prescribed shock is a pure innovation with no
history attached, both are zero and (8) reduces to r_i = b_eff(u) u.

Step 5, conditional variance by the law of total variance:

    Var(eps_i | e) = kappa_i^2 dt (1 - rho_iX^2)          [orthogonal part]
                   + (gamma_i - b_diff)^2 Var(Y|e)        [split uncertainty] (9)

Two caveats the caller must not skip:

1. The return is d(Psi) and d(Psi) carries a mean-reversion pull
   -alpha*dt*Psi_t that depends on the CURRENT LEVEL, not on the shock.
   Strip it from r_X before splitting, and add name i's own pull back at the
   end. Check the size of alpha*dt for your fitted alpha - it is not
   automatically negligible at dt = 1/252.

2. The result is a DISTRIBUTION, not a number. kappa_i sqrt(1 - rho_iX^2)
   is independent of the shock, and the split itself is uncertain. Use
   conditional_moments() and quote a quantile, not just the mean, whenever the
   number is going to be used as a stress.
"""

import numpy as np

__all__ = ["diffusive_beta", "jump_share", "effective_beta",
           "conditional_moments"]


def diffusive_beta(betai, kappai, rhoix, sigma):
    """b_diff = beta_i + kappa_i rho_iX / sigma.

    E[ sigma beta_i dZ + kappa_i dW_i | sigma dZ = d ] = b_diff * d,
    using E[dW_i | dZ] = rho_iX dZ. Note rho_iX pulls the idiosyncratic
    Brownian into the systematic response - a name can have beta_i = 1 and
    still respond more than one-for-one if rho_iX > 0.
    """
    return betai + kappai * rhoix / sigma


def _ade_pdf(y, pprob, eta1, eta2):
    """Kou asymmetric double exponential density."""
    return np.where(y >= 0,
                    pprob * eta1 * np.exp(-eta1 * np.clip(y, 0, None)),
                    (1.0 - pprob) * eta2 * np.exp(eta2 * np.clip(y, None, 0)))


def jump_share(u, sigma, lamb, pprob, eta1, eta2, dt=1.0 / 252,
               grid=20001, span=12.0):
    """w(u) = E[Y | u] / u, the share of the move attributable to the jump.

    Numerical integration of the one-jump branch on a grid. Returns
    (w, E[Y|u], P(jump|u)).
    """
    u = float(u)
    sd = sigma * np.sqrt(dt)
    p_jump_prior = 1.0 - np.exp(-lamb * dt)

    lo, hi = u - span * sd, u + span * sd
    lo, hi = min(lo, -abs(u) * 3 - 10 * sd), max(hi, abs(u) * 3 + 10 * sd)
    y = np.linspace(lo, hi, grid)

    phi = np.exp(-0.5 * ((u - y) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    fy = _ade_pdf(y, pprob, eta1, eta2)

    dens_jump = np.trapezoid(fy * phi, y) * p_jump_prior
    num_y = np.trapezoid(y * fy * phi, y) * p_jump_prior

    phi0 = np.exp(-0.5 * (u / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    dens_nojump = (1.0 - p_jump_prior) * phi0

    num_y2 = np.trapezoid(y * y * fy * phi, y) * p_jump_prior

    total = dens_jump + dens_nojump
    if total <= 0:
        return 1.0, u, 1.0, 0.0                 # numerically all-jump
    ey = num_y / total
    ey2 = num_y2 / total                        # E[Y^2 | e]; Y = 0 on the
    var_y = max(0.0, ey2 - ey * ey)             # no-jump branch contributes 0
    return (ey / u if u != 0 else 0.0), ey, dens_jump / total, var_y


def effective_beta(u, betai, kappai, rhoix, gammai,
                   sigma, lamb, pprob, eta1, eta2, dt=1.0 / 252):
    """The single number to multiply the prescribed shock by.

    b_eff(u) = b_diff + (gamma_i - b_diff) * w(u)
    """
    b_d = diffusive_beta(betai, kappai, rhoix, sigma)
    w, ey, pj, vy = jump_share(u, sigma, lamb, pprob, eta1, eta2, dt)
    return b_d + (gammai - b_d) * w, dict(b_diff=b_d, b_jump=gammai,
                                          w=w, E_Y=ey, P_jump=pj, Var_Y=vy)


def conditional_moments(u, betai, kappai, rhoix, gammai,
                        sigma, lamb, pprob, eta1, eta2, dt=1.0 / 252,
                        mui=0.0, psi_i=0.0, alpha=0.0):
    """Mean and sd of r_i given r_X = u, plus the pull and drift terms.

    The sd is the part CAPM has no analogue for: kappa_i sqrt(1 - rho_iX^2)
    is orthogonal to the systematic move and survives the conditioning.
    """
    b_eff, parts = effective_beta(u, betai, kappai, rhoix, gammai,
                                  sigma, lamb, pprob, eta1, eta2, dt)
    mean = b_eff * u + mui * dt - alpha * dt * psi_i

    # Var(eps_i | e) = E[Var(eps_i | D,Y) | e] + Var(E[eps_i | D,Y] | e)
    #                = kappa^2 dt (1 - rho^2)   +  (gamma_i - b_diff)^2 Var(Y|e)
    #
    # The second term is the uncertainty in the jump/diffusion SPLIT. It
    # vanishes for tiny shocks (certainly diffusion) and for huge ones
    # (certainly a jump), and PEAKS in the ambiguous middle where the move
    # could plausibly be either. Omitting it understates dispersion exactly
    # where the translation is least trustworthy.
    var_orth = (kappai ** 2) * dt * max(0.0, 1.0 - rhoix ** 2)
    var_split = (gammai - parts["b_diff"]) ** 2 * parts["Var_Y"]
    sd = float(np.sqrt(var_orth + var_split))
    return mean, sd, dict(b_eff=b_eff, sd_orthogonal=float(np.sqrt(var_orth)),
                          sd_split=float(np.sqrt(var_split)), **parts)


if __name__ == "__main__":
    # Illustrative parameters, NOT estimates. Systematic annualised vol 20%,
    # 10 jumps a year, mean down-jump 1/40 = 2.5%.
    SYS = dict(sigma=0.20, lamb=10.0, pprob=0.40, eta1=50.0, eta2=40.0)
    NAMES = {
        "low gamma  ": dict(betai=1.10, kappai=0.25, rhoix=0.20, gammai=1.10),
        "mid gamma  ": dict(betai=1.10, kappai=0.25, rhoix=0.20, gammai=2.00),
        "high gamma ": dict(betai=1.10, kappai=0.25, rhoix=0.20, gammai=3.50),
    }
    dt = 1.0 / 252
    shocks = [-0.005, -0.01, -0.02, -0.03, -0.05, -0.10]

    print("systematic: sigma=%.0f%%/yr (daily %.2f%%), lambda=%.0f/yr, "
          "mean down jump=%.1f%%"
          % (SYS["sigma"] * 100, SYS["sigma"] * np.sqrt(dt) * 100,
             SYS["lamb"], 100 / SYS["eta2"]))
    print()
    print("%-12s %8s %8s %8s %9s %9s %8s %9s"
          % ("name", "shock", "P(jump)", "w(u)", "b_eff", "E[r_i]", "sd",
             "5% qtl"))
    print("-" * 78)
    for nm, kw in NAMES.items():
        b_d = diffusive_beta(kw["betai"], kw["kappai"], kw["rhoix"],
                             SYS["sigma"])
        print("%-12s b_diff = %.3f   b_jump = gamma_i = %.3f"
              % (nm.strip(), b_d, kw["gammai"]))
        for u in shocks:
            mean, sd, parts = conditional_moments(
                u, sigma=SYS["sigma"], lamb=SYS["lamb"], pprob=SYS["pprob"],
                eta1=SYS["eta1"], eta2=SYS["eta2"], dt=dt, **kw)
            print("%-12s %7.1f%% %8.3f %8.3f %9.3f %8.2f%% %7.2f%% %8.2f%%"
                  % ("", u * 100, parts["P_jump"], parts["w"],
                     parts["b_eff"], mean * 100, sd * 100,
                     (mean - 1.645 * sd) * 100))
        print()
