"""Why alpha was never estimated: the Beta cap and the trend, separately.

Draft 7 reports alpha near 0.5 with a tight interval, and Figure 6 shows
Psi_SPX drifting upward instead of reverting. Two independent defects produce
both, and FIXING EITHER ONE ALONE CHANGES NOTHING:

  1. THE PRIOR CAP. Every alpha prior in draft 7 is a Beta, support (0, 1).
     alpha = 1 is a half-life of 175 trading days, so a model of a process
     that reverts within a year could not be written down.

  2. THE TREND. Psi was built as the cumulative index return less the Jensen
     term 0.5 sigma^2 (~1%/yr) while the index rises ~10%/yr. The residual
     ~9%/yr is a linear trend inside a process the model says reverts to zero.

This script fits alpha by conditional MLE on simulated OU paths - exact
transition exp(-alpha dt), no jumps, no priors, so nothing but the two defects
under test can move the answer - and crosses the two fixes.

Run: python poc/alpha_prior_cap.py
"""
import numpy as np

DT = 1.0 / 252.0
N_STEPS = 504          # the paper's two-year estimation window
SIGMA = 0.1437         # the fitted systematic volatility
TREND = 0.09           # per year, the index's drift net of the Jensen term
N_REPS = 40

GRID_CAP1 = np.exp(np.linspace(np.log(1e-3), np.log(1.0), 900))
GRID_WIDE = np.exp(np.linspace(np.log(1e-3), np.log(250.0), 900))


def fit_alpha(psi, grid, dt=DT):
    """Conditional MLE of alpha over a grid, profiling out the innovation sd.

    The EXACT OU transition, exp(-alpha dt), not the engine's Euler
    (1 - alpha dt). The Euler form changes sign at alpha = 252 and would
    confound the question with a discretisation artefact.
    """
    x, y = psi[:-1], psi[1:]
    best_ll, best_a = -np.inf, None
    for a in grid:
        resid = y - np.exp(-a * dt) * x
        s2 = max(resid.var(), 1e-300)
        ll = -0.5 * len(resid) * (np.log(2.0 * np.pi * s2) + 1.0)
        if ll > best_ll:
            best_ll, best_a = ll, a
    return best_a


def ou_path(alpha, rng, n=N_STEPS, sigma=SIGMA, dt=DT):
    rho = np.exp(-alpha * dt)
    sd = sigma * np.sqrt((1.0 - np.exp(-2.0 * alpha * dt)) / (2.0 * alpha))
    psi = np.zeros(n + 1)
    psi[0] = rng.normal(0.0, sigma / np.sqrt(2.0 * alpha))
    for t in range(n):
        psi[t + 1] = rho * psi[t] + sd * rng.normal()
    return psi


def demean(series):
    """What systematic_psi_returns() does, applied to a level series."""
    d = np.diff(series)
    return np.concatenate(([0.0], np.cumsum(d - d.mean())))


def main():
    print("alpha recovery, %d replications, %d days, sigma %.4f, trend %.0f%%/yr"
          % (N_REPS, N_STEPS, SIGMA, 100 * TREND))
    for a_true in (5.0, 50.0):
        sd_inf = SIGMA / np.sqrt(2.0 * a_true)
        print("\ntrue alpha %.0f   half-life %.1f days   stationary sd %.4f"
              "   trend over window %.4f (%.1f sd)"
              % (a_true, np.log(2.0) / a_true * 252, sd_inf,
                 TREND * N_STEPS * DT, TREND * N_STEPS * DT / sd_inf))
        print("  %-18s %10s %10s %10s" % ("series", "cap 1", "cap 250", "de-meaned"))
        for label, trend in (("trend %.0f%%/yr" % (100 * TREND), TREND),
                             ("no trend", 0.0)):
            got = {"cap1": [], "wide": [], "dem": []}
            for rep in range(N_REPS):
                rng = np.random.default_rng(1000 + rep)
                psi = ou_path(a_true, rng)
                obs = psi + trend * DT * np.arange(N_STEPS + 1)
                got["cap1"].append(fit_alpha(obs, GRID_CAP1))
                got["wide"].append(fit_alpha(obs, GRID_WIDE))
                got["dem"].append(fit_alpha(demean(obs), GRID_WIDE))
            print("  %-18s %10.3f %10.3f %10.3f"
                  % (label, np.median(got["cap1"]), np.median(got["wide"]),
                     np.median(got["dem"])))

    print("\nREADING IT. 1.000 is the boundary of the capped grid, not an "
          "estimate: under the cap alpha is pinned there even on a clean\n"
          "mean-zero path, so the published alpha measured the prior. "
          "Widening the cap on a TRENDED series still misses badly (3.3\n"
          "against a true 50). Only de-meaning and widening together recover "
          "it. The 'no trend' row confirms the estimator itself is\n"
          "unbiased, so nothing here is an artefact of the grid or the "
          "profiling.")


if __name__ == "__main__":
    main()
