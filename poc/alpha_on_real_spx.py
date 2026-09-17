"""What is alpha, once the estimator's own bias is taken out?

WHY THIS EXISTS. Table 1 reports alpha ~ 0.68 with the 95% upper bound at
0.99 on every date. Draft 7's alpha priors were all Beta, support (0, 1), and
alpha = 1 is a 175-trading-day half-life - the model could not express
reversion inside a year, so 0.68 is where the prior stopped, not where the
data pointed. The raw estimator cannot settle it either: the near-unit-root
AR(1) bias is roughly +4/T_years in ABSOLUTE terms whatever alpha is (see
poc/alpha_ar1_bias.py), so a 252-day fit returns about alpha + 4 and cannot
come back below ~4 for any true alpha.

METHOD: INDIRECT INFERENCE, NOT AN EXPANSION. An earlier draft regressed the
per-window alpha_hats on 1/T and read alpha off the intercept. That fails -
the bias carries a 1/T^2 term, so the line over-predicts at the long end and
the intercept lands LOW: 0.30 against a true 0.68, with the fitted slope at
5.4-8.1 rather than the 4 the theory gives. Indirect inference needs no
expansion. Simulate at a candidate alpha, run THE SAME block procedure, and
keep the alpha whose simulated alpha_hat vector matches the observed one.
On 19 years of simulated data that is near unbiased at every scale tried:

      true alpha    0.68    2.00    5.00   15.00
      recovered     0.74    1.92    4.84   14.69

and it barely cares what jump parameters the calibration assumes - inverting
lambda-11 data under a lambda-77 table moves alpha by a few percent.

WHY NO TEXTBOOK CORRECTION APPLIES. Psi is rebuilt as cumsum(de-meaned
returns) inside each window, and the de-meaned returns sum to zero BY
CONSTRUCTION, so Psi[0] ~= 0 and Psi[-1] = 0 EXACTLY. The series the
estimator sees is a BRIDGE pinned at both ends, not a free OU path - and
this is true of the production estimator too, which builds its observed
series the same way. That, plus the near-unit-root bias, is why alpha_hat
comes out far too HIGH at short windows and too LOW at long ones (at a true
5.0: 9.41, 7.22, 5.64, 4.74 across 252 to 2016 days). The bias changes
sign, so no single expansion describes it and the correction has to be
calibrated on the procedure itself. Blocks are non-overlapping so the
per-window numbers are not re-reading the same data.

    python poc/alpha_on_real_spx.py
    SELFTEST=5.0 python poc/alpha_on_real_spx.py      # recover a known alpha
    START=2003-07-01 python poc/alpha_on_real_spx.py  # is alpha stable?
    SYMBOL=^SPX WINDOWS=252,504,1008,2016 GRIDN=28 REPS=60 python poc/...
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Library.DataAccess import get_aligned_price_panel              # noqa: E402
from Library.RiskEngineKimYi2025 import systematic_psi_returns      # noqa: E402

DT = 1.0 / 252.0


def block_alpha_vector(r, windows):
    """Mean alpha_hat per window over non-overlapping blocks.

    r is (n,) or (n, reps); the return is one number per window, averaged
    over blocks and reps. Blocks whose rho leaves (0, 1) carry no alpha and
    are dropped rather than clipped - clipping would invent mean reversion.
    """
    r2 = r[:, None] if r.ndim == 1 else r
    out = []
    for n in windows:
        nb = len(r2) // n
        acc = []
        for i in range(nb):
            psi = np.cumsum(systematic_psi_returns(r2[i * n:(i + 1) * n]), axis=0)
            x, y = psi[:-1], psi[1:]
            xm, ym = x.mean(0), y.mean(0)
            den = ((x - xm) ** 2).sum(0)
            rho = np.divide(((x - xm) * (y - ym)).sum(0), den,
                            out=np.full(np.shape(den), np.nan), where=den > 0)
            acc.append(np.where((rho > 0) & (rho < 1),
                                -np.log(np.clip(rho, 1e-12, None)) / DT, np.nan))
        out.append(np.nanmean(np.concatenate(acc)) if acc else np.nan)
    return np.array(out)


def simulate_returns(alpha, sigma, lam, p, eta1, eta2, n, reps, rng):
    """n returns x reps, from an exact-transition OU with ADED jumps.

    Psi is built with n+1 rows and differenced, so each block's cumsum
    reproduces Psi - Psi[block start] exactly as it does on real returns.
    """
    rho = np.exp(-alpha * DT)
    sd = sigma * np.sqrt((1 - rho ** 2) / (2 * alpha))
    psi = np.empty((n + 1, reps))
    psi[0] = rng.normal(0, sigma / np.sqrt(2 * alpha), reps)   # stationary start
    for t in range(1, n + 1):
        step = rho * psi[t - 1] + rng.normal(0, sd, reps)
        nj = rng.poisson(lam * DT, reps)
        hit = nj > 0
        if hit.any():
            m = nj[hit]
            tot = int(m.sum())
            sz = np.where(rng.random(tot) < p, rng.exponential(1 / eta1, tot),
                          -rng.exponential(1 / eta2, tot))
            step[hit] += np.bincount(np.repeat(np.arange(m.size), m),
                                     weights=sz, minlength=m.size)
        psi[t] = step
    return np.diff(psi, axis=0)


def calibration(grid, n_hist, reps, pars, windows, seed=11):
    """E[alpha_hat vector] at each candidate alpha, same procedure as the data."""
    tab = []
    for k, a in enumerate(grid):
        r = simulate_returns(a, *pars, n_hist, reps, np.random.default_rng(seed + k))
        tab.append(block_alpha_vector(r, windows))
    return np.array(tab)


def invert(obs, grid, tab):
    """The alpha whose simulated alpha_hat vector is closest to the observed."""
    ok = ~np.isnan(obs)
    d = ((tab[:, ok] - obs[ok]) ** 2).sum(1)
    k = int(np.argmin(d))
    if 0 < k < len(grid) - 1:                     # parabolic refine in log alpha
        lg, dd = np.log(grid[k - 1:k + 2]), d[k - 1:k + 2]
        den = dd[0] - 2 * dd[1] + dd[2]
        if den > 0:
            return float(np.exp(lg[1] - 0.5 * (lg[2] - lg[0]) * (dd[2] - dd[0])
                                / (2 * den)))
    return float(grid[k])


def main():
    windows = [int(x) for x in
               os.environ.get("WINDOWS", "252,504,1008,2016").split(",")]
    gridn = int(os.environ.get("GRIDN", 24))
    reps = int(os.environ.get("REPS", 40))
    boot = int(os.environ.get("BOOT", 120))
    selftest = os.environ.get("SELFTEST")

    if selftest:
        a_true = float(selftest)
        n_hist, sigma = 4900, 0.15
        r = simulate_returns(a_true, sigma, 11.0, 0.40, 50.5, 26.5, n_hist, 1,
                             np.random.default_rng(2024))[:, 0]
        print("SELFTEST: simulated %d sessions at a true alpha of %.3f\n"
              % (n_hist, a_true))
    else:
        sym = os.environ.get("SYMBOL", "^SPX")
        px, _ = get_aligned_price_panel([sym], reference=sym, verbose=False)
        ser = px[sym]
        # alpha over the FULL history assumes one alpha for the whole span. To
        # test that, refit on halves: START=1980-01-01 END=2003-06-30, then
        # START=2003-07-01. If they disagree, alpha is not a constant and it
        # cannot simply be fixed in the rolling window.
        start, end = os.environ.get("START"), os.environ.get("END")
        if start:
            ser = ser[ser.index >= start]
        if end:
            ser = ser[ser.index <= end]
        r = ser.pct_change().dropna().to_numpy()
        n_hist = len(r)
        if n_hist < max(windows) * 2:
            raise SystemExit("only %d sessions after filtering - need at least "
                             "two blocks of the longest window (%d)"
                             % (n_hist, max(windows)))
        sigma = float(r.std(ddof=1)) * np.sqrt(252.0)
        print("%s: %d returns, %.1f years (%s to %s), sigma %.3f\n"
              % (sym, n_hist, n_hist * DT, ser.index[0].date(),
                 ser.index[-1].date(), sigma))
        a_true = None

    # Jump parameters enter only through the calibration, and the recovered
    # alpha is nearly invariant to them (lambda 11 vs 77 moves it a few
    # percent), so rough values are fine. Override if a fit says otherwise.
    pars = (sigma,
            float(os.environ.get("LAMB", 11.0)), float(os.environ.get("PPROB", 0.40)),
            float(os.environ.get("ETA1", 50.5)), float(os.environ.get("ETA2", 26.5)))

    obs = block_alpha_vector(r, windows)
    print("  %8s %8s %12s %12s" % ("window", "blocks", "alpha_hat", "half-life"))
    for n, a in zip(windows, obs):
        print("  %7dd %8d %12.3f %11.0fd"
              % (n, n_hist // n, a, np.log(2) / a * 252 if a > 0 else np.nan))

    grid = np.exp(np.linspace(np.log(0.02), np.log(60.0), gridn))
    print("\ncalibrating: %d candidate alphas x %d paths x %d sessions ..."
          % (gridn, reps, n_hist))
    tab = calibration(grid, n_hist, reps, pars, windows)

    # A short window SATURATES at small alpha: below roughly alpha = 1 its
    # alpha_hat is all bias and stops moving, so the grid cannot bracket an
    # observation that lands under that floor. That is informative about the
    # window, not a failure - what matters is whether the LONGEST window,
    # which carries the identification, is bracketed.
    lo_t, hi_t = np.nanmin(tab, axis=0), np.nanmax(tab, axis=0)
    tol = 0.02 * np.abs(obs)
    out = [w for w, o, l, h in zip(windows, obs, lo_t, hi_t)
           if o < l - tol[windows.index(w)] or o > h + tol[windows.index(w)]]
    if out:
        worst = windows[-1] in out
        print("\n  %s no candidate alpha reproduces the %s window%s."
              % ("WARNING:" if worst else "note:",
                 ", ".join("%dd" % w for w in out),
                 "" if len(out) == 1 else "s"))
        print("  %s" % ("The longest window is among them, so the correction is\n  extrapolating and the number below should NOT be quoted."
                        if worst else
                        "The longest window is still bracketed, so this is the\n  short window saturating at low alpha rather than a failure."))

    a_hat = invert(obs, grid, tab)

    # Parametric bootstrap AT the recovered alpha: the honest interval is the
    # spread of what this procedure returns when alpha really is a_hat.
    rng = np.random.default_rng(4242)
    draws = np.array([invert(block_alpha_vector(
        simulate_returns(a_hat, *pars, n_hist, 1, rng)[:, 0], windows), grid, tab)
        for _ in range(boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])

    print("\n  bias-corrected alpha = %.3f   95%% [%.3f, %.3f]" % (a_hat, lo, hi))
    print("  half-life %.0f trading days   95%% [%.0f, %.0f]"
          % (np.log(2) / a_hat * 252,
             np.log(2) / hi * 252, np.log(2) / lo * 252))
    if a_true is not None:
        print("  (true alpha was %.3f - inside the interval: %s)"
              % (a_true, "yes" if lo <= a_true <= hi else "NO"))
    print("  alpha*T > 32 needs %.1f years to estimate this well; you have %.1f"
          % (32.0 / a_hat, n_hist * DT))
    if hi / lo > 4:
        print("\n  The interval spans more than a factor of four, so this says"
              "\n  which REGIME alpha is in, not what alpha is.")


if __name__ == "__main__":
    main()
