"""What is alpha, once the estimator's own bias is taken out?

WHY THIS EXISTS. Table 1 reports alpha ~ 0.68 with the 95% upper bound at
0.99 on every date. Draft 7's alpha priors were all Beta, support (0, 1), and
alpha = 1 is a 175-trading-day half-life - the model could not express
reversion inside a year, so 0.68 is where the prior stopped, not where the
data pointed.

Nor can the raw estimator settle it. poc/alpha_ar1_bias.py measures the
near-unit-root AR(1) bias at about +4/T_years in ABSOLUTE terms whatever
alpha is, so a 252-day fit returns roughly alpha + 4 and cannot come back
below ~4 for any true alpha. The published 0.68 and an unconstrained fit
disagree by construction, and neither is a measurement.

WHAT THIS DOES. OLS on Psi is the exact MLE for alpha, so no prior and no
MCMC are needed. For each window length it fits alpha on every NON-
OVERLAPPING block of that length in the history and averages, giving

    E[alpha_hat(T)] ~= alpha + c / T,       c ~= 4

Regressing the per-window means on 1/T then reads alpha off the INTERCEPT
and c off the slope. Two things make the answer credible rather than merely
arithmetic:

  * c should come back near 4. If it does not, the 1/T story is wrong and
    the intercept means nothing - report that, do not quote the intercept.
  * the jackknife column is an independent correction. It should agree with
    the intercept. Where it does not, say so.

Blocks are non-overlapping ON PURPOSE. Nested windows all ending today share
their data, so their alpha_hats are strongly correlated and the regression
would understate its own uncertainty.

    python poc/alpha_on_real_spx.py
    SYMBOL=^SPX WINDOWS=252,504,1008,2016 python poc/alpha_on_real_spx.py

Set MKTDEPTH_DATA_MODE=live to pull fresh prices instead of the snapshot.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Library.DataAccess import get_aligned_price_panel              # noqa: E402
from Library.RiskEngineKimYi2025 import systematic_psi_returns      # noqa: E402

DT = 1.0 / 252.0


def ols_alpha(psi):
    """alpha = -log(rho)/dt from the AR(1) regression, intercept included."""
    x, y = psi[:-1], psi[1:]
    if len(x) < 20:
        return np.nan
    xm, ym = x.mean(), y.mean()
    den = ((x - xm) ** 2).sum()
    if den <= 0:
        return np.nan
    rho = ((x - xm) * (y - ym)).sum() / den
    if not 0 < rho < 1:
        return np.nan            # explosive or sign-flipped: no alpha to read
    return -np.log(rho) / DT


def alpha_of_block(returns):
    """De-mean exactly as the estimator does, cumulate to Psi, then OLS."""
    return ols_alpha(np.cumsum(systematic_psi_returns(np.asarray(returns))))


def jackknife_block(returns, m=4):
    a_full = alpha_of_block(returns)
    b = len(returns) // m
    parts = [alpha_of_block(returns[i * b:(i + 1) * b]) for i in range(m)]
    if np.isnan(a_full) or np.any(np.isnan(parts)):
        return np.nan
    return (m * a_full - np.mean(parts)) / (m - 1)


def main():
    sym = os.environ.get("SYMBOL", "^SPX")
    windows = [int(x) for x in
               os.environ.get("WINDOWS", "252,504,1008,2016").split(",")]

    px = get_aligned_price_panel([sym], reference=sym, verbose=False)
    r = px[sym].pct_change().dropna().to_numpy()
    print("%s: %d returns, %.1f years\n" % (sym, len(r), len(r) * DT))

    print("  %7s %7s %10s %10s %10s %11s"
          % ("window", "blocks", "alpha_hat", "sd", "half-life", "jackknife"))
    xs, ys = [], []
    for n in windows:
        nb = len(r) // n
        if nb < 2:
            print("  %6dd %7d   (need at least 2 non-overlapping blocks)" % (n, nb))
            continue
        a = np.array([alpha_of_block(r[i * n:(i + 1) * n]) for i in range(nb)])
        j = np.array([jackknife_block(r[i * n:(i + 1) * n]) for i in range(nb)])
        a, j = a[~np.isnan(a)], j[~np.isnan(j)]
        if a.size == 0:
            print("  %6dd %7d   (no block produced a usable rho)" % (n, nb))
            continue
        print("  %6dd %7d %10.3f %10.3f %9.0fd %11.3f"
              % (n, a.size, a.mean(), a.std(ddof=1) if a.size > 1 else np.nan,
                 np.log(2) / a.mean() * 252, j.mean() if j.size else np.nan))
        xs.append(1.0 / (n * DT))
        ys.append(a.mean())

    if len(xs) >= 3:
        slope, intercept = np.polyfit(xs, ys, 1)
        print("\n  alpha_hat(T) = %.3f + %.2f / T      (fit over %d windows)"
              % (intercept, slope, len(xs)))
        print("  slope should be near 4 - it is the AR(1) bias. %s"
              % ("looks right, so read the intercept" if 2 <= slope <= 7 else
                 "IT IS NOT, so the intercept is not interpretable"))
        if intercept > 0:
            print("  bias-free alpha = %.3f -> half-life %.0f trading days"
                  % (intercept, np.log(2) / intercept * 252))
            print("  alpha*T > 32 needs %.1f years of data to estimate it well"
                  % (32.0 / intercept))
        else:
            print("  intercept is NEGATIVE: the data is consistent with no "
                  "mean reversion at all over this span.")
    else:
        print("\n  need 3+ usable windows to extrapolate")


if __name__ == "__main__":
    main()
