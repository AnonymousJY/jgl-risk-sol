"""Where alpha's bias comes from, and what it costs.

THE QUESTION THIS SETTLES. Two window studies disagreed. At a true alpha of
15 the posterior came back at 15.8 on 252 days (+5%) and coverage was fine.
At a true alpha of 0.68 - the value Table 1 of the paper reports - the same
code came back at 5.74 (+745%) with coverage 1/40. Same estimator, same
priors, same density. The obvious suspect was the alpha prior, a LogNormal
with median 25, sitting 37x above a truth of 0.68.

IT IS NOT THE PRIOR. OLS on Psi is the exact MLE for alpha (checked earlier:
identical to every digit), so this script runs OLS - no prior, no MCMC, no
Appendix B truncation - and reproduces the posterior almost exactly:

                    OLS      posterior
      252d         5.445       5.744
      504d         2.903       3.130
     1008d         1.742       1.904
     2016d         1.197       1.133

The bias is the classic near-unit-root AR(1) bias: rho_hat is biased DOWN by
about (1+3rho)/T, and alpha = -log(rho)/dt turns that into an UPWARD bias of

      bias(alpha) ~= 4 / T_years        IN ABSOLUTE TERMS, whatever alpha is

which this script confirms at both scales (4.58/2.18/1.11/0.52 at alpha 15
against 4/2/1/0.5 predicted; 5.32/2.56/1.23/0.58 at alpha 0.68).

WHAT ACTUALLY GOVERNS EVERYTHING IS alpha*T - the number of mean-reversion
e-foldings the window contains, not the number of days:

      rel bias ~= 4/(alpha*T)      rel sd ~= sqrt(2/(alpha*T))
      bias/sd  ~= 2*sqrt(2)/sqrt(alpha*T)

bias/sd below 0.5 needs alpha*T > 32. A 12-day half-life reaches that in 2.2
years; a one-year half-life needs 46 years. That single ratio reproduces the
coverage in BOTH window studies, which is why "252 days" was never the
answer to a question posed in days.

TWO SIDE FINDINGS.
  1. Appendix B's <=1-jump truncation is a LAMBDA effect, not a constant.
     posterior/OLS is 0.83 at lamb 77 (-17% on alpha) and 1.04 at lamb 11.
  2. The jackknife removes essentially all of it where alpha*T is large:
     14.97 against a true 15.0 at 2016 days, and still 14.55 at 252 days.
     At alpha 0.68 it goes NEGATIVE at 252 days - do not apply it blind.
"""
import numpy as np

DT = 1.0 / 252.0


def paths(alpha, sigma, lam, p, eta1, eta2, n, reps, rng, jumps=True):
    """reps independent Psi paths, exact OU transition, true Poisson counts."""
    rho = np.exp(-alpha * DT)
    sd = sigma * np.sqrt((1 - rho ** 2) / (2 * alpha))
    psi = np.empty((n + 1, reps))
    psi[0] = rng.normal(0, sigma / np.sqrt(2 * alpha), reps)   # stationary start
    for k in range(n):
        step = rho * psi[k] + rng.normal(0, sd, reps)
        if jumps:
            nj = rng.poisson(lam * DT, reps)
            hit = nj > 0
            if hit.any():
                m = nj[hit]
                tot_n = int(m.sum())
                sz = np.where(rng.random(tot_n) < p,
                              rng.exponential(1 / eta1, tot_n),
                              -rng.exponential(1 / eta2, tot_n))
                idx = np.repeat(np.arange(m.size), m)
                step[hit] += np.bincount(idx, weights=sz, minlength=m.size)
        psi[k + 1] = step
    return psi


def ols_alpha(psi, intercept=True):
    x, y = psi[:-1], psi[1:]
    if intercept:
        xm, ym = x.mean(0), y.mean(0)
        num = ((x - xm) * (y - ym)).sum(0)
        den = ((x - xm) ** 2).sum(0)
    else:
        num, den = (x * y).sum(0), (x * x).sum(0)
    return -np.log(np.clip(num / den, 1e-12, None)) / DT


def jackknife(psi, m=4, intercept=True):
    """(m * a_full - mean(a_blocks)) / (m - 1); removes the O(1/T) term."""
    n = psi.shape[0] - 1
    b = n // m
    blocks = [ols_alpha(psi[i * b:(i + 1) * b + 1], intercept) for i in range(m)]
    return (m * ols_alpha(psi, intercept) - np.mean(blocks, axis=0)) / (m - 1)


SCALES = [(0.68, 0.15, 11.0, 0.40, 50.5, 26.5,
           "Table 1 scale   alpha 0.68  lamb 11"),
          (15.0, 0.105, 76.99, 0.575, 78.59, 60.68,
           "window-study    alpha 15    lamb 77")]


def main(reps=4000):
    for alpha, sigma, lam, p, e1, e2, tag in SCALES:
        print("\n%s   (one-step rho = %.6f, half-life %.1f trading days)"
              % (tag, np.exp(-alpha * DT), np.log(2) / alpha * 252))
        print("  %6s %8s %10s %10s %9s %10s %9s"
              % ("window", "alpha*T", "no-jump", "with-jump", "bias",
                 "4/T_yr", "jackknife"))
        for n in (252, 504, 1008, 2016):
            rng = np.random.default_rng(12345)
            a0 = ols_alpha(paths(alpha, sigma, lam, p, e1, e2, n, reps,
                                 rng, jumps=False)).mean()
            rng = np.random.default_rng(12345)
            pk = paths(alpha, sigma, lam, p, e1, e2, n, reps, rng, jumps=True)
            a1, aj = ols_alpha(pk).mean(), jackknife(pk).mean()
            print("  %5dd %8.2f %10.3f %10.3f %8.1f%% %10.3f %9.3f"
                  % (n, alpha * n * DT, a0, a1, 100 * (a1 - alpha) / alpha,
                     4.0 / (n * DT), aj))

    print("\nwhat alpha*T buys, and what it costs to get there")
    print("  %8s %10s %9s %9s" % ("alpha*T", "rel bias", "rel sd", "bias/sd"))
    for aT in (0.68, 5.44, 15.0, 32.0, 120.0):
        print("  %8.2f %9.0f%% %8.0f%% %9.2f"
              % (aT, 100 * 4 / aT, 100 * np.sqrt(2 / aT),
                 2 * np.sqrt(2) / np.sqrt(aT)))
    print("\n  bias/sd < 0.5 needs alpha*T > 32:")
    for hl in (12, 63, 252):
        a = np.log(2) / (hl / 252.0)
        print("    half-life %3d trading days -> alpha %6.2f -> %5.1f years of data"
              % (hl, a, 32.0 / a))


if __name__ == "__main__":
    main()
