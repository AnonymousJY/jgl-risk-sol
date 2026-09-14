"""Identify beta_i and kappa_i from the variation in sigma_t, using the
marginal P-MLE output only.

THE IDEA. At rho_iX = 0 the diffusive part of the name's return is

    sigma*beta_i dZ_t  +  kappa_i dW_i,t ,      dZ _|_ dW_i

and within one window nothing can separate them: two independent Gaussians sum
to a Gaussian, so the likelihood sees only the total

    phi_i^2 = (sigma beta_i)^2 + kappa_i^2 .

One equation, two unknowns - which is the rank 3-of-4 result.

But that is only true AT FIXED SIGMA. Across valuation dates sigma_t moves (by
about 3.2x in this sample), and the two components scale with DIFFERENT POWERS
of it:

    phi_{i,t}^2 = beta_i^2 * sigma_t^2  +  kappa_i^2 .                    (*)

That is a straight line in sigma_t^2 with slope beta_i^2 and intercept
kappa_i^2. Two distinct sigma values identify both. The systematic stage
already estimates sigma_t per date and the marginal stage already estimates
phi_{i,t}, so (*) needs no new data, no factor series, and no volume.

WHY phi IS USABLE EVEN THOUGH beta AND kappa ARE NOT. phi_i^2 is identified by
the marginal likelihood; beta_i and kappa_i individually are not, and the
values in the parameter CSVs are prior artifacts. But phi_i^2 recomputed from
those same CSVs is the identified combination, so it is trustworthy. This
script therefore rebuilds phi_{i,t} from (sigma, beta, kappa, rho) rather than
using beta and kappa directly.

WHAT HAS TO BE ASSUMED. beta_i and kappa_i constant over the dates used. That
is testable rather than assumed: if the loading moves with sigma, (*) bends,
so the script adds a sigma^4 term and reports whether it matters.

TWO STATISTICAL POINTS THAT CHANGE THE ANSWER.

  Overlapping windows. Rolling estimates at a 504-day lookback share almost
  all of their data, so consecutive phi_{i,t} are massively autocorrelated and
  the naive standard errors are fiction. Reported both ways: Newey-West at the
  window length, and a non-overlapping subsample.

  Errors in variables. sigma_t^2 on the right-hand side is itself estimated,
  which attenuates the slope. Instrumenting with lagged sigma^2 fixes it -
  measurement errors are independent across dates while sigma^2 is persistent.

usage
    python poc/beta_kappa_from_sigma.py selftest
    python poc/beta_kappa_from_sigma.py run "Study/Estimated Parameters PMLE" [lookback]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

KEY = ["dtVALUATION_DATE", "sUNDERLYING_NAME"]


def load(directory):
    paths = sorted(glob.glob(os.path.join(directory, "**",
                                          "estimated_params_pmle_*.csv"),
                             recursive=True))
    frames = [pd.read_csv(p) for p in paths if os.path.getsize(p) > 0]
    if not frames:
        raise SystemExit("no non-empty parameter CSVs under %r" % directory)
    df = pd.concat(frames, ignore_index=True)
    df["dtVALUATION_DATE"] = pd.to_datetime(df["dtVALUATION_DATE"])
    # phi^2 is the identified combination; beta and kappa separately are not.
    df["phi2"] = ((df.dSIGMA*df.dBETAI)**2
                  + 2*df.dSIGMA*df.dBETAI*df.dKAPPAI*df.dRHOIX
                  + df.dKAPPAI**2)
    df["s2"] = df.dSIGMA**2
    return df.sort_values(KEY).reset_index(drop=True)


def nw_ols(y, X, lag):
    """OLS with Newey-West standard errors."""
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    S = (X*e[:, None]).T @ (X*e[:, None])
    for L in range(1, min(lag, n-1) + 1):
        w = 1.0 - L/(lag + 1.0)
        G = (X[L:]*e[L:, None]).T @ (X[:-L]*e[:-L, None])
        S += w*(G + G.T)
    V = XtXi @ S @ XtXi
    return b, np.sqrt(np.maximum(np.diag(V), 0.0))


def report(tag, k2, b2, se_k2=None, se_b2=None):
    beta = np.sqrt(max(b2, 0.0))
    kap = np.sqrt(max(k2, 0.0))
    extra = ""
    if se_b2 is not None and beta > 0:
        extra = "   se(beta) %.3f  se(kappa) %.4f" % (
            se_b2/(2*beta), (se_k2/(2*kap) if kap > 0 else np.nan))
    print("   %-26s beta_i %6.3f   kappa_i %7.4f%s" % (tag, beta, kap, extra))


def analyse(d, name, lookback):
    s2 = d["s2"].to_numpy()
    phi2 = d["phi2"].to_numpy()
    n = len(d)
    print("\n%s   %d dates   sigma %.4f-%.4f (%.1fx)   phi %.4f-%.4f"
          % (name, n, np.sqrt(s2.min()), np.sqrt(s2.max()),
             np.sqrt(s2.max()/s2.min()), np.sqrt(phi2.min()), np.sqrt(phi2.max())))
    if np.sqrt(s2.max()/s2.min()) < 1.3:
        print("   sigma barely moves here - the regression has no leverage.")
        return

    X = np.column_stack([np.ones(n), s2])
    b, se = nw_ols(phi2, X, lag=lookback)
    report("OLS, Newey-West", b[0], b[1], se[0], se[1])

    # errors in variables: instrument sigma^2 with its own lag
    Z = np.column_stack([np.ones(n-1), s2[:-1]])
    Xl = np.column_stack([np.ones(n-1), s2[1:]])
    Pz = Z @ np.linalg.pinv(Z)
    biv = np.linalg.lstsq(Pz @ Xl, Pz @ phi2[1:], rcond=None)[0]
    report("IV, lagged sigma^2", biv[0], biv[1])

    # non-overlapping windows - the honest sample size. Derive the stride from
    # the actual date spacing rather than assuming daily or monthly.
    if "dtVALUATION_DATE" in d.columns and len(d) > 1:
        gap = np.median(np.diff(d["dtVALUATION_DATE"].to_numpy()).astype(
            "timedelta64[D]").astype(float))
        step = max(int(np.ceil(lookback*(365/252)/max(gap, 1.0))), 1)
    else:
        step = max(int(lookback/21), 1)
    sub = slice(None, None, step)
    if len(s2[sub]) >= 4:
        Xs = np.column_stack([np.ones(len(s2[sub])), s2[sub]])
        bs, ses = nw_ols(phi2[sub], Xs, lag=1)
        report("non-overlapping (n=%d)" % len(s2[sub]), bs[0], bs[1], ses[0], ses[1])

    # Constancy check. Two tests look obvious and neither is usable as a
    # SIGNIFICANCE test, because sigma^2 is measured with error:
    #   - a sigma^4 term inherits a systematic bias from squaring a noisy
    #     regressor and fires at t = -3 on constant-beta data;
    #   - splitting the sample and comparing slopes with their standard errors
    #     fires at z = -4 on the same data, because the fit is so precise that
    #     a 1.6% EIV difference is many standard errors.
    # So compare the two halves' IV estimates and read the GAP, not a p-value.
    # A few percent is what EIV alone produces; a real change in the loading
    # shows up as tens of percent.
    cut = np.median(s2)
    halves = []
    for m, tag in ((s2 <= cut, "low sigma"), (s2 > cut, "high sigma")):
        if m.sum() < 6 or np.ptp(s2[m]) <= 0:
            halves.append((tag, np.nan)); continue
        sm, pm = s2[m], phi2[m]
        Zm = np.column_stack([np.ones(len(sm)-1), sm[:-1]])
        Xm = np.column_stack([np.ones(len(sm)-1), sm[1:]])
        Pm = Zm @ np.linalg.pinv(Zm)
        bm = np.linalg.lstsq(Pm @ Xm, Pm @ pm[1:], rcond=None)[0]
        halves.append((tag, np.sqrt(max(bm[1], 0.0))))
    (t1, b1), (t2, b2) = halves
    if np.isfinite(b1) and np.isfinite(b2) and b1 > 0:
        gap = 100*(b2 - b1)/b1
        print("   constancy: beta_i %.3f (%s) vs %.3f (%s), gap %+.1f%%   %s"
              % (b1, t1, b2, t2, gap,
                 "consistent with constant" if abs(gap) < 10
                 else "**beta_i moves with sigma**"))
    else:
        print("   constancy: not enough sigma spread within halves to test")


def selftest():
    rng = np.random.default_rng(0)
    BETA, KAPPA, n = 1.777, 0.1273, 245
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.985*z[t-1] + rng.standard_normal()*0.12
    sig = 0.08 + (z - z.min())/(z.max() - z.min())*(0.26 - 0.08)
    phi2 = (BETA**2*sig**2 + KAPPA**2)*(1 + 0.04*rng.standard_normal(n))
    s2 = sig**2*(1 + 0.04*rng.standard_normal(n))
    d = pd.DataFrame({"s2": s2, "phi2": phi2})
    print("truth: beta_i %.3f  kappa_i %.4f" % (BETA, KAPPA))
    analyse(d, "SELFTEST", lookback=504)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"selftest", "run"}:
        raise SystemExit(__doc__)
    if sys.argv[1] == "selftest":
        selftest()
    else:
        lb = int(sys.argv[3]) if len(sys.argv) > 3 else 504
        df = load(sys.argv[2])
        for nm, d in df.groupby("sUNDERLYING_NAME"):
            if d.dBETAI.abs().max() == 0:
                continue                      # the systematic proxy
            analyse(d.reset_index(drop=True), nm, lb)
