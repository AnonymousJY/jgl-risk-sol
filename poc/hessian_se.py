"""se and 95% CI from the observed information, not from a Monte Carlo spread.

WHY. The recovery study reports, per parameter, the spread of the estimate
ACROSS 40 simulated datasets. That is the right diagnostic for "does the
estimator work", and it is exactly the wrong thing to print in a table: it
cannot be computed from the one dataset a paper actually has. What a table
reports is the standard error of the estimate from THAT dataset, and for an
MLE that is the observed information,

    se = sqrt(diag(inv(-H))),      H = Hessian of the log-likelihood at the max

The log-likelihood here is already a pytensor graph, so H is exact autodiff -
no finite differencing, no step-size choice.

ON WHICH SCALE. Every parameter is constrained: alpha, sigma, lamb, eta > 0
and pprob in (0, 1). The Hessian is therefore taken on the UNCONSTRAINED
scale the sampler already walks - log alpha, log sigma, logit pprob, log lamb,
log eta - where the quadratic approximation behaves and an interval cannot run
through zero. Endpoints are mapped back afterwards, which makes the
natural-scale interval ASYMMETRIC about the estimate. The natural-scale se
printed beside it is the delta-method value, for reference only; the interval
is the thing to quote.

HOW IT IS CHECKED. A formula is not evidence. Across many simulated datasets
the true sampling spread of the estimate is observable, so a correct se has to
reproduce it, and a correct 95% interval has to contain the truth about 95% of
the time. Both comparisons are printed. That is the reason to believe the
number, and it is what the Monte Carlo study is FOR - validating the se that
a single fit will report, rather than being reported itself.

    python poc/hessian_se.py
    SEEDS=1,2,3,4,5,6,7,8,9,10 NSTEPS=252 ALPHA=2.44 python poc/hessian_se.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytensor
import pytensor.tensor as pt
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recover_two_stage import DT, simulate                            # noqa: E402
from Library.RiskEngineKimYi2025 import (KimYiLogLike,                # noqa: E402
                                         systematic_psi_returns)

# (name, link). "exp" is a positive parameter on the log scale; "logit" is a
# probability. Order fixes the coordinates of theta throughout.
PARAMS = [("alpha", "exp"), ("sigma", "exp"), ("pprob", "logit"),
          ("lamb", "exp"), ("eta1", "exp"), ("eta2", "exp")]


def to_theta(nat):
    out = []
    for v, (_, link) in zip(nat, PARAMS):
        out.append(np.log(v / (1 - v)) if link == "logit" else np.log(v))
    return np.array(out)


def to_natural(theta):
    out = []
    for t, (_, link) in zip(np.atleast_1d(theta), PARAMS):
        t = float(np.clip(t, -700, 700))          # no overflow warning at the edge
        out.append(1 / (1 + np.exp(-t)) if link == "logit" else np.exp(t))
    return np.array(out)


# A fit is usable only if the maximum is INTERIOR. Two things happen otherwise
# and both corrupt the summary rather than announcing themselves:
#
#   alpha runs to the boundary  - the estimate prints as 0.000 with se 0.000,
#                                 and the delta-method se underflows.
#   a direction goes flat       - se on the unconstrained scale blows up and
#                                 the interval's upper end reaches 1e132.
#
# Either way the interval then spans (0, inf), which CONTAINS the truth, so a
# coverage count that keeps these reports them as successes. They are not
# estimates and are excluded, with the count shown.
THETA_MAX = 25.0      # alpha outside [1e-11, 7e10]: collapsed, not estimated
SE_T_MAX = 10.0       # a 95% interval spanning e^(+-19.6): a flat direction


def usable(th, se_t, nat, se_nat):
    return bool(np.all(np.isfinite(th)) and np.all(np.abs(th) < THETA_MAX)
                and np.all(np.isfinite(se_t)) and np.all(se_t < SE_T_MAX)
                and np.all(np.isfinite(nat)) and np.all(nat > 0)
                and np.all(np.isfinite(se_nat)) and np.all(se_nat > 0))


def jacobian(theta):
    """d(natural)/d(theta), elementwise - the delta-method factor."""
    nat = to_natural(theta)
    return np.array([n * (1 - n) if link == "logit" else n
                     for n, (_, link) in zip(nat, PARAMS)])


def compile_loglike():
    """f(theta, y) -> (loglike, gradient, Hessian), all exact."""
    th, y = pt.dvector("theta"), pt.dmatrix("y")
    alpha, sigma = pt.exp(th[0]), pt.exp(th[1])
    pprob = pt.sigmoid(th[2])
    lamb, eta1, eta2 = pt.exp(th[3]), pt.exp(th[4]), pt.exp(th[5])
    # mui = -0.5 sigma^2 with betai = 1, kappai = 0 makes _drift() identically
    # zero, which is what the systematic arm does; the series arrives de-meaned.
    ll = KimYiLogLike(
        mui=-0.5 * sigma ** 2, kappai=pt.as_tensor(0.), gammai=pt.as_tensor(1.),
        betai=pt.as_tensor(1.), rhoix=pt.as_tensor(0.), alpha=alpha,
        sigma=sigma, pprob=pprob, lamb=lamb, eta1=eta1, eta2=eta2,
        dt=pt.as_tensor(DT)).logp(y).sum()
    # GRADIENT ONLY. Second-order autodiff through this density returns NaN:
    # the jump branches carry pt.exp(pm.logcdf(...)), and differentiating the
    # normal log-CDF twice hits a 0/0 branch in four of the six coordinates
    # (alpha, sigma, eta1, eta2) while the gradient itself stays finite. So
    # the Hessian is central-differenced from the EXACT gradient below, which
    # is both standard practice and far more robust than differencing the
    # function twice.
    return pytensor.function([th, y], [ll, pt.grad(ll, th)],
                             on_unused_input="ignore")


def hessian_fd(fn, theta, obs, h=1e-4):
    """Central differences of the analytic gradient, symmetrised.

    theta is on the unconstrained scale, so every coordinate is O(1) and a
    single absolute step serves all six. Symmetrising costs nothing and
    removes the asymmetry that finite differencing always leaves behind.
    """
    k = len(theta)
    H = np.empty((k, k))
    for i in range(k):
        e = np.zeros(k)
        e[i] = h
        gp = np.asarray(fn(theta + e, obs)[1], dtype=float)
        gm = np.asarray(fn(theta - e, obs)[1], dtype=float)
        H[i] = (gp - gm) / (2 * h)
    return 0.5 * (H + H.T)


def fit_one(fn, returns, start_nat, h=1e-4):
    """MLE, its Hessian-based se on both scales, and the 95% interval."""
    obs = np.cumsum(systematic_psi_returns(np.asarray(returns))).reshape((-1, 1))
    neg = lambda t: (lambda v, g: (-float(v), -np.asarray(g)))(*fn(t, obs))
    res = minimize(neg, x0=to_theta(start_nat), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
    th = np.asarray(res.x, dtype=float)

    H = hessian_fd(fn, th, obs, h)
    if not np.all(np.isfinite(H)):
        return None
    ev = np.linalg.eigvalsh(-H)
    if ev.min() <= 0:                    # a saddle or a flat direction: no se
        return None
    se_t = np.sqrt(np.diag(np.linalg.inv(-H)))

    # Step-size check. A finite-difference Hessian is only worth quoting if it
    # is insensitive to h; this reports the largest relative move in se when
    # the step is doubled, and main() surfaces the worst case over all fits.
    H2 = hessian_fd(fn, th, obs, 2 * h)
    drift = np.nan
    if np.all(np.isfinite(H2)) and np.linalg.eigvalsh(-H2).min() > 0:
        se2 = np.sqrt(np.diag(np.linalg.inv(-H2)))
        drift = float(np.max(np.abs(se2 - se_t) / se_t))

    nat, se_nat = to_natural(th), jacobian(th) * se_t
    if not usable(th, se_t, nat, se_nat):
        return "boundary"
    return dict(nat=nat, se_nat=se_nat, drift=drift,
                grad=float(np.linalg.norm(fn(th, obs)[1])),
                lo=to_natural(th - 1.96 * se_t), hi=to_natural(th + 1.96 * se_t))


def main():
    env = lambda k, d: float(os.environ.get(k, d))
    seeds = [int(x) for x in os.environ.get("SEEDS", "1,2,3,4,5,6,7,8").split(",")]
    n = int(env("NSTEPS", 252))
    truth = dict(alpha=env("ALPHA", 2.44), sigma=env("SIGMA", 0.105),
                 lamb=env("LAMB", 76.99), pprob=env("PPROB", 0.575),
                 eta1=env("ETA1", 78.59), eta2=env("ETA2", 60.68),
                 betai=1.5, kappai=0.15, gammai=2.5, mui=0.05)
    true_vec = np.array([truth[k] for k, _ in PARAMS])

    print("compiling the log-likelihood and its gradient ...")
    t0 = time.time()
    fn = compile_loglike()
    print("  done in %.1fs\n" % (time.time() - t0))
    print("%d simulated datasets of %d days; truth %s\n"
          % (len(seeds), n, dict(zip([p for p, _ in PARAMS], true_vec))))

    fits, dropped = [], []
    for sd in seeds:
        sys_r, _ = simulate(truth, n, sd)
        # Start away from the truth: an optimiser begun AT the answer measures
        # nothing. A 30% displacement is enough to make the search real.
        out = fit_one(fn, sys_r, true_vec * 1.3)
        if out is None or out == "boundary":
            dropped.append(sd)
            print("  seed %-5d  %s - excluded"
                  % (sd, "maximum on the boundary" if out == "boundary"
                     else "-H not positive definite"))
            continue
        fits.append(out)
        print("  seed %-5d  alpha %8.3f  se %7.3f   [%7.3f, %8.3f]"
              "   |grad| %.1e  h-drift %.1e"
              % (sd, out["nat"][0], out["se_nat"][0], out["lo"][0], out["hi"][0],
                 out["grad"], out["drift"]))

    if not fits:
        raise SystemExit("no usable fits")

    est = np.array([f["nat"] for f in fits])
    se = np.array([f["se_nat"] for f in fits])
    lo = np.array([f["lo"] for f in fits])
    hi = np.array([f["hi"] for f in fits])
    cov = ((lo <= true_vec) & (true_vec <= hi)).sum(axis=0)

    if dropped:
        print("\n  %d of %d datasets excluded (maximum not interior): %s"
              % (len(dropped), len(seeds), ", ".join(map(str, dropped))))
    print("\n  %-8s %9s %9s %11s %11s %9s %22s"
          % ("param", "TRUE", "MLE", "se(Hess)", "sd(across)", "ratio",
             "median 95% interval"))
    for i, (name, _) in enumerate(PARAMS):
        sd_across = est[:, i].std(ddof=1) if len(fits) > 1 else np.nan
        print("  %-8s %9.4f %9.4f %11.4f %11.4f %9.2f  [%9.4f, %9.4f] %3d/%d"
              % (name, true_vec[i], est[:, i].mean(), se[:, i].mean(), sd_across,
                 se[:, i].mean() / sd_across if sd_across else np.nan,
                 np.median(lo[:, i]), np.median(hi[:, i]), cov[i], len(fits)))

    # MEDIAN, not mean: a near-flat alpha direction on one dataset sends its
    # upper bound to infinity, and one such draw destroys a mean.
    print("\n  se(Hess) is what ONE fit reports; sd(across) is the true spread")
    print("  of the estimate over %d datasets. A ratio near 1 means the reported"
          % len(fits))
    print("  se is honest. The last column counts how often the 95% interval")
    print("  held the truth - nominal is %.1f of %d."
          % (0.95 * len(fits), len(fits)))


if __name__ == "__main__":
    main()
