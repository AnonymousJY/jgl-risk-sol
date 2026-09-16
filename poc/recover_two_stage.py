"""Does the P-MLE recover parameters it generated itself?

Simulates from the model, runs the PRODUCTION functions
(pmle_kimyirisk_systematic then pmle_kimyirisk_joint) and reports bias and
credible-interval coverage per parameter. This is the harness the September
2026 prior investigation was built on; every number in the commit messages for
that work came from here.

WHAT IT ESTABLISHED.

  1. The joint likelihood recovers the idiosyncratic block. Over 9 seeds:
     beta_i +0.1%, kappa_i +1.1%, gamma_i -0.6%, coverage 89-100%. Holds
     whether stage 2 is given the FITTED or the TRUE systematic parameters, so
     the two-stage split does not leak.

  2. mu_i does NOT recover: 22% coverage, intervals about 4x too narrow.
     Still open.

  3. The systematic block failed on tight priors and recovers on wide ones.
     At the paper's own FULL_SAMPLE calibration, 504 days:

                      lamb   eta1   eta2   sigma   pprob   alpha
         TRUTH        77.0   78.6   60.7  0.1050   0.575    15.0
         old priors   35.6   51.7   30.9  0.1239   0.599    11.5
         new priors   89.9   78.4   64.6  0.1034   0.554    13.2

     Coverage went 0/0/0/1/3 to 3/3/3/3/2, and alpha from 3/5 to 5/5.

  4. It was the PRIORS, not the <=1-jump truncation in Appendix B. Profiling
     lambda with no prior at all peaks at 72.0 against a true 77.0, so the
     truncation is worth about 6% and the priors were worth the other 48%.

DESIGN NOTES THAT COST TIME TO LEARN.

  - DISPLACE THE TRUTH FROM THE PRIOR MEANS. The first version of this study
    set the true lambda, eta1, eta2 equal to their prior means and reported
    100% coverage. A likelihood carrying no information would have scored the
    same. Any "recovery" at a truth sitting on its prior mean is worthless.
  - The DGP uses the same EXACT transition exp(-alpha*dt) the likelihood
    inverts, so a discrepancy is estimation and not discretisation.
  - The index and the name share BOTH the systematic Brownian Z and the jump
    dJ. KimYiRiskEngine.random() draws Z per asset and so cannot be used here.
  - alpha's precision is se ~ sqrt(2*alpha/T_years), a function of CALENDAR
    SPAN only. At alpha=20 over 2 years that is +-23% and no prior, sampler
    or reparameterisation improves it. Judge alpha against that, not against
    zero.

Run:  JGL_DRAWS=800 JGL_CHAINS=4 python poc/recover_two_stage.py
      SEEDS=7,99,2024 NSTEPS=504 ALPHA=15 LAMB=76.99 python poc/recover_two_stage.py
      PLOTS=out/recovery python poc/recover_two_stage.py      # ArviZ posteriors

PLOTS=<dir> writes, per seed, an ArviZ posterior panel with the TRUE value
drawn on each parameter, plus a trace panel and an az.summary CSV carrying
r_hat and ess. Read the posterior panel first: a bias shows as the mass
sitting to one side of the reference line, while the mu_i pathology shows as a
NARROW posterior that misses it - a different failure needing a different fix.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Library.RiskEngineKimYi2025 import (          # noqa: E402
    pmle_kimyirisk_joint, pmle_kimyirisk_systematic, prior_moments,
    SYSTEMATIC_PRIORS,
)

DT = 1.0 / 252.0

# posterior variable name -> key in the truth dict. The CONSTRAINED variables
# are the ones to plot: alpha_rv is alpha itself, and betai_rv/kappai_rv/
# gamma_rv are on the natural scale, while "betai"/"kappai"/"gammai" are their
# logs and would put the reference line in the wrong place.
SYS_VARS = [("alpha_rv", "alpha"), ("sigma", "sigma"), ("pprob_rv", "pprob"),
            ("lamb", "lamb"), ("eta1", "eta1"), ("eta2", "eta2")]
JOINT_VARS = [("mui", "mui"), ("betai_rv", "betai"),
              ("kappai_rv", "kappai"), ("gamma_rv", "gammai")]


def save_plots(idata, varmap, truth, outdir, tag):
    """ArviZ posterior + trace panels with the true value marked, and a CSV."""
    import arviz as az
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = [v for v, _ in varmap if v in idata.posterior]

    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0))
    axes = np.atleast_1d(axes)
    for ax, (v, key) in zip(axes, [p for p in varmap if p[0] in names]):
        # one variable at a time: ref_val's dict form has changed across ArviZ
        # versions, a scalar has not.
        az.plot_posterior(idata, var_names=[v], ref_val=float(truth[key]),
                          ax=ax, hdi_prob=0.95)
        ax.set_title("%s  (true %.4g)" % (v, truth[key]), fontsize=10)
    fig.suptitle("%s - posterior vs truth" % tag, fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / ("%s_posterior.png" % tag), dpi=130)
    plt.close(fig)

    # One row per parameter, density left and trace right, CHAINS OVERLAID -
    # four densities that sit on top of each other is the convergence check,
    # and compact=True would collapse them into one. lines= puts the true
    # value on the density as a vertical reference.
    lines = [(v, {}, [float(truth[key])]) for v, key in varmap if v in names]
    axes = az.plot_trace(idata, var_names=names, compact=False, lines=lines,
                         figsize=(11, 2.0 * len(names)))
    fig = axes.ravel()[0].figure
    for ax, (v, key) in zip(axes[:, 0], [p for p in varmap if p[0] in names]):
        ax.set_title("%s   (true %.4g)" % (v, truth[key]), fontsize=10)
    fig.suptitle("%s - posterior density (truth marked) and traces" % tag,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(outdir / ("%s_trace.png" % tag), dpi=120)
    plt.close(fig)

    summ = az.summary(idata, var_names=names, hdi_prob=0.95)
    summ["true"] = [truth[k] for v, k in varmap if v in names]
    summ.to_csv(outdir / ("%s_summary.csv" % tag))
    return summ


def simulate(truth, n_steps, seed):
    """(sys_returns, idi_returns) as KimYiLogLike reads them.

    logp forms Psi_k = cumsum(r)[k] - k*m*dt and quasi-differences with
    exp(-alpha*dt), so the recursion below matches it exactly. r_0 = Psi_0.
    """
    a, s = truth["alpha"], truth["sigma"]
    lam, p, e1, e2 = truth["lamb"], truth["pprob"], truth["eta1"], truth["eta2"]
    b, k, g, mu = truth["betai"], truth["kappai"], truth["gammai"], truth["mui"]
    rng = np.random.default_rng(seed)

    z, w = rng.normal(size=n_steps), rng.normal(size=n_steps)
    n_jumps = rng.poisson(lam * DT, n_steps)          # TRUE Poisson, not <=1
    tot = int(n_jumps.sum())
    y = np.where(rng.random(tot) < p, rng.exponential(1 / e1, tot),
                 -rng.exponential(1 / e2, tot))
    dj = np.zeros(n_steps)
    np.add.at(dj, np.repeat(np.arange(n_steps), n_jumps), y)

    psi_m, psi_i = np.zeros(n_steps), np.zeros(n_steps)
    decay = np.exp(-a * DT)
    for t in range(1, n_steps):
        psi_m[t] = decay * psi_m[t - 1] + s * np.sqrt(DT) * z[t] + dj[t]
        psi_i[t] = (decay * psi_i[t - 1] + s * b * np.sqrt(DT) * z[t]
                    + k * np.sqrt(DT) * w[t] + g * dj[t])

    mi = mu + 0.5 * (s * b) ** 2                      # the drift stage 2 removes
    sys_r = np.diff(psi_m, prepend=0.0)
    idi_r = np.diff(psi_i, prepend=0.0) + mi * DT
    idi_r[0] = psi_i[0]
    return sys_r, idi_r


def _row(name, true_v, ests, los, his):
    e = np.asarray(ests)
    cov = int(np.sum((np.asarray(los) <= true_v) & (true_v <= np.asarray(his))))
    print("  %-9s %11.4f %11.4f %8.1f%% %6d/%d"
          % (name, true_v, e.mean(), 100 * (e.mean() - true_v) / abs(true_v),
             cov, len(e)))


def main():
    env = lambda k, d: float(os.environ.get(k, d))                   # noqa: E731
    truth = dict(alpha=env("ALPHA", 15.0), sigma=env("SIGMA", 0.105),
                 lamb=env("LAMB", 76.99), pprob=env("PPROB", 0.575),
                 eta1=env("ETA1", 78.59), eta2=env("ETA2", 60.68),
                 betai=env("BETAI", 1.50), kappai=env("KAPPAI", 0.15),
                 gammai=env("GAMMAI", 2.50), mui=env("MUI", 0.05))
    seeds = [int(x) for x in os.environ.get("SEEDS", "7,99,2024,555,31337").split(",")]
    n_steps = int(os.environ.get("NSTEPS", 504))
    draws = int(os.environ.get("NMC", 600))
    plots = os.environ.get("PLOTS", "").strip()

    print("truth:", {k: round(v, 4) for k, v in truth.items()})
    print("prior means:", {k: round(prior_moments(v)[0], 3)
                           for k, v in SYSTEMATIC_PRIORS.items()})
    print("\nCHECK THE TWO LINES ABOVE AGAINST EACH OTHER. A parameter whose truth")
    print("sits on its prior mean is not being tested.\n")
    print("lam*dt = %.4f  ->  P(2+ jumps/day) = %.2f%%   alpha se ~ sqrt(2a/T) = %.2f"
          % (truth["lamb"] * DT,
             100 * (1 - np.exp(-truth["lamb"] * DT) * (1 + truth["lamb"] * DT)),
             np.sqrt(2 * truth["alpha"] / (n_steps * DT))))
    if plots:
        print("writing ArviZ panels to %s/" % plots)

    sys_res, joint_res = [], {"fitted": [], "oracle": []}
    for sd in seeds:
        sys_r, idi_r = simulate(truth, n_steps, sd)
        r1 = pmle_kimyirisk_systematic(sys_returns=sys_r, delta_t=np.array(DT),
                                       seed_number=np.uint64(sd), n_mc_paths=draws,
                                       return_idata=bool(plots))
        if plots:
            r1, id1 = r1
            save_plots(id1, SYS_VARS, truth, plots, "systematic_seed%d" % sd)
        sys_res.append(r1)
        fitted = {k: float(v.dMEAN) for k, v in r1.items()}
        oracle = {"dALPHA": truth["alpha"], "dSIGMA": truth["sigma"],
                  "dPPROB": truth["pprob"], "dLAMB": truth["lamb"],
                  "dETA1": truth["eta1"], "dETA2": truth["eta2"]}
        for tag, ps in (("fitted", fitted), ("oracle", oracle)):
            out = pmle_kimyirisk_joint(
                idi_returns=idi_r, sys_returns=sys_r, params_sys=ps,
                delta_t=np.array(DT), seed_number=np.uint64(sd), n_mc_paths=draws,
                return_idata=bool(plots))
            if plots:
                out, id2 = out
                save_plots(id2, JOINT_VARS, truth, plots,
                           "joint_%s_seed%d" % (tag, sd))
            joint_res[tag].append(out)

    hdr = "  %-9s %11s %11s %9s %8s" % ("param", "TRUE", "estimate", "bias", "cover")
    print("\nSTAGE 1  systematic\n" + hdr)
    for key, t in (("dALPHA", "alpha"), ("dSIGMA", "sigma"), ("dPPROB", "pprob"),
                   ("dLAMB", "lamb"), ("dETA1", "eta1"), ("dETA2", "eta2")):
        _row(key, truth[t], [float(r[key].dMEAN) for r in sys_res],
             [float(r[key].dCI_LOWER) for r in sys_res],
             [float(r[key].dCI_UPPER) for r in sys_res])
    for tag in ("fitted", "oracle"):
        print("\nSTAGE 2  joint, %s systematic parameters\n" % tag + hdr)
        for key, t in (("dBETAI", "betai"), ("dKAPPAI", "kappai"),
                       ("dGAMMAI", "gammai"), ("dMUI", "mui")):
            R = joint_res[tag]
            _row(key, truth[t], [float(r[key].dMEAN) for r in R],
                 [float(r[key].dCI_LOWER) for r in R],
                 [float(r[key].dCI_UPPER) for r in R])


if __name__ == "__main__":
    main()
