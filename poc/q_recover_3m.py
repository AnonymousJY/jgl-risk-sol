"""Does a 3M skew curve calibrate back to the true Q jump parameters?

Simulate an implied volatility surface at ONE expiry from known (p, lamb,
eta1, eta2), then calibrate those four back. sigma is measure-invariant under
Girsanov so it is held at its P value and never fitted.

The answer is yes - the objective AT the truth is exactly 0, so the truth is
the global optimum in every case with jump content in it. What decides whether
you SEE that is the optimizer, and it fails in two separate ways that this
script makes visible side by side.

  SCALING. Production searches (p, lamb, eta1, eta2) on their natural scales,
  which span 0.2 to 22. SLSQP takes one finite-difference step and one
  tolerance for every coordinate, so it halts early - 35% to 105% away at some
  tenors. Dividing through by x0 fixes it and leaks nothing about the truth.

  STARTING POINT. Even rescaled, a poor start lands in a shallow secondary
  basin. On the equity-smirk shape the default start stops at objective
  9.3e-09 with p 0.063 and eta1 29.6, against a truth of 0.150 and 40.0 that
  scores exactly 0. The codebase already knows this failure mode - it is why
  IDIOSYNCRATIC_X0_HINTS exists - and a short multi-start removes the need for
  anyone to know which date needs a hint.

The remaining exception is not the optimizer. A FLAT smile carries almost no
information about jump structure, so it fits to a few hundredths of a vol
point from parameters nowhere near the truth. Fitting well and identifying are
different things, and the objective-at-truth column is what tells them apart.

    python poc/q_recover_3m.py
    TENOR=0.5 NSTRIKES=21 python poc/q_recover_3m.py
    SHAPES=2 python poc/q_recover_3m.py        # one shape only, to go fast
    NSTARTS=1 python poc/q_recover_3m.py       # production's single start
    PLOTS=out python poc/q_recover_3m.py       # also write the figure

Budget about a minute per start per shape: the default 5 starts x 4 shapes is
roughly 15 minutes.
"""
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Library.SkewCalibrationKimYi2025 import KimYiSkewCalibrationSystematic  # noqa: E402

# Production's SLSQP start, used both as the first start and as the SCALE the
# search is divided through by - a quantity the calibrator already has.
X0 = np.array([0.20, 5.00, 22.00, 7.00])
BOUNDS = [(0., 1.), (1e-4, None), (1.5, None), (0.5, None)]
STARTS = [X0,
          np.array([0.15, 10.0, 40.0, 10.0]),   # equity smirk
          np.array([0.50, 10.0, 15.0, 15.0]),   # symmetric, larger jumps
          np.array([0.80, 8.00, 12.0, 30.0]),   # call-leaning
          np.array([0.35, 2.00, 30.0, 30.0])]   # sparse, small jumps

SHAPES = [("1 both wings up",   dict(pprob=0.50, lamb=10.0, eta1=12.0, eta2=12.0)),
          ("2 put up call down", dict(pprob=0.15, lamb=10.0, eta1=40.0, eta2=10.0)),
          ("3 put down call up", dict(pprob=0.85, lamb=10.0, eta1=10.0, eta2=40.0)),
          ("4 both flat",        dict(pprob=0.50, lamb=0.50, eta1=40.0, eta2=40.0))]


def fitter(T, n, phii, s0, r, q, mkt):
    width = 3.0 * 0.45 * np.sqrt(T)
    k = s0 * np.exp(np.linspace(-width, width, n))
    col = lambda v: np.full(n, float(v))
    return KimYiSkewCalibrationSystematic(
        mkt_imp_vol=mkt, und_price=col(s0), und_strike=k, risk_free_rate=col(r),
        dividend_yield=col(q), time_to_expiry=col(T),
        is_call_option=k > s0 * np.exp((r - q) * T),
        option_weights=np.ones((n, 1)) / n, sigma=np.array(phii))


def fit(f, x0):
    """One rescaled SLSQP run. Returns (parameters, objective)."""
    sc = X0
    bnds = [(lo / s, None if hi is None else hi / s)
            for (lo, hi), s in zip(BOUNDS, sc)]
    r = minimize(lambda u: f.target(u * sc), x0=x0 / sc, method="SLSQP",
                 bounds=bnds, tol=1e-12, options={"maxiter": 200})
    return np.asarray(r.x, dtype=float) * sc, float(r.fun)


# Categorical slots 1-3 of the validated palette, which clear the all-pairs
# colour-vision gates. Target is the thick line underneath; the two fits sit
# on top so a fit that lands on the truth reads as a coloured core in a blue
# halo rather than vanishing.
C_TRUE, C_ONE, C_BEST = "#2a78d6", "#eb6834", "#1baf7a"


def save_plots(out_dir, T, panels):
    """One figure, one panel per shape: the target surface and both fits."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("\n  matplotlib not installed - skipping plots")
        return
    os.makedirs(out_dir, exist_ok=True)
    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.3 * rows),
                             squeeze=False)
    for ax, pan in zip(axes.ravel(), panels):
        m = pan["m"]
        ax.plot(m, pan["tgt"], color=C_TRUE, lw=3.4, label="target (true parameters)",
                solid_capstyle="round", zorder=1)
        ax.plot(m, pan["one"], color=C_ONE, lw=2.0, label="one start (production)",
                zorder=3)
        ax.plot(m, pan["best"], color=C_BEST, lw=2.0, ls=(0, (5, 3)),
                label="best of %d start%s" % (pan["nstart"],
                                              "" if pan["nstart"] == 1 else "s"),
                zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([0.6, 0.8, 1.0, 1.3, 1.7])
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: "%.2f" % v))
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(m[0], m[-1])
        # open a band at the bottom so the parameter box never sits on a curve
        lo = min(pan["tgt"].min(), pan["one"].min(), pan["best"].min())
        hi = max(pan["tgt"].max(), pan["one"].max(), pan["best"].max())
        pad = max(hi - lo, 1e-6)
        ax.set_ylim(lo - 0.42 * pad, hi + 0.08 * pad)
        ax.set_title(pan["label"], fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel("moneyness K/S", fontsize=9)
        ax.set_ylabel("implied volatility (%)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.25, lw=.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fmt = lambda v: "p %.3f  lam %.2f  e1 %.2f  e2 %.2f" % tuple(v)
        ax.text(.02, .035,
                "true   " + fmt(pan["true"]) + "\n" +
                "one    " + fmt(pan["xone"]) + "   (%.3f vp)\n" % pan["rone"] +
                "best   " + fmt(pan["xbest"]) + "   (%.3f vp)" % pan["rbest"],
                transform=ax.transAxes, fontsize=7.2, family="monospace",
                va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                       ec="#dcdbd5", lw=.8, alpha=.92))
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Q-measure recovery at %.2f yr - simulated data" % T,
                 fontsize=13, fontweight="bold", x=.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "q_recover_%dd.png" % round(T * 365))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("\n  plot written: %s" % path)


def main():
    env = lambda k, d: float(os.environ.get(k, d))
    T, n = env("TENOR", 0.25), int(env("NSTRIKES", 15))
    phii, s0, r, q = env("PHII", 0.30), env("SPOT", 100.), env("RATE", 0.04), env("DIVY", 0.015)
    # Each start costs roughly a minute, so make the ladder tunable: NSTARTS=1
    # is production's single default start and shows the failure this script
    # exists to document.
    starts = STARTS[:max(1, int(env("NSTARTS", len(STARTS))))]
    pick = os.environ.get("SHAPES")
    shapes = ([s for s in SHAPES if s[0][0] in pick.split(",")] if pick else SHAPES)

    print("tenor %.3f yr, %d strikes (+-3 sd), phi_i %.2f, %d starts\n"
          % (T, n, phii, len(starts)))
    print("  %-20s %7s %7s %7s %7s %11s %9s"
          % ("", "p", "lamb", "eta1", "eta2", "objective", "misfit"))
    panels = []

    for label, jump in shapes:
        true = np.array([jump["pprob"], jump["lamb"], jump["eta1"], jump["eta2"]])
        f0 = fitter(T, n, phii, s0, r, q, np.zeros(n))
        tgt = np.asarray(f0.model_vol(true)).reshape(-1)
        if not np.all(np.isfinite(tgt)):
            print("  %-20s  IV inversion failed at the truth" % label)
            continue
        f = fitter(T, n, phii, s0, r, q, tgt)

        print("  %s" % label)
        print("  %-20s %7.3f %7.2f %7.2f %7.2f %11.3e %9s"
              % ("true", true[0], true[1], true[2], true[3],
                 float(f.target(true)), ""))

        single, obj1 = fit(f, X0)
        best, objb = single, obj1
        for x0 in starts[1:]:
            cand, o = fit(f, x0)
            if o < objb:
                best, objb = cand, o
        rms = lambda x: 100 * np.sqrt(np.mean(
            (tgt - np.asarray(f.model_vol(x)).reshape(-1)) ** 2))
        for tag, x, o in (("one start (production)", single, obj1),
                          ("best of %d starts" % len(starts), best, objb)):
            print("  %-20s %7.3f %7.2f %7.2f %7.2f %11.3e %8.3f vp"
                  % (tag, x[0], x[1], x[2], x[3], o, rms(x)))
        width = 3.0 * 0.45 * np.sqrt(T)
        order = np.argsort(np.concatenate(
            [np.flatnonzero(~f.is_call_option.reshape(-1)),
             np.flatnonzero(f.is_call_option.reshape(-1))]))
        curve = lambda x: np.asarray(f.model_vol(x)).reshape(-1)[order] * 100
        panels.append(dict(
            label=label, m=np.exp(np.linspace(-width, width, n)),
            tgt=tgt[order] * 100, one=curve(single), best=curve(best),
            true=true, xone=single, xbest=best, rone=rms(single),
            rbest=rms(best), nstart=len(starts)))

        err = 100 * np.max(np.abs(best - true) / true)
        print("  %-20s worst parameter error %.1f%%%s\n"
              % ("", err,
                 "   <- fits but does not identify" if err > 20 and objb < 1e-6 else ""))

    out = os.environ.get("PLOTS")
    if out and panels:
        save_plots(out, T, panels)


if __name__ == "__main__":
    main()
