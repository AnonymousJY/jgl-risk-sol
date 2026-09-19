"""Does a single name's 3M smile calibrate back to its true gamma_i*?

The systematic study asks whether an index surface identifies the four jump
parameters. This asks the question one level down: given those - a name is
calibrated AFTER the index, with (p*, lamb*, eta1*, eta2*) already fixed -
does the name's own smile identify gamma_i*, the one parameter left?

WHAT gamma_i* DOES. It reaches the pricer as eta1*/gamma_i* and eta2*/gamma_i*,
dividing both decay rates by the same factor. A decay rate is the reciprocal of
a mean jump size, so gamma_i* > 1 is a name that jumps FURTHER than the index
in both directions and gamma_i* < 1 one that jumps less. It is a volume knob,
not a balance knob: it cannot tilt a smile towards one wing, which is why a
name whose skew leans the other way from the index cannot be fitted by
gamma_i* at all, and why the answer here is about magnitude rather than shape.

WHY THIS ONE IS EASY. It is one parameter on an interval, so there is no
starting value to get wrong and no flat direction to stall on: bounded Brent
on [0.1, 10] needs neither. What Brent guarantees is a minimum INSIDE the
interval, not the global one - that follows only where the objective is
unimodal there, which is a property of this surface and not of the algorithm.
So the objective is plotted across the whole interval rather than asserted,
and in every case here it has a single well. The contrast with the systematic
problem, where four parameters share one curved valley floor, is the
dimension, not the model.

WHAT beta_i, kappa_i AND sigma DO HERE. Nothing, individually. The name's
price reaches the pricer through psi_vol, so the three enter only as

    phi_i = sqrt((sigma beta_i)^2 + 2 sigma beta_i kappa_i rho_iX + kappa_i^2)

and any (beta_i, kappa_i, rho_iX) with the same phi_i gives the same surface
to the last bit - five such triples at sigma 0.30 agree to 0.000e+00 vol
points. Given phi_i, sigma itself cancels: the same phi_i at sigma 0.10, 0.30,
0.60 and 1.00 gives the identical smile. That is the flat manifold the
calibrator already knows about; the option surface sees total diffusion, never
its decomposition. So the inputs below are (beta_i, kappa_i, rho_iX) - the
model's own parameters, from the P-measure fit - and phi_i is derived from
them, rather than a collapsed number appearing from nowhere. phi_i is a
diffusion coefficient, measure-invariant under Girsanov, and never fitted.

    python poc/q_gamma_3m.py
    PLOTS=poc/out python poc/q_gamma_3m.py
    GAMMAS=0.5,1.61,3.0 python poc/q_gamma_3m.py
    SHAPES=2,3 TOL=0.5 python poc/q_gamma_3m.py
    BETAI=1.2 KAPPAI=0.27 python poc/q_gamma_3m.py   # same phi_i, same answer

All four index shapes are run by default, one figure each, and the figure
names the four Q jump parameters the name was calibrated against - gamma_i*
means nothing without them. One parameter per fit, so the whole thing is
seconds, against the quarter of an hour a systematic shape costs.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Library.OptionPricerKimYi2025 import psi_vol  # noqa: E402
from Library.SkewCalibrationKimYi2025 import (  # noqa: E402
    KimYiSkewCalibrationIdiosyncratic)
from poc.q_multistart import SCENARIOS  # noqa: E402

# Production's search interval for gamma_i, from
# Scripts/skew_calibration_kimyi2025.py's bounds_idiosyncratic. The upper end
# is open there; Brent needs a finite one, and nothing sane sits above a name
# that jumps ten times the index.
G_LO, G_HI = 0.1, 10.0

# The true gamma_i* values to recover. 1.61 is the one the paper reports for
# COIN, so the ladder brackets a real number rather than a round one.
GAMMAS = (0.50, 1.00, 1.61, 3.00)

# The name's structural parameters, from the P side. rho_iX is 0 because that
# is where the P-measure work left it; kappa_i is then what makes phi_i 0.45 at
# sigma 0.30, a name whose total diffusion is half again the index's.
BETAI, KAPPAI, RHOIX = 1.00, 0.3354, 0.00

C_TRUE, C_FIT = "#2a78d6", "#1baf7a"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def fitter(T, n, jump, phii, s0, r, q, mkt):
    """A name's calibrator at one expiry, with the index's Q jumps fixed."""
    width = 3.0 * 0.45 * np.sqrt(T)
    k = s0 * np.exp(np.linspace(-width, width, n))
    col = lambda v: np.full(n, float(v))
    return KimYiSkewCalibrationIdiosyncratic(
        sigma=np.array(jump["sigma"]), pprob=np.array(jump["pprob"]),
        lamb=np.array(jump["lamb"]), eta1=np.array(jump["eta1"]),
        eta2=np.array(jump["eta2"]),
        mkt_imp_vol=mkt, und_price=col(s0), und_strike=k, risk_free_rate=col(r),
        dividend_yield=col(q), time_to_expiry=col(T),
        is_call_option=k > s0 * np.exp((r - q) * T),
        option_weights=np.ones((n, 1)) / n, phii=phii)


def objective(f, g):
    return float(f.target(np.array([float(g)])))


def recover(f):
    """Bounded Brent on gamma_i. One parameter, so no starting value at all.

    Brent returns a minimum within [G_LO, G_HI]; that it is THE minimum rests
    on the objective being unimodal there, which curve() below plots.
    """
    r = minimize_scalar(lambda g: objective(f, g), bounds=(G_LO, G_HI),
                        method="bounded", options={"xatol": 1e-8})
    return float(r.x), objective(f, r.x)


def curve(f, n=41):
    """The objective across the whole interval, for the lower panel."""
    gs = np.exp(np.linspace(np.log(G_LO), np.log(G_HI), n))
    return gs, np.array([objective(f, g) for g in gs])


def qtag(jump):
    """The index's Q jump parameters, as the figures and the log both name them."""
    return (r"$p^*$ %.3f   $\lambda^*$ %.2f   $\eta_1^*$ %.2f   $\eta_2^*$ %.2f"
            % (jump["pprob"], jump["lamb"], jump["eta1"], jump["eta2"]))


def phi_of(betai, kappai, rhoix, sigma):
    """The only combination of the three the option price depends on."""
    return float(psi_vol(betai=np.array(betai), kappai=np.array(kappai),
                         rhoix=np.array(rhoix), sigma=np.array(sigma)))


def save_plots(out_dir, T, panels, label, jump, phii):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, NullFormatter
    except ImportError:
        print("\n  matplotlib not installed - skipping plots")
        return
    os.makedirs(out_dir, exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.0), squeeze=False,
                             facecolor=SURFACE)
    for j, pan in enumerate(panels):
        ax = axes[0][j]
        ax.plot(pan["m"], pan["tgt"], color=C_TRUE, lw=3.4, zorder=1,
                solid_capstyle="round", label="target (true $\\gamma_i^*$)")
        ax.plot(pan["m"], pan["fit"], color=C_FIT, lw=2.0, ls=(0, (5, 3)),
                zorder=2, label="calibrated $\\widehat{\\gamma}_i$")
        ax.set_xscale("log")
        ax.set_xticks([0.6, 0.8, 1.0, 1.3, 1.7])
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: "%.2f" % v))
        ax.get_xaxis().set_minor_formatter(NullFormatter())
        ax.set_xlim(pan["m"][0], pan["m"][-1])
        lo, hi = min(pan["tgt"].min(), pan["fit"].min()), max(pan["tgt"].max(),
                                                             pan["fit"].max())
        pad = max(hi - lo, 1e-6)
        ax.set_ylim(lo - 0.28 * pad, hi + 0.08 * pad)
        ax.set_title("$\\gamma_i^*$ = %.2f" % pan["true"], fontsize=11,
                     loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("moneyness $K/S$", fontsize=9, color=MUTED)
        ax.set_ylabel("implied volatility (%)", fontsize=9, color=MUTED)
        ax.text(.03, .04, "$\\widehat{\\gamma}_i$ = %.4f      error %.3f%%"
                % (pan["fit_g"], pan["err"]), transform=ax.transAxes,
                fontsize=8.5, va="bottom", color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec="#dcdbd5",
                          lw=.8, alpha=.93))

        ax = axes[1][j]
        # Below gamma_i ~ 0.25 the model price falls close enough to the
        # arbitrage bound that the IV inversion fails and target() returns a
        # flat penalty. That is "no fit here", not a misfit, and plotting it
        # would compress the real structure into nothing.
        gs, ob = pan["gs"], np.maximum(pan["obj"], 1e-18)
        ok = ob < 1e5
        ax.plot(gs[ok], ob[ok], color="#4a4a48", lw=1.7, zorder=3)
        if (~ok).any():
            top = ob[ok].max() * 30
            ax.plot(gs[~ok], np.full((~ok).sum(), top), ls="none", marker="x",
                    ms=5, mew=1.4, color="#8d8b84", zorder=3,
                    label="no fit (IV inversion failed)")
            ax.set_ylim(ob[ok].min() / 30, top * 3)
        ax.axvline(pan["true"], color=C_TRUE, lw=2.2, ls=(0, (4, 3)), zorder=2,
                   label="true $\\gamma_i^*$")
        ax.axvline(pan["fit_g"], color=C_FIT, lw=1.6, zorder=4,
                   label="$\\widehat{\\gamma}_i$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(G_LO, G_HI)
        ax.set_xticks([0.1, 0.3, 1.0, 3.0, 10.0])
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        ax.get_xaxis().set_minor_formatter(NullFormatter())
        ax.set_xlabel("$\\gamma_i$", fontsize=10, color=MUTED)
        ax.set_ylabel("misfit  $\\mathcal{G}(\\gamma_i)$", fontsize=9,
                      color=MUTED)
        ax.set_title("the objective across the whole interval", fontsize=10,
                     loc="left", color=INK)
        ax.grid(alpha=.25, lw=.7)
        if j == 0:
            # Lower left: the penalty marks sit along the top and the
            # curve climbs to the right, so this is the only empty corner.
            ax.legend(fontsize=8, frameon=False, loc="lower left",
                      labelcolor=INK)
    for row in axes:
        for a in row:
            a.set_facecolor(SURFACE)
            a.tick_params(labelsize=8, colors=MUTED)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper right",
                      labelcolor=INK)
    fig.suptitle("Recovering a name's $\\gamma_i^*$ at %.2f yr  -  simulated data"
                 % T, fontsize=12.5, fontweight="bold", x=.008, y=.975,
                 ha="left", color=INK)
    # The index fit the name is calibrated against, spelled out: gamma_i* means
    # nothing without the four numbers it is measured relative to.
    fig.text(.008, .933, "index held at shape %s, %s:      %s          name's"
             " $\\varphi_i$ %.3f" % (label[0], label[2:], qtag(jump), phii),
             fontsize=10.5, ha="left", color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.912])
    path = os.path.join(out_dir, "q_gamma_%dd_shape%s.png"
                        % (round(T * 365), label[0]))
    fig.savefig(path, dpi=165, facecolor=SURFACE)
    plt.close(fig)
    print("\n  plot written: %s" % path)


def one_shape(label, jump, cfg, gammas, tol, out):
    """The gamma_i* ladder for one index shape, plus its cross-skew check."""
    T, n, phii, s0, r, q = cfg
    print("\n%s   index held at  p* %.3f  lamb* %.2f  eta1* %.2f  eta2* %.2f"
          % (label, jump["pprob"], jump["lamb"], jump["eta1"], jump["eta2"]))
    print("  %10s %12s %12s %13s %9s"
          % ("true g*", "recovered", "error", "misfit", "verdict"))

    panels, rows = [], []
    for gtrue in gammas:
        tgt = np.asarray(fitter(T, n, jump, phii, s0, r, q, np.zeros(n))
                         .model_vol(np.array([gtrue]))).reshape(-1)
        if not np.all(np.isfinite(tgt)):
            print("  %10.3f   IV inversion failed at the truth" % gtrue)
            continue
        f = fitter(T, n, jump, phii, s0, r, q, tgt)
        ghat, obj = recover(f)
        err = 100 * abs(ghat - gtrue) / gtrue
        print("  %10.3f %12.5f %11.4f%% %13.3e %9s"
              % (gtrue, ghat, err, obj, "PASS" if err <= tol else "FAIL"))
        rows.append((gtrue, ghat, err, obj))

        gs, ob = curve(f)
        order = np.argsort(np.concatenate(
            [np.flatnonzero(~f.is_call_option.reshape(-1)),
             np.flatnonzero(f.is_call_option.reshape(-1))]))
        width = 3.0 * 0.45 * np.sqrt(T)
        panels.append(dict(
            true=gtrue, fit_g=ghat, err=err,
            m=np.exp(np.linspace(-width, width, n)), tgt=tgt[order] * 100,
            fit=np.asarray(f.model_vol(np.array([ghat]))).reshape(-1)[order] * 100,
            gs=gs, obj=ob))

    if out and panels:
        save_plots(out, T, panels, label, jump, phii)
    return rows


def cross_check(label, jump, other, cfg):
    """What gamma_i* cannot do: reach a name that leans the other way.

    gamma_i* scales both decay rates together, so it moves a name's jumps
    further out or closer in but never tilts them towards one wing. Generating
    the name's smile from ANOTHER shape's jumps at gamma_i* = 1 and then asking
    this shape's gamma_i* to fit it prices that limit, in vol points.
    """
    T, n, phii, s0, r, q = cfg
    clab, cjump = other
    tgt = np.asarray(fitter(T, n, cjump, phii, s0, r, q, np.zeros(n))
                     .model_vol(np.array([1.0]))).reshape(-1)
    if not np.all(np.isfinite(tgt)):
        return None
    ghat, obj = recover(fitter(T, n, jump, phii, s0, r, q, tgt))
    print("  a name shaped like %-20s best gamma_i %6.3f, residual %7.3f vp"
          % (clab + ":", ghat, 100 * np.sqrt(2 * obj)))
    return clab, ghat, obj


def main():
    env = lambda k, d: float(os.environ.get(k, d))
    T, n = env("TENOR", 0.25), int(env("NSTRIKES", 11))
    sigma = env("SIGMA", 0.30)
    betai, kappai = env("BETAI", BETAI), env("KAPPAI", KAPPAI)
    rhoix = env("RHOIX", RHOIX)
    phii = phi_of(betai, kappai, rhoix, sigma)
    s0, r, q = env("SPOT", 100.), env("RATE", 0.04), env("DIVY", 0.015)
    tol = env("TOL", 1.0)
    gammas = [float(v) for v in os.environ["GAMMAS"].split(",")] \
        if os.environ.get("GAMMAS") else list(GAMMAS)
    pick = os.environ.get("SHAPES")
    shapes = [(lab, dict(j, sigma=sigma)) for lab, j in SCENARIOS
              if not pick or lab[0] in pick.split(",")]
    cfg = (T, n, phii, s0, r, q)
    out = os.environ.get("PLOTS")

    print("tenor %.3f yr, %d strikes, S0 %.0f, r %.3f, q %.3f, pass at %.1f%%"
          % (T, n, s0, r, q, tol))
    print("name: beta_i %.3f, kappa_i %.4f, rho_iX %.2f, sigma %.2f  ->"
          "  phi_i %.5f" % (betai, kappai, rhoix, sigma, phii))
    print("      the surface depends on those four only through phi_i, and"
          " given phi_i")
    print("      sigma cancels, so phi_i is the whole of the name's diffusion"
          " input")
    print("gamma_i searched on [%.1f, %.1f] by bounded Brent - one parameter,"
          " so no starting value" % (G_LO, G_HI))

    summary = []
    for label, jump in shapes:
        rows = one_shape(label, jump, cfg, gammas, tol, out)
        summary.append((label, rows))

    print("\n  worst error over the gamma_i* ladder, by index shape")
    for label, rows in summary:
        worst = max((r[2] for r in rows), default=float("nan"))
        print("  %-22s %8.4f%%   %s"
              % (label, worst, "PASS" if worst <= tol else "FAIL"))
    print("\n  The objective at the true gamma_i* is 0 by construction, so the")
    print("  misfit column is how close the search got to it, and the lower")
    print("  panels show whether that minimum is sharp or merely somewhere.")

    # The boundary of the claim, once, against the shape that leans opposite.
    if len(shapes) > 1:
        print("\n  gamma_i* is a magnitude, not a direction. Fitting a smile that")
        print("  leans the other way is not something a larger or smaller"
              " gamma_i* can do:")
        pairs = {"1": "4", "2": "3", "3": "2", "4": "1"}
        by_key = {lab[0]: (lab, j) for lab, j in shapes}
        for label, jump in shapes:
            other = by_key.get(pairs[label[0]])
            if other:
                cross_check(label, jump, other, cfg)


if __name__ == "__main__":
    main()
