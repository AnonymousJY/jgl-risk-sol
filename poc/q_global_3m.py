"""Recovering the true Q jump parameters at one fixed 3M expiry, in each of
the four skew shapes, without a starting value that knows the answer.

WHAT poc/q_multistart.py LEFT UNRESOLVED. Random starts recovered shape 1 ten
times out of ten and shape 2 none out of ten. That looked like a statement
about the model. It is not. The failed fits all sat at objective ~1e-08 with
q*lamb and eta2 pinned at their true values and only (p, eta1) adrift, which
is the signature of a FLAT DIRECTION, not of a rival optimum.

WHAT THE LANDSCAPE ACTUALLY LOOKS LIKE. Profile the objective in p - fix p,
optimise (lamb, eta1, eta2) at that p, and plot the result against p - and
the picture is unambiguous. On shape 2 the profile runs

    p     0.02      0.11      0.20      0.28      0.46      0.54      0.63
    obj   5.3e-08   1.4e-09   8.2e-10   4.7e-09   1.4e-08   8.2e-08   3.2e-04

One minimum, and it sits at the truth, at the bottom of a valley 50 p-points
wide whose floor never rises above 2e-08 before the wall at p ~ 0.55. Along
that floor lamb climbs 8.9 -> 15.4 and eta1 climbs 21.9 -> 70.1 together,
holding (1-p)*lamb and eta2 fixed: the surface prices the DOWN branch
precisely and buys the UP branch's contribution with whatever (p, eta1) pair
delivers it.

That is why a four-parameter local search stalls. The valley floor is curved,
so from a point on it every coordinate direction is uphill; SLSQP is at a
point where it cannot improve, and stops. A straight line from such a point to
the truth leaves the valley and climbs to 2.9e-06 midway, which is what makes
the two look like separate basins. They are one basin with a bent floor.

WHAT THIS SCRIPT DOES. It follows the floor instead of fighting it: a coarse
scan over p with the other three optimised out, a bounded one-dimensional
refinement between the neighbours of the best node, and a four-parameter
polish if anything is left to gain. The scan is deterministic, identical
across all four shapes, and carries nothing about any truth - the (lamb, eta1,
eta2) search starts at production's own x0 and then continues from the
previous node. The grid is spaced so that no node coincides with any of the
four true p values; the script prints how far the nearest node was, so the
claim rests on the refinement rather than on a lucky grid.

For contrast each shape is also fitted the plain way, from production's own
starting vector - which is the fit the calibrator would actually produce.
NLOCAL=k substitutes the best of k random starts over the support
q_multistart draws from. Both fits are plotted against the target smile, and
the profile itself is plotted underneath, so the failure and the fix are
visible in the same figure.

    python poc/q_global_3m.py
    PLOTS=poc/out python poc/q_global_3m.py         # also write the figures
    SCENARIOS=2 ECHO=1 python poc/q_global_3m.py    # one shape, node by node
    PGRID=20 python poc/q_global_3m.py              # finer scan, slower
    NLOCAL=3 python poc/q_global_3m.py              # random-start contrast

Budget roughly 30-40 s per grid node, plus about ten more nodes' worth for the
refinement: the default 12 nodes x 4 shapes is 45 minutes to an hour. NLOCAL
adds several minutes per draw.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poc.q_multistart import (BOUNDS, SCALE, SCENARIOS,  # noqa: E402
                              draw_starts, fit, fitter)

# The p grid. Endpoints inside the open unit interval, and a node count chosen
# so that no node lands on 0.15, 0.50 or 0.85 - the three true values among the
# four shapes. main() prints the realised distance, so a future change to
# either list is caught rather than assumed away.
P_LO, P_HI = 0.02, 0.98

# target() returns this flat value when Newton's IV inversion fails at any
# strike, so it marks "no fit here" rather than a comparable objective.
PENALTY = 1e5

# Below this the profile bottom is already at the numerical floor and a
# four-parameter polish has nothing left to find, so it is skipped.
POLISH_ABOVE = 1e-12

S3 = SCALE[1:]                                    # scale for (lamb, eta1, eta2)
B3 = [(lo / s, None if hi is None else hi / s)
      for (lo, hi), s in zip(BOUNDS[1:], S3)]


def profile(f, p, u0):
    """min over (lamb, eta1, eta2) at fixed p. Returns (objective, the three)."""
    g = lambda u: float(f.target(np.concatenate(([p], np.asarray(u) * S3))))
    r = minimize(g, x0=u0, method="SLSQP", bounds=B3, tol=1e-13,
                 options={"maxiter": 200})
    return float(r.fun), np.asarray(r.x, dtype=float) * S3


def scan(f, grid, echo=False):
    """The profile over the whole grid, each node continuing from the last.

    The first node starts at production's x0 for (lamb, eta1, eta2), which is
    the same vector for every shape and so cannot encode a particular truth.
    """
    u0 = np.ones(3)
    rows = []
    for p in grid:
        o, x3 = profile(f, p, u0)
        if o >= PENALTY:
            # The inner solve walked into the region where the IV inversion
            # fails. Retry from production x0 and, either way, do not hand a
            # broken iterate to the next node.
            o, x3 = profile(f, p, np.ones(3))
        u0 = x3 / S3 if o < PENALTY else np.ones(3)
        rows.append((p, o, *x3))
        if echo:
            print("    %6.3f %11.4e %9.3f %9.3f %9.3f" % (p, o, *x3), flush=True)
    return np.asarray(rows)


def refine(f, rows, tol=1e-3):
    """Minimise the profile in p between the neighbours of the best node.

    One dimension, and the flat direction has already been optimised away, so
    bounded Brent is well posed here where a four-parameter local search is
    not. Every evaluation starts the inner solve from the SAME (lamb, eta1,
    eta2) - the best node's - so the profile is a deterministic function of p.
    """
    i = int(np.argmin(rows[:, 1]))
    lo = rows[max(i - 1, 0), 0]
    hi = rows[min(i + 1, len(rows) - 1), 0]
    u0 = rows[i, 2:] / S3
    r = minimize_scalar(lambda p: profile(f, p, u0)[0], bounds=(lo, hi),
                        method="bounded", options={"xatol": tol})
    p = float(r.x)
    return np.concatenate(([p], profile(f, p, u0)[1])), float(r.fun)


C_TRUE, C_ONE, C_BEST = "#2a78d6", "#eb6834", "#1baf7a"


def save_plots(out_dir, T, panels):
    """Two rows per shape: the fitted smile above, the p profile below."""
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
    fig, axes = plt.subplots(2, n, figsize=(5.3 * n, 8.0), squeeze=False)

    for j, pan in enumerate(panels):
        ax = axes[0][j]
        m = pan["m"]
        ax.plot(m, pan["tgt"], color=C_TRUE, lw=3.4, solid_capstyle="round",
                label="target (true parameters)", zorder=1)
        ax.plot(m, pan["one"], color=C_ONE, lw=2.0, label=pan["onelab"], zorder=3)
        ax.plot(m, pan["glo"], color=C_BEST, lw=2.0, ls=(0, (5, 3)),
                label="profile scan + polish", zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([0.6, 0.8, 1.0, 1.3, 1.7])
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: "%.2f" % v))
        ax.get_xaxis().set_minor_formatter(NullFormatter())
        ax.set_xlim(m[0], m[-1])
        lo = min(pan["tgt"].min(), pan["one"].min(), pan["glo"].min())
        hi = max(pan["tgt"].max(), pan["one"].max(), pan["glo"].max())
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
                "true    " + fmt(pan["true"]) + "\n" +
                "local   " + fmt(pan["xone"]) + "   err %5.1f%%\n" % pan["eone"] +
                "global  " + fmt(pan["xglo"]) + "   err %5.1f%%" % pan["eglo"],
                transform=ax.transAxes, fontsize=7.2, family="monospace",
                va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                       ec="#dcdbd5", lw=.8, alpha=.92))

        ax = axes[1][j]
        g, o = pan["grid"], np.maximum(pan["prof"], 1e-18)
        # Nodes where the IV inversion failed carry a flat penalty, not an
        # objective. Plotting them would compress six decades of real
        # structure into nothing, so they are drawn as open marks on the
        # ceiling instead.
        bad = o >= PENALTY
        ax.plot(g[~bad], o[~bad], color="#4a4a48", lw=1.6, marker="o", ms=3.2,
                zorder=2)
        if bad.any():
            top = o[~bad].max() * 30
            ax.plot(g[bad], np.full(bad.sum(), top), ls="none", marker="x",
                    ms=5, mew=1.4, color="#8d8b84", zorder=2,
                    label="no fit (IV inversion failed)")
            ax.set_ylim(o[~bad].min() / 30, top * 3)
        ax.axvline(pan["true"][0], color=C_TRUE, lw=2.2, ls=(0, (4, 3)), zorder=1,
                   label="true p")
        ax.axvline(pan["xglo"][0], color=C_BEST, lw=1.6, zorder=3,
                   label="p after polish")
        ax.axvline(pan["xone"][0], color=C_ONE, lw=1.6, zorder=3,
                   label="p from the local fit")
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_title("objective profiled in p", fontsize=10, loc="left")
        ax.set_xlabel("p  (up-jump probability), lamb/eta1/eta2 optimised out",
                      fontsize=9)
        ax.set_ylabel("profile objective (log scale)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.25, lw=.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if j == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper right")

    axes[0][0].legend(fontsize=8, frameon=False, loc="upper right")
    # One panel is a narrow figure, so the title has to shrink with it or it
    # runs off the canvas.
    fig.suptitle("Q-measure recovery at %.2f yr, one fixed expiry - simulated data"
                 % T, fontsize=13 if n > 1 else 9.5, fontweight="bold",
                 x=.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = os.path.join(out_dir, "q_global_%dd.png" % round(T * 365))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("\n  plot written: %s" % path)


def main():
    env = lambda k, d: float(os.environ.get(k, d))
    T, n = env("TENOR", 0.25), int(env("NSTRIKES", 11))
    phii, s0 = env("PHII", 0.30), env("SPOT", 100.)
    r, q = env("RATE", 0.04), env("DIVY", 0.015)
    ngrid = int(env("PGRID", 12))
    nlocal = int(env("NLOCAL", 0))
    echo = bool(os.environ.get("ECHO"))
    rng = np.random.default_rng(int(env("SEED", 20240114)))
    pick = os.environ.get("SCENARIOS")
    scen = ([s for s in SCENARIOS if s[0][0] in pick.split(",")] if pick else SCENARIOS)
    grid = np.linspace(P_LO, P_HI, ngrid)

    print("tenor %.3f yr, %d strikes, phi_i %.2f" % (T, n, phii))
    print("p scan: %d nodes on [%.2f, %.2f], (lamb, eta1, eta2) optimised out at"
          % (ngrid, P_LO, P_HI))
    print("each node, starting from production x0 %s and continuing\n"
          % np.array2string(SCALE[1:], precision=2))

    panels, summary = [], []
    for label, jump in scen:
        true = np.array([jump["pprob"], jump["lamb"], jump["eta1"], jump["eta2"]])
        tgt = np.asarray(fitter(T, n, phii, s0, r, q, np.zeros(n))
                         .model_vol(true)).reshape(-1)
        if not np.all(np.isfinite(tgt)):
            print("  %-20s  IV inversion failed at the truth" % label)
            continue
        f = fitter(T, n, phii, s0, r, q, tgt)
        gap = float(np.min(np.abs(grid - true[0])))

        print("%s   true p %.3f lam %.2f e1 %.2f e2 %.2f   objective at truth %.2e"
              % (label, *true, float(f.target(true))))
        print("  nearest grid node is %.3f away from the true p" % gap)
        if echo:
            print("    %6s %11s %9s %9s %9s"
                  % ("p", "profile", "lamb", "eta1", "eta2"))

        t0 = time.time()
        rows = scan(f, grid, echo=echo)
        node = rows[int(np.argmin(rows[:, 1]))]
        xref, oref = refine(f, rows)
        xglo, oglo = xref, oref
        if oref > POLISH_ABOVE:
            cand, o = fit(f, xref)
            if o < oglo:                    # the polish may help, never hurt
                xglo, oglo = cand, o
        tscan = time.time() - t0

        # The plain way, for contrast. Production's own x0 by default: it is
        # deterministic, it is what the calibrator actually does, and it is
        # the same vector in every scenario so it encodes no truth. NLOCAL=k
        # substitutes the best of k random starts over q_multistart's support,
        # which is slower - an unbounded lamb lets SLSQP wander into the
        # region where the Kou jump-count bound, and so the price, is an order
        # of magnitude more expensive.
        xone, oone = None, np.inf
        for x0 in (draw_starts(rng, nlocal) if nlocal else [SCALE]):
            cand, o = fit(f, x0)
            if o < oone:
                xone, oone = cand, o

        eglo = 100 * np.max(np.abs(xglo - true) / true)
        eone = 100 * np.max(np.abs(xone - true) / true)
        print("  %-24s %7s %7s %7s %7s %11s %8s"
              % ("", "p", "lamb", "eta1", "eta2", "objective", "err"))
        print("  %-24s %7.3f %7.2f %7.2f %7.2f %11.3e %8s"
              % ("true", *true, float(f.target(true)), ""))
        print("  %-24s %7.3f %7.2f %7.2f %7.2f %11.3e %7.1f%%"
              % ("local, best of %d random" % nlocal if nlocal
                 else "local, production x0", *xone, oone, eone))
        print("  %-24s %7.3f %7.2f %7.2f %7.2f %11.3e %8s"
              % ("best grid node", node[0], node[2], node[3], node[4], node[1], ""))
        print("  %-24s %7.3f %7.2f %7.2f %7.2f %11.3e %7.1f%%"
              % ("after refining p", *xref, oref,
                 100 * np.max(np.abs(xref - true) / true)))
        print("  %-24s %7.3f %7.2f %7.2f %7.2f %11.3e %7.1f%%   (%.0fs)\n"
              % ("+ four-way polish", *xglo, oglo, eglo, tscan))

        width = 3.0 * 0.45 * np.sqrt(T)
        order = np.argsort(np.concatenate(
            [np.flatnonzero(~f.is_call_option.reshape(-1)),
             np.flatnonzero(f.is_call_option.reshape(-1))]))
        curve = lambda x: np.asarray(f.model_vol(x)).reshape(-1)[order] * 100
        panels.append(dict(
            label=label, m=np.exp(np.linspace(-width, width, n)),
            tgt=tgt[order] * 100, one=curve(xone), glo=curve(xglo),
            true=true, xone=xone, xglo=xglo, eone=eone, eglo=eglo,
            grid=rows[:, 0], prof=rows[:, 1],
            onelab="local, best of %d random" % nlocal if nlocal
                   else "local, production x0"))
        summary.append((label, eone, eglo, gap))

    print("\n  %-20s %14s %14s %12s"
          % ("scenario", "local err", "global err", "grid gap"))
    for label, eone, eglo, gap in summary:
        print("  %-20s %13.1f%% %13.1f%% %12.3f" % (label, eone, eglo, gap))
    print("\n  'grid gap' is the distance from the true p to the nearest node of")
    print("  the scan. No node is the answer, so what recovers the truth is the")
    print("  refinement and polish that follow the scan, not the grid.")

    out = os.environ.get("PLOTS")
    if out and panels:
        save_plots(out, T, panels)


if __name__ == "__main__":
    main()
