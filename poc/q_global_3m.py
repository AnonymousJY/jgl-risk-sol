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

Each shape is also fitted the plain way, from production's own starting vector
- the fit the calibrator would actually produce. NLOCAL=k substitutes the best
of k random starts over the support q_multistart draws from.

NEITHER ROUTE WINS EVERYWHERE, so the script reports both and keeps whichever
scored the lower objective, which is a choice available without knowing the
truth. The plain fit takes the shapes where production's start already sits in
the right basin; the scan takes the ones where it does not. Both are plotted
against the target smile, and the profile itself is plotted underneath, so the
failure and the fix are visible in the same figure.

    python poc/q_global_3m.py
    PLOTS=poc/out python poc/q_global_3m.py         # also write the figures
    SCENARIOS=2 ECHO=1 python poc/q_global_3m.py    # one shape, node by node
    PGRID=20 python poc/q_global_3m.py              # finer scan, slower
    NLOCAL=3 python poc/q_global_3m.py              # random-start contrast

Every node is solved twice and the refinement costs about ten nodes more, so
budget roughly a minute and a half per grid node: the default 10 nodes x 4
shapes runs one to two hours depending on the machine. NLOCAL adds several
minutes per draw.
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


X0 = np.ones(3)                                   # production x0, in S3 units


def profile(f, p, starts):
    """min over (lamb, eta1, eta2) at fixed p, best of the starts given.

    The inner problem has flat directions of its own - at p near 1 the down
    branch carries almost no weight, so eta2 is barely determined - and a
    single start can stall on the wrong branch of them. Taking the better of
    two starts is what keeps the profile a profile rather than one branch of
    an upper envelope.
    """
    g = lambda u: float(f.target(np.concatenate(([p], np.asarray(u) * S3))))
    best, bobj = None, np.inf
    for u0 in starts:
        r = minimize(g, x0=u0, method="SLSQP", bounds=B3, tol=1e-13,
                     options={"maxiter": 200})
        if float(r.fun) < bobj:
            best, bobj = np.asarray(r.x, dtype=float) * S3, float(r.fun)
    return bobj, best


def scan(f, grid, echo=False):
    """The profile over the whole grid.

    Each node is solved twice: from production's x0, which is the same vector
    for every shape and so encodes no truth, and from the previous node's
    answer. Continuing alone is what went wrong the first time this was
    written - on the put-down/call-up shape the chain picked up eta2 ~ 78 in
    the uninformative left half of the grid and carried it all the way across,
    putting the profile three decades above its true floor at high p and
    hiding the minimum entirely.
    """
    u0 = X0
    rows = []
    for p in grid:
        starts = [X0] if np.array_equal(u0, X0) else [X0, u0]
        o, x3 = profile(f, p, starts)
        u0 = x3 / S3 if o < PENALTY else X0
        rows.append((p, o, *x3))
        if echo:
            print("    %6.3f %11.4e %9.3f %9.3f %9.3f" % (p, o, *x3), flush=True)
    return np.asarray(rows)


def refine(f, rows, tol=1e-3):
    """Minimise the profile in p between the neighbours of the best node.

    One dimension, and the flat direction has already been optimised away, so
    bounded Brent is well posed here where a four-parameter local search is
    not. Every evaluation uses the SAME pair of inner starts - production's x0
    and the best node's answer - so the profile is a deterministic function
    of p rather than of the path Brent happens to take to it.
    """
    i = int(np.argmin(rows[:, 1]))
    lo = rows[max(i - 1, 0), 0]
    hi = rows[min(i + 1, len(rows) - 1), 0]
    starts = [X0, rows[i, 2:] / S3]
    r = minimize_scalar(lambda p: profile(f, p, starts)[0], bounds=(lo, hi),
                        method="bounded", options={"xatol": tol})
    p = float(r.x)
    return np.concatenate(([p], profile(f, p, starts)[1])), float(r.fun)


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
    ngrid = int(env("PGRID", 10))
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
        # Neither route wins everywhere - the plain fit takes the shapes where
        # production's start already sits in the right basin, the scan takes
        # the ones where it does not - and the objective says which is which
        # without anyone having to know the truth.
        xbst, obst, hbst = ((xone, oone, "the plain fit") if oone <= oglo
                            else (xglo, oglo, "the scan"))
        ebst = 100 * np.max(np.abs(xbst - true) / true)
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
        print("  %-24s %7.3f %7.2f %7.2f %7.2f %11.3e %7.1f%%   (%.0fs)"
              % ("+ four-way polish", *xglo, oglo, eglo, tscan))
        print("  %-24s %7.3f %7.2f %7.2f %7.2f %11.3e %7.1f%%   <- %s\n"
              % ("lower objective wins", *xbst, obst, ebst, hbst))

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
        summary.append((label, eone, eglo, ebst, gap))

    print("\n  %-20s %11s %11s %11s %10s"
          % ("scenario", "plain fit", "scan", "kept", "grid gap"))
    for label, eone, eglo, ebst, gap in summary:
        print("  %-20s %10.1f%% %10.1f%% %10.1f%% %10.3f"
              % (label, eone, eglo, ebst, gap))
    print("\n  'kept' is whichever of the two scored the lower objective, which")
    print("  is a choice the calibrator can make without knowing the truth.")
    print("  'grid gap' is the distance from the true p to the nearest node of")
    print("  the scan. No node is the answer, so what recovers the truth is the")
    print("  refinement and polish that follow the scan, not the grid.")

    out = os.environ.get("PLOTS")
    if out and panels:
        save_plots(out, T, panels)


if __name__ == "__main__":
    main()
