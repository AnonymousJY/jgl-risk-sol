"""A picture of why the four-parameter search stalls, and what the scan does.

Three panels, all of the put-up/call-down shape at 3M, all in the (p, eta1)
plane with lamb and eta2 minimised out at every point - so what is drawn is the
lowest misfit reachable at each (p, eta1), not a slice at arbitrary lamb and
eta2, and the low region really is the set of near-perfect fits.

  A  the landscape, with a descent from a starting guess drawn on it. The
     near-perfect region is a long CURVED band, and the descent walks into it
     and halts on its floor, well away from the truth, at a misfit small
     enough that the fitted smile is indistinguishable by eye.
  B  step 1 of the algorithm: p is held at each grid value in turn and the
     other three are fitted, which steps along the band instead of across it.
  C  steps 2 and 3: the same information as a curve - the misfit profiled in p
     - with the bracket the refinement searches, and where it lands.

    python poc/q_valley_3m.py                    # compute the field and plot
    OUT=poc/out python poc/q_valley_3m.py
    NP=48 NE=36 python poc/q_valley_3m.py        # finer field, slower
    FIELD=poc/out/valley.npz python poc/q_valley_3m.py   # reuse a saved field

The field is the expensive part: NP x NE two-parameter minimisations, about
half a second each, so the default 40 x 32 takes ten to fifteen minutes. It is
cached to OUT/valley.npz and reused when FIELD points at it.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poc.q_global_3m import S3, inner_starts, profile  # noqa: E402
from poc.q_multistart import BOUNDS, SCALE, fitter  # noqa: E402

T, NSTRIKES, PHII, SPOT, RATE, DIVY = 0.25, 11, 0.30, 100., 0.04, 0.015
TRUE = np.array([0.15, 10.0, 40.0, 10.0])        # the put-up/call-down shape
DESCENT_START = np.array([0.55, 18.0, 62.0, 14.0])

P_LO, P_HI = 0.02, 0.98
E1_LO, E1_HI = 8.0, 90.0

# lamb and eta2, the two minimised out at each point of the field
S2 = SCALE[[1, 3]]
B2 = [(lo / s, None if hi is None else hi / s)
      for (lo, hi), s in zip([BOUNDS[1], BOUNDS[3]], S2)]


def build(f, ps, e1s):
    """Lowest misfit at each (p, eta1), minimising over (lamb, eta2)."""
    Z = np.empty((len(e1s), len(ps)))
    row0 = np.array([10.0, 10.0]) / S2
    t0 = time.time()
    for a, e1 in enumerate(e1s):
        u = row0
        for b, p in enumerate(ps):
            g = lambda v: float(f.target(np.array(
                [p, v[0] * S2[0], e1, v[1] * S2[1]])))
            # A picture, not a result: three good digits on a log scale is
            # plenty, and the row continuation means every solve starts close.
            r = minimize(g, x0=u, method="SLSQP", bounds=B2, tol=1e-9,
                         options={"maxiter": 50})
            Z[a, b] = float(r.fun)
            u = np.asarray(r.x, float)
            if b == 0:
                row0 = u
        print("    row %2d/%2d  eta1 %6.2f  (%.0fs)"
              % (a + 1, len(e1s), e1, time.time() - t0), flush=True)
    return Z


def descent(f, x0):
    """The four-parameter SLSQP path, as it is actually walked."""
    path = [np.asarray(x0, float)]
    bnds = [(lo / s, None if hi is None else hi / s)
            for (lo, hi), s in zip(BOUNDS, SCALE)]
    minimize(lambda u: float(f.target(u * SCALE)), x0=np.asarray(x0) / SCALE,
             method="SLSQP", bounds=bnds, tol=1e-11,
             options={"maxiter": 150},
             callback=lambda u: path.append(np.asarray(u, float) * SCALE))
    return np.array(path)


# Sequential blue for magnitude; categorical slots 2 and 3 for the two routes,
# slot 1 for the truth. The field never goes past step 450, so a fully
# saturated slot-1 marker with a surface ring still reads on top of it.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
        "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab"]
C_TRUE, C_ONE, C_BEST = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def draw(out_dir, ps, e1s, Z, path, rows, xref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

    def leg(a, loc):
        """A legend that stays readable where it sits on the dark end."""
        h = a.legend(fontsize=8, loc=loc, labelcolor=INK, framealpha=.93,
                     facecolor=SURFACE, edgecolor="#dcdbd5", borderpad=.5)
        h.get_frame().set_linewidth(.8)
        return h

    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    L = np.log10(np.clip(Z, 1e-12, None))
    levels = np.linspace(np.floor(L.min()), min(np.ceil(L.max()), -2.0), 11)
    norm = BoundaryNorm(levels, cmap.N)

    fig = plt.figure(figsize=(15.6, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, .05], hspace=.62, wspace=.24,
                          left=.045, right=.99, top=.855, bottom=.08)
    ax = [fig.add_subplot(gs[0, j]) for j in range(3)]
    cax = fig.add_subplot(gs[1, 0:2])
    for a in ax:
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=8, colors=MUTED)

    def field(a, alpha=1.0):
        return a.contourf(ps, e1s, L, levels=levels, cmap=cmap, norm=norm,
                          alpha=alpha, extend="max")

    # --- A: the landscape, and a descent that stops on its floor -----------
    cf = field(ax[0])
    ax[0].plot(path[:, 0], path[:, 2], color=C_ONE, lw=2.0, marker="o", ms=4,
               mec=SURFACE, mew=1.2, zorder=4, label="four-parameter descent")
    ax[0].plot(path[-1, 0], path[-1, 2], marker="o", ms=11, color=C_ONE,
               mec=SURFACE, mew=2, zorder=5, label="where it stops")
    ax[0].plot(*TRUE[[0, 2]], marker="*", ms=19, color=C_TRUE, mec=SURFACE,
               mew=1.6, zorder=6, label="true parameters")
    ax[0].set_title("A   the misfit landscape", fontsize=11, loc="left",
                    fontweight="bold", color=INK)
    leg(ax[0], "lower right")

    # --- B: step 1, holding p and fitting the rest -------------------------
    field(ax[1], alpha=.45)
    for p in rows[:, 0]:
        ax[1].axvline(p, color=MUTED, lw=.7, ls=(0, (2, 3)), zorder=2)
    ax[1].plot(rows[:, 0], rows[:, 3], color=C_BEST, lw=2.0, marker="o", ms=5,
               mec=SURFACE, mew=1.2, zorder=4, label="best fit at each fixed $p$")
    i = int(np.argmin(rows[:, 1]))
    ax[1].plot(rows[i, 0], rows[i, 3], marker="o", ms=12, mfc="none",
               mec=C_BEST, mew=2.4, zorder=5, label="lowest of them")
    ax[1].plot(*TRUE[[0, 2]], marker="*", ms=19, color=C_TRUE, mec=SURFACE,
               mew=1.6, zorder=6)
    ax[1].set_title("B   step 1: hold $p$, fit the other three",
                    fontsize=11, loc="left", fontweight="bold", color=INK)
    leg(ax[1], "lower right")

    for a in ax[:2]:
        a.set_xlim(ps[0], ps[-1])
        a.set_ylim(e1s[0], e1s[-1])
        a.set_xlabel("$p$   (up-jump probability)", fontsize=9, color=MUTED)
    ax[0].set_ylabel(r"$\eta_1$   (up-jump decay rate)", fontsize=9, color=MUTED)

    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("misfit  $\\log_{10}\\mathcal{G}$   -   lighter is a better fit"
                 " (the pale band is the set of near-perfect fits)",
                 fontsize=8.5, color=MUTED, labelpad=4)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED, length=2)
    cb.outline.set_visible(False)

    # --- C: steps 2 and 3, the same thing as a curve -----------------------
    g = np.maximum(rows[:, 1], 1e-16)
    lo = rows[max(i - 1, 0), 0]
    hi = rows[min(i + 1, len(rows) - 1), 0]
    ax[2].axvspan(lo, hi, color=C_BEST, alpha=.10, zorder=1,
                  label="step 2 searches here")
    ax[2].plot(rows[:, 0], g, color="#4a4a48", lw=1.6, marker="o", ms=4.5,
               mec=SURFACE, mew=1.0, zorder=3)
    ax[2].plot(rows[i, 0], g[i], marker="o", ms=12, mfc="none", mec=C_BEST,
               mew=2.4, zorder=4)
    ax[2].axvline(TRUE[0], color=C_TRUE, lw=2.0, ls=(0, (4, 3)), zorder=2,
                  label="true $p$")
    ax[2].axvline(xref[0], color=C_BEST, lw=1.8, zorder=5,
                  label="$\\widehat p$ after step 2")
    ax[2].set_yscale("log")
    ax[2].set_xlim(ps[0], ps[-1])
    ax[2].set_xlabel("$p$   (up-jump probability)", fontsize=9, color=MUTED)
    ax[2].set_ylabel(r"$g(p)=\min_{\lambda,\eta_1,\eta_2}\mathcal{G}$",
                     fontsize=9, color=MUTED)
    ax[2].set_title("C   steps 2-3: the same thing as a curve",
                    fontsize=11, loc="left", fontweight="bold", color=INK)
    ax[2].grid(alpha=.25, lw=.7)
    leg(ax[2], "upper left")

    fig.suptitle("Why the direct search stalls, and what holding $p$ fixed does"
                 "  -  simulated 3-month smile",
                 fontsize=13, fontweight="bold", x=.008, y=.965, ha="left",
                 color=INK)
    os.makedirs(out_dir, exist_ok=True)
    path_png = os.path.join(out_dir, "q_valley_3m.png")
    fig.savefig(path_png, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("\n  plot written: %s" % path_png)


def main():
    env = lambda k, d: int(os.environ.get(k, d))
    out = os.environ.get("OUT", "poc/out")
    npg, neg = env("NP", 40), env("NE", 32)

    tgt = np.asarray(fitter(T, NSTRIKES, PHII, SPOT, RATE, DIVY,
                            np.zeros(NSTRIKES)).model_vol(TRUE)).reshape(-1)
    f = fitter(T, NSTRIKES, PHII, SPOT, RATE, DIVY, tgt)

    cached = os.environ.get("FIELD")
    if cached and os.path.exists(cached):
        d = np.load(cached)
        ps, e1s, Z = d["ps"], d["e1s"], d["Z"]
        print("  field read from %s  (%d x %d)" % (cached, len(ps), len(e1s)))
    else:
        ps = np.linspace(P_LO, P_HI, npg)
        e1s = np.linspace(E1_LO, E1_HI, neg)
        print("  building the %d x %d field" % (npg, neg))
        Z = build(f, ps, e1s)
        os.makedirs(out, exist_ok=True)
        np.savez(os.path.join(out, "valley.npz"), ps=ps, e1s=e1s, Z=Z)

    print("  descent from p %.2f lam %.1f e1 %.1f e2 %.1f"
          % tuple(DESCENT_START))
    path = descent(f, DESCENT_START)
    print("    stops at p %.3f lam %.2f e1 %.2f e2 %.2f, misfit %.2e, err %.0f%%"
          % (*path[-1], float(f.target(path[-1])),
             100 * np.max(np.abs(path[-1] - TRUE) / TRUE)))

    starts = inner_starts() / S3
    grid = np.linspace(P_LO, P_HI, 10)
    rows, prev = [], None
    for p in grid:
        o, v = profile(f, p, list(starts) + ([prev] if prev is not None else []))
        prev = v / S3
        rows.append((p, o, *v))
        print("    p %.3f  g %.3e  lam %7.3f e1 %7.3f e2 %7.3f" % (p, o, *v),
              flush=True)
    rows = np.asarray(rows)
    # rows columns: p, g, lamb, eta1, eta2 -> eta1 is column 3
    from poc.q_global_3m import refine
    xref, oref = refine(f, rows, starts)
    print("  step 2 lands at p %.4f lam %.3f e1 %.3f e2 %.3f, misfit %.2e"
          % (*xref, oref))

    draw(out, ps, e1s, Z, path, rows, xref)


if __name__ == "__main__":
    main()
