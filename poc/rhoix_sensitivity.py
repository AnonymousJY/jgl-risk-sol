"""Does the unidentified direction in rho_iX move any price?

rho_iX is never identified. Across 594 valuation dates and six names the
posterior/prior width ratio sits at 1.03-1.09 - the posterior is WIDER than the
prior - and the shift is under half a standard deviation. That is structural,
not weak data: given the systematic fit, a single name's returns offer the
likelihood exactly two second moments,

    Var(r_i)      = (sigma beta_i)^2 + 2 sigma beta_i kappa_i rho_iX + kappa_i^2
    Cov(r_i, r_X) = sigma^2 beta_i + sigma kappa_i rho_iX

against three unknowns. The diffusion is Gaussian so second moments exhaust it,
and the jump channel is driven by the common Y, so it speaks to gamma_i and not
to the diffusion correlation. One dimension is flat, and no amount of history
or higher frequency changes that.

WHAT IS IDENTIFIED. Exactly the two combinations above, and it is worth writing
them in the form the pricing uses:

    phi_i    = sqrt(Var(r_i))                          the Black-Scholes vol
    b_i      = beta_i + kappa_i rho_iX / sigma         the diffusion channel of
             = Cov(r_i, r_X) / sigma^2                 the translation

Both are functions of the identified moments alone. So walking along the flat
direction cannot move phi_i or the translation - only R_ij, through

    F_ij = sigma^2 b_i b_j  -  kappa_i kappa_j rho_iX rho_jX      (rho_ij = 0)

whose second term is the unidentified part.

THE WALK. Fix phi_i and b_i at their fitted values and choose a new rho. Then

    phi^2 = sigma^2 b^2 + kappa^2 (1 - rho^2)

    =>  kappa = sqrt((phi^2 - sigma^2 b^2) / (1 - rho^2))
        beta  = b - kappa rho / sigma

which is just the regression decomposition: explained variance sigma^2 b^2 plus
residual kappa^2 (1 - rho^2). Every scenario below reprices a DIFFERENT
(beta_i, kappa_i, rho_iX) that fits the data exactly as well as the fitted one.

COMMON RANDOM NUMBERS. Every scenario prices on the same seed and the same path
count, so a difference between rows is signal, not Monte-Carlo noise. Do not
compare these numbers against a run made with a different seed.

    python poc/rhoix_sensitivity.py --window 20090309:20090915
    python poc/rhoix_sensitivity.py --names CCL,RCL,LUV --window 20200407:20210219
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poc.autocallable_stress as A                            # noqa: E402


def identified(name, sigma):
    """(phi_i, b_i) - the two combinations the data actually pins down."""
    d = A.IDIO[name]
    b, k, r = d["dBETAI"], d["dKAPPAI"], d["dRHOIX"]
    phi = np.sqrt((sigma * b) ** 2 + 2 * sigma * b * k * r + k * k)
    return float(phi), float(b + k * r / sigma)


def resolve(phi, b_diff, rho, sigma):
    """(beta, kappa) that reproduce phi and b_diff at this rho."""
    resid = phi * phi - (sigma * b_diff) ** 2
    if resid < 0:
        raise SystemExit("phi^2 < (sigma*b)^2 - no residual variance left; the "
                         "fitted pair is inconsistent with this sigma")
    k = np.sqrt(resid / (1.0 - rho * rho))
    return float(b_diff - k * rho / sigma), float(k)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default=None)
    ap.add_argument("--vintage", default="full", choices=tuple(A.IDIO_BY_VINTAGE))
    ap.add_argument("--window", default=None, metavar="BEG:END")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--arm", default=A.ARM)
    ap.add_argument("--paths", type=int, default=200000)
    ap.add_argument("--periods", type=int, default=12)
    ap.add_argument("--autocall", type=float, default=1.0)
    ap.add_argument("--coupon-barrier", type=float, default=0.70)
    ap.add_argument("--coupon", type=float, default=0.02)
    ap.add_argument("--protection", type=float, default=0.50)
    ap.add_argument("--notional", type=float, default=1_000_000.)
    ap.add_argument("--rate", type=float, default=0.03)
    ap.add_argument("--dividend", type=float, default=0.0)
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.05, 0.95],
                    help="PRIOR quantiles of rho_iX to walk to. The prior is "
                         "wide, so these show the structural size of the flat "
                         "direction; --rhos is the realistic band.")
    ap.add_argument("--rhos", type=float, nargs="+", default=None,
                    help="explicit rho_iX values instead of prior quantiles, "
                         "e.g. --rhos 0.20 0.40 for a band around the fit")
    a = ap.parse_args()
    a.engine, a.horizon, a.translate = "bsm", 1, True
    a.vol_bump, a.corr_bump = 0.01, 0.05
    a.iv_strike, a.iv_expiry = 100.0 * a.protection, a.periods / 4.0

    if a.names:
        A.set_names([x.strip() for x in a.names.split(",") if x.strip()])
    if a.window:
        beg, _, end = a.window.partition(":")
        A.set_window(beg, end, a.arm)
    elif a.as_of:
        A.set_asof(a.as_of, a.arm)
    else:
        A.set_vintage(a.vintage)

    sigma = A.SYS_Q["sigma"]
    base = {n: identified(n, sigma) for n in A.NAMES}
    fitted = {n: A.IDIO[n]["dRHOIX"] for n in A.NAMES}

    # The prior is 2*Beta(5,2) - 1, per the estimator's report.
    rhos = [("fitted", None)]
    if a.rhos:
        rhos += [("set", float(r)) for r in a.rhos]
    else:
        for q in a.quantiles:
            rhos.append(("prior q%02d" % round(100 * q),
                         float(2 * beta_dist.ppf(q, 5, 2) - 1)))

    spot0 = np.full(len(A.NAMES), 100.)
    print()
    print("=" * 78)
    print("rho_iX sensitivity :: %s :: %s" % (" / ".join(A.NAMES), A.VINTAGE))
    print("=" * 78)
    print("  sigma %.4f   fitted rho_iX: %s"
          % (sigma, ", ".join("%s %.4f" % (n, fitted[n]) for n in A.NAMES)))
    print("  phi_i and b_i are HELD at their fitted values in every row;")
    print("  beta_i and kappa_i are re-solved so the fit is unchanged.")
    print("  Common random numbers: differences are signal, not MC noise.")
    print()
    print("  %-11s %8s %9s %9s %26s %13s %13s"
          % ("rho_iX", "phi chk", "kappa", "beta", "R_ij", "note", "put"))
    print("  " + "-" * 96)

    rows = []
    for label, rho in rhos:
        for n in A.NAMES:
            phi, b = base[n]
            r = fitted[n] if rho is None else rho
            bet, kap = resolve(phi, b, r, sigma)
            A.IDIO[n] = dict(A.IDIO[n], dBETAI=bet, dKAPPAI=kap, dRHOIX=r)
        rij, ph = A.total_diffusion_corr(sigma)
        drift = max(abs(ph[i] - base[n][0]) for i, n in enumerate(A.NAMES))
        put, pv = A.put_value(spot0, spot0, a)
        shown = fitted[A.NAMES[0]] if rho is None else rho
        print("  %-11s %8.1e %9.4f %9.4f  %24s %13s %13s"
              % (label + (" %.3f" % shown), drift,
                 A.IDIO[A.NAMES[0]]["dKAPPAI"], A.IDIO[A.NAMES[0]]["dBETAI"],
                 " ".join("%.4f" % v for v in rij),
                 format(pv, ",.0f"), format(put, ",.0f")))
        rows.append((label, shown, pv, put, list(rij)))

    print()
    print("  phi chk is max |phi_i(row) - phi_i(fitted)|; it must be ~1e-15. If")
    print("  it is, the walk stayed on the flat direction and every row fits")
    print("  the data exactly as well as the fitted one.")
    print()
    print("  Move against the fitted row, on common random numbers:")
    print()
    print("  %-17s %13s %13s %10s %10s"
          % ("rho_iX", "d note", "d put", "d put %", "d R_ij"))
    print("  " + "-" * 68)
    b_pv, b_put, b_rij = rows[0][2], rows[0][3], rows[0][4]
    worst = 0.0
    for label, shown, pv, put, rij in rows[1:]:
        dr = max(abs(x - y) for x, y in zip(rij, b_rij))
        pct = 100.0 * (put - b_put) / b_put if b_put else float("nan")
        worst = max(worst, abs(pct))
        print("  %-17s %13s %13s %9.2f%% %10.4f"
              % (label + " %.3f" % shown, format(pv - b_pv, "+,.0f"),
                 format(put - b_put, "+,.0f"), pct, dr))
    print()
    if worst < 1.0:
        print("  Under 1% across the prior's whole plausible range. The")
        print("  non-identification is a limitation to state, not a result to")
        print("  qualify: phi_i and the translation are untouched by")
        print("  construction, and R_ij carries too little of it to matter.")
    else:
        print("  Above 1%. R_ij carries enough of the unidentified direction to")
        print("  affect the answer, so the prices need an interval rather than a")
        print("  point - or the joint estimation that would identify rho_iX.")
    print()


if __name__ == "__main__":
    main()
