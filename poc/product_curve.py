"""Does the two-beta translation beat a regression beta? The decisive test.

    python poc/product_curve.py --names AAPL,MSFT,KO --anchor hybrid --priors skew
    python poc/product_curve.py --names @poc/names.txt --date 20260821

Prints, per name, exactly what the proposed Excel function would return - and
then the one statistic that decides whether the function is worth shipping.

--------------------------------------------------------------------------
The test
--------------------------------------------------------------------------
The model gives a name TWO loadings on the systematic factor:

    b_diff = beta_i + kappa_i rho_iX / sigma      ordinary days
    gamma_i                                       gaps
    b_eff(u) = b_diff + (gamma_i - b_diff) w(u)   what applies at shock u

A regression beta on daily returns is set by the ~247 diffusion days in a
252-day window, so it estimates roughly b_diff and is blind to gamma_i.

That is only worth selling if gamma_i carries information b_diff does not.
Three ways it could fail, and this script reports all three:

  1. NO CURVATURE. If b_eff(-20%) / b_eff(-1%) is near 1 for every name, the
     multiplier does not move with shock size and the whole construction
     collapses to a single beta.

  2. NO SPREAD. If gamma_i is close to b_diff for every name, the gap loading
     is the ordinary loading and there is nothing extra to report.

  3. NO CROSS-SECTIONAL INFORMATION - the one that actually kills it. If
     gamma_i / b_diff is roughly the SAME CONSTANT for every name, then a risk
     manager reproduces every number here by multiplying their existing beta by
     that constant. The model would be right and still worthless, because the
     per-name part is not per-name. What the product needs is for that ratio to
     VARY: some names gap harder than their beta implies, others softer.

So the headline number is the cross-sectional dispersion of gamma_i / b_diff,
not the level of anything.

The shock grid includes the model's own h-day systematic ES(99.7%), so the
columns are shocks that can actually occur rather than round numbers.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.DataAccess import get_pmle_params, available_pmle_dates   # noqa: E402
from Library.RiskEngineKimYi2025 import (                              # noqa: E402
    SYSTEMATIC_PRIOR_SETS, FULL_SAMPLE,
)
from poc.translate_shock import (                                      # noqa: E402
    diffusive_beta, effective_beta, conditional_moments,
)
from poc.estimate_idiosyncratic import name_store_id, full_sample_series  # noqa: E402
from poc.estimate_systematic import store_id, SYSTEMATIC_ID            # noqa: E402

SYS_KEYS = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]

# Model h-day systematic ES(99.7%) under the FULL_SAMPLE anchor, from
# the simulation in the ES/horizon work. Used only to place the grid.
ES_997 = {1: -0.0823, 2: -0.0987, 5: -0.1277, 10: -0.1606}
GRID = [-0.01, -0.02, ES_997[1], -0.10, ES_997[10], -0.20]
GRID_LABEL = ["-1%", "-2%", "ES1d", "-10%", "ES10d", "-20%"]


def load(name, date, anchor, tag, priors):
    drawer = name_store_id(name, anchor, tag, priors)
    dates = available_pmle_dates(drawer)
    if not dates:
        return None, None
    d = date if (date and date in dates) else dates[-1]
    s = get_pmle_params(d, drawer)
    idio = {k: float(s[k]) for k in ("dBETAI", "dKAPPAI", "dGAMMAI",
                                     "dRHOIX", "dMUI")}
    sysp = {k: float(s[k]) for k in SYS_KEYS}
    return dict(date=d, **idio, **sysp), d


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True,
                    help="comma-separated tickers, or @file")
    ap.add_argument("--date", default=None,
                    help="valuation date YYYYMMDD; default the latest fitted")
    ap.add_argument("--anchor", choices=("hybrid", "rolling", "full"),
                    default="hybrid")
    ap.add_argument("--priors", choices=tuple(SYSTEMATIC_PRIOR_SETS),
                    default="skew")
    a = ap.parse_args()

    if a.names.startswith("@"):
        with open(a.names[1:]) as fh:
            names = [x.strip().upper() for x in fh if x.strip()
                     and not x.startswith("#")]
    else:
        names = [x.strip().upper() for x in a.names.split(",") if x.strip()]

    priors = SYSTEMATIC_PRIOR_SETS[a.priors]

    rows, missing = [], []
    for nm in names:
        p, d = load(nm, a.date, a.anchor, a.priors, priors)
        if p is None:
            missing.append(nm)
            continue
        rows.append((nm, p))
    for nm in missing:
        print("  no fits on disk for %s (drawer %s)"
              % (nm, name_store_id(nm, a.anchor, a.priors, priors)))
    if not rows:
        raise SystemExit("nothing to report - run poc/estimate_idiosyncratic.py first")

    print("=" * 78)
    print("What the Excel function would return   (anchor %s, priors %s)"
          % (a.anchor, a.priors))
    print("=" * 78)
    hdr = "  %-7s %7s %7s" % ("name", "b_diff", "gamma")
    for lab in GRID_LABEL:
        hdr += " %8s" % lab
    print(hdr + "     <- b_eff(u)")
    print("  " + "-" * 74)

    summary = []
    for nm, p in rows:
        bd = diffusive_beta(p["dBETAI"], p["dKAPPAI"], p["dRHOIX"], p["dSIGMA"])
        line = "  %-7s %7.3f %7.3f" % (nm, bd, p["dGAMMAI"])
        beffs = []
        for u in GRID:
            be, _ = effective_beta(u, p["dBETAI"], p["dKAPPAI"], p["dRHOIX"],
                                   p["dGAMMAI"], p["dSIGMA"], p["dLAMB"],
                                   p["dPPROB"], p["dETA1"], p["dETA2"])
            beffs.append(be)
            line += " %8.3f" % be
        print(line)
        summary.append(dict(name=nm, b_diff=bd, gamma=p["dGAMMAI"],
                            curv=beffs[-1] / beffs[0] if beffs[0] else np.nan,
                            ratio=p["dGAMMAI"] / bd if bd else np.nan,
                            date=p["date"]))

    print("\n  the same rows as SHOCKED RETURNS (b_eff(u) * u):")
    print("  " + "-" * 74)
    for nm, p in rows:
        line = "  %-7s %15s" % (nm, "")
        for u in GRID:
            m, sd, _ = conditional_moments(
                u, p["dBETAI"], p["dKAPPAI"], p["dRHOIX"], p["dGAMMAI"],
                p["dSIGMA"], p["dLAMB"], p["dPPROB"], p["dETA1"], p["dETA2"],
                mui=p["dMUI"])
            line += " %7.1f%%" % (100 * m)
        print(line)
    print("  (conditional sd omitted above; it is large and reported below)")

    df = pd.DataFrame(summary)
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    print("\n1. CURVATURE  b_eff(-20%%) / b_eff(-1%%)")
    print("   median %.3f   range %.3f - %.3f"
          % (df.curv.median(), df.curv.min(), df.curv.max()))
    print("   near 1.0 for every name  ->  the multiplier does not move;")
    print("   the whole two-beta construction collapses to one beta.")

    print("\n2. SPREAD  gamma_i vs b_diff")
    print("   b_diff  median %.3f  range %.3f - %.3f"
          % (df.b_diff.median(), df.b_diff.min(), df.b_diff.max()))
    print("   gamma   median %.3f  range %.3f - %.3f"
          % (df.gamma.median(), df.gamma.min(), df.gamma.max()))

    print("\n3. CROSS-SECTIONAL INFORMATION  gamma_i / b_diff   <- THE TEST")
    r = df.ratio.dropna()
    print("   median %.3f   sd %.3f   CV %.3f   range %.3f - %.3f"
          % (r.median(), r.std(), r.std() / abs(r.mean()), r.min(), r.max()))
    if len(df) > 2:
        rho = df[["b_diff", "gamma"]].corr(method="spearman").iloc[0, 1]
        print("   Spearman rank corr(b_diff, gamma) = %+.3f" % rho)
        print("   near +1.0 with a tight CV means gamma_i just re-scales beta:")
        print("   a client reproduces every number by multiplying their own")
        print("   beta by %.2f, and there is no product." % r.median())
    print("""
   What you need to see: a CV well above ~0.15 and a rank correlation
   materially below 1. That is names re-ordering between ordinary and gap
   conditions - something a regression beta cannot produce, and the only
   version of this that a risk manager cannot rebuild in one cell.""")

    out = os.path.join(_REPO_ROOT, "poc", "product_curve_%s_%s.csv"
                       % (a.priors, a.anchor))
    df.to_csv(out, index=False)
    print("\n  written to %s" % os.path.relpath(out, _REPO_ROOT))


if __name__ == "__main__":
    main()
