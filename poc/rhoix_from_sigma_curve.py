"""Recover rho_iX from the curvature of phi_i against sigma. No new fits.

phi_i is identified by the PUBLISHED likelihood - it is the one combination of
beta_i, kappa_i and rho_iX that the name's own increments pin down, which is
why it moves only 1-2% across arms whose rho_iX priors differ by a factor of
three. sigma is identified by the systematic fit. Both are already on disk.

Holding beta_i, kappa_i and rho_iX fixed while sigma moves,

    phi_i^2(sigma) = beta_i^2 sigma^2 + (2 beta_i kappa_i rho_iX) sigma
                     + kappa_i^2

is a QUADRATIC in sigma: three coefficients, three unknowns, so

    beta_i  = sqrt(c2),   kappa_i = sqrt(c0),
    rho_iX  = c1 / (2 sqrt(c2) sqrt(c0))

with no prior on rho_iX anywhere and no change to the model.

TWO THINGS DECIDE WHETHER THIS WORKS, and both are reported before the answer.

  PRECISION ON phi_i. The design [sigma^2, sigma, 1] over sigma in
  0.08 to 0.26 has a condition number near 600, so the split is sensitive to
  noise in phi_i. On the real sigma path the recovery is unbiased at every
  noise level but the spread grows fast: at 0.5% noise rho_iX lands within
  +/-0.04, at 1% within +/-0.09, at 2% within +/-0.19 - which is no answer at
  all. The cross-ARM spread in phi_i is the honest estimate of that noise, so
  pass two or more --idio-priors and the run will measure it.

  OVERLAP. 504-day windows stepped 21 business days share 96% of their data.
  245 windows are not 245 observations - the effective count is closer to
  245 * 21/504, about ten. Standard errors therefore come from a MOVING BLOCK
  bootstrap with block length lookback/step, not from the regression formula,
  which would report intervals several times too tight.

    python poc/rhoix_from_sigma_curve.py --names C,BAC,JPM \\
        --idio-priors thm31-wide,econ,thm31-high
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import PMLE_DIR                              # noqa: E402
from Library.Logging import report as _report                        # noqa: E402
from Library.RiskEngineKimYi2025 import (                            # noqa: E402
    SYSTEMATIC_PRIOR_SETS, IDIOSYNCRATIC_PRIOR_SETS,
)
from Library.TableHeatmap import render as heat                      # noqa: E402
from poc.estimate_systematic import store_id                         # noqa: E402

BASE_DAYS = 252
_LOG = _report(__name__)


def _load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dtVALUATION_DATE"])
    return df.sort_values("dt").reset_index(drop=True)


def phi_series(idf, sdf):
    """phi_i per window, each from THAT window's sigma.

    Combining by-year means instead is what turns a working design into a
    broken one: phi_i^2 is nonlinear in its components, so a mean of
    components is not the component of means, and each window's phi_i belongs
    with its own sigma rather than the year's.
    """
    m = idf.merge(sdf[["dt", "dSIGMA"]], on="dt", how="inner")
    s, b = m["dSIGMA"].to_numpy(), m["dBETAI"].to_numpy()
    k, r = m["dKAPPAI"].to_numpy(), m["dRHOIX"].to_numpy()
    m["phi"] = np.sqrt(np.maximum((s * b) ** 2 + 2 * s * b * k * r + k ** 2, 0.0))
    return m[["dt", "dSIGMA", "phi"]]


def solve(sigma, phi2):
    """Least squares on [sigma^2, sigma, 1] -> (beta_i, kappa_i, rho_iX, r2)."""
    X = np.column_stack([sigma ** 2, sigma, np.ones_like(sigma)])
    c, *_ = np.linalg.lstsq(X, phi2, rcond=None)
    if c[0] <= 0 or c[2] <= 0:
        return (np.nan,) * 4
    beta = np.sqrt(c[0])
    kappa = np.sqrt(c[2])
    rho = c[1] / (2.0 * beta * kappa)
    resid = phi2 - X @ c
    denom = ((phi2 - phi2.mean()) ** 2).sum()
    return beta, kappa, rho, (1.0 - resid @ resid / denom if denom > 0 else np.nan)


def block_bootstrap(sigma, phi2, block, n=2000, seed=20240114):
    """Moving block bootstrap. Overlapping windows are not independent draws."""
    rng = np.random.default_rng(seed)
    N = len(sigma)
    block = max(1, min(int(block), N))
    starts = np.arange(N - block + 1)
    out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in rng.choice(starts,
                                                  size=int(np.ceil(N / block)))])[:N]
        out.append(solve(sigma[idx], phi2[idx]))
    a = np.array(out, dtype=float)
    return a[np.isfinite(a).all(axis=1)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True)
    ap.add_argument("--priors", default="alpha-pprob-eta-flat",
                    help="systematic arm supplying sigma")
    ap.add_argument("--idio-priors", default="thm31-wide",
                    help="comma-separated idiosyncratic arms. The FIRST is the "
                         "one estimated on; the rest measure phi_i's noise")
    ap.add_argument("--anchor", default="hybrid")
    ap.add_argument("--lookback", type=int, default=504)
    ap.add_argument("--step", type=int, default=21,
                    help="business days between windows, for the block length")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--color", dest="color", action="store_true", default=None)
    ap.add_argument("--no-color", dest="color", action="store_false")
    a = ap.parse_args()

    names = [s.strip().upper() for s in a.names.split(",") if s.strip()]
    arms = [s.strip() for s in a.idio_priors.split(",") if s.strip()]
    for arm in arms:
        if arm not in IDIOSYNCRATIC_PRIOR_SETS:
            raise SystemExit("unknown idiosyncratic arm %r" % arm)

    sdrawer = store_id(a.priors, SYSTEMATIC_PRIOR_SETS[a.priors], a.lookback)
    sdf = _load(os.path.join(PMLE_DIR, sdrawer, "*.csv"))
    if sdf is None:
        raise SystemExit("no systematic drawer %s" % sdrawer)

    _LOG.info("=" * 78)
    _LOG.info("rho_iX FROM THE CURVATURE OF phi_i AGAINST sigma")
    _LOG.info("systematic drawer %s" % sdrawer)
    _LOG.info("idiosyncratic arms: %s   (first is the estimate, rest measure "
              "noise)" % ", ".join(arms))
    _LOG.info("=" * 78)

    s = sdf["dSIGMA"].to_numpy()
    X = np.column_stack([s ** 2, s, np.ones_like(s)])
    block = max(1, int(round(a.lookback / float(a.step))))
    _LOG.info("\n  sigma %.4f to %.4f (%.1fx)   condition number %.0f"
              % (s.min(), s.max(), s.max() / s.min(), np.linalg.cond(X)))
    _LOG.info("  %d windows, %d-day lookback stepped %d days -> block length %d,"
              % (len(s), a.lookback, a.step, block))
    _LOG.info("  i.e. about %d independent observations, not %d."
              % (max(1, len(s) // block), len(s)))

    for nm in names:
        _LOG.info("\n" + "=" * 78)
        _LOG.info("%s" % nm)
        _LOG.info("=" * 78)
        series = {}
        for arm in arms:
            pat = os.path.join(PMLE_DIR,
                               "%s__%s__*%s*lb%d" % (nm, a.anchor,
                                                     arm.replace("-", ""),
                                                     a.lookback), "*.csv")
            hit = glob.glob(pat)
            if not hit:
                pat = os.path.join(PMLE_DIR, "*%s*%s*lb%d"
                                   % (nm, arm.replace("-", ""), a.lookback),
                                   "*.csv")
                hit = glob.glob(pat)
            if not hit:
                _LOG.info("  arm %-12s no drawer found - skipped" % arm)
                continue
            idf = _load(os.path.dirname(hit[0]) + "/*.csv")
            series[arm] = phi_series(idf, sdf)

        if not series:
            _LOG.info("  no drawers for %s" % nm)
            continue

        # --- how noisy is phi_i? the cross-arm spread is the measurement ---
        if len(series) > 1:
            base = None
            rel = {}
            for arm, d in series.items():
                v = d.set_index("dt")["phi"]
                if base is None:
                    base = v
                    continue
                j = pd.concat([base, v], axis=1, join="inner").dropna()
                rel[arm] = float((100 * (j.iloc[:, 1] / j.iloc[:, 0] - 1)).abs().median())
            _LOG.info("\n  phi_i across arms, median |relative difference| vs %s:"
                      % arms[0])
            worst = 0.0
            for arm, v in rel.items():
                _LOG.info("    %-14s %6.2f%%" % (arm, v))
                worst = max(worst, v)
            verdict = ("usable - rho_iX to about +/-0.05" if worst < 0.6 else
                       "marginal - rho_iX to about +/-0.10" if worst < 1.2 else
                       "TOO NOISY - the interval will not exclude anything")
            _LOG.info("    -> phi_i noise about %.2f%%: %s" % (worst, verdict))
        else:
            _LOG.info("\n  only one arm given, so phi_i's noise is not measured."
                      "\n  Pass more arms to --idio-priors to get that number.")

        d = series[arms[0]]
        sg = d["dSIGMA"].to_numpy()
        phi2 = d["phi"].to_numpy() ** 2
        beta, kappa, rho, r2 = solve(sg, phi2)
        if not np.isfinite(rho):
            _LOG.info("\n  the quadratic has a non-positive sigma^2 or constant"
                      " term, so beta_i or kappa_i has no real root."
                      "\n  That is the design failing, not an estimate.")
            continue

        bs = block_bootstrap(sg, phi2, block, n=a.boot)
        _LOG.info("\n  point estimates and BLOCK bootstrap intervals (%d draws)"
                  % len(bs))
        t = pd.DataFrame(index=["beta_i", "kappa_i", "rho_iX"])
        for i, nm2 in enumerate(t.index):
            col = bs[:, i]
            t.loc[nm2, "estimate"] = [beta, kappa, rho][i]
            t.loc[nm2, "boot_mean"] = col.mean()
            t.loc[nm2, "boot_sd"] = col.std(ddof=1)
            t.loc[nm2, "lo95"] = np.percentile(col, 2.5)
            t.loc[nm2, "hi95"] = np.percentile(col, 97.5)
        _LOG.info(heat(t.astype(float).round(4), decimals=4, color=a.color,
                       index_width=9))
        _LOG.info("  R2 %.4f   n %d windows   block %d" % (r2, len(sg), block))
        lo, hi = t.loc["rho_iX", "lo95"], t.loc["rho_iX", "hi95"]
        if hi - lo > 1.0:
            _LOG.info("  the 95%% interval spans %.2f - wider than most of the"
                      " admissible range, so this is not an estimate." % (hi - lo))
        elif lo > 0:
            _LOG.info("  the interval excludes zero from above: rho_iX > 0.")
        elif hi < 0:
            _LOG.info("  the interval lies below zero, which contradicts the"
                      " economic restriction rho_iX > 0 - read it as the design"
                      " failing rather than as a negative correlation.")
        else:
            _LOG.info("  the interval straddles zero: no sign can be claimed.")
    _LOG.info("")


if __name__ == "__main__":
    main()
