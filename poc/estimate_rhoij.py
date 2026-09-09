"""Step 1c :: cross-name correlation rho_{i,j} for the simulation.

    python poc/estimate_rhoij.py --names C,BAC,JPM
    python poc/estimate_rhoij.py --names @poc/names.txt --step 21
    python poc/estimate_rhoij.py --names C,BAC,JPM --report-only

rho_{i,j} is not produced by the P-MLE, which fits one name at a time. This
estimates it as the realised correlation of the pair's returns over the same
252-day window the per-name fits use, on the same aligned calendar.

KimYiRiskEngine consumes it as the correlation of the TOTAL diffusion: the
simulation draws one Gaussian per name and couples them only through the
Cholesky factor L, then scales by phi_i. So the number in that slot is the
correlation of r_i with r_j, not a residual after removing the systematic
factor.

The shared jump Y carries its own covariance gamma_i*gamma_j*lamb*E[Y^2] on
top of L. --deduct-jump subtracts that from the realised covariance before
normalising; without it the simulated correlation exceeds the realised one by
that amount.

Output: one CSV, dates down, pairs across, in np.triu_indices(n, k=1) order -
the order get_corr_mat reads. rhoij_params() turns a row into the list of
ParametersConstant the engine's rhoij argument takes.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.Parameters import ParametersConstant              # noqa: E402
from Library.StatisticsMC import get_corr_mat                  # noqa: E402
from Library.DataAccess import get_aligned_price_panel         # noqa: E402
from Library.StudyWindow import (                              # noqa: E402
    SYSTEMATIC_ID, DATE_FMT, LOOKBACK, BASE_DAYS, BEG, END, valuation_dates,
)

SEP = "|"


def pair_columns(names):
    """Pair labels in np.triu_indices order - the order get_corr_mat reads."""
    rows, cols = np.triu_indices(len(names), k=1)
    return [names[i] + SEP + names[j] for i, j in zip(rows, cols)]


def out_path(names, lookback, deduct_jump):
    tag = "_dedj" if deduct_jump else ""
    stem = "-".join(names) if len(names) <= 6 else "%dnames" % len(names)
    return os.path.join(_REPO_ROOT, "poc",
                        "rho_ij_%s_%d%s.csv" % (stem, lookback, tag))


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------
def window_corr(rets, names, end_ts, lookback):
    """Correlation of the pair's returns over the trailing window.

    Complete cases only: every name must be observed on every date used. A
    correlation built that way is a Gram matrix and is positive semi-definite
    by construction, which is what the engine's Cholesky needs. Pairwise
    deletion would give a matrix that need not be.
    """
    block = rets.loc[rets.index <= end_ts, names].dropna(how="any")
    if len(block) < lookback:
        return None, len(block)
    block = block.iloc[-lookback:]
    return block.corr().to_numpy(), len(block)


def jump_covariance(names, gammai, lamb, pprob, eta1, eta2):
    """gamma_i * gamma_j * lamb * E[Y^2] per unit time, as a full matrix."""
    ey2 = 2.0 * pprob / eta1**2 + 2.0 * (1.0 - pprob) / eta2**2
    g = np.array([gammai[n] for n in names], dtype=float).reshape((-1, 1))
    return (g @ g.T) * lamb * ey2


def deduct(corr, sd, jump_cov, dt):
    """Remove the shared-jump covariance and renormalise to a correlation.

    sd is the realised per-period standard deviation of each name's returns,
    so corr * outer(sd, sd) is the realised covariance over one period. The
    jump term is quoted per unit time and scaled by dt to match.
    """
    cov = corr * np.outer(sd, sd) - jump_cov * dt
    d = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    out = cov / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return np.clip(out, -1.0, 1.0)


def nearest_psd(mat, floor=1e-8):
    """Eigenvalue clip, then renormalise back to unit diagonal."""
    w, v = np.linalg.eigh((mat + mat.T) / 2.0)
    if w.min() >= floor:
        return mat, float(w.min()), False
    fixed = v @ np.diag(np.clip(w, floor, None)) @ v.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed, float(w.min()), True


def run(names, dates, lookback, jump_cov=None):
    panel, _ = get_aligned_price_panel([SYSTEMATIC_ID] + names,
                                       reference=SYSTEMATIC_ID)
    rets = panel.pct_change().dropna(how="all")

    rows, cols = np.triu_indices(len(names), k=1)
    labels = pair_columns(names)
    dt = 1.0 / BASE_DAYS

    out, skipped = [], []
    for d in dates:
        ts = pd.to_datetime(d, format=DATE_FMT)
        corr, n_obs = window_corr(rets, names, ts, lookback)
        if corr is None:
            skipped.append((d, n_obs))
            continue

        if jump_cov is not None:
            block = rets.loc[rets.index <= ts, names].dropna(how="any")
            sd = block.iloc[-lookback:].std(ddof=1).to_numpy()
            corr = deduct(corr, sd, jump_cov, dt)

        corr, min_eig, repaired = nearest_psd(corr)
        rec = {"DATE": d, "N_OBS": n_obs, "MIN_EIG": min_eig,
               "REPAIRED": int(repaired)}
        rec.update(dict(zip(labels, corr[rows, cols])))
        out.append(rec)

    if skipped:
        print("  %d date(s) skipped for short history (first %s: %d complete "
              "observations, need %d)"
              % (len(skipped), skipped[0][0], skipped[0][1], lookback))
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).set_index("DATE")


# ---------------------------------------------------------------------------
# consumption
# ---------------------------------------------------------------------------
def rhoij_vector(row, names):
    """The upper triangle as a plain array, in get_corr_mat's order."""
    return np.array([float(row[lab]) for lab in pair_columns(names)])


def rhoij_params(row, names):
    """The list KimYiRiskEngine's rhoij argument takes."""
    return [ParametersConstant(np.array(v)) for v in rhoij_vector(row, names)]


def load(names, lookback=LOOKBACK, deduct_jump=False):
    return pd.read_csv(out_path(names, lookback, deduct_jump), index_col="DATE")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True,
                    help="comma separated, or @file with one per line")
    ap.add_argument("--beg", default=BEG)
    ap.add_argument("--end", default=END)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--deduct-jump", action="store_true",
                    help="subtract gamma_i*gamma_j*lamb*E[Y^2] before "
                         "normalising; needs --gammai and the systematic four")
    ap.add_argument("--gammai", default=None,
                    help="NAME=value,NAME=value for --deduct-jump")
    ap.add_argument("--lamb", type=float, default=None)
    ap.add_argument("--pprob", type=float, default=None)
    ap.add_argument("--eta1", type=float, default=None)
    ap.add_argument("--eta2", type=float, default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    if a.names.startswith("@"):
        with open(a.names[1:]) as fh:
            names = [ln.strip().upper() for ln in fh
                     if ln.strip() and not ln.startswith("#")]
    else:
        names = [n.strip().upper() for n in a.names.split(",") if n.strip()]
    if len(names) < 2:
        raise SystemExit("need at least two names")

    jump_cov = None
    if a.deduct_jump:
        missing = [k for k in ("gammai", "lamb", "pprob", "eta1", "eta2")
                   if getattr(a, k) is None]
        if missing:
            raise SystemExit("--deduct-jump needs --%s"
                             % ", --".join(missing))
        gammai = {}
        for item in a.gammai.split(","):
            k, v = item.split("=")
            gammai[k.strip().upper()] = float(v)
        absent = [n for n in names if n not in gammai]
        if absent:
            raise SystemExit("--gammai missing %s" % ", ".join(absent))
        jump_cov = jump_covariance(names, gammai, a.lamb, a.pprob,
                                   a.eta1, a.eta2)

    path = out_path(names, a.lookback, a.deduct_jump)

    print("=" * 72)
    print("Step 1c :: cross-name correlation")
    print("=" * 72)
    print("  names    : %s" % ", ".join(names))
    print("  pairs    : %d" % (len(names) * (len(names) - 1) // 2))
    print("  window   : %s -> %s every %d business days, %d-day lookback"
          % (a.beg, a.end, a.step, a.lookback))
    print("  jump term: %s" % ("deducted" if jump_cov is not None
                               else "left in - the engine adds it on top of L"))
    print("  file     : %s" % os.path.relpath(path, _REPO_ROOT))
    print()

    if a.report_only:
        if not os.path.exists(path):
            raise SystemExit("nothing on disk at %s" % path)
        df = pd.read_csv(path, index_col="DATE")
    else:
        df = run(names, valuation_dates(a.beg, a.end, a.step), a.lookback,
                 jump_cov)
        if not len(df):
            raise SystemExit("no date had %d complete observations"
                             % a.lookback)
        df.to_csv(path)
        with open(path.replace(".csv", ".json"), "w") as fh:
            json.dump({"names": names, "lookback": a.lookback,
                       "step": a.step, "beg": a.beg, "end": a.end,
                       "deduct_jump": bool(a.deduct_jump),
                       "pair_order": pair_columns(names)}, fh, indent=2)

    labels = pair_columns(names)
    print("  %d date(s) estimated, %s -> %s"
          % (len(df), df.index[0], df.index[-1]))
    if df["REPAIRED"].sum():
        print("  %d matrix(es) eigenvalue-clipped to restore PSD"
              % int(df["REPAIRED"].sum()))
    print()

    stats = df[labels].agg(["mean", "min", "max"]).T
    width = max(len(x) for x in labels)
    print("  %-*s   %8s %8s %8s" % (width, "pair", "mean", "min", "max"))
    print("  " + "-" * (width + 28))
    for lab, r in stats.iterrows():
        print("  %-*s   %8.4f %8.4f %8.4f"
              % (width, lab, r["mean"], r["min"], r["max"]))

    last = df.iloc[-1]
    mat = get_corr_mat(rhoij_vector(last, names).reshape((-1, 1)), len(names))
    np.linalg.cholesky(mat)
    print("\n  latest (%s), Cholesky OK:" % df.index[-1])
    print("  %-*s %s" % (width, "", " ".join("%8s" % n for n in names)))
    for i, n in enumerate(names):
        print("  %-*s %s" % (width, n,
                             " ".join("%8.4f" % v for v in mat[i])))


if __name__ == "__main__":
    main()
