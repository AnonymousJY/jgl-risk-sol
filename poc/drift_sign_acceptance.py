"""Acceptance test for the drift sign flip in Theorem 3.1.

The cross term in m_i changed from -sigma*beta_i*kappa_i*rho_iX to
+sigma*beta_i*kappa_i*rho_iX in both KimYiLogLike._drift() (estimation) and the
KimYiDistribution.drift_dt setter (simulation). This script says what that
should and should not do, and then checks it against a re-run.

WHAT SHOULD HAPPEN, AND WHY.

mui enters the idiosyncratic likelihood ONLY through _drift(), and it enters
additively, with an unconstrained prior. So the flip is an exact
reparameterisation

    (mui, theta) -> (mui + 2 sigma beta_i kappa_i rho_iX, theta),

whose Jacobian is 1. The profile likelihood over theta = (beta_i, kappa_i,
rho_iX, gamma_i) is therefore IDENTICAL, and the only asymmetry between the two
fits is the prior ratio at the shifted mui - of order mui * 0.03 log units
against a likelihood over hundreds of observations.

    dMUI          falls by exactly 2 * dSIGMA * dBETAI * dKAPPAI * dRHOIX
    everything else                unchanged to MCMC noise

AND THE VaR SHOULD NOT MOVE AT ALL. est_liquidity_process() forms
Psi = cumsum(r) - drift * t and random() forms r = dPsi + drift, so the two are
exact inverses. Re-fitting mui under the new convention reproduces the same
drift_dt, hence the same simulated paths. If VaR moves, the two sites were not
flipped in step.

THE ONE THING THAT CAN SPOIL IT. Scripts/mcmc_empirical_bayes.py builds the
mui prior as Normal(mean of historical dMUI, prior_scale * sd of historical
dMUI). That history was produced under the old sign, so it now sits
2 sigma beta kappa rho too high. If sd(dMUI) across the history is comparable
to the shift the prior is wide enough to be pulled and nothing else moves; if
it is several times smaller, the prior fights the likelihood and beta_i and
kappa_i absorb part of the shift - which is exactly the contamination this
whole exercise is trying to avoid. `predict` reports that ratio.

usage
    python poc/drift_sign_acceptance.py selftest
    python poc/drift_sign_acceptance.py predict <before_dir>
    python poc/drift_sign_acceptance.py verify  <before_dir> <after_dir>

<before_dir> and <after_dir> are searched recursively for
estimated_params_pmle_*.csv, so "Study/Estimated Parameters PMLE" works.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

KEY = ["dtVALUATION_DATE", "sUNDERLYING_NAME"]
CROSS = ["dSIGMA", "dBETAI", "dKAPPAI", "dRHOIX"]
OTHERS = ["dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX", "dALPHA", "dSIGMA",
          "dPPROB", "dLAMB", "dETA1", "dETA2"]


def load(directory):
    """Every estimated_params_pmle_*.csv under `directory`, one row each."""
    paths = sorted(glob.glob(os.path.join(directory, "**",
                                          "estimated_params_pmle_*.csv"),
                             recursive=True))
    frames = []
    for p in paths:
        if os.path.getsize(p) == 0:
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("no non-empty parameter CSVs under %r" % directory)
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in KEY + CROSS + ["dMUI"] if c not in df.columns]
    if missing:
        raise SystemExit("columns missing from the CSVs: %s" % missing)
    return df.sort_values(KEY).reset_index(drop=True)


def shift(df):
    """2 * sigma * beta_i * kappa_i * rho_iX, the amount dMUI must fall by."""
    return 2.0 * df.dSIGMA * df.dBETAI * df.dKAPPAI * df.dRHOIX


def ci_width(df, col):
    lo, hi = col + "_CI_LOWER", col + "_CI_UPPER"
    if lo in df.columns and hi in df.columns:
        return (df[hi] - df[lo]).abs()
    return pd.Series(np.nan, index=df.index)


# ---------------------------------------------------------------- selftest
def selftest():
    """No data. Demonstrates the two claims directly."""
    rng = np.random.default_rng(0)
    n = 10
    sig = rng.uniform(0.10, 0.40, n)
    bet = rng.uniform(0.20, 2.00, n)
    kap = rng.uniform(0.05, 0.60, n)
    rho = rng.uniform(-0.95, 0.95, n)
    m_hat = rng.uniform(-0.20, 0.40, n)        # the empirical mean return

    # Estimation pins _drift()(mui) = m_hat under whichever convention is in
    # force, so solve for mui both ways.
    mui_old = m_hat - 0.5 * (sig * bet) ** 2 + sig * bet * kap * rho
    mui_new = m_hat - 0.5 * (sig * bet) ** 2 - sig * bet * kap * rho

    print("CLAIM 1  dMUI falls by exactly 2 sigma beta kappa rho")
    got = mui_new - mui_old
    want = -2.0 * sig * bet * kap * rho
    print("   max |observed - predicted| = %.3e   %s"
          % (np.abs(got - want).max(),
             "PASS" if np.allclose(got, want) else "**FAIL**"))

    print("\nCLAIM 2  the simulated drift is unchanged, so VaR is unchanged")
    drift_old = mui_old + 0.5 * (sig * bet) ** 2 - sig * bet * kap * rho
    drift_new = mui_new + 0.5 * (sig * bet) ** 2 + sig * bet * kap * rho
    print("   max |drift_new - drift_old| = %.3e   %s"
          % (np.abs(drift_new - drift_old).max(),
             "PASS" if np.allclose(drift_new, drift_old) else "**FAIL**"))
    print("   (both equal the empirical mean return, by construction)")

    print("\nCLAIM 3  flipping only ONE of the two sites breaks it")
    half = mui_new + 0.5 * (sig * bet) ** 2 - sig * bet * kap * rho
    err = np.abs(half - drift_old)
    print("   every simulated return shifts by 2 sigma beta kappa rho.")
    print("   over this random draw:  max %.4f /yr (%.2f bp/day),"
          "  median %.4f /yr (%.2f bp/day)"
          % (err.max(), err.max() / 252.0 * 1e4,
             np.median(err), np.median(err) / 252.0 * 1e4))
    print("   at the fitted scale 2 sigma beta kappa rho ~ 0.033 /yr this is")
    print("   1.3 bp/day - immaterial for a one-day VaR, 3.3% over a year.")


# ----------------------------------------------------------------- predict
def predict(before_dir):
    df = load(before_dir)
    d = shift(df)
    out = pd.DataFrame({
        "date": df.dtVALUATION_DATE,
        "name": df.sUNDERLYING_NAME,
        "dMUI_old": df.dMUI,
        "shift": -d,
        "dMUI_new": df.dMUI - d,
    })
    if "dMUI_CI_LOWER" in df.columns:
        out["shift/CIwidth"] = d / ci_width(df, "dMUI").replace(0, np.nan)
    print(out.to_string(index=False, float_format=lambda x: "%+.5f" % x))

    print("\nEMPIRICAL-BAYES PRIOR CHECK (Scripts/mcmc_empirical_bayes.py:107)")
    sd = df.dMUI.std(ddof=1)
    mean_shift = d.mean()
    print("   sd of historical dMUI          %+.5f" % sd)
    print("   mean shift to be absorbed      %+.5f" % -mean_shift)
    if sd > 0:
        print("   shift / prior sd               %.2f" % (mean_shift / sd))
        if mean_shift / sd > 1.0:
            print("   -> the EB prior on mui is NARROWER than the shift. It will")
            print("      fight the likelihood and beta_i / kappa_i will absorb")
            print("      part of it. Regenerate the dMUI history under the new")
            print("      sign before running the EB arm, or run the plain")
            print("      Normal(0, 1) arm instead.")
        else:
            print("   -> the EB prior is wide enough; nothing else should move.")


# ------------------------------------------------------------------ verify
def verify(before_dir, after_dir):
    a = load(before_dir)
    b = load(after_dir)
    m = a.merge(b, on=KEY, suffixes=("_b", "_a"))
    if m.empty:
        raise SystemExit("no (date, underlying) rows in common")
    print("%d rows matched\n" % len(m))

    pred = 2.0 * m.dSIGMA_b * m.dBETAI_b * m.dKAPPAI_b * m.dRHOIX_b
    got = m.dMUI_b - m.dMUI_a
    tol = np.maximum(0.15 * pred.abs(), 0.01)
    ok = (got - pred).abs() <= tol
    print("(a) dMUI fell by 2 sigma beta kappa rho")
    print(pd.DataFrame({
        "name": m.sUNDERLYING_NAME,
        "predicted_fall": pred,
        "actual_fall": got,
        "ok": np.where(ok, "PASS", "**FAIL**"),
    }).to_string(index=False, float_format=lambda x: "%+.5f" % x))

    print("\n(b) nothing else moved  (tolerance: 10% of the 95% CI width)")
    rows = []
    for c in OTHERS:
        if c + "_b" not in m.columns:
            continue
        w = ((m.get(c + "_CI_UPPER_b", np.nan)
              - m.get(c + "_CI_LOWER_b", np.nan)).abs())
        move = (m[c + "_a"] - m[c + "_b"]).abs()
        lim = np.where(np.isfinite(w) & (w > 0), 0.10 * w,
                       0.02 * m[c + "_b"].abs().clip(lower=1e-8))
        rows.append((c, move.max(), float(np.nanmin(lim)),
                     "PASS" if (move <= lim).all() else "**FAIL**"))
    print(pd.DataFrame(rows, columns=["param", "max_move", "tolerance",
                                      "result"]).to_string(index=False))
    print("\n(c) VaR: compare the two runs' VaR output separately. It should")
    print("    be identical, not merely close - the drift is a pure round trip.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"selftest", "predict", "verify"}:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    if mode == "selftest":
        selftest()
    elif mode == "predict":
        predict(sys.argv[2])
    else:
        verify(sys.argv[2], sys.argv[3])
