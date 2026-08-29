"""
skew_target.py — invert: find η₁ and η₂ that achieve target skew steepening.

For each desired skew change in vol points (e.g. +1, +2, +5), bisect on a
single anti-symmetric knob α that moves

    η₁ -> η₁ · (1 + α)
    η₂ -> η₂ · (1 - α)

and reports the resulting η values, the new σ, and the achieved skew change.

Two tables are printed:

  (1) Raw — α is solved so the skew slope IV(90%) − IV(110%) increases by
      the target. ATM is allowed to move; the script reports by how much.

  (2) ATM-anchored — for the same α candidate σ is bisected inside the loop
      so IV(100%) stays at the baseline value. ΔIV100 ≈ 0 by construction;
      Δskew is "pure" skew steepening at fixed ATM.

Both runs use the same scalar α-knob; change `perturb()` to use a different
parameterization (e.g. η₂-only, or proportional joint scaling) and the rest
of the script still works.

Run from the repository root:

    python Scripts/skew_target.py
"""

import os
import sys
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.DataAccess import get_pmle_params
from Library.OptionPricerKimYi2025 import kimyi_call
from Library.OptionPricerBSM1973 import bsm_call_price
from Library.RootFinder import bisection

# ---------- baseline calibration -------------------------------------------
BASELINE_DATE = "20250409"
p = get_pmle_params(BASELINE_DATE, "^SPX")
BASELINE = dict(
    kappai=0.0, gammai=1.0, betai=1.0, rhoix=0.0,
    sigma=float(p.dSIGMA),
    pprob=float(p.dPPROB),
    lamb=float(p.dLAMB),
    eta1=float(p.dETA1),
    eta2=float(p.dETA2),
)

S0 = 100.0
T = 30.0 / 365.0
R = 0.05
Q = 0.0

SKEW_TARGETS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]   # vol points of steepening

# bracket for the outer α-bisection: α ∈ (α_lo, α_hi)
ALPHA_LO = -0.50
ALPHA_HI = 0.95


# ---------- helpers (self-contained; mirror skew_sensitivity.py) -----------
def _arr(x):
    return np.asarray(x, dtype=np.float64).reshape((-1,))


def kimyi_call_price(params, K):
    kwargs = {k: _arr([v]) for k, v in params.items()}
    return kimyi_call(
        und_price=_arr([S0]), und_strike=_arr([K]),
        risk_free_rate=_arr([R]), dividend_yield=_arr([Q]),
        time_to_expiry=_arr([T]),
        **kwargs,
    ).item()


def bsm_iv(price, K, vol_lo=0.001, vol_hi=5.0, vol_init=0.20):
    def f(sigma):
        return bsm_call_price(
            und_price=_arr([S0]), und_strike=_arr([K]),
            volatility=sigma,
            risk_free_rate=_arr([R]), dividend_yield=_arr([Q]),
            time_to_expiry=_arr([T]),
        ).reshape((-1, 1))
    return bisection(
        func=f,
        lower_value=np.array([[vol_lo]]),
        upper_value=np.array([[vol_hi]]),
        target_value=np.array([[price]]),
        initial_value=np.array([[vol_init]]),
    ).item()


def anchor_atm(params, target_atm_iv, sigma_lo=1e-4, sigma_hi=3.0):
    def f(sigma_arr):
        scen = {**params, "sigma": float(sigma_arr.flat[0])}
        return np.array([[bsm_iv(kimyi_call_price(scen, S0), S0)]])
    new_sigma = bisection(
        func=f,
        lower_value=np.array([[sigma_lo]]),
        upper_value=np.array([[sigma_hi]]),
        target_value=np.array([[target_atm_iv]]),
        initial_value=np.array([[params["sigma"]]]),
    ).item()
    return {**params, "sigma": new_sigma}


# ---------- α-parameterization ---------------------------------------------
def perturb(alpha):
    """Anti-symmetric η bump: η₁ up by (1+α), η₂ down by (1-α). α=0 is baseline.
    Replace this function to try a different parameterization (e.g. η₂-only)
    while keeping the rest of the script identical."""
    return {
        **BASELINE,
        "eta1": BASELINE["eta1"] * (1.0 + alpha),
        "eta2": BASELINE["eta2"] * (1.0 - alpha),
    }


def skew_for_alpha(alpha, anchor, target_atm):
    """Skew slope IV(90%) − IV(110%) in vol points for a given α."""
    params = perturb(alpha)
    if anchor:
        params = anchor_atm(params, target_atm)
    iv90 = bsm_iv(kimyi_call_price(params, 0.90 * S0), 0.90 * S0)
    iv110 = bsm_iv(kimyi_call_price(params, 1.10 * S0), 1.10 * S0)
    return (iv90 - iv110) * 100.0, params


def find_alpha(target_dskew, baseline_skew, anchor, target_atm,
               alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI):
    """Bisect α so the resulting skew slope = baseline + target_dskew."""
    target_skew = baseline_skew + target_dskew

    def f(alpha_arr):
        sk, _ = skew_for_alpha(float(alpha_arr.flat[0]), anchor, target_atm)
        return np.array([[sk]])

    alpha = bisection(
        func=f,
        lower_value=np.array([[alpha_lo]]),
        upper_value=np.array([[alpha_hi]]),
        target_value=np.array([[target_skew]]),
        initial_value=np.array([[0.0]]),
    ).item()
    return alpha


# ---------- report ---------------------------------------------------------
def _build_row(target, alpha, params, base_iv):
    iv90 = bsm_iv(kimyi_call_price(params, 0.90 * S0), 0.90 * S0)
    iv100 = bsm_iv(kimyi_call_price(params, 1.00 * S0), 1.00 * S0)
    iv110 = bsm_iv(kimyi_call_price(params, 1.10 * S0), 1.10 * S0)
    return dict(
        target=target, alpha=alpha,
        eta1=params["eta1"], eta2=params["eta2"], sigma=params["sigma"],
        iv90=iv90*100, iv100=iv100*100, iv110=iv110*100,
        div100=(iv100 - base_iv[1.00])*100,
        skew=(iv90 - iv110)*100,
        dskew=(iv90 - iv110)*100 - (base_iv[0.90] - base_iv[1.10])*100,
    )


def _print_table(title, rows):
    print(f"\n=== {title} ===")
    hdr = (f"{'target dskew':>12s}  {'alpha':>8s}  "
           f"{'eta1':>8s}  {'eta2':>8s}  {'sigma':>7s}  "
           f"{'IV100':>6s}  {'dIV100':>7s}  {'skew':>6s}  {'dskew':>7s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['target']:+12.2f}  {r['alpha']:+8.4f}  "
              f"{r['eta1']:8.3f}  {r['eta2']:8.3f}  {r['sigma']:7.4f}  "
              f"{r['iv100']:6.2f}  {r['div100']:+7.2f}  {r['skew']:6.2f}  {r['dskew']:+7.2f}")


def main():
    print(f"\nBaseline: SPX {BASELINE_DATE}   "
          f"sigma={BASELINE['sigma']:.3f}  lamb={BASELINE['lamb']:.2f}  "
          f"eta1={BASELINE['eta1']:.2f}  eta2={BASELINE['eta2']:.2f}  p={BASELINE['pprob']:.2f}")
    print(f"Option: S={S0}  T={T*365:.0f}d  r={R:.2%}  q={Q:.2%}")

    # baseline IV curve + skew
    base_iv = {m: bsm_iv(kimyi_call_price(BASELINE, m * S0), m * S0)
               for m in (0.90, 1.00, 1.10)}
    base_skew = (base_iv[0.90] - base_iv[1.10]) * 100
    target_atm = base_iv[1.00]
    print(f"Baseline IVs (vol pts): IV90={base_iv[0.90]*100:.2f}  "
          f"IV100={base_iv[1.00]*100:.2f}  IV110={base_iv[1.10]*100:.2f}  "
          f"skew={base_skew:+.2f}")

    print("\nParameterization: eta1 -> eta1*(1+α),  eta2 -> eta2*(1-α)  "
          "(anti-symmetric; α=0 is baseline)")

    # --- raw: no ATM compensation -----------------------------------------
    raw_rows = []
    for tgt in SKEW_TARGETS:
        alpha = find_alpha(tgt, base_skew, anchor=False, target_atm=target_atm)
        params = perturb(alpha)
        raw_rows.append(_build_row(tgt, alpha, params, base_iv))
    _print_table("Raw α solutions (ATM allowed to move; dIV100 shows the drift)",
                 raw_rows)

    # --- anchored: σ bisected inside α-loop to hold IV100 ------------------
    anchored_rows = []
    for tgt in SKEW_TARGETS:
        alpha = find_alpha(tgt, base_skew, anchor=True, target_atm=target_atm)
        params = anchor_atm(perturb(alpha), target_atm)
        anchored_rows.append(_build_row(tgt, alpha, params, base_iv))
    _print_table("ATM-anchored α solutions (σ bisected so dIV100 ≈ 0)",
                 anchored_rows)


if __name__ == "__main__":
    main()
