"""
skew_target_idi.py — per-asset solver for uniform idiosyncratic skew steepening.

A systematic perturbation in (η₁, η₂) cannot produce a uniform vol-point skew
shift across the cross-section, because each asset's response depends on its
own (γ_i, β_i, κ_i, ρ_{i,X}) and on its IV level. To get the same vol-point
steepening on every idiosyncratic name you need a per-asset adjustment.

This script does that. For each asset in IDI_ASSETS, and for each skew target
in SKEW_TARGETS, it solves a 2-D inverse problem:

    find (γ_i, κ_i)  such that
        IV(90%) − IV(110%) on asset i   =  baseline_skew + target_Δskew
        IV(100%) on asset i             =  baseline_ATM_IV

The outer loop bisects γ_i to hit the skew target; the inner loop bisects κ_i
to anchor that asset's ATM IV. Systematic parameters (η₁, η₂, σ, λ, p) stay
at the SPX calibration and are common across all assets.

Run from the repository root:

    python Scripts/skew_target_idi.py
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
from Scripts.load_portfolio import get_idiosyncratic_ids

# ---------- configuration ---------------------------------------------------
BASELINE_DATE = "20250409"
IDI_ASSETS = get_idiosyncratic_ids()   # e.g. ["COIN"]
SKEW_TARGETS = [0.5, 1.0, 2.0, 3.0, 5.0]   # vol points of steepening

S0 = 100.0
T = 30.0 / 365.0
R = 0.05
Q = 0.0

# bisection brackets
GAMMA_LO, GAMMA_HI = 0.10, 10.0
KAPPA_LO, KAPPA_HI = 1e-4, 3.0


# ---------- systematic baseline (common across all idiosyncratic assets) ---
p_sys = get_pmle_params(BASELINE_DATE, "^SPX")
SYS = dict(
    sigma=float(p_sys.dSIGMA),
    pprob=float(p_sys.dPPROB),
    lamb=float(p_sys.dLAMB),
    eta1=float(p_sys.dETA1),
    eta2=float(p_sys.dETA2),
)


def load_idi_baseline(asset, date):
    """Asset i's full parameter set: systematic + its own idiosyncratic fit."""
    p = get_pmle_params(date, asset)
    return {
        **SYS,
        "kappai": float(p.dKAPPAI),
        "gammai": float(p.dGAMMAI),
        "betai": float(p.dBETAI),
        "rhoix": float(p.dRHOIX),
    }


# ---------- pricing + IV inversion (same recipe as the other scripts) ------
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


def bsm_iv(price, K, vol_lo=1e-3, vol_hi=5.0, vol_init=0.20):
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


def iv_at(params, m):
    return bsm_iv(kimyi_call_price(params, m * S0), m * S0)


# ---------- inner solve: bisect κ_i so IV(100%) hits target ----------------
def anchor_atm_via_kappai(params, target_atm_iv,
                          kap_lo=KAPPA_LO, kap_hi=KAPPA_HI):
    def f(kap_arr):
        scen = {**params, "kappai": float(kap_arr.flat[0])}
        return np.array([[bsm_iv(kimyi_call_price(scen, S0), S0)]])
    new_kap = bisection(
        func=f,
        lower_value=np.array([[kap_lo]]),
        upper_value=np.array([[kap_hi]]),
        target_value=np.array([[target_atm_iv]]),
        initial_value=np.array([[params["kappai"]]]),
    ).item()
    return {**params, "kappai": new_kap}


# ---------- outer solve: bisect γ_i so Δskew hits target -------------------
def find_gammai(target_dskew, baseline_params, baseline_skew_pts, target_atm,
                gamma_lo=GAMMA_LO, gamma_hi=GAMMA_HI):
    """For one asset and one target, return (γ_i, κ_i) that delivers the
    target Δskew at fixed ATM IV."""
    target_skew = baseline_skew_pts + target_dskew

    def f(gam_arr):
        scen = {**baseline_params, "gammai": float(gam_arr.flat[0])}
        scen = anchor_atm_via_kappai(scen, target_atm)
        skew_pts = (iv_at(scen, 0.90) - iv_at(scen, 1.10)) * 100
        return np.array([[skew_pts]])

    new_gamma = bisection(
        func=f,
        lower_value=np.array([[gamma_lo]]),
        upper_value=np.array([[gamma_hi]]),
        target_value=np.array([[target_skew]]),
        initial_value=np.array([[baseline_params["gammai"]]]),
    ).item()

    # rebuild the final (γ_i, κ_i) pair for reporting
    final = {**baseline_params, "gammai": new_gamma}
    final = anchor_atm_via_kappai(final, target_atm)
    return final


# ---------- report ---------------------------------------------------------
def run_for_asset(asset):
    print(f"\n{'='*100}")
    print(f"Asset: {asset}   (idiosyncratic baseline loaded from {BASELINE_DATE})")
    print(f"{'='*100}")

    base = load_idi_baseline(asset, BASELINE_DATE)
    print(f"Baseline params: gamma_i={base['gammai']:.3f}  beta_i={base['betai']:.3f}  "
          f"kappa_i={base['kappai']:.3f}  rho_iX={base['rhoix']:.3f}")

    iv90 = iv_at(base, 0.90); iv100 = iv_at(base, 1.00); iv110 = iv_at(base, 1.10)
    base_skew_pts = (iv90 - iv110) * 100
    print(f"Baseline IVs (vol pts): IV90={iv90*100:.2f}  IV100={iv100*100:.2f}  "
          f"IV110={iv110*100:.2f}  skew={base_skew_pts:+.2f}")

    hdr = (f"\n{'target dskew':>12s}  {'gamma_i':>8s}  {'kappa_i':>8s}  "
           f"{'IV90':>6s}  {'IV100':>6s}  {'IV110':>6s}  "
           f"{'dIV100':>7s}  {'skew':>6s}  {'achieved':>9s}")
    print(hdr); print("-" * len(hdr))
    for tgt in SKEW_TARGETS:
        final = find_gammai(tgt, base, base_skew_pts, iv100)
        f_iv90  = iv_at(final, 0.90)
        f_iv100 = iv_at(final, 1.00)
        f_iv110 = iv_at(final, 1.10)
        f_skew_pts = (f_iv90 - f_iv110) * 100
        achieved = f_skew_pts - base_skew_pts
        print(f"{tgt:+12.2f}  {final['gammai']:8.3f}  {final['kappai']:8.4f}  "
              f"{f_iv90*100:6.2f}  {f_iv100*100:6.2f}  {f_iv110*100:6.2f}  "
              f"{(f_iv100 - iv100)*100:+7.2f}  {f_skew_pts:6.2f}  {achieved:+9.2f}")


def main():
    print(f"Systematic baseline (SPX {BASELINE_DATE}): "
          f"sigma={SYS['sigma']:.3f}  lamb={SYS['lamb']:.2f}  "
          f"eta1={SYS['eta1']:.2f}  eta2={SYS['eta2']:.2f}  p={SYS['pprob']:.2f}")
    print(f"Option: S={S0}  T={T*365:.0f}d  r={R:.2%}  q={Q:.2%}")
    print(f"Targets (vol points of skew steepening): {SKEW_TARGETS}")

    for asset in IDI_ASSETS:
        run_for_asset(asset)


if __name__ == "__main__":
    main()
