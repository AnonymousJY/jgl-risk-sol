"""
skew_sensitivity.py — translate KimYi jump parameters into skew vol points.

Reads the cached SPX P-MLE calibration, prices European calls at a moneyness
grid, inverts to Black-Scholes implied vols, and reports per-wing vol-point
shifts and the IV(90%) − IV(110%) skew slope under several parameter
perturbations.

Two tables are printed:

  (1) Raw scenarios — η, p, γ_i, κ_i, etc. perturbed with no compensation;
      ATM and the wings move freely. ΔIV90, ΔIV100, ΔIV110 show how each
      strike's implied vol shifts in vol points vs. the baseline.

  (2) ATM-anchored variants — for the same parameter perturbations, σ is
      bisected so that IV(100%) equals the baseline ATM IV. ΔIV100 is
      therefore ≈ 0 by construction; the remaining ΔIV90 and ΔIV110 isolate
      the pure shape change. The Δskew column is the steepening you actually
      bought at fixed ATM.

Run from the repository root after the conda env is built:

    python Scripts/skew_sensitivity.py

Edit BASELINE_DATE and the scenarios dict to experiment further.
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
BASELINE_DATE = "20250409"          # any cached SPX date works
p = get_pmle_params(BASELINE_DATE, "^SPX")
BASELINE = dict(
    # systematic asset: fixed structural values
    kappai=0.0, gammai=1.0, betai=1.0, rhoix=0.0,
    # fitted jump-diffusion parameters
    sigma=float(p.dSIGMA),
    pprob=float(p.dPPROB),
    lamb=float(p.dLAMB),
    eta1=float(p.dETA1),
    eta2=float(p.dETA2),
)

# ---------- market context --------------------------------------------------
S0 = 100.0
T = 30.0 / 365.0          # 30-day option
R = 0.05
Q = 0.0
MONEYNESS = (0.90, 1.00, 1.10)


def _arr(x):
    return np.asarray(x, dtype=np.float64).reshape((-1,))


def kimyi_call_price(params, K):
    """Single KimYi call price for one strike."""
    kwargs = {k: _arr([v]) for k, v in params.items()}
    return kimyi_call(
        und_price=_arr([S0]), und_strike=_arr([K]),
        risk_free_rate=_arr([R]), dividend_yield=_arr([Q]),
        time_to_expiry=_arr([T]),
        **kwargs,
    ).item()


def bsm_iv(price, K, vol_lo=0.001, vol_hi=5.0, vol_init=0.20):
    """Invert BSM to get IV from a call price (uses the project's bisection)."""
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


def iv_curve(params):
    return {m: bsm_iv(kimyi_call_price(params, m * S0), m * S0) for m in MONEYNESS}


def anchor_atm(params, target_atm_iv,
               sigma_lo=1e-4, sigma_hi=3.0):
    """Bisect on σ so the perturbed params reproduce target_atm_iv at K = S0."""
    def f(sigma_arr):
        scen = {**params, "sigma": float(sigma_arr.flat[0])}
        atm_price = kimyi_call_price(scen, S0)
        return np.array([[bsm_iv(atm_price, S0)]])
    new_sigma = bisection(
        func=f,
        lower_value=np.array([[sigma_lo]]),
        upper_value=np.array([[sigma_hi]]),
        target_value=np.array([[target_atm_iv]]),
        initial_value=np.array([[params["sigma"]]]),
    ).item()
    return {**params, "sigma": new_sigma}


# ---------- scenarios -------------------------------------------------------
scenarios = {
    "baseline":                                 BASELINE,
    "eta2 down 30%   (larger down-jumps)":      {**BASELINE, "eta2":  BASELINE["eta2"] * 0.7},
    "eta1 up   30%   (smaller up-jumps)":       {**BASELINE, "eta1":  BASELINE["eta1"] * 1.3},
    "eta1 up & eta2 down 30% (both wings)":     {**BASELINE, "eta1":  BASELINE["eta1"] * 1.3,
                                                             "eta2":  BASELINE["eta2"] * 0.7},
    "pprob -> 0.20   (more down-jumps)":        {**BASELINE, "pprob": 0.20},
    # the user's two proposed scenarios — included for comparison
    "USER: eta1 down 30% & eta2 up 30%":        {**BASELINE, "eta1":  BASELINE["eta1"] * 0.7,
                                                             "eta2":  BASELINE["eta2"] * 1.3},
    "USER: eta1 up & eta2 up 30% (both)":       {**BASELINE, "eta1":  BASELINE["eta1"] * 1.3,
                                                             "eta2":  BASELINE["eta2"] * 1.3},
}


# ---------- run + report ----------------------------------------------------
def _print_table(title, rows, baseline_iv):
    print(f"\n=== {title} ===")
    header = (f"{'scenario':45s}  {'sigma':>6s}  "
              f"{'IV90':>6s}  {'IV100':>6s}  {'IV110':>6s}  "
              f"{'dIV90':>7s}  {'dIV100':>7s}  {'dIV110':>7s}  "
              f"{'skew':>6s}  {'dskew':>7s}")
    print(header); print("-" * len(header))
    for r in rows:
        print(f"{r['name']:45s}  {r['sigma']:6.3f}  "
              f"{r['iv90']:6.2f}  {r['iv100']:6.2f}  {r['iv110']:6.2f}  "
              f"{r['div90']:+7.2f}  {r['div100']:+7.2f}  {r['div110']:+7.2f}  "
              f"{r['skew']:6.2f}  {r['dskew']:+7.2f}")


def _row(name, params, baseline_iv):
    iv = iv_curve(params)
    iv90, iv100, iv110 = iv[0.90] * 100, iv[1.00] * 100, iv[1.10] * 100
    bsk = (baseline_iv[0.90] - baseline_iv[1.10]) * 100
    return dict(
        name=name, sigma=params["sigma"],
        iv90=iv90, iv100=iv100, iv110=iv110,
        div90=iv90 - baseline_iv[0.90] * 100,
        div100=iv100 - baseline_iv[1.00] * 100,
        div110=iv110 - baseline_iv[1.10] * 100,
        skew=iv90 - iv110,
        dskew=(iv90 - iv110) - bsk,
    )


def main():
    print(f"\nBaseline: SPX {BASELINE_DATE}   "
          f"sigma={BASELINE['sigma']:.3f}  lamb={BASELINE['lamb']:.2f}  "
          f"eta1={BASELINE['eta1']:.2f}  eta2={BASELINE['eta2']:.2f}  p={BASELINE['pprob']:.2f}")
    print(f"Option: S={S0}  T={T*365:.0f}d  r={R:.2%}  q={Q:.2%}")

    # baseline IV curve drives every delta + the ATM target
    base_iv = iv_curve(BASELINE)
    target_atm = base_iv[1.00]
    print(f"Baseline IVs (vol pts): IV90={base_iv[0.90]*100:.2f}  "
          f"IV100={base_iv[1.00]*100:.2f}  IV110={base_iv[1.10]*100:.2f}  "
          f"skew={(base_iv[0.90]-base_iv[1.10])*100:+.2f}")

    raw_rows = [_row(name, params, base_iv) for name, params in scenarios.items()]
    _print_table("Raw scenarios (no ATM compensation)", raw_rows, base_iv)

    anchored_rows = []
    for name, params in scenarios.items():
        if name == "baseline":
            anchored_rows.append(_row(name, params, base_iv))
            continue
        try:
            anchored_params = anchor_atm(params, target_atm)
            row_name = name + "  [σ-anchored]"
            anchored_rows.append(_row(row_name, anchored_params, base_iv))
        except Exception as exc:
            anchored_rows.append(dict(
                name=name + "  [anchor failed]", sigma=float("nan"),
                iv90=float("nan"), iv100=float("nan"), iv110=float("nan"),
                div90=float("nan"), div100=float("nan"), div110=float("nan"),
                skew=float("nan"), dskew=float("nan"),
            ))
    _print_table("ATM-anchored variants (σ bisected to hold IV100 at baseline)",
                 anchored_rows, base_iv)


if __name__ == "__main__":
    main()
