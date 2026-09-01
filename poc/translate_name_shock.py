"""
Fit one name's idiosyncratic loadings on a chosen window, then translate a
prescribed systematic shock grid into that name's shocks.

    python poc/translate_name_shock.py AMZN --date 20090630
    python poc/translate_name_shock.py AMZN --date 20090630 --shocks -5,-10,-20,-30,-40

WHICH WINDOW. The valuation date fixes a 252-day lookback and BOTH stages read
it:

  --date in the crisis   both the systematic parameters and the name's loadings
                         describe the crisis. Answers "how did this name behave
                         in the GFC" - the right choice for studying whether
                         loadings rise in stress.

  --date recent, with    current loadings against crisis systematic dynamics.
  --sys-date in crisis   Answers "what would TODAY's version of this name do in
                         a GFC-like event" - the stress-testing convention, and
                         what MAR33.5 does: current portfolio, stress-period
                         risk factors. For a name whose business has changed
                         (AMZN in 2008 was a $20bn retailer with no meaningful
                         AWS) these give very different answers, and neither is
                         wrong - they answer different questions. State which.

THE TRANSLATION. A prescribed systematic shock reaches the name through two
channels with different coefficients:

    b_i    = beta_i + kappa_i rho_iX / sigma        diffusion
    gamma_i                                          jump

and the shock splits into the two, D + J = shock, so

    r_i = b_i D + gamma_i J

The split is a property of the SYSTEMATIC parameters alone - it does not depend
on the name - so it is computed once and reused across names. See
poc/translate_shock.py for the derivation of b_i and of the split.

HORIZON. Each shock is labelled with the horizon at which it is the 1% quantile
under the systematic parameters in force. That labelling is regime-dependent:
the same -20% is roughly a 10-day event on GFC parameters and a 34-day event on
typical ones. The horizon is reported so the grid is never quoted without it.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.DataAccess import (                                  # noqa: E402
    get_aligned_price_panel, get_pmle_params, pmle_params_exists,
)
from Library.RiskEngineKimYi2025 import (                          # noqa: E402
    pmle_kimyirisk_idiosyncratic,
)

SYSTEMATIC_ID = "^SPX"
SYSTEMATIC_PARAMS = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]
LOOKBACK = 252
BASE_DAYS = 252
DATE_FMT = "%Y%m%d"


# ---------------------------------------------------------------------------
# The systematic split - name-independent, computed once
# ---------------------------------------------------------------------------

def split_shock(shocks, sys_params, n_paths=2_000_000, max_h=260,
                quantile=0.01, seed=11):
    """For each shock, find the horizon at which it is the `quantile` quantile,
    and decompose it into diffusion and jump parts.

    Psi is linear in its two drivers, so simulating the diffusion-only and
    jump-only processes side by side and adding them reproduces Psi exactly,
    while keeping the two contributions separable. Conditioning on the total
    landing near the shock then gives E[D | shock] and E[J | shock].

    Returns {shock: (horizon, D, J, phi)}. A shock the process cannot reach is
    absent from the dict: the OU variance saturates at sigma/sqrt(2 alpha), so
    large shocks have NO horizon at which they become a `quantile` event.
    """
    al = float(sys_params["dALPHA"]); s = float(sys_params["dSIGMA"])
    la = float(sys_params["dLAMB"]);  p = float(sys_params["dPPROB"])
    e1 = float(sys_params["dETA1"]);  e2 = float(sys_params["dETA2"])

    rng = np.random.default_rng(seed)
    dt = 1.0 / BASE_DAYS
    pd_ = np.zeros(n_paths); pj = np.zeros(n_paths)
    todo = sorted(shocks, key=abs)
    out = {}

    for t in range(1, max_h + 1):
        Z = rng.standard_normal(n_paths)
        n = rng.poisson(la * dt, n_paths)
        up = rng.random(n_paths) < p
        Y = np.zeros(n_paths); m = n > 0
        Y[m] = np.where(up[m], rng.gamma(n[m], 1 / e1), -rng.gamma(n[m], 1 / e2))
        pd_ = (1 - al * dt) * pd_ + s * np.sqrt(dt) * Z
        pj = (1 - al * dt) * pj + Y
        tot = pd_ + pj
        q = np.quantile(tot, quantile)
        for sh in list(todo):
            if q <= sh:                       # sh is negative
                band = max(0.004, abs(sh) * 0.08)
                mk = np.abs(tot - sh) < band
                if mk.sum() < 500:
                    continue
                D, J = float(pd_[mk].mean()), float(pj[mk].mean())
                out[sh] = (t, D, J, J / sh)
                todo.remove(sh)
        if not todo:
            break
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--date", required=True,
                    help="valuation date YYYYMMDD; fixes the 252-day window "
                         "for the name's loadings")
    ap.add_argument("--sys-date", default=None,
                    help="valuation date for the SYSTEMATIC parameters. "
                         "Defaults to --date. Set it to a crisis date while "
                         "--date stays recent for the stress-testing "
                         "convention: current loadings, crisis dynamics.")
    ap.add_argument("--shocks", default="-5,-10,-20,-30,-40",
                    help="prescribed systematic shocks in percent")
    ap.add_argument("--quantile", type=float, default=0.01,
                    help="probability level defining the horizon labels")
    a = ap.parse_args()

    sys_date = a.sys_date or a.date
    shocks = [float(x) / 100.0 for x in a.shocks.split(",")]

    print("=" * 74)
    print("Shock translation :: %s" % a.name)
    print("=" * 74)

    if not pmle_params_exists(sys_date, SYSTEMATIC_ID):
        raise SystemExit(
            "No systematic estimate for %s. Run:\n"
            "    python poc/estimate_systematic.py --beg %s --end %s --step 1"
            % (sys_date, sys_date, sys_date))
    ss = get_pmle_params(sys_date, SYSTEMATIC_ID)
    sys_params = {k: float(ss[k]) for k in SYSTEMATIC_PARAMS}

    print("  systematic window : %s" % sys_date)
    print("    " + "  ".join("%s=%.4f" % (k[1:].lower(), v)
                             for k, v in sys_params.items()))
    sd_inf = sys_params["dSIGMA"] / np.sqrt(2 * sys_params["dALPHA"])
    print("    OU ceiling sd_inf = sigma/sqrt(2 alpha) = %.1f%%  "
          "(shocks far beyond this are unreachable)" % (sd_inf * 100))
    print("  name window       : %s  (252-day lookback)" % a.date)
    if sys_date != a.date:
        print("    NOTE current-loadings / crisis-dynamics mode")
    print()

    # ---- returns -----------------------------------------------------------
    panel, _ = get_aligned_price_panel([SYSTEMATIC_ID, a.name],
                                       reference=SYSTEMATIC_ID, verbose=False)
    rets = panel.pct_change().dropna()
    upto = rets.loc[rets.index <= pd.to_datetime(a.date, format=DATE_FMT)]
    if len(upto) < LOOKBACK:
        raise SystemExit("only %d returns available before %s; need %d"
                         % (len(upto), a.date, LOOKBACK))
    rv = upto[a.name].iloc[-LOOKBACK:].to_numpy()
    print("  fitting %s on %s -> %s"
          % (a.name, upto.index[-LOOKBACK].date(), upto.index[-1].date()))

    res = pmle_kimyirisk_idiosyncratic(
        idi_returns=rv, params_sys=sys_params,
        delta_t=np.array(1 / BASE_DAYS), n_mc_paths=10_000)

    def val(k):
        return float(res[k].dMEAN) if hasattr(res[k], "dMEAN") else float(res[k])

    beta, kappa, rho, gamma = (val("dBETAI"), val("dKAPPAI"),
                               val("dRHOIX"), val("dGAMMAI"))
    b_i = beta + kappa * rho / sys_params["dSIGMA"]
    print()
    print("  beta_i = %.4f   kappa_i = %.4f   rho_iX = %.4f   gamma_i = %.4f"
          % (beta, kappa, rho, gamma))
    print("  b_i = beta_i + kappa_i rho_iX / sigma = %.4f" % b_i)
    print()

    # ---- translate ---------------------------------------------------------
    sp = split_shock(shocks, sys_params, quantile=a.quantile)
    print("  horizon labels: the point at which each shock is the %.1f%% "
          "quantile" % (a.quantile * 100))
    print()
    print("  %8s %9s %11s %10s %8s %9s %10s"
          % ("shock", "horizon", "diffusion", "jump", "phi", "coeff", a.name))
    print("  " + "-" * 70)
    for sh in shocks:
        if sh not in sp:
            print("  %7.0f%%   beyond the OU ceiling - no horizon reaches it"
                  % (sh * 100))
            continue
        h, D, J, phi = sp[sh]
        coeff = (1 - phi) * b_i + phi * gamma
        r = D * b_i + J * gamma
        print("  %7.0f%% %8dd %10.2f%% %9.2f%% %8.3f %9.3f %9.2f%%"
              % (sh * 100, h, D * 100, J * 100, phi, coeff, r * 100))
    print()
    print("  r = b_i x diffusion + gamma_i x jump.  The split is a property of")
    print("  the systematic parameters only, so it is shared across all names.")


if __name__ == "__main__":
    main()
