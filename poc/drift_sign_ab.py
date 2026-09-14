"""Same-seed A/B on the Theorem 3.1 drift sign.

The before/after route through the parameter cache cannot settle this: the
predicted shift in mui is 1.5-2.2% of its own 95% credible interval, which is
far below the difference between two MCMC runs. This script removes the noise
instead of trying to average it away.

For each (valuation date, name) it fits the idiosyncratic block TWICE IN ONE
PROCESS with the SAME SEED - once with KimYiLogLike._drift monkeypatched back
to the printed minus convention, once with the plus that ships - and compares.
Identical seed, identical data, identical everything except the sign, so the
two posteriors should differ only by the reparameterisation

    mui_minus - mui_plus = 2 * sigma * beta_i * kappa_i * rho_iX

and beta_i, kappa_i, rho_iX and gamma_i should not move at all. Whatever
residual disagreement remains in those four is the sampler's own tolerance,
and it bounds how much of the mui move could be anything other than the flip.

Touches neither Scripts/config_skew.py nor the parameter cache: it reads the
cached systematic CSVs, fits in memory, and writes nothing.

usage
    python poc/drift_sign_ab.py [n_dates] [name ...]

    n_dates  how many of the most recent valuation dates to fit (default 3)
    name     underlyings to fit (default: whatever get_idiosyncratic_ids gives)
"""
import os
import sys

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Scripts")
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Scripts.load_portfolio import get_idiosyncratic_ids
from Scripts.run_pmle_kimyi2025 import SYSTEMATIC_PARAMS
from Library.DataAccess import get_price_panel, get_pmle_params_dict
from Library import RiskEngineKimYi2025 as eng

# Mirrors the configuration block of Scripts/run_pmle_kimyi2025.py.
VALUATION_BEG, VALUATION_END, FMT = "20250331", "20250417", "%Y%m%d"
LOOKBACK = 252
DELTA_T = np.array(1 / 252)
SEED = np.uint64(20240114)
N_MC = int(10_000)
SYSTEMATIC_ID = "^SPX"

IDI_PARAMS = ["dMUI", "dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX"]


def _drift_minus(self):
    """KimYiLogLike._drift exactly as it stood before commit 7bd2e76."""
    drift = self.mui
    drift += 0.5 * (self.sigma * self.betai) ** 2
    drift -= self.sigma * self.betai * self.kappai * self.rhoix
    return drift.reshape((-1, 1))


def fit(returns, params_sys, minus):
    """One idiosyncratic fit. `minus` selects the printed (old) convention."""
    original = eng.KimYiLogLike._drift
    if minus:
        eng.KimYiLogLike._drift = _drift_minus
    try:
        return eng.pmle_kimyirisk_idiosyncratic(
            idi_returns=returns, params_sys=params_sys, delta_t=DELTA_T,
            seed_number=SEED, n_mc_paths=N_MC,
        )
    finally:
        eng.KimYiLogLike._drift = original


def main():
    n_dates = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    names = sys.argv[2:] or get_idiosyncratic_ids()

    window = [d.strftime(FMT) for d in pd.bdate_range(
        pd.to_datetime(VALUATION_BEG, format=FMT),
        pd.to_datetime(VALUATION_END, format=FMT))][-n_dates:]

    price = get_price_panel([SYSTEMATIC_ID] + list(names))
    rets = price.pct_change().dropna()

    print("dates %s   names %s   seed %d\n" % (window, list(names), int(SEED)))
    rows = []
    for dt in window:
        params_sys = get_pmle_params_dict(dt, SYSTEMATIC_ID,
                                          params=SYSTEMATIC_PARAMS)
        for nm in names:
            r = rets.loc[rets.index <= dt, nm].iloc[-LOOKBACK:].to_numpy()
            old = fit(r, params_sys, minus=True)
            new = fit(r, params_sys, minus=False)
            row = {"date": dt, "name": nm}
            for p in IDI_PARAMS:
                row[p + "_old"] = old[p].dMEAN
                row[p + "_new"] = new[p].dMEAN
            row["sigma"] = float(np.asarray(params_sys["dSIGMA"]).ravel()[0])
            rows.append(row)
            print("  %s %-8s  dMUI %+.6f -> %+.6f   (fall %+.6f)"
                  % (dt, nm, row["dMUI_old"], row["dMUI_new"],
                     row["dMUI_old"] - row["dMUI_new"]))

    d = pd.DataFrame(rows)
    # Predicted from the average of the two fits' beta, kappa, rho - they
    # should agree to the sampler's tolerance, and the check below says so.
    b = 0.5 * (d.dBETAI_old + d.dBETAI_new)
    k = 0.5 * (d.dKAPPAI_old + d.dKAPPAI_new)
    rho = 0.5 * (d.dRHOIX_old + d.dRHOIX_new)
    pred = 2.0 * d.sigma * b * k * rho
    got = d.dMUI_old - d.dMUI_new

    print("\n(a) dMUI fell by 2 sigma beta kappa rho")
    t = pd.DataFrame({"date": d.date, "name": d.name,
                      "predicted": pred, "actual": got,
                      "residual": got - pred})
    print(t.to_string(index=False, float_format=lambda x: "%+.6f" % x))
    scale = pred.abs().mean()
    worst = (got - pred).abs().max()
    ok = worst <= max(0.05 * scale, 1e-6)
    print("   largest residual %.3e against a mean shift of %.3e   %s"
          % (worst, scale, "PASS" if ok else "**FAIL**"))

    print("\n(b) nothing else moved")
    out = []
    for p in ["dKAPPAI", "dGAMMAI", "dBETAI", "dRHOIX"]:
        mv = (d[p + "_new"] - d[p + "_old"]).abs()
        rel = mv / d[p + "_old"].abs().clip(lower=1e-12)
        out.append((p, mv.max(), rel.max(),
                    "PASS" if rel.max() < 0.01 else "CHECK"))
    print(pd.DataFrame(out, columns=["param", "max_abs_move", "max_rel_move",
                                     "result"]).to_string(
        index=False, float_format=lambda x: "%.3e" % x))
    print("\n   A non-zero move here is the sampler's own run-to-run tolerance,")
    print("   not the sign. It bounds how much of the dMUI move in (a) could")
    print("   be anything other than the reparameterisation.")


if __name__ == "__main__":
    main()
