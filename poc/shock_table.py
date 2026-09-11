"""Prescribed SPX shocks, translated into each name's equivalent shock.

A systematic shock x reaches name i through two channels with different
coefficients, and the split between them is a property of the SYSTEMATIC
parameters alone:

    b_diff,i = beta_i + kappa_i rho_iX / sigma        diffusion
    gamma_i                                            jump

so the name's move is b_diff * E[D|x] + E[exp(gamma_i Y) - 1 | x], compounded
over the horizon rather than applied once at it. All of that is
shock_to_name.name_shock, which is used here unchanged; this file only reads
parameters and lays out the table.

WHERE THE PARAMETERS COME FROM. One row of one idiosyncratic drawer. The
drawer row carries the systematic parameters the name was CONDITIONED on
alongside its own five, so a single read cannot mix a name fitted under one
systematic arm with systematic parameters from another - which is the whole
reason it reads the drawer rather than re-fitting.

    python poc/shock_table.py --names HYG --priors alpha-pprob-flat --lookback 504
    python poc/shock_table.py --names C,BAC,JPM --date 20090630
    python poc/shock_table.py --names HYG --compare-idio paper,rhoix-flat

WHY --compare-idio EXISTS. b_diff is beta_i + kappa_i rho_iX / sigma, so this
translation runs straight through the one parameter the likelihood pins worst.
On HYG at 504 days the two rho_iX arms give b_i of 0.377 and 0.179 - a factor
of two on every number in this table. Quoting an equivalent shock without
saying which arm produced it is quoting a coin flip, so the comparison is one
flag away and the header always names the arm.

THE HORIZON LABEL. A shock is not plausible or implausible on its own, only at
a horizon: -20% is roughly a 10-day event on GFC systematic parameters and a
34-day event on calm ones. Each row is labelled with the horizon at which it
is the --quantile tail under the systematic parameters in force, so the grid
is never quoted bare.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import (                                  # noqa: E402
    get_pmle_params, available_pmle_dates,
)

SYS_KEYS = ["dSIGMA", "dLAMB", "dPPROB", "dETA1", "dETA2", "dALPHA"]
IDIO_KEYS = ["dBETAI", "dKAPPAI", "dGAMMAI", "dRHOIX", "dMUI"]


def load_row(drawer, date=None):
    """One drawer row: the name's five parameters AND the systematic six."""
    dates = available_pmle_dates(drawer)
    if not dates:
        raise SystemExit("drawer %s holds no dates" % drawer)
    if date is None:
        date = sorted(dates)[-1]
    elif date not in set(dates):
        nearest = min(dates, key=lambda d: abs(int(d) - int(date)))
        print("  %s not in %s - using nearest, %s" % (date, drawer, nearest))
        date = nearest
    s = get_pmle_params(date, drawer)
    return date, {k: float(s[k]) for k in SYS_KEYS}, \
        {k: float(s[k]) for k in IDIO_KEYS}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True)
    ap.add_argument("--priors", default="alpha-pprob-flat",
                    help="systematic arm the names were conditioned on")
    ap.add_argument("--idio-priors", default="paper")
    ap.add_argument("--compare-idio", default=None,
                    help="comma-separated idiosyncratic arms to show side by "
                         "side, e.g. 'paper,rhoix-flat'")
    ap.add_argument("--anchor", default="hybrid")
    ap.add_argument("--lookback", type=int, default=504)
    ap.add_argument("--date", default=None,
                    help="YYYYMMDD; default is the drawer's last date")
    ap.add_argument("--shocks", default="-5,-10,-20,-30,-40",
                    help="prescribed SYSTEMATIC shocks, percent")
    ap.add_argument("--quantile", type=float, default=0.01)
    a = ap.parse_args()

    from Library.RiskEngineKimYi2025 import SYSTEMATIC_PRIOR_SETS
    import poc.estimate_idiosyncratic as EI
    from poc.shock_to_name import name_shock, model_horizon

    names = [n.strip().upper() for n in a.names.split(",") if n.strip()]
    shocks = [float(s) / 100.0 for s in a.shocks.split(",") if s.strip()]
    arms = ([x.strip() for x in a.compare_idio.split(",")]
            if a.compare_idio else [a.idio_priors])
    from Library.RiskEngineKimYi2025 import IDIOSYNCRATIC_PRIOR_SETS
    bad = [x for x in arms if x not in IDIOSYNCRATIC_PRIOR_SETS]
    if bad:
        raise SystemExit("unknown idiosyncratic arm(s) %s; have %s"
                         % (bad, sorted(IDIOSYNCRATIC_PRIOR_SETS)))
    if a.priors not in SYSTEMATIC_PRIOR_SETS:
        raise SystemExit("unknown systematic arm %r" % a.priors)
    priors = SYSTEMATIC_PRIOR_SETS[a.priors]

    print()
    print("=" * 78)
    print("equivalent shocks :: prescribed SPX move -> each name")
    print("=" * 78)

    for arm in arms:
        EI.IDIO_TAG = arm          # name_store_id reads this module global
        rows = {}
        for nm in names:
            drawer = EI.name_store_id(nm, a.anchor, a.priors, priors,
                                      a.lookback)
            try:
                dt, sysp, idio = load_row(drawer, a.date)
            except SystemExit as exc:
                print("  %-6s SKIPPED - %s" % (nm, exc))
                continue
            rows[nm] = (dt, sysp, idio, drawer)
        if not rows:
            continue

        any_sys = next(iter(rows.values()))[1]
        print()
        print("  idiosyncratic arm: %s      systematic arm: %s, %d-day window"
              % (arm, a.priors, a.lookback))
        print("  %-6s %-10s %8s %8s %8s %8s %8s %8s"
              % ("name", "date", "beta_i", "kappa_i", "rho_iX", "b_diff",
                 "gamma_i", "g/b"))
        print("  " + "-" * 70)
        for nm, (dt, sysp, idio, _) in rows.items():
            b_diff = (idio["dBETAI"]
                      + idio["dKAPPAI"] * idio["dRHOIX"] / sysp["dSIGMA"])
            print("  %-6s %-10s %8.4f %8.4f %+8.4f %8.4f %8.4f %8.2f"
                  % (nm, dt, idio["dBETAI"], idio["dKAPPAI"], idio["dRHOIX"],
                     b_diff, idio["dGAMMAI"],
                     idio["dGAMMAI"] / b_diff if b_diff else float("nan")))

        print()
        print("  %-9s %8s   %s"
              % ("SPX", "horizon", "  ".join("%16s" % n for n in rows)))
        print("  " + "-" * (20 + 18 * len(rows)))
        for x in shocks:
            try:
                h = int(model_horizon(x, any_sys, alpha=a.quantile))
            except Exception:                                     # noqa: BLE001
                h = 1
            cells = []
            for nm, (dt, sysp, idio, _) in rows.items():
                r = name_shock(x, sysp, idio, horizon_days=h)
                cells.append("%16s" % ("%+7.1f%% +-%4.1f"
                                       % (100 * r["y"], 100 * r["sd"])))
            print("  %+8.1f%% %7dd   %s" % (100 * x, h, "  ".join(cells)))
        print()
        print("  Each cell is the name's equivalent move, +- one sd of the")
        print("  dispersion around it. horizon is where that SPX shock is the")
        print("  %.0f%% tail under these systematic parameters." % (100 * a.quantile))

    if len(arms) > 1:
        print()
        print("  The two blocks differ ONLY in the rho_iX prior. Any gap")
        print("  between them is what that prior is worth in shock units, and")
        print("  it is the number to put in front of a referee rather than an")
        print("  identification ratio.")
    print()


if __name__ == "__main__":
    main()
