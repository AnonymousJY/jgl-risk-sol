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

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)

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
        _LOG.info("  %s not in %s - using nearest, %s" % (date, drawer, nearest))
        date = nearest
    s = get_pmle_params(date, drawer)
    return date, {k: float(s[k]) for k in SYS_KEYS}, \
        {k: float(s[k]) for k in IDIO_KEYS}


# ---------------------------------------------------------------------------
# The calibrated ladder.
# ---------------------------------------------------------------------------
# Five rungs, each carrying TWO anchors that were derived without reference to
# each other: one from the realised SPX record 2007-01-03 to 2026-09-10 (4,953
# days), one from the fitted model simulated at 1,000,000 paths per year
# (poc/systematic_tail_mc.py).
#
#   +/-x   what it is                 realised anchor        model anchor
#   -----  ------------------------   --------------------   -------------------
#     2%   routine bad day            1 in 12 days           -
#     4%   severe day, average year   pooled ES(2.5%) -3.93  20-yr mean sim
#                                                            ES(2.5%) -3.90
#     7%   severe day, crisis year    pooled ES(0.5%) -6.53  worst-year sim
#                                                            ES(2.5%) -6.96
#    10%   crisis extreme             1 in 1,651 days        worst-year sim
#                                                            ES(1%)   -9.85
#    12%   the ceiling                worst day    -11.98    worst-year sim
#                                                            ES(0.5%) -12.10
#
# THE CEILING IS THE RESULT. The model's severest through-the-cycle 0.5% ES is
# -12.096% from the 2009 window; the worst day the sample contains is -11.984%
# on 2020-03-16. They agree to 0.112pp - 0.9% - and the 2009 window ends in
# 2009, eleven years before that day happened. Two independent routes to the
# same ceiling is what justifies stopping there. Above it, +/-15% and +/-20%
# exceed both anchors and have never occurred in twenty years: prescribe them
# as labelled reverse-stress, not as rungs.
#
# SYMMETRIC, DELIBERATELY. The realised down/up ES ratio FALLS toward one as
# the tail is cut further out - 1.075 at 2.5%, 1.033 at 1%, 1.002 at 0.5% -
# and in the three crisis years it shows no down-skew at all: 0.994 in 2008,
# 0.941 in 2009, 1.051 in 2020. The fitted model asserts 1.275 on average and
# never drops below 1.055, but that comes from pprob, whose prior support
# under alpha-pprob-eta-flat is U(0.05, 0.45) and CANNOT reach 0.5. The
# asymmetry is prior, not data, so the ladder does not inherit it.
#
# SET THROUGH THE CYCLE, NOT OFF THE CURRENT WINDOW. Simulated ES(2.5%) runs
# -2.363 in 2014 to -6.961 in 2009, a factor of 2.95, and it LAGS: its
# correlation with the realised ES of the same year is 0.389, with the
# previous year 0.780, with the mean of the two previous years 0.820. A
# 504-day window is a two-year trailing average, so a ladder anchored on the
# current fit is procyclical. The rungs above use worst-year figures.
LADDER = (-12.0, -10.0, -7.0, -4.0, -2.0, 2.0, 4.0, 7.0, 10.0, 12.0)


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
    ap.add_argument("--ladder", action="store_true",
                    help="use the calibrated ladder instead of --shocks: "
                         "+/-2, 4, 7, 10, 12 percent. Every rung carries two "
                         "anchors derived without reference to each other - "
                         "one from the realised SPX record 2007-2026, one "
                         "from the fitted model - see LADDER below")
    ap.add_argument("--quantile", type=float, default=0.01)
    ap.add_argument("--jump-response", choices=("power", "linear", "exp"),
                    default="power",
                    help="power is E[(1+J)^gamma_i - 1|x], the only form that keeps a name above -100% AND sends every name to zero when the factor does. linear is gamma_i E[J|x], the convention Appendix "
                         "B's transition densities were fitted under. exp is "
                         "E[exp(gamma_i Y)-1|x], the SDE's form; it adds "
                         "curvature the estimates do not contain and "
                         "understates the downside. Run shock_to_name.py "
                         "--selfcheck to see it on the systematic factor.")
    ap.add_argument("--compare-response", action="store_true",
                    help="show both responses side by side")
    a = ap.parse_args()

    from Library.RiskEngineKimYi2025 import SYSTEMATIC_PRIOR_SETS
    import poc.estimate_idiosyncratic as EI
    from poc.shock_to_name import name_shock, model_horizon

    names = [n.strip().upper() for n in a.names.split(",") if n.strip()]
    if a.ladder:
        shocks = [x / 100.0 for x in LADDER]
    else:
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

    _LOG.info("")
    _LOG.info("=" * 78)
    _LOG.info("equivalent shocks :: prescribed SPX move -> each name")
    _LOG.info("=" * 78)

    for arm in arms:
        EI.IDIO_TAG = arm          # name_store_id reads this module global
        rows = {}
        for nm in names:
            drawer = EI.name_store_id(nm, a.anchor, a.priors, priors,
                                      a.lookback)
            try:
                dt, sysp, idio = load_row(drawer, a.date)
            except SystemExit as exc:
                _LOG.info("  %-6s SKIPPED - %s" % (nm, exc))
                continue
            rows[nm] = (dt, sysp, idio, drawer)
        if not rows:
            continue

        any_sys = next(iter(rows.values()))[1]
        _LOG.info("")
        _LOG.info("  idiosyncratic arm: %s      systematic arm: %s, %d-day window"
              % (arm, a.priors, a.lookback))
        _LOG.info("  %-6s %-10s %8s %8s %8s %8s %8s %8s"
              % ("name", "date", "beta_i", "kappa_i", "rho_iX", "b_diff",
                 "gamma_i", "g/b"))
        _LOG.info("  " + "-" * 70)
        for nm, (dt, sysp, idio, _) in rows.items():
            b_diff = (idio["dBETAI"]
                      + idio["dKAPPAI"] * idio["dRHOIX"] / sysp["dSIGMA"])
            _LOG.info("  %-6s %-10s %8.4f %8.4f %+8.4f %8.4f %8.4f %8.2f"
                  % (nm, dt, idio["dBETAI"], idio["dKAPPAI"], idio["dRHOIX"],
                     b_diff, idio["dGAMMAI"],
                     idio["dGAMMAI"] / b_diff if b_diff else float("nan")))

        _LOG.info("")
        _LOG.info("  %-9s %8s   %s"
              % ("SPX", "horizon", "  ".join("%16s" % n for n in rows)))
        _LOG.info("  " + "-" * (20 + 18 * len(rows)))
        for x in shocks:
            try:
                h = int(model_horizon(x, any_sys, alpha=a.quantile))
            except Exception:                                     # noqa: BLE001
                h = 1
            cells = []
            for nm, (dt, sysp, idio, _) in rows.items():
                r = name_shock(x, sysp, idio, horizon_days=h,
                               jump_response=a.jump_response)
                if a.compare_response:
                    other = "exp" if a.jump_response == "linear" else "linear"
                    try:
                        r2 = name_shock(x, sysp, idio, horizon_days=h,
                                        jump_response=other)
                        cells.append("%16s" % ("%+7.1f%% /%+6.1f%%"
                                               % (100 * r["y"], 100 * r2["y"])))
                        continue
                    except Exception:                         # noqa: BLE001
                        pass
                cells.append("%16s" % ("%+7.1f%% +-%4.1f"
                                       % (100 * r["y"], 100 * r["sd"])))
            _LOG.info("  %+8.1f%% %7dd   %s" % (100 * x, h, "  ".join(cells)))
        _LOG.info("")
        _LOG.info("  Each cell is the name's equivalent move, +- one sd of the")
        _LOG.info("  dispersion around it. horizon is where that SPX shock is the")
        _LOG.info("  %.0f%% tail under these systematic parameters." % (100 * a.quantile))
        _LOG.info("  jump response: %s" % a.jump_response)
        if a.compare_response:
            other = "exp" if a.jump_response == "linear" else "linear"
            _LOG.info("  Cells are %s / %s. The gap between them is the Jensen"
                      % (a.jump_response, other))
            _LOG.info("  term Appendix B's transition density drops. It is")
            _LOG.info("  always positive, so exp always reads the smaller loss.")

    if len(arms) > 1:
        _LOG.info("")
        _LOG.info("  The two blocks differ ONLY in the rho_iX prior. Any gap")
        _LOG.info("  between them is what that prior is worth in shock units, and")
        _LOG.info("  it is the number to put in front of a referee rather than an")
        _LOG.info("  identification ratio.")
    _LOG.info("")


if __name__ == "__main__":
    main()
