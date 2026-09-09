"""Worst-of autocallable on three plain Black-Scholes assets.

No liquidity model, no jumps: three assets at the same flat implied volatility
with the same pairwise correlation, priced through
ExoticEngineBlackScholesMerton. This is the reference case - it is what the
structure's spot and vega profiles are SUPPOSED to look like, and it is the
yardstick the model-based run has to be read against.

    python poc/autocallable_bsm.py
    python poc/autocallable_bsm.py --vol 0.25 --corr 0.7 --paths 400000
"""
import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.Parameters import ParametersConstant                    # noqa: E402
from Library.Random import RandomMT19937                             # noqa: E402
from Library.StatisticsMC import StatisticsMCMean                     # noqa: E402
from Library.PathDependent import PathDependentWorstOfAutocallable    # noqa: E402
from Library.ExoticEngine import ExoticEngineBlackScholesMerton       # noqa: E402

C = lambda v: ParametersConstant(np.array(float(v)))
N_ASSETS = 3


def price(spots, initial, a, vol, seed, protection=None, corr=None):
    prod = PathDependentWorstOfAutocallable(
        fixing_times=np.arange(1, a.periods + 1) / 4.,
        initial_levels=np.asarray(initial, dtype=float),
        autocall_barriers=a.autocall, coupon_barriers=a.coupon_barrier,
        coupon_amounts=a.coupon,
        protection_barrier=a.protection if protection is None else protection,
        notional=a.notional, memory=True)
    n_pairs = N_ASSETS * (N_ASSETS - 1) // 2
    eng = ExoticEngineBlackScholesMerton(
        the_product=prod, risk_free_rate=C(a.rate),
        dividend_yield=[C(a.dividend)] * N_ASSETS,
        imp_volatility=[C(vol)] * N_ASSETS,
        rand_generator=RandomMT19937(np.uint64(seed)),
        spot_price=np.asarray(spots, dtype=float),
        number_of_paths=np.uint64(a.paths),
        correlations=[C(a.corr if corr is None else corr)] * n_pairs)
    g = StatisticsMCMean()
    eng.do_simulation(gatherer=g)
    return g.get_result_so_far()[0, 0]


def put_value(spots, initial, a, vol, seed, corr=None):
    """The embedded knock-in put, by removing it: the same note with the
    protection barrier at zero can never write down, so it redeems at par
    whenever it survives. Same coupons, same autocall, same seed - the
    difference is the barrier feature and nothing else."""
    protected = price(spots, initial, a, vol, seed, protection=0.0, corr=corr)
    actual = price(spots, initial, a, vol, seed, corr=corr)
    return protected - actual, actual


def put_greeks(spots, initial, a, seed=20240114):
    """Put, its vega per 1 vol point, and its correlation P&L per 10 bps.

    Both are CENTRAL differences over a bump far larger than the reported unit,
    then scaled back. A difference of two Monte Carlo prices carries noise that
    barely shrinks with the bump, so differencing at the reported unit - 1 vol
    point, and especially 10 bps of correlation - would return almost pure
    noise. Central also drops the second-order bias a one-sided bump of this
    size would carry.

    Correlation is the weaker of the two for variance reduction: bumping rho
    changes the Cholesky factor, so the same normals no longer map to the same
    asset paths. The bump is clipped to keep rho inside (-1/(n-1), 1).
    """
    put, pv = put_value(spots, initial, a, a.vol, seed)

    up, _ = put_value(spots, initial, a, a.vol + a.bump, seed)
    dn, _ = put_value(spots, initial, a, a.vol - a.bump, seed)
    vega = (up - dn) / (2.0 * a.bump) * 0.01

    cb = min(a.corr_bump, 0.999 - a.corr, a.corr + 1.0 / (N_ASSETS - 1) - 1e-6)
    cup, _ = put_value(spots, initial, a, a.vol, seed, corr=a.corr + cb)
    cdn, _ = put_value(spots, initial, a, a.vol, seed, corr=a.corr - cb)
    cega = (cup - cdn) / (2.0 * cb) * 0.0010

    return put, pv, vega, cega


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vol", type=float, default=0.20)
    ap.add_argument("--corr", type=float, default=0.50)
    ap.add_argument("--rate", type=float, default=0.04)
    ap.add_argument("--dividend", type=float, default=0.0)
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--periods", type=int, default=12, help="quarterly observations")
    ap.add_argument("--autocall", type=float, default=1.00)
    ap.add_argument("--coupon-barrier", type=float, default=0.70)
    ap.add_argument("--coupon", type=float, default=0.02, help="per period")
    ap.add_argument("--protection", type=float, default=0.50)
    ap.add_argument("--notional", type=float, default=100.)
    ap.add_argument("--bump", type=float, default=0.02,
                    help="half-bump in vol for the central vega difference")
    ap.add_argument("--corr-bump", type=float, default=0.05,
                    help="half-bump in correlation for the central corr "
                         "difference; the reported number is per 10 bps")
    a = ap.parse_args()

    spot0 = np.full(N_ASSETS, 100.)
    shocks = [-0.60, -0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20,
              -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]

    print("=" * 70)
    print("Worst-of autocallable :: 3 identical Black-Scholes assets")
    print("=" * 70)
    print("  %dy quarterly, autocall %.0f%%, coupon %.2f%%/q above %.0f%%, "
          "protection %.0f%%, memory"
          % (a.periods / 4, 100 * a.autocall, 100 * a.coupon,
             100 * a.coupon_barrier, 100 * a.protection))
    print("  vol %.2f on every name, pairwise correlation %.2f, r %.2f%%, q %.2f%%"
          % (a.vol, a.corr, 100 * a.rate, 100 * a.dividend))
    print("  %s paths" % f"{a.paths:,}")

    put0, base, vega0, cega0 = put_greeks(spot0, spot0, a)
    print("\n  note  %.4f   embedded put  %.4f   vega  %.4f   corr  %+.4f"
          % (base, put0, vega0, cega0))
    print("  The issuer is short the note and therefore long the put, so these")
    print("  are the ISSUER's sensitivities.")
    print("  vega  per +1 vol point on all three names   (central, +/-%.0f pts)"
          % (100 * a.bump))
    print("  corr  per +10 bps of pairwise correlation   (central, +/-%.0f bps)"
          % (10000 * a.corr_bump))

    print("\n  %6s %6s %10s %9s %10s %8s %9s" %
          ("shock", "spot", "note", "put", "put P&L", "vega", "corr"))
    print("  " + "-" * 66)
    for x in shocks:
        sp = spot0 * (1.0 + x)
        put, pv, vg, cg = put_greeks(sp, spot0, a)
        print("  %+5.0f%% %6.0f %10.4f %9.4f %+10.4f %8.4f %+9.4f"
              % (100 * x, sp[0], pv, put, put - put0, vg, cg))


if __name__ == "__main__":
    main()
