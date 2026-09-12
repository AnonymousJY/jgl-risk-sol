"""Should alpha be split into a systematic and a per-name decay?

WHAT THE MODEL CURRENTLY DOES. alpha is the only parameter in KimYiLogLike
that couples one observation to the next:

    observed_data = np.cumsum(sys_returns)              # a LEVEL, not returns
    diff_y_x      = y - (1 - alpha * dt) * x

So alpha is the decay of the de-drifted CUMULATIVE return - the transitory
component the liquidity story is about - and everything else in the
likelihood (the variance, the drift, the two jump branches) acts inside a
single increment. _dist_loglike_idiosyncratic then passes the SYSTEMATIC alpha
straight through, so alpha_i = alpha_X is an imposed restriction, never an
estimate. Whatever is decided about it, the paper should say that.

WHY SPLITTING IS WELL POSED, UNLIKE rho_ix. rho_ix cannot be separated because
it appears only inside the product sigma*beta_i*kappa_i*rho_ix - no amount of
return data moves it on its own. alpha_i is not in that position: it has its
own term, on the name's own series, and a different alpha_i produces a
different likelihood at fixed beta_i, kappa_i and rho_ix. The obstacle is not
structural.

WHAT THE OBSTACLE ACTUALLY IS. Over a one-year window an OU pull toward zero
and a small drift produce nearly the same cumulative path, and mu_i is FREE in
the idiosyncratic block under a Normal(0,1) prior. The systematic block pins
mu at zero and still cannot identify alpha in 252 days - which is the warning,
not the reassurance. So alpha_i, if it is estimated at all, is a full-sample
per-name quantity like alpha_X, and the hypothesis to test is alpha_i =
alpha_X rather than anything a rolling window could show.

WHAT THIS SCRIPT IS. Not the MLE - a cheap probe of the same object, so the
question of whether the modelling work is worth doing gets answered before the
work is done. It regresses each name's de-drifted cumulative return on its own
lag over the full sample, reports the implied alpha = (1 - phi) / dt with a
standard error and a half-life, and does the same on rolling windows to show
what a window can and cannot see. If the cross-name spread is inside the
standard errors, splitting alpha buys nothing and the restriction stands on
its own evidence.

    python poc/alpha_split_probe.py --names C,BAC,JPM,CCL,RCL,LUV
    python poc/alpha_split_probe.py --names C,BAC,JPM --rolling

THE LEVEL IS NOT INTERPRETABLE, AND BY A LOT. An AR(1) slope near a unit root
is biased downward (Dickey-Fuller), so alpha = (1 - phi) * 252 is biased UP,
and de-trending makes it worse. Simulated at the full sample's length, this
estimator returns:

    true alpha    estimated
       0.000        0.504 +- 0.242      <- a PURE RANDOM WALK reads 0.5
       0.036        0.445 +- 0.252
       0.250        0.660 +- 0.295
       0.500        0.900 +- 0.418
       2.000        2.423 +- 0.510

Student-t(3) innovations move the 0.036 case to 0.555. So a reading of "0.5"
here means "no detectable reversion", not a 1.4-year half-life, and the only
thing worth reading is the DIFFERENCE between series of the same length, where
the bias is common and cancels. The script simulates that null itself and
prints the band, so no one has to remember this paragraph.

AND IT SHOWS WHY THIS CANNOT BE A ROLLING QUANTITY. At a true alpha of 0.036,
a 252-day window returns a median of 9.6 with a 5-95% range of [3.0, 20.6] -
pure noise centred two orders of magnitude above the truth. That is the same
verdict the P-MLE identification table gives for alpha, reached independently
and without a prior anywhere near it.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import get_aligned_price_panel            # noqa: E402
from Library.StudyWindow import SYSTEMATIC_ID, BASE_DAYS          # noqa: E402

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)


def ar1_alpha(level, base_days=BASE_DAYS, detrend=True):
    """alpha = (1 - phi) * base_days from y_t = c + phi y_{t-1}, plus its se."""
    y = np.asarray(level, dtype=float)
    if detrend:                       # the drift the model removes separately
        t = np.arange(len(y))
        y = y - np.polyval(np.polyfit(t, y, 1), t)
    x0, x1 = y[:-1], y[1:]
    n = len(x0)
    if n < 30 or x0.std() == 0:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(n), x0])
    beta, *_ = np.linalg.lstsq(X, x1, rcond=None)
    resid = x1 - X @ beta
    s2 = float(resid @ resid) / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    phi, se_phi = float(beta[1]), float(np.sqrt(cov[1, 1]))
    alpha = (1.0 - phi) * base_days
    se = se_phi * base_days
    hl = np.log(2.0) / alpha if alpha > 0 else np.inf
    return alpha, se, hl


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="C,BAC,JPM,CCL,RCL,LUV")
    ap.add_argument("--rolling", action="store_true",
                    help="also show what a 252-day window sees, which is the "
                         "point about why this has to be full-sample")
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--no-detrend", dest="detrend", action="store_false")
    ap.add_argument("--null-reps", type=int, default=400,
                    help="random walks simulated to calibrate the bias band")
    a = ap.parse_args()

    names = [n.strip().upper() for n in a.names.split(",") if n.strip()]
    panel, _ = get_aligned_price_panel([SYSTEMATIC_ID] + names,
                                       reference=SYSTEMATIC_ID)
    # COMMON DATES ONLY. The bias cancels between two series of the same
    # length and does not between series of different ones, and a name listed
    # in 2021 against SPX from 2007 would otherwise be compared on a bias gap
    # rather than on its dynamics.
    rets = panel.pct_change().dropna(how="any")

    _LOG.info("")
    _LOG.info("=" * 78)
    _LOG.info("alpha split probe :: AR(1) on the de-drifted cumulative return")
    _LOG.info("=" * 78)
    _LOG.info("  alpha = (1 - phi) * %d, so a LARGER alpha is FASTER reversion." % BASE_DAYS)
    _LOG.info("  Full-sample alpha_X from the systematic fit is 0.036 (half-life")
    _LOG.info("  19.3 years) - close enough to a unit root that 'the systematic")
    _LOG.info("  process reverts quickly' is not what the market series says, and")
    _LOG.info("  the direction of any difference is an open question, not a prior.")
    _LOG.info("")
    _LOG.info("  %-8s %7s %10s %9s %11s %9s"
          % ("series", "n", "alpha", "se", "half-life", "phi"))
    _LOG.info("  " + "-" * 62)

    out = {}
    for sym in [SYSTEMATIC_ID] + names:
        if sym not in rets:
            _LOG.info("  %-8s  not in the panel" % sym)
            continue
        r = rets[sym].dropna()
        al, se, hl = ar1_alpha(np.cumsum(r.to_numpy()), detrend=a.detrend)
        out[sym] = (al, se)
        _LOG.info("  %-8s %7d %10.4f %9.4f %11s %9.6f"
              % (sym, len(r), al, se,
                 ("%.1f yr" % hl) if np.isfinite(hl) else "inf",
                 1 - al / BASE_DAYS))

    # --- the null, simulated at this sample's own length --------------------
    n_obs = len(rets)
    rng = np.random.default_rng(20240114)
    null = []
    for _ in range(a.null_reps):
        y = np.cumsum(rng.normal(0.0, 1.0, n_obs))
        v, _, _ = ar1_alpha(y, detrend=a.detrend)
        if np.isfinite(v):
            null.append(v)
    null = np.array(null)
    if len(null):
        _LOG.info("")
        _LOG.info("  NULL, %d simulated random walks of length %d - no reversion"
              % (len(null), n_obs))
        _LOG.info("  at all, put through this same estimator:")
        _LOG.info("     median %.3f   5%%-95%% [%.3f, %.3f]"
              % (np.median(null), np.quantile(null, 0.05),
                 np.quantile(null, 0.95)))
        _LOG.info("  Anything inside that band is indistinguishable from a random")
        _LOG.info("  walk. The level above is bias; only the differences below are")
        _LOG.info("  worth reading.")

    sysal = out.get(SYSTEMATIC_ID, (np.nan, np.nan))
    _LOG.info("")
    _LOG.info("  Each name against the systematic, in standard errors of the")
    _LOG.info("  difference. Inside about 2 and the restriction alpha_i = alpha_X")
    _LOG.info("  is not being contradicted by that name:")
    _LOG.info("  %-8s %11s %11s %9s   %s"
          % ("name", "alpha_i", "- alpha_X", "t", "reading"))
    _LOG.info("  " + "-" * 66)
    for sym in names:
        if sym not in out or not np.isfinite(sysal[0]):
            continue
        d = out[sym][0] - sysal[0]
        sd = np.sqrt(out[sym][1] ** 2 + sysal[1] ** 2)
        t = d / sd if sd > 0 else np.nan
        if not np.isfinite(t):
            read = "-"
        elif abs(t) < 2:
            read = "consistent with alpha_X"
        elif t > 0:
            read = "reverts FASTER than the market"
        else:
            read = "reverts SLOWER than the market"
        _LOG.info("  %-8s %11.4f %11.4f %9.2f   %s"
              % (sym, out[sym][0], d, t, read))

    if a.rolling:
        _LOG.info("")
        _LOG.info("  What a %d-day window sees. If this column is wide and"
              % a.lookback)
        _LOG.info("  unstable while the full-sample figure above is not, that is")
        _LOG.info("  the whole argument for estimating alpha_i once rather than")
        _LOG.info("  rolling it:")
        _LOG.info("  %-8s %10s %10s %10s %10s"
              % ("series", "median", "sd", "5%", "95%"))
        _LOG.info("  " + "-" * 52)
        for sym in [SYSTEMATIC_ID] + names:
            if sym not in rets:
                continue
            r = rets[sym].dropna().to_numpy()
            vals = []
            for i in range(a.lookback, len(r), 21):
                al, _, _ = ar1_alpha(np.cumsum(r[i - a.lookback:i]),
                                     detrend=a.detrend)
                if np.isfinite(al):
                    vals.append(al)
            if not vals:
                continue
            v = np.array(vals)
            _LOG.info("  %-8s %10.3f %10.3f %10.3f %10.3f"
                  % (sym, np.median(v), v.std(ddof=1),
                     np.quantile(v, 0.05), np.quantile(v, 0.95)))
    _LOG.info("")


if __name__ == "__main__":
    main()
