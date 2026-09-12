"""Hypothetical worst-of autocallable on C / BAC / JPM: price and stress it.

Prices the note under the liquidity-adjusted dynamics and reprices it after a
prescribed SYSTEMATIC spot shock, which each name feels differently: the shock
reaches name i through the two-channel translation in shock_to_name.py, so its
beta, its correlation with the systematic diffusion and its gap loading all
decide how much of x it takes.

Spot shocks only. The implied-volatility surface is held fixed, which is not
what happens in a real liquidity event - the surface steepens with gamma_i -
so treat these as delta-space numbers until the option history is in.

    python poc/autocallable_stress.py
    python poc/autocallable_stress.py --paths 500000 --coupon 0.025
"""
import argparse
import os
import glob
import re
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Library.Parameters import ParametersConstant                 # noqa: E402
from Library.Random import RandomMT19937                          # noqa: E402
from Library.StatisticsMC import StatisticsMCMean                  # noqa: E402
from Library.PathDependent import PathDependentWorstOfAutocallable  # noqa: E402
from Library.ExoticEngine import (ExoticEngineKimYi,               # noqa: E402
                                  ExoticEngineBlackScholesMerton)
from poc.shock_to_name import (name_shock, model_horizon,           # noqa: E402
                               systematic_es, es_ladder_compounded,
                               model_horizon_from_ladder, HORIZON_LADDER)

from Library.Logging import report as _report  # noqa: E402

_LOG = _report(__name__)

_DP = {"money": 4, "vega": 4, "corr": 4, "delta": 4}

C = lambda v: ParametersConstant(np.array(float(v)))

# Fitted P-measure parameters. BOTH blocks are the TIGHT-PRIOR (skew-tight)
# arm at DAILY resolution, read on 2026-09-10 from the rebuilt store:
#
#   ^SPX__skewtight_913f43b7                     5,131 daily dates
#   {C,BAC,JPM}__rolling__skewtight_913f43b7     same grid, --anchor rolling
#
# The store was rebuilt from empty on 2026-09-09 (11h29m) so that exactly one
# drawer per series exists, every one carrying a manifest. The previous store
# held thirteen SPX drawers, five of them untagged, and one of those - a
# superseded pre-bugfix run - had silently supplied the 2009 and covid rows.
#
# THE DAILY RUN VALIDATED THE MONTHLY GRID. Full-sample means over 5,131 daily
# dates against 245 monthly ones from the 10,000-draw archive:
#
#            daily(5131)   monthly(245)
#   alpha       0.4898       0.4898
#   sigma       0.1513       0.1512
#   pprob       0.5652       0.5652
#   lambda      8.8065       8.7657
#   eta1       50.2951      50.2986
#   eta2       26.6333      26.6165
#
# Twenty times the compute moved nothing. Daily resolution is for the
# backfilled SERIES, which needs a value on every date; for a vintage MEAN the
# monthly grid was already right.
#
# VINTAGE WINDOWS. A fit dated t describes the 252 trading days BEHIND t, so a
# crisis vintage is not the crisis dates - it is the valuation dates whose
# window is loaded with the crisis, which sits later than the event.
#
#   gfc        20090309-20090915   Wikipedia dates the bear market 2007-10-09
#              (137 dates)         to 2009-03-09, S&P 1565.15 -> 676.53,
#                                  -56.78%. That is 17 months - LONGER than the
#                                  252-day window - so no window holds all of
#                                  it and the acute phase has to be chosen.
#                                  These are the dates whose window spans
#                                  Lehman (2008-09-15) through the trough.
#
#   covid      20200407-20210219   Wikipedia dates the crash 2020-02-20 to
#              (229 dates)         2020-04-07. Seven weeks fits inside a
#                                  window, so this is every date whose window
#                                  contains all of it.
#
#   gfc_cal, covid_cal are the calendar-year means, kept because they are what
#   was reported before and the dilution is worth seeing: calendar 2020 opens
#   with three months whose windows are entirely pre-crash, and it reads sigma
#   0.1378 / lambda 27.87 against the loaded 0.1573 / 31.77.
#
# ON ALPHA. It reads 0.487-0.497 across every vintage, which is not the data
# speaking. The tight prior is Beta(9.5, 9.5): prior mean 0.5, prior sd 0.1118,
# against a fitted posterior sd of 0.1106 - 99% of the prior width retained, so
# a 252-day window updates alpha almost not at all. The wide arm behaves the
# same (Beta(2,2), prior sd 0.2236, posterior 0.2052). Treat alpha as an
# assumption and check anything leaning on it against another value. Contrast
# sigma, which travels 43 posterior sd across the sample.
NAMES = ["C", "BAC", "JPM"]


def _pairs(names):
    """Upper-triangle order over names - the order total_diffusion_corr emits."""
    return ["%s-%s" % (names[i], names[j])
            for i in range(len(names)) for j in range(i + 1, len(names))]


def set_names(names):
    """Repoint the basket. NAMES and PAIRS are read by name everywhere else."""
    global NAMES, PAIRS
    NAMES = list(names)
    PAIRS = _pairs(NAMES)

IDIO_BY_VINTAGE = {
    "full": {
        "C":   dict(dBETAI=1.3090, dKAPPAI=0.2020, dGAMMAI=2.2737, dRHOIX=0.2951, dMUI=0.0173),
        "BAC": dict(dBETAI=1.2784, dKAPPAI=0.1946, dGAMMAI=2.2367, dRHOIX=0.2916, dMUI=0.0563),
        "JPM": dict(dBETAI=1.1109, dKAPPAI=0.1599, dGAMMAI=2.0100, dRHOIX=0.2795, dMUI=0.1223),
    },
    "gfc": {
        "C":   dict(dBETAI=1.7989, dKAPPAI=0.7753, dGAMMAI=5.8761, dRHOIX=0.4055, dMUI=-0.7154),
        "BAC": dict(dBETAI=1.9253, dKAPPAI=0.7899, dGAMMAI=4.4615, dRHOIX=0.3842, dMUI=-0.3060),
        "JPM": dict(dBETAI=1.4839, dKAPPAI=0.5411, dGAMMAI=3.9006, dRHOIX=0.3346, dMUI=-0.1711),
    },
    "gfc_cal": {
        "C":   dict(dBETAI=1.7880, dKAPPAI=0.6870, dGAMMAI=5.7347, dRHOIX=0.3879, dMUI=-0.6437),
        "BAC": dict(dBETAI=1.8225, dKAPPAI=0.6742, dGAMMAI=4.8565, dRHOIX=0.3706, dMUI=-0.3938),
        "JPM": dict(dBETAI=1.4650, dKAPPAI=0.4848, dGAMMAI=3.9496, dRHOIX=0.3276, dMUI=-0.1450),
    },
    "covid": {
        "C":   dict(dBETAI=1.4775, dKAPPAI=0.2346, dGAMMAI=2.2481, dRHOIX=0.3072, dMUI=-0.2713),
        "BAC": dict(dBETAI=1.4022, dKAPPAI=0.2167, dGAMMAI=1.8896, dRHOIX=0.3006, dMUI=-0.0413),
        "JPM": dict(dBETAI=1.2455, dKAPPAI=0.1858, dGAMMAI=1.8268, dRHOIX=0.2893, dMUI=-0.0448),
    },
    "covid_cal": {
        "C":   dict(dBETAI=1.4310, dKAPPAI=0.1999, dGAMMAI=2.0662, dRHOIX=0.3023, dMUI=-0.1468),
        "BAC": dict(dBETAI=1.3588, dKAPPAI=0.1859, dGAMMAI=1.7535, dRHOIX=0.2962, dMUI=0.0442),
        "JPM": dict(dBETAI=1.2072, dKAPPAI=0.1595, dGAMMAI=1.7006, dRHOIX=0.2856, dMUI=0.0461),
    },
}

SYS_BY_VINTAGE = {
    "full": dict(dALPHA=0.4898, dSIGMA=0.1513, dPPROB=0.5652,
                 dLAMB=8.8065, dETA1=50.2951, dETA2=26.6333),
    "gfc": dict(dALPHA=0.4961, dSIGMA=0.3942, dPPROB=0.5524,
                dLAMB=12.0943, dETA1=39.5485, dETA2=24.3207),
    "gfc_cal": dict(dALPHA=0.4973, dSIGMA=0.3555, dPPROB=0.5541,
                    dLAMB=12.7647, dETA1=40.1140, dETA2=24.4524),
    "covid": dict(dALPHA=0.4938, dSIGMA=0.1573, dPPROB=0.5218,
                  dLAMB=31.7683, dETA1=36.7096, dETA2=25.6968),
    "covid_cal": dict(dALPHA=0.4878, dSIGMA=0.1378, dPPROB=0.5138,
                      dLAMB=27.8695, dETA1=40.3223, dETA2=26.4494),
}

VINTAGE = os.environ.get("STRESS_VINTAGE", "full")
IDIO = IDIO_BY_VINTAGE[VINTAGE]
SYS_P = SYS_BY_VINTAGE[VINTAGE]
# No option data, so nothing to calibrate Q on: Q is taken to BE the estimated
# P set, which is the explicit assumption that the jump risk premium is zero.
# On the bsm engine only sigma is read from it - phi_i and R_ij are the whole
# specification and no jump block is used at all.
SYS_Q = dict(sigma=SYS_P["dSIGMA"], pprob=SYS_P["dPPROB"], lamb=SYS_P["dLAMB"],
             eta1=SYS_P["dETA1"], eta2=SYS_P["dETA2"], alpha=SYS_P["dALPHA"])


def set_vintage(v):
    """Repoint the module-level parameter blocks. Everything downstream reads
    IDIO / SYS_P / SYS_Q by name, so this is all that has to change."""
    global IDIO, SYS_P, SYS_Q, VINTAGE
    VINTAGE = v
    IDIO = IDIO_BY_VINTAGE[v]
    SYS_P = SYS_BY_VINTAGE[v]
    SYS_Q.update(sigma=SYS_P["dSIGMA"], pprob=SYS_P["dPPROB"],
                 lamb=SYS_P["dLAMB"], eta1=SYS_P["dETA1"],
                 eta2=SYS_P["dETA2"], alpha=SYS_P["dALPHA"])



# ----------------------------------------------------------------------------
# Reading a vintage straight out of the drawers.
#
# The blocks above are transcribed numbers, and transcription is exactly how
# the 2009 and covid rows spent days carrying values from a superseded
# pre-bugfix drawer. --as-of removes the step: it reads the parameters for one
# valuation date out of the store and prints which drawer and which date they
# came from, so the provenance is in the output rather than in a comment.
# ----------------------------------------------------------------------------
ARM = "skewtight_913f43b7"


def _drawer_dates(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, "*.csv")):
        m = re.search(r"_(\d{8})\.csv$", os.path.basename(f))
        if m:
            out[m.group(1)] = f
    return out


def _asof_row(drawer, date):
    """The row for `date`, or the latest date at or before it.

    As-of rather than exact: the grid has gaps wherever a fit failed, and
    silently pricing off a date the caller did not ask for is only safe if the
    date actually used is reported. main() prints it.
    """
    from Library.DataAccess import PMLE_DIR
    folder = os.path.join(PMLE_DIR, drawer)
    if not os.path.isdir(folder):
        raise SystemExit("no drawer %s in %s" % (drawer, PMLE_DIR))
    dates = _drawer_dates(folder)
    if not dates:
        raise SystemExit("drawer %s holds no dated CSVs" % drawer)
    usable = sorted(d for d in dates if d <= date)
    if not usable:
        raise SystemExit("drawer %s has nothing at or before %s (earliest %s)"
                         % (drawer, date, min(dates)))
    got = usable[-1]
    return got, pd.read_csv(dates[got]).iloc[0]


def _window_mean(drawer, beg, end, cols):
    """Mean of `cols` over every dated CSV in [beg, end]. Returns (n, dict)."""
    from Library.DataAccess import PMLE_DIR
    folder = os.path.join(PMLE_DIR, drawer)
    if not os.path.isdir(folder):
        raise SystemExit("no drawer %s in %s" % (drawer, PMLE_DIR))
    files = [f for d, f in sorted(_drawer_dates(folder).items())
             if beg <= d <= end]
    if not files:
        raise SystemExit("drawer %s has no dates in %s..%s" % (drawer, beg, end))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return len(files), {c: float(df[c].mean()) for c in cols}


def set_window(beg, end, arm=ARM):
    """A vintage averaged over a date window, read from the drawers.

    Same idea as --as-of but over a range: it is what the transcribed gfc and
    covid blocks are, computed rather than copied. That matters for a basket
    the blocks do not cover - the point of --names - and it removes the
    transcription step that put a superseded drawer's numbers into the crisis
    vintages for days.
    """
    global IDIO, SYS_P, SYS_Q, VINTAGE
    sysc = ("dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2")
    idic = ("dBETAI", "dKAPPAI", "dGAMMAI", "dRHOIX", "dMUI")
    nsys, SYS_P = _window_mean("^SPX__%s" % arm, beg, end, sysc)
    counts, idio = {}, {}
    for n in NAMES:
        counts[n], idio[n] = _window_mean("%s__rolling__%s" % (n, arm),
                                          beg, end, idic)
    IDIO = idio
    VINTAGE = "window %s-%s" % (beg, end)
    SYS_Q.update(sigma=SYS_P["dSIGMA"], pprob=SYS_P["dPPROB"],
                 lamb=SYS_P["dLAMB"], eta1=SYS_P["dETA1"],
                 eta2=SYS_P["dETA2"], alpha=SYS_P["dALPHA"])
    return nsys, counts


def set_asof(date, arm=ARM):
    """Point the parameter blocks at one valuation date in the store."""
    global IDIO, SYS_P, SYS_Q, VINTAGE
    if date in ("latest", "today"):
        date = "99999999"
    sys_date, row = _asof_row("^SPX__%s" % arm, date)
    SYS_P = {k: float(row[k]) for k in
             ("dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2")}
    idio, name_dates = {}, {}
    for n in NAMES:
        d, r = _asof_row("%s__rolling__%s" % (n, arm), date)
        name_dates[n] = d
        idio[n] = {k: float(r[k]) for k in
                   ("dBETAI", "dKAPPAI", "dGAMMAI", "dRHOIX", "dMUI")}
    IDIO = idio
    VINTAGE = "as-of %s" % sys_date
    SYS_Q.update(sigma=SYS_P["dSIGMA"], pprob=SYS_P["dPPROB"],
                 lamb=SYS_P["dLAMB"], eta1=SYS_P["dETA1"],
                 eta2=SYS_P["dETA2"], alpha=SYS_P["dALPHA"])
    return sys_date, name_dates


# -50% to +20% in 5% steps. The range is deliberately asymmetric: with the
# shocks instantaneous there is no compounding to damp the upside, so a large
# positive shock passes through gamma_i ungoverned - exp(gamma_i Y) - 1 has no
# ceiling - and past about +20% the translated y_i stop being scenarios. The
# downside needs no such cap; the -100% floor binds on its own.
SHOCKS = [round(-0.50 + 0.05 * k, 10) for k in range(15)]

# The prescribed shock is applied UNCHANGED to C, BAC and JPM: all three move
# together by x. No systematic-to-name translation, so beta_i, gamma_i and
# rho_iX do not enter the shock - they still price the note, but they no longer
# decide how much of the shock each name feels. --translate restores the
# model's two-channel route.

# Shocks are INSTANTANEOUS. Each is applied as a single Appendix B increment
# at the full shock size - no holding period, no compounding over a horizon.
#
# There was a ladder here mapping shock size to a holding period (5% -> 1 day,
# 60% -> 90 days). Six of its seven entries were invented, and the choice moved
# the deepest put by 14%, so it was an undocumented free parameter sitting in
# the middle of every row. Instantaneous is the assumption that needs no
# defending: it is the model's own one-period increment, used as written.
#
# --horizon h re-enables a holding period, translating per day and compounding
# over h days. Anything other than 1 is an assumption that has to be justified.


def phi(sig, b, k, r):
    return np.sqrt((sig * b) ** 2 + 2 * sig * b * k * r + k ** 2)


def total_diffusion_corr(sig, vol_scale=1.0):
    """R_ij the engine wants: the correlation of the TOTAL diffusion.

    rho_ij = 0 is the modelling assumption, but the engine multiplies its
    Cholesky factor by the scalar phi_i, so the slot takes
    (F_ij + kappa_i kappa_j rho_ij) / (phi_i phi_j), and at rho_ij = 0 that is
    the common-factor part, not zero.

    vol_scale must be applied to kappa here as well as to sigma. Scaling only
    sigma leaves R_ij drifting with the bump, which turns a vega into a mixed
    vega-plus-correlation sensitivity - and for a worst-of, correlation is the
    larger of the two.
    """
    sig = sig * vol_scale
    b = np.array([IDIO[n]["dBETAI"] for n in NAMES])
    k = np.array([IDIO[n]["dKAPPAI"] for n in NAMES]) * vol_scale
    r = np.array([IDIO[n]["dRHOIX"] for n in NAMES])
    ph = phi(sig, b, k, r)
    out = []
    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            F = sig ** 2 * b[i] * b[j] + sig * b[i] * k[j] * r[j] + sig * b[j] * k[i] * r[i]
            out.append(F / (ph[i] * ph[j]))
    return out, ph


def build_bsm(spots, product, paths, a, seed=20240114, vol_shift=0.0,
              corr_shift=0.0):
    """The note under Black-Scholes at the model's own diffusion parameters.

    Corollary 3.2 already collapses each name's dynamics onto one effective
    Brownian motion with volatility

        phi_i = sqrt((sigma beta_i)^2 + 2 sigma beta_i kappa_i rho_iX + kappa_i^2)

    and the pair correlation of those effective Brownian motions is

        R_ij = (sigma^2 beta_i beta_j + sigma beta_i kappa_j rho_jX
                + sigma beta_j kappa_i rho_iX + kappa_i kappa_j rho_ij)
               / (phi_i phi_j)

    at rho_ij = 0. So phi_i IS the volatility and R_ij IS the correlation the
    exotic needs - they go straight into ExoticEngineBlackScholesMerton. All
    P-measure estimates; with no option data there is nothing to calibrate a Q
    jump block on, and none is used.

    vol_shift is added to phi_i in absolute vol and corr_shift to R_ij; both
    take a scalar to move everything together, or a vector to move one name -
    or one PAIR - at a time. Pairs are in numpy triu order: (C,BAC), (C,JPM),
    (BAC,JPM).
    """
    n = len(NAMES)
    rij, ph = total_diffusion_corr(SYS_Q["sigma"])
    cs = np.broadcast_to(np.asarray(corr_shift, dtype=float), (len(rij),))
    rij = [min(0.999, max(-0.499, v + w)) for v, w in zip(rij, cs)]
    return ExoticEngineBlackScholesMerton(
        the_product=product, risk_free_rate=C(a.rate),
        dividend_yield=[C(a.dividend)] * n,
        imp_volatility=[C(v + w) for v, w in
                        zip(ph, np.broadcast_to(np.asarray(vol_shift, dtype=float),
                                                (n,)))],
        rand_generator=RandomMT19937(np.uint64(seed)),
        spot_price=np.asarray(spots, dtype=float),
        number_of_paths=np.uint64(paths),
        correlations=[C(v) for v in rij])


def build(spots, product, paths, seed=20240114, vol_scale=1.0,
          corr_shift=0.0):
    """vol_scale multiplies sigma AND every kappa_i by the same factor.

    phi_i^2 = (sigma beta_i)^2 + 2 sigma beta_i kappa_i rho_iX + kappa_i^2 is
    homogeneous of degree 2 in (sigma, kappa), so a common factor c sends
    phi_i -> c phi_i exactly. F_ij scales by c^2 and phi_i phi_j by c^2 too, so
    R_ij is INVARIANT: this is a pure volatility bump with the correlation and
    the jump block held fixed, which is what a vega should be.
    """
    n = len(NAMES)
    rij, _ = total_diffusion_corr(SYS_Q["sigma"], vol_scale)
    # corr_shift moves the TOTAL diffusion correlation the engine uses, which
    # is the quantity comparable to a plain correlation bump. The model's own
    # free parameter is rho_ij, and it reaches R_ij only through
    # d R_ij / d rho_ij = kappa_i kappa_j / (phi_i phi_j) - about 0.39 for
    # these three - so 10 bps of rho_ij is under 4 bps of R_ij.
    rij = [min(0.999, max(-0.499, v + corr_shift)) for v in rij]
    return ExoticEngineKimYi(
        the_product=product, risk_free_rate=C(0.04), dividend_yield=[C(0.)] * n,  # kimyi path
        sigma=C(SYS_Q["sigma"] * vol_scale),
        kappai=[C(IDIO[x]["dKAPPAI"] * vol_scale) for x in NAMES],
        rhoij=[C(v) for v in rij],
        betai=[C(IDIO[x]["dBETAI"]) for x in NAMES],
        rhoix=[C(IDIO[x]["dRHOIX"]) for x in NAMES],
        gammai=[C(IDIO[x]["dGAMMAI"]) for x in NAMES],
        p_prob=C(SYS_Q["pprob"]), lamb=C(SYS_Q["lamb"]),
        eta1=C(SYS_Q["eta1"]), eta2=C(SYS_Q["eta2"]),
        generator=RandomMT19937(np.uint64(seed)),
        spot=np.asarray(spots, dtype=float), number_of_paths=np.uint64(paths))


def price(spots, initial, args, seed=20240114, protection=None, vol_shift=0.0,
          corr_shift=0.0):
    prod = PathDependentWorstOfAutocallable(
        fixing_times=np.arange(1, args.periods + 1) / 4.,
        initial_levels=np.asarray(initial, dtype=float),
        autocall_barriers=args.autocall, coupon_barriers=args.coupon_barrier,
        coupon_amounts=args.coupon,
        protection_barrier=args.protection if protection is None else protection,
        notional=args.notional, memory=True)
    g = StatisticsMCMean()
    eng = (build_bsm(spots, prod, args.paths, args, seed, vol_shift, corr_shift)
           if args.engine == "bsm"
           else build(spots, prod, args.paths, seed,
                      1.0 + float(np.mean(vol_shift)) / 0.2983,
                      float(np.mean(corr_shift))))
    eng.do_simulation(gatherer=g)
    return g.get_result_so_far()[0, 0]


def put_value(spots, initial, args, seed=20240114, vol_shift=0.0,
              corr_shift=0.0):
    """The embedded knock-in put, by removing it.

    Setting the protection barrier to zero makes the maturity test unbreachable,
    so the note redeems at par whenever it survives to maturity: same coupons,
    same autocall, no downside participation. What that version is worth over
    the real note IS the put the investor is short and the issuer is long. Same
    seed on both legs, so the difference is the feature and not MC noise.
    """
    protected = price(spots, initial, args, seed, protection=0.0,
                      vol_shift=vol_shift, corr_shift=corr_shift)
    actual = price(spots, initial, args, seed, vol_shift=vol_shift,
                   corr_shift=corr_shift)
    return protected - actual, actual


PAIRS = _pairs(NAMES)                      # numpy triu order over NAMES


def put_greeks(spots, initial, args, phibar=None, seed=20240114):
    """Put, vega per NAME, correlation P&L per PAIR.

    Both greeks are reported one risk factor at a time - bump that name's phi_i
    or that pair's R_ij and hold everything else - because that is how the
    exposure is hedged, and because on a worst-of the factors are not
    interchangeable: the name most likely to BE the worst carries most of the
    vega, and the pair most likely to contain it carries most of the
    correlation risk. A parallel bump averages that away.

    Vega is one-sided at the reported bump, the desk definition. Correlation is
    central over a much wider bump scaled back to 10 bps - a one-sided
    difference of two Monte Carlo prices 10 bps apart is almost all noise.
    A single-pair bump of +/-5 points keeps the 3x3 matrix comfortably positive
    definite here (1 + 2abc - a^2 - b^2 - c^2 runs 0.30 to 0.35 against 0.33 at
    base), so the Cholesky in the engine is safe.
    """
    b, cb = args.vol_bump, args.corr_bump
    n, npair = len(NAMES), len(PAIRS)
    put, pv = put_value(spots, initial, args, seed)

    vegas = []
    for i in range(n):
        sh = np.zeros(n)
        sh[i] = b
        up, _ = put_value(spots, initial, args, seed, vol_shift=sh)
        vegas.append(up - put)

    cegas = []
    for k in range(npair):
        sh = np.zeros(npair)
        sh[k] = cb
        cup, _ = put_value(spots, initial, args, seed, corr_shift=+sh)
        cdn, _ = put_value(spots, initial, args, seed, corr_shift=-sh)
        cegas.append((cup - cdn) / (2.0 * cb) * 0.0010)

    return put, pv, np.array(vegas), np.array(cegas)


def _cached_es_ladder(rungs, args):
    """The compounded ES ladder depends only on SYS_P, alpha, the rungs and the
    path count - none of which change when the product terms or the pricing
    path count do. Caching it keeps a 34-second Monte Carlo off every rerun."""
    import hashlib
    import json
    key = hashlib.sha1(json.dumps(
        [sorted(SYS_P.items()), args.es_alpha, list(rungs), args.es_paths],
        sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_es_ladder_cache", key + ".json")
    if os.path.exists(path):
        return {int(k): tuple(v) for k, v in json.load(open(path)).items()}
    lad = es_ladder_compounded(SYS_P, args.es_alpha, rungs,
                               n_paths=args.es_paths)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({str(k): list(v) for k, v in lad.items()}, open(path, "w"))
    return lad



# ----------------------------------------------------------------------------
# Conditional colour. Off unless stdout is a terminal, so a redirect to a file
# or a pipe into grep stays plain text - ANSI escapes in a log are worse than
# no colour at all. --color always forces it on, --color never off.
# ----------------------------------------------------------------------------
_ANSI = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
         "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
         "cyan": "\033[36m", "bred": "\033[91m", "bgreen": "\033[92m"}
_COLOR = [False]


def _use_color(mode):
    _COLOR[0] = (mode == "always" or
                 (mode == "auto" and sys.stdout.isatty()))


def _paint(text, *codes):
    """Wrap in ANSI. A code is either a name in _ANSI or "fgN" for 256-colour N."""
    if not _COLOR[0] or not codes:
        return text
    lead = "".join("\033[38;5;%sm" % c[2:] if c.startswith("fg") else _ANSI[c]
                   for c in codes)
    return lead + text + _ANSI["reset"]


def _sign_color(v):
    """Green for a gain to the issuer, red for a loss, dim at the flat point."""
    if abs(v) < 1e-9:
        return ("dim",)
    return ("green",) if v > 0 else ("red",)


# Grey through deep blue to cyan. A heat map wants intensity to read at a
# glance; bold-versus-normal does not, in most terminal themes, which is what
# the first version used and why the peak was no easier to find than before.
_RAMP = (244, 25, 32, 39, 51)


def _heat(v, vmax):
    """Colour a cell by its size relative to the column's maximum.

    Where vega peaks was a live question about this table, and the point of
    the ramp is that the answer should be visible without reading fifteen rows
    of digits. Bold is kept for the top band on top of the colour, so the peak
    itself still stands out on a monochrome terminal.
    """
    if vmax <= 0:
        return ()
    f = min(1.0, abs(v) / vmax)
    idx = min(len(_RAMP) - 1, int(f * len(_RAMP)))
    codes = ("fg%d" % _RAMP[idx],)
    if f >= 0.85:
        codes += ("bold",)
    return codes


def _dp(ref):
    """Decimals worth showing for a column whose values are the size of ref."""
    a = abs(ref)
    return 0 if a >= 1000 else (2 if a >= 10 else 4)


def _m(v, signed=False, kind="money"):
    """Money, with thousands separators, at a precision fixed for the run.

    At --notional 100 every field is two digits and four decimals are the
    signal; at $1mm the note runs to seven digits and four decimals are noise
    that pushes the columns apart. So precision follows size - but it is
    decided ONCE per column, not per value. Per value puts 2,927.15 in a
    column of whole thousands, which is harder to read than the ragged
    decimals it was meant to fix.

    Three scales, because they differ by orders of magnitude: the note and the
    put, a vega, and a correlation bump worth a few tens of dollars against a
    note worth a million.
    """
    sign = "+" if signed and v >= 0 else ""
    return sign + format(v, ",.%df" % _DP[kind])


def _set_precision(base, vega0, cega0, deltas):
    _DP["money"] = _dp(base)
    _DP["vega"] = _dp(max(abs(v) for v in vega0))
    _DP["corr"] = _dp(max(abs(v) for v in cega0))
    _DP["delta"] = _dp(max(abs(v) for v in deltas))



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--periods", type=int, default=12, help="quarterly observations")
    ap.add_argument("--autocall", type=float, default=1.00)
    ap.add_argument("--coupon-barrier", type=float, default=0.70)
    ap.add_argument("--coupon", type=float, default=0.02, help="per period")
    ap.add_argument("--protection", type=float, default=0.50)
    ap.add_argument("--notional", type=float, default=100.)
    ap.add_argument("--horizon", type=int, default=1,
                    help="holding period in days over which the shock plays "
                         "out; 1 (default) is an instantaneous shock")
    ap.add_argument("--vol-bump", type=float, default=0.01,
                    help="vol bump defining vega, in absolute vol; the "
                         "reported vega is the price change for that bump")
    ap.add_argument("--no-translate", dest="translate", action="store_false",
                    help="apply the systematic shock directly to every name "
                         "instead of routing it through the model's "
                         "systematic -> name translation")
    ap.set_defaults(translate=True)
    ap.add_argument("--names", default=None,
                    help="comma-separated basket, e.g. CCL,RCL,LUV. Needs "
                         "--as-of or --window, since the transcribed vintages "
                         "only cover C,BAC,JPM.")
    ap.add_argument("--window", default=None, metavar="BEG:END",
                    help="vintage averaged over a date window read from the "
                         "drawers, e.g. 20090309:20090915. Works for any "
                         "basket; no transcription.")
    ap.add_argument("--color", choices=("auto", "always", "never"),
                    default="auto",
                    help="conditional colour in the tables: sign on the P&L "
                         "columns, bold at each greek column's peak and dim "
                         "near its floor. auto = only when stdout is a tty")
    ap.add_argument("--as-of", default=None,
                    help="read the parameters for ONE valuation date out of "
                         "the drawers (YYYYMMDD, or 'latest') instead of using "
                         "a transcribed vintage. As-of: the latest date at or "
                         "before the one asked for is used, and reported.")
    ap.add_argument("--arm", default=ARM,
                    help="drawer suffix to read with --as-of "
                         "(default %(default)s)")
    ap.add_argument("--vintage", choices=tuple(IDIO_BY_VINTAGE), default="full",
                    help="which vintage of the rolling fit to use: full-sample "
                         "means, or the 2009 yearly means")
    ap.add_argument("--engine", choices=("bsm", "kimyi"), default="bsm",
                    help="bsm: back the model out into an implied vol per name "
                         "and price the exotic with "
                         "ExoticEngineBlackScholesMerton. kimyi: run the jump "
                         "process straight through the path generator.")
    ap.add_argument("--rate", type=float, default=0.04)
    ap.add_argument("--dividend", type=float, default=0.0)
    ap.add_argument("--iv-strike", type=float, default=None,
                    help="absolute strike for the implied-vol backout; "
                         "defaults to the protection barrier level")
    ap.add_argument("--iv-expiry", type=float, default=None,
                    help="expiry for the implied-vol backout in years; "
                         "defaults to the note tenor")
    ap.add_argument("--corr-bump", type=float, default=0.05,
                    help="half-bump in R_ij for the central correlation "
                         "difference; the reported number is per 10 bps")
    ap.add_argument("--es-mode", choices=("compound", "sum"), default="compound",
                    help="how the h-day move is aggregated from daily moves. "
                         "compound: 1+X_h = prod(1+r_k), by Monte Carlo. "
                         "sum: X_h = sum(r_k), by the FFT stack.")
    ap.add_argument("--es-horizons", type=int, nargs="+",
                    default=[1, 5, 10, 15, 20],
                    help="horizons in days at which the systematic ES is taken")
    ap.add_argument("--es-paths", type=int, default=1_000_000,
                    help="paths for the compounded ES ladder")
    ap.add_argument("--es-alpha", type=float, default=0.025,
                    help="expected-shortfall level defining h*: 0.025 is the "
                         "97.5%% ES for down-shocks and its 2.5%% mirror for up")
    a = ap.parse_args()
    _use_color(a.color)
    if a.names:
        set_names([x.strip() for x in a.names.split(",") if x.strip()])
        if not a.as_of and not a.window:
            raise SystemExit(
                "--names needs --as-of or --window: the transcribed vintages "
                "cover only %s." % ",".join(sorted(IDIO_BY_VINTAGE["full"])))
    asof_note = None
    win_note = None
    if a.window:
        beg, _, end = a.window.partition(":")
        if not end:
            raise SystemExit("--window wants BEG:END, e.g. 20090309:20090915")
        win_note = (beg, end, a.arm) + set_window(beg, end, a.arm)
    elif a.as_of:
        sd, nd = set_asof(a.as_of, a.arm)
        asof_note = (sd, nd, a.arm)
    else:
        set_vintage(a.vintage)
    if a.iv_strike is None:
        a.iv_strike = 100.0 * a.protection
    if a.iv_expiry is None:
        a.iv_expiry = a.periods / 4.0

    spot0 = np.full(len(NAMES), 100.)
    rij, ph = total_diffusion_corr(SYS_Q["sigma"])

    _LOG.info("=" * 74)
    _LOG.info("Worst-of autocallable :: %s" % " / ".join(NAMES))
    _LOG.info("=" * 74)
    _LOG.info("  %dy quarterly, autocall %.0f%%, coupon %.2f%%/q above %.0f%%, "
          "protection %.0f%%, memory"
          % (a.periods / 4, 100 * a.autocall, 100 * a.coupon,
             100 * a.coupon_barrier, 100 * a.protection))
    _LOG.info("  %s paths, engine %s, parameter vintage %s"
          % (f"{a.paths:,}", a.engine, VINTAGE))
    if a.engine == "bsm":
        _LOG.info("  Priced under Black-Scholes at the model's own diffusion")
        _LOG.info("  parameters: phi_i is the volatility, R_ij the correlation,")
        _LOG.info("  both from the P-measure estimates. No option data, so no Q")
        _LOG.info("  jump block is calibrated and none is used.")
    _LOG.info("")
    if win_note:
        beg, end, arm, nsys, counts = win_note
        _LOG.info("  Vintage AVERAGED over a window, read from the drawers -")
        _LOG.info("  computed, not transcribed.")
        _LOG.info("    window      %s .. %s" % (beg, end))
        _LOG.info("    systematic  ^SPX__%s  %d dates" % (arm, nsys))
        for n in NAMES:
            _LOG.info("    %-11s %s__rolling__%s  %d dates"
                  % (n, n, arm, counts[n])
                  + ("   <- fewer than the systematic fit" if counts[n] < nsys
                     else ""))
    elif asof_note:
        sd, nd, arm = asof_note
        _LOG.info("  ONE valuation date, read from the drawers - not a transcribed")
        _LOG.info("  vintage, and not an average over a window.")
        _LOG.info("    systematic  ^SPX__%s  %s" % (arm, sd))
        for n in NAMES:
            _LOG.info("    %-11s %s__rolling__%s  %s"
                  % (n, n, arm, nd[n])
                  + ("   <- not the requested date" if nd[n] != sd else ""))
    else:
        _LOG.info("  Parameters: skew-tight arm on BOTH blocks, daily store rebuilt")
        _LOG.info("  2026-09-09, name fits conditioned on the same systematic drawer.")
        _LOG.info("  gfc = dates whose 252d window spans Lehman to the 2009-03-09")
        _LOG.info("  trough; covid = dates whose window holds the 2020-02-20/04-07")
        _LOG.info("  crash. alpha sits on its prior in every vintage - an assumption.")

    # The ES ladder has to carry every prescribed h as well as its own rungs,
    # since ES(h) is reported at the horizon the shock is actually applied over.
    # The ES ladder and the model-implied horizon h* are no longer reported, so
    # the Monte Carlo behind them is not run. systematic_es / model_horizon in
    # shock_to_name still compute them if the columns are ever wanted back.

    put0, base, vega0, cega0 = put_greeks(spot0, spot0, a)

    # Two delta numbers, because they answer different questions.
    #
    #   d_pct   the ISSUER's P&L if ONE name moves +1% and the other two do
    #           not. One-sided on purpose: this note is convex enough at these
    #           volatilities that the central difference is not what a 1% move
    #           actually costs, and the P&L is the number a desk sizes against.
    #   d_unit  dV/dS, the per-unit-spot delta - what you hedge with, in
    #           shares, and NOT a per-1% figure.
    #
    # Both are the issuer's: short the note, so a name rallying is a loss.
    bump = 0.01
    d_pct, d_unit = [], []
    for i in range(len(NAMES)):
        up, dn = spot0.copy(), spot0.copy()
        up[i] *= 1 + bump
        dn[i] *= 1 - bump
        v_up, v_dn = price(up, spot0, a), price(dn, spot0, a)
        d_pct.append(-(v_up - base))
        d_unit.append(-(v_up - v_dn) / (2 * bump * spot0[i]))
    d_pct = np.array(d_pct)
    d_unit = np.array(d_unit)
    _LOG.info("")
    _LOG.info("  name   beta   kappa   rho_iX   gamma    phi_i")
    for n, p in zip(NAMES, ph):
        d = IDIO[n]
        _LOG.info("  %-5s %6.4f %7.4f %8.4f %7.4f %8.4f"
              % (n, d["dBETAI"], d["dKAPPAI"], d["dRHOIX"], d["dGAMMAI"], p))
    _LOG.info("  R_ij: %s" % ", ".join("%.4f" % v for v in rij))
    _LOG.info("  sigma %.4f  lambda %.2f  eta %.2f/%.2f"
          % (SYS_P["dSIGMA"], SYS_P["dLAMB"], SYS_P["dETA1"], SYS_P["dETA2"]))
    _set_precision(base, vega0, cega0, list(d_pct) + list(d_unit))
    _LOG.info("")
    _LOG.info("  note %s   embedded put %s" % (_m(base), _m(put0)))
    _LOG.info("  vega  $ per +1 vol point:  %s"
          % "  ".join("%s %s" % (n, _m(v, signed=True, kind="vega"))
                       for n, v in zip(NAMES, vega0)))
    _LOG.info("  corr  $ per +10 bps:      %s"
          % "  ".join("%s %s" % (p, _m(v, signed=True, kind="corr"))
                       for p, v in zip(PAIRS, cega0)))
    _LOG.info("  issuer P&L per +1%% on one name: %s"
          % ", ".join("%s %s" % (n, _m(d, signed=True, kind="delta"))
                       for n, d in zip(NAMES, d_pct)))
    _LOG.info("  issuer delta dV/dS (per unit spot, for hedging): %s"
          % ", ".join("%s %s" % (n, _m(d, signed=True, kind="delta"))
                       for n, d in zip(NAMES, d_unit)))
    _LOG.info("  The put is the same note with protection removed, less this one.")
    _LOG.info("  The issuer is SHORT the note and therefore LONG that put.")
    _LOG.info("")
    if a.translate:
        _LOG.info("  x is the SYSTEMATIC shock, relative. Each name feels it through the")
        _LOG.info("  model's two-channel translation, so y_i differs by name.")
    else:
        _LOG.info("  Shocks are RELATIVE and applied UNCHANGED to all %d names:"
              % len(NAMES))
        _LOG.info("  %+.0f%% means %s each fall %.0f%%."
              % (100 * SHOCKS[0], ", ".join(NAMES), -100 * SHOCKS[0]))
    _LOG.info("  Shocks are INSTANTANEOUS - one Appendix B increment at the full")
    _LOG.info("  shock size, no holding period. y_i is what each name feels after")
    _LOG.info("  the model's systematic -> name translation.")
    _LOG.info("  EVERY greek column below is a DOLLAR amount at this notional,")
    _LOG.info("  not a parameter: cr_* is P&L per +10 bps of R_ij, NOT R_ij.")
    _LOG.info("  vega_i is the put's change for +%.0f vol point on THAT name's phi_i,"
          % (100 * a.vol_bump))
    _LOG.info("  the other two held; corr_k is +10 bps on THAT pair's R_ij, the")
    _LOG.info("  other two pairs held. dl_i is the issuer's P&L for a further")
    _LOG.info("  +1% on THAT name from the shocked spot, the other two held.")
    def _abbr(pair):
        i, j = pair.split("-")
        return (i[:3] + j[:3])[:6]

    cols = ([("x", 6)] + [("y_" + n, 8) for n in NAMES]
            + [("note", 13), ("put", 12), ("put P&L", 13)]
            + [("dl_" + n, 9) for n in NAMES]
            + [("vg_" + n, 9) for n in NAMES]
            + [("cr_" + _abbr(p), 9) for p in PAIRS])
    hfmt = "  " + " ".join("%%%ds" % w for _, w in cols)
    _LOG.info(hfmt % tuple(h for h, _ in cols))
    _LOG.info("  " + "-" * (sum(w for _, w in cols) + len(cols) - 1))
    # Collected before printing, because the heat map needs each column's
    # maximum and the rows are what produce it.
    ladder = []
    for x in SHOCKS:
        h = max(1, a.horizon)
        # x and y are both SIMPLE returns. Appendix B's Psi increment is
        # dS_i/S_i-, and the jump enters it as exp(gamma_i Y) - 1, so the
        # translation takes a relative shock in and returns a relative shock
        # out. No log conversion in either direction.
        if a.translate:
            ys = np.array([name_shock(x, SYS_P, IDIO[n], horizon_days=h)["y"]
                           for n in NAMES])
        else:
            ys = np.full(len(NAMES), x)
        if (ys <= -1.0).any():
            # The diffusion channel of the translation is LINEAR in the return,
            # b_diff * E[D|x] with b_diff = beta_i + kappa_i rho_iX / sigma, and
            # unlike the jump channel exp(gamma_i Y) - 1 it has no floor at
            # -100%. On the 2009 vintage b_diff is 2.54 for C, so a deep
            # systematic shock over a long horizon drives y_i through -1 and the
            # shocked price negative. Those rows are outside the translation's
            # range, not a scenario - refusing them beats pricing log(negative).
            ladder.append((x, ys, None, None, None, None, None))
            continue
        shocked = spot0 * (1.0 + ys)
        put, pv, vega, cega = put_greeks(shocked, spot0, a)
        # Issuer P&L for a further +1% on ONE name FROM THE SHOCKED spot,
        # the other two held. One-sided, matching the summary block: it is
        # what the next 1% costs from here, not a derivative at the origin.
        dl = []
        for i in range(len(NAMES)):
            up = shocked.copy()
            up[i] *= 1.0 + 0.01
            dl.append(-(price(up, spot0, a) - pv))
        ladder.append((x, ys, pv, put, put - put0, np.array(dl), (vega, cega)))

    live = [r for r in ladder if r[2] is not None]
    n = len(NAMES)
    dmax = [max(abs(r[5][i]) for r in live) for i in range(n)] if live else [0] * n
    vmax = [max(r[6][0][i] for r in live) for i in range(n)] if live else [0] * n
    cmax = [max(abs(r[6][1][i]) for r in live) for i in range(len(PAIRS))] \
        if live else [0] * len(PAIRS)

    for x, ys, pv, put, pnl, dl, greeks in ladder:
        xs = _paint("%+5.0f%%" % (100 * x), *(("red",) if x < 0 else
                                              ("green",) if x > 0 else ("dim",)))
        yss = " ".join("%+7.1f%%" % (100 * r) for r in ys)
        if pv is None:
            _LOG.info("  %s %s   %s" % (xs, yss,
                  _paint("-- y_i through -100%, outside range", "yellow")))
            continue
        vega, cega = greeks
        _LOG.info("  %s %s %13s %12s %s %s %s %s"
              % (xs, yss, _m(pv), _m(put),
                 _paint("%13s" % _m(pnl, signed=True), *_sign_color(pnl)),
                 " ".join(_paint("%9s" % _m(v, signed=True, kind="delta"),
                                 *_heat(v, dmax[i])) for i, v in enumerate(dl)),
                 " ".join(_paint("%9s" % _m(v, kind="vega"),
                                 *_heat(v, vmax[i])) for i, v in enumerate(vega)),
                 " ".join(_paint("%9s" % _m(v, signed=True, kind="corr"),
                                 *_heat(v, cmax[i])) for i, v in enumerate(cega))))

    # ---------------------------------------------------------------- ES table
    # A second pass where the shock is not prescribed but taken from the
    # model's own expected shortfall at horizon h. h sizes the SHOCK; it is not
    # a holding period for the translation, which stays instantaneous like the
    # table above. So the reading is "a 97.5% ES move over h days, delivered at
    # once".
    es_rungs = tuple(a.es_horizons)
    lad = _cached_es_ladder(es_rungs, a)
    _LOG.info("\n  systematic expected-shortfall shocks")
    _LOG.info("  x is no longer prescribed: it is the model's own %.1f%% expected"
          % (100 * (1 - a.es_alpha)))
    _LOG.info("  shortfall of the systematic factor over h days, and its %.1f%%"
          % (100 * a.es_alpha))
    _LOG.info("  mirror on the up side. The h-day move is built by compounding")
    _LOG.info("  daily moves, 1 + X_h = prod_k (1 + r_k).")
    _LOG.info("  h sizes the SHOCK ONLY. The translation into y_i stays")
    _LOG.info("  instantaneous, as above - h is not a holding period here.")
    ecols = ([("h", 5), ("side", 6), ("ES(h)", 9)]
             + [("y_" + n, 8) for n in NAMES]
             + [("note", 13), ("put", 12), ("put P&L", 13)])
    efmt = "  " + " ".join("%%%ds" % w for _, w in ecols)
    _LOG.info(efmt % tuple(h for h, _ in ecols))
    _LOG.info("  " + "-" * (sum(w for _, w in ecols) + len(ecols) - 1))
    for h in es_rungs:
        for side, esv in (("97.5%", lad[h][0]), ("2.5%", lad[h][1])):
            ys = np.array([name_shock(esv, SYS_P, IDIO[n], horizon_days=1)["y"]
                           for n in NAMES])
            if (ys <= -1.0).any():
                _LOG.info("  %4dd %6s %+8.2f%%   -- y_i through -100%%, outside range"
                      % (h, side, 100 * esv))
                continue
            put, pv = put_value(spot0 * (1.0 + ys), spot0, a)
            pnl = put - put0
            _LOG.info("  %4dd %6s %+8.2f%% %s %13s %12s %s"
                  % (h, side, 100 * esv,
                     " ".join("%+7.1f%%" % (100 * r) for r in ys),
                     _m(pv), _m(put),
                     _paint("%13s" % _m(pnl, signed=True), *_sign_color(pnl))))

    _LOG.info("\n  put P&L is the ISSUER's, who is long the put: positive in a selloff.")
    _LOG.info("  It is not the issuer's whole P&L - the note also carries the bond")
    _LOG.info("  and coupon legs - but it is the part the barrier creates, and the")
    _LOG.info("  part that cannot be rehedged through a gap.")
    _LOG.info("  The vega column sizes what a frozen surface is hiding: multiply it")
    _LOG.info("  by the vol points you expect the surface to add in that scenario.")
    _LOG.info("  Upside rows are unbounded by construction: the jump reaches name i")
    _LOG.info("  as exp(gamma_i Y) - 1, floored at -100% but with no ceiling, so a")
    _LOG.info("  large positive instantaneous shock compounds through gamma_i. Read")
    _LOG.info("  the deep up-shocks as the model speaking, not as scenarios.")


if __name__ == "__main__":
    main()
