"""Does an observed liquidity scale identify rho_iX?  A pre-test. No refitting.

WHY THIS EXISTS. rho_iX is not identified by the published likelihood - see
poc/rhoix_identification_proof.py, which shows the Fisher information has rank
3 of 5 and the likelihood is flat to machine precision along the fibre. The
deficit is exactly ONE, so exactly one extra equation closes it. Brunetti and
Caldarera (2006) and Feng, Hung and Wang (2014) both get that equation from the
same place: market liquidity as a MEASURED, time-varying series rather than a
constant diffusion coefficient. This script asks whether that equation is
actually present in the data before anyone changes the model to use it.

THE TEST. Under the model the name's diffusive loading on the market is

    b_i = beta_i + kappa_i rho_iX / sigma

so if sigma is replaced by an OBSERVED scale s_t, the name's beta against the
market is not constant - it falls as s_t rises, at a rate that is exactly
kappa_i rho_iX. That makes the whole thing one linear regression:

    r_i,t = alpha + beta_i * r_t + (kappa_i rho_iX) * (r_t / s_t) + e_t

    H0: coefficient on r_t / s_t is zero   <=>   rho_iX = 0 (since kappa_i > 0)

The coefficient IS kappa_i rho_iX in annualised units. Combined with kperp from
the conditional fit, which needs no rho_iX prior,

    kappa_i = sqrt(kperp^2 + (kappa_i rho_iX)^2),   rho_iX = (kappa_i rho_iX)/kappa_i

- a point estimate, with no prior anywhere.

WHY THIS IS NOT THE ROUTE THAT ALREADY FAILED. Stage 3 of the conditional fit
regressed b_i on 1/sigma across 245 overlapping 504-day windows and got the
WRONG SIGN at all three names, because beta_i itself rises with sigma over the
cycle (b_i climbs 60-130% into 2009) and that bias exceeds the signal. Here s_t
is observed DAILY, and the contamination lives at the multi-year scale. Running
the same regression at 21, 63 and 252 day scales separates the two: a
coefficient that survives at 21 days is the kappa_i rho_iX term, one that
appears only at 252 days is the same state-dependence as before.

WHAT ANSWERS THE ACTUAL QUESTION ABOUT VOLUME. Three scales are built. "rv"
uses PRICES ONLY - trailing realised volatility of the market. "bc" and
"amihud" use VOLUME. If the volume-based scales do no better than rv, then
volume adds nothing that the price series did not already carry, and the answer
to "will adding SPX volume identify rho_iX" is no.

CONTROLS, because a t statistic on its own means nothing here:
  - block placebo: s_t is block-shuffled, preserving its autocorrelation and
    marginal distribution but destroying its alignment with returns.
  - lagged scale: s_{t-1} in place of s_t, which breaks any same-day mechanical
    coupling between the regressor and the dependent variable and attenuates
    classical measurement error in the proxy.
  - subsamples, because one GFC-driven number is not an estimate.

VALIDATED BEFORE USE, on simulated data with a known rho_iX. Given the TRUE
scale the regression returns rho_iX 0.010 / 0.259 / 0.507 / 0.753 against
truths of 0.00 / 0.25 / 0.50 / 0.75. Given a scale that has to be ESTIMATED
from returns, as here, the 21-day window returns 0.490 against a truth of 0.50
while the 252-day window attenuates to 0.348 - staleness, and the reason the
21-day row is the primary one. Under the null the block placebo returns
p = 0.163, and with signal present p = 0.000. So a null result from this script
is evidence of absence, not of low power.

    python poc/rhoix_volume_pretest.py --names C,BAC,JPM --volume-symbol SPY
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import get_aligned_price_panel                # noqa: E402
from Library.Logging import report as _report                         # noqa: E402
from Library.TableHeatmap import render as heat                       # noqa: E402

_LOG = _report(__name__)
ANNUAL = 252.0
SYSTEMATIC_ID = "^SPX"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_volume(symbol, csv_path=None):
    """Daily volume. The committed snapshots hold adjusted close only, so this
    is a live pull - the one thing in the pre-test that needs the network. Pass
    --volume-csv (date in the first column, volume in the second) if the box
    estimations run on has no outbound access."""
    if csv_path:
        v = pd.read_csv(csv_path, index_col=0, parse_dates=True).iloc[:, 0]
        v.index = pd.to_datetime(v.index)
        return v.astype(float).replace(0.0, np.nan)
    import FinanceDataReader as fdr
    df = fdr.DataReader(symbol)
    if "Volume" not in df.columns:
        raise SystemExit("%s has no Volume column" % symbol)
    v = df["Volume"].astype(float)
    v.index = pd.to_datetime(v.index)
    nz = (v > 0).mean()
    if nz < 0.5:
        raise SystemExit(
            "%s reports volume on only %.0f%% of days - index volume is often "
            "blank. Try SPY." % (symbol, 100 * nz))
    return v.replace(0.0, np.nan)


def build_scales(mkt_ret, volume, windows, detrend=252):
    """Observed proxies for the systematic diffusion scale, ANNUALISED.

    rv      trailing realised volatility of the market. PRICE ONLY.
    bc      Brunetti-Caldarera / Feng et al: |r_t| / volume, trailing mean.
    amihud  Amihud (2002) illiquidity: |r_t| / dollar volume, trailing mean.

    bc and amihud are rescaled to share rv's sample mean so that the regression
    coefficients below are in the same units as sigma and therefore directly
    comparable. That rescaling cannot manufacture a t statistic - it is a
    change of units - but it does mean the LEVEL of kappa_i rho_iX from a
    volume scale inherits rv's calibration.
    """
    px = (1.0 + mkt_ret).cumprod()
    v = volume.reindex(mkt_ret.index)
    # DETREND FIRST. r_t/V_t has units of 1/shares and inherits volume's
    # secular growth inverted, so a scale built from it drifts across a
    # 20-year sample even when market liquidity does not. Dividing volume by
    # its own trailing year makes the ratio DIMENSIONLESS and stationary -
    # today's volume relative to its own recent normal - which is what the
    # measure was always meant to mean. Returns are already unitless, so only
    # the volume side needs this.
    v_rel = v / v.rolling(detrend).mean()
    dv_rel = (v * px) / (v * px).rolling(detrend).mean()
    raw = {
        "bc": mkt_ret.abs() / v_rel,
        "amihud": mkt_ret.abs() / dv_rel,
    }
    out = {}
    for w in windows:
        rv = mkt_ret.rolling(w).std() * np.sqrt(ANNUAL)
        out[("rv", w)] = rv
        for k, r in raw.items():
            s = r.rolling(w).mean()
            s = s / s.mean() * rv.mean()
            out[(k, w)] = s
    return out


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def ols_nw(y, X, lags=None):
    """OLS with Newey-West standard errors. Returns (coef, se, t, r2, n)."""
    n, k = X.shape
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    xtx_inv = np.linalg.pinv(X.T @ X)
    b = xtx_inv @ (X.T @ y)
    e = y - X @ b

    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))

    ss_res = float(e @ e)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return b, se, b / np.where(se > 0, se, np.nan), 1.0 - ss_res / ss_tot, n


def conditional_beta(name_ret, mkt_ret, scale, truncate=None):
    """r_i = a + beta_i r_t + (kappa_i rho_iX) (r_t / s_t) + e.

    TRUNCATE IS NOT OPTIONAL IN PRACTICE. The common jump makes OLS on r_t
    return a variance-weighted BLEND of b_i and gamma_i, and the weights move
    with s_t: a high s_t tilts the blend toward b_i, a low one toward gamma_i.
    Since gamma_i > b_i the blend FALLS as s_t rises - the same direction as
    kappa_i rho_iX / s_t. With the paper's own SPX parameters the jump channel
    is 42% of daily market variance, and on simulated data with rho_iX SET TO
    ZERO this returns an implied rho of 0.438 at t = 9.48. A textbook false
    positive.

    Mancini truncation - drop days where |r_t| > k s_t sqrt(dt), which are the
    jump days - removes it: the same simulation returns 0.016 at t = 0.51 for
    k = 3, while a true rho_iX of 0.50 comes back as 0.500 and its t statistic
    IMPROVES from 8.5 to 20.9, because the jump days were noise for this
    purpose. It costs about 2% of observations.
    """
    d = pd.concat([name_ret, mkt_ret, scale], axis=1).dropna()
    if truncate:
        d = d[d.iloc[:, 1].abs() <= truncate * d.iloc[:, 2] / np.sqrt(ANNUAL)]
    if len(d) < 250:
        return None
    y = d.iloc[:, 0].to_numpy()
    r = d.iloc[:, 1].to_numpy()
    s = d.iloc[:, 2].to_numpy()
    X = np.column_stack([np.ones_like(r), r, r / s])
    b, se, t, r2, n = ols_nw(y, X)
    return dict(beta=b[1], se_beta=se[1], kr=b[2], se_kr=se[2], t_kr=t[2],
                r2=r2, n=n)


def within_year(name_ret, mkt_ret, scale, truncate=3.0):
    """Same regression, but beta_i is free to differ EVERY YEAR.

    b_i,t = beta_i(t) + kappa_i rho_iX / s_t. beta_i drifts over the cycle -
    C's b_i runs 1.74 in 2009 to 1.05 in 2025 - and any scale that also drifts
    is confounded with it. That is Route 3's failure returning at the
    low-frequency end, and it is NOT specific to volume: on simulated data with
    rho_iX = 0, a drifting beta_i produces a spurious coefficient at t = -3.4
    even with a PRICE-ONLY scale.

    Interacting r_t with year dummies lets beta_i move freely across years, so
    kappa_i rho_iX is identified only from WITHIN-year variation in s_t, where
    beta_i is constant. On the same simulation this restores the null at all
    three scales (t = -0.78, 0.29, 1.63 against a truth of zero).

    IT COSTS MAGNITUDE. Within-year variation is a small slice of the total, so
    errors-in-variables in the proxy bites hard: with a true rho_iX of 0.50 the
    three scales return 0.29, 0.56 and 0.20. Treat this as a TEST of whether
    the signal exists, not as an estimator of how big it is.
    """
    d = pd.concat([name_ret.rename("y"), mkt_ret.rename("r"),
                   scale.rename("s")], axis=1).dropna()
    if truncate:
        d = d[d["r"].abs() <= truncate * d["s"] / np.sqrt(ANNUAL)]
    if len(d) < 250:
        return None
    yr = pd.get_dummies(d.index.year).to_numpy(float)
    X = np.column_stack([yr * d["r"].to_numpy()[:, None],
                         (d["r"] / d["s"]).to_numpy()])
    b, se, t, r2, n = ols_nw(d["y"].to_numpy(), X)
    # b[:-1] are the per-year betas; report their mean so the column is
    # comparable with the pooled rows rather than NaN.
    return dict(kr=b[-1], se_kr=se[-1], t_kr=t[-1], r2=r2, n=n,
                beta=float(np.mean(b[:-1])))


def block_placebo(name_ret, mkt_ret, scale, n_draws=500, block=21,
                  seed=20240114, truncate=3.0):
    """Block-shuffle the scale. Same autocorrelation and marginal, no alignment.

    A plain i.i.d. shuffle would destroy the persistence of s_t and understate
    the null distribution of the t statistic, making a spurious result look
    significant. Blocks preserve it.
    """
    rng = np.random.default_rng(seed)
    d = pd.concat([name_ret, mkt_ret, scale], axis=1).dropna()
    s = d.iloc[:, 2].to_numpy()
    N = len(s)
    starts = np.arange(max(1, N - block + 1))
    out = []
    for _ in range(n_draws):
        idx = np.concatenate([np.arange(a, a + block)
                              for a in rng.choice(starts,
                                                  size=int(np.ceil(N / block)))])[:N]
        sh = pd.Series(s[idx], index=d.index)
        res = within_year(d.iloc[:, 0], d.iloc[:, 1], sh.rename("s"),
                          truncate=truncate)
        if res:
            out.append(res["t_kr"])
    return np.array(out)


def variance_quadratic(name_ret, mkt_ret, volume):
    """Brunetti-Caldarera's own moment with the rho_iX cross term restored.

        E(u_i,t^2) = kappa_i^2 + 2 sigma beta_i kappa_i rho_iX v_t
                     + (sigma beta_i)^2 v_t^2

    REPORTED WITH A HEALTH WARNING. v_t = r_t / volume_t inherits the market
    return's SIGN, and u_i,t retains market exposure, so u^2 is driven by r^2
    while v is driven by r. The cross moment E(u^2 v) then picks up E(r^3),
    which is negative for equity indices - a spurious NEGATIVE linear term, and
    therefore a spurious negative rho_iX, out of nothing but skewness. The
    skew diagnostic below is what says whether that is what happened.
    """
    d = pd.concat([name_ret, mkt_ret, volume.rename("vol")], axis=1).dropna()
    if len(d) < 250:
        return None
    v = (d.iloc[:, 1] / d["vol"]).to_numpy()
    v = v / np.std(v)
    y0 = d.iloc[:, 0].to_numpy()
    X0 = np.column_stack([np.ones_like(v), v])
    b0, *_ = ols_nw(y0, X0)
    u = y0 - X0 @ b0

    X = np.column_stack([np.ones_like(v), v, v ** 2])
    b, se, t, r2, n = ols_nw(u ** 2, X)
    c0, c1, c2 = b
    rho = (c1 / (2.0 * np.sqrt(c0 * c2))
           if c0 > 0 and c2 > 0 else np.nan)
    mkt_skew = float(pd.Series(d.iloc[:, 1]).skew())
    return dict(c0=c0, c1=c1, t_c1=t[1], c2=c2, rho=rho, r2=r2, n=n,
                mkt_skew=mkt_skew)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="C,BAC,JPM")
    ap.add_argument("--volume-symbol", default="SPY",
                    help="index volume is often blank; SPY is the usual stand-in")
    ap.add_argument("--volume-csv", default=None,
                    help="local volume CSV, if this box has no outbound access")
    ap.add_argument("--windows", default="21,63,252",
                    help="trailing windows for the scale, in business days")
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--truncate", type=float, default=3.0,
                    help="Mancini jump truncation: drop |r_t| > k s_t sqrt(dt). "
                         "Without it the common jump manufactures a rho_iX of "
                         "0.44 out of nothing - see conditional_beta.__doc__")
    ap.add_argument("--detrend", type=int, default=252,
                    help="window for normalising volume to its own recent "
                         "normal, making the measure dimensionless")
    ap.add_argument("--placebo", type=int, default=500)
    ap.add_argument("--kperp", default="C:0.2419,BAC:0.2390,JPM:0.1896",
                    help="kperp per name from the conditional fit, to convert "
                         "kappa_i rho_iX into rho_iX")
    ap.add_argument("--color", dest="color", action="store_true", default=None)
    ap.add_argument("--no-color", dest="color", action="store_false")
    a = ap.parse_args()

    names = [s.strip().upper() for s in a.names.split(",") if s.strip()]
    windows = [int(s) for s in a.windows.split(",") if s.strip()]
    kperp = {}
    for part in a.kperp.split(","):
        if ":" in part:
            k, v = part.split(":")
            kperp[k.strip().upper()] = float(v)

    px, _ = get_aligned_price_panel([SYSTEMATIC_ID] + names,
                                    reference=SYSTEMATIC_ID)
    px = px[px.index >= pd.Timestamp(a.start)]
    ret = px.pct_change().dropna()
    mkt = ret[SYSTEMATIC_ID]

    vol = load_volume(a.volume_symbol, a.volume_csv)
    scales = build_scales(mkt, vol, windows, detrend=a.detrend)

    _LOG.info("=" * 78)
    _LOG.info("DOES AN OBSERVED LIQUIDITY SCALE IDENTIFY rho_iX?")
    _LOG.info("=" * 78)
    _LOG.info("  %s .. %s   %d trading days   volume from %s"
              % (ret.index[0].date(), ret.index[-1].date(), len(ret),
                 a.volume_symbol))
    _LOG.info("  H0: the coefficient on r_t/s_t is zero, i.e. rho_iX = 0.")
    _LOG.info("  The model predicts it is kappa_i rho_iX > 0 - a POSITIVE sign.")
    _LOG.info("  Stage 3 across windows gave NEGATIVE at all three names; that")
    _LOG.info("  is the state-dependence this test is built to avoid.\n")

    corr = pd.DataFrame(index=[k for k in ("rv", "bc", "amihud")],
                        columns=[str(w) for w in windows], dtype=float)
    base = {w: scales[("rv", w)] for w in windows}
    for k in ("rv", "bc", "amihud"):
        for w in windows:
            j = pd.concat([scales[(k, w)], base[w]], axis=1).dropna()
            corr.loc[k, str(w)] = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
    _LOG.info("  correlation of each scale with the price-only scale (rv):")
    _LOG.info(heat(corr.astype(float).round(3), decimals=3, color=a.color,
                   index_width=8))
    _LOG.info("  A volume scale that correlates ~1.0 with rv carries no extra")
    _LOG.info("  information, whatever its t statistic.\n")

    for nm in names:
        _LOG.info("=" * 78)
        _LOG.info("%s" % nm)
        _LOG.info("=" * 78)
        rows = []
        for k in ("rv", "bc", "amihud"):
            for w in windows:
                for lag, tr, fn, tag in (
                        (0, a.truncate, within_year, "WITHIN"),
                        (1, a.truncate, within_year, "WITHINlag"),
                        (0, a.truncate, conditional_beta, "pooled"),
                        (0, None, conditional_beta, "pooledRAW")):
                    s = scales[(k, w)].shift(lag).rename("s")
                    r = fn(ret[nm], mkt, s, truncate=tr)
                    if r is None:
                        continue
                    kap = np.sqrt(kperp.get(nm, np.nan) ** 2 + r["kr"] ** 2)
                    rows.append(dict(
                        scale=k, window=w, timing=tag,
                        beta_i=r["beta"], kappa_rho=r["kr"],
                        se=r["se_kr"], t=r["t_kr"],
                        rho_iX=r["kr"] / kap if np.isfinite(kap) else np.nan,
                        r2=r["r2"]))
        t = pd.DataFrame(rows)
        t.index = [("%s/%d/%s" % (r.scale, r.window, r.timing))
                   for r in t.itertuples()]
        _LOG.info(heat(t[["beta_i", "kappa_rho", "se", "t", "rho_iX", "r2"]]
                       .astype(float).round(4), decimals=4, color=a.color,
                       index_width=18))

        s21 = scales[("rv", windows[0])].rename("s")
        null = block_placebo(ret[nm], mkt, s21, n_draws=a.placebo,
                             truncate=a.truncate)
        obs = within_year(ret[nm], mkt, s21, truncate=a.truncate)
        if len(null) and obs:
            p = float((np.abs(null) >= abs(obs["t_kr"])).mean())
            _LOG.info("  block placebo on rv/%d: observed t %.2f, "
                      "null |t| 95th pct %.2f, p = %.3f"
                      % (windows[0], obs["t_kr"], np.percentile(np.abs(null), 95), p))
            if p > 0.05:
                _LOG.info("  -> indistinguishable from a shuffled scale. No signal.")

        q = variance_quadratic(ret[nm], mkt, vol)
        if q:
            _LOG.info("  BC variance quadratic: c1 %+.4f (t %+.2f), implied "
                      "rho %.3f, R2 %.3f" % (q["c1"], q["t_c1"], q["rho"], q["r2"]))
            _LOG.info("     market return skew %+.2f - if c1 and the skew share"
                      " a sign, suspect the E(r^3) artefact, not rho_iX."
                      % q["mkt_skew"])
        _LOG.info("")

    _LOG.info("=" * 78)
    _LOG.info("HOW TO READ THIS")
    _LOG.info("=" * 78)
    _LOG.info("  A usable result needs ALL of: a POSITIVE coefficient on")
    _LOG.info("  r_t/s_t; survival at the 21-day window, not only at 252;")
    _LOG.info("  survival when the scale is lagged; a placebo p below 0.05;")
    _LOG.info("  an implied rho_iX inside the identified set [0, rho_bar]")
    _LOG.info("  (0.674 C, 0.671 BAC, 0.752 JPM); and presence in the WITHIN")
    _LOG.info("  rows, not only the pooled ones - that gap is beta_i drift.")
    _LOG.info("  A large pooledRAW row beside a null WITHIN row is the jump")
    _LOG.info("  artefact plus the drift, not a signal.")
    _LOG.info("  Anything less and the model should not be changed: the")
    _LOG.info("  bracket stands and rho_iX is reported as a set, not a number.")


if __name__ == "__main__":
    main()
