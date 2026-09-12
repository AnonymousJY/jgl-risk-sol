"""Expected shortfall by simulation, from each year's estimated parameters,
read against what that year actually did.

Paths come from KimYiRiskEngine.random - the repo's own simulator, the same
one the VaR run uses. One simulation per year at that year's MEAN fitted
parameters, then the ES straight off the sample, printed beside the ES of the
realised SPX returns for the same year.

The comparison is the point. A simulated ES the year's own returns do not
support means the prescribed shock is being read against a distribution that
does not believe in it.

ES at level a averages the worst round(a*N) draws. From N simulated paths that
carries sampling error, so se = sd(tail)/sqrt(a*N) is printed; from a year of
~252 realised days the same tail is ~6 observations, so its count is printed
instead and its third decimal means nothing.

Every ES and quantile is SIGNED and in percent: a down-side ES of -7.617 is a
day of -7.617%. The heat ramp shades on |v| (render's abs_shade), so the
severest year is still the darkest cell without the number having to lie
about its sign.

    python poc/systematic_tail_mc.py
    python poc/systematic_tail_mc.py --paths 1000000
    python poc/systematic_tail_mc.py --horizon 10 --no-color
"""
import argparse
import glob
import io
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import PMLE_DIR, get_aligned_price_panel     # noqa: E402
from Library.Logging import report as _report                        # noqa: E402
from Library.Parameters import ParametersConstant                    # noqa: E402
from Library.Random import RandomMT19937                             # noqa: E402
from Library.RiskEngineKimYi2025 import (KimYiRiskEngine,            # noqa: E402
                                         SYSTEMATIC_PRIOR_SETS)
from Library.StatisticsMC import (                                   # noqa: E402
    StatisticsMCQuantile, StatisticsMCConditionalQuantile,
)
from Library.TableHeatmap import (                                   # noqa: E402
    render as heat, legend as heat_legend, to_html, to_eml,
)
from poc.estimate_systematic import store_id                         # noqa: E402

# Heat shading: on for a terminal, off when piped or NO_COLOR is set.
# --no-color / --color override. Same convention as the PMLE scripts.
COLOR = None

SYSTEMATIC_ID = "^SPX"
BASE_DAYS = 252
SEED = 20240114
COLS = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]

_LOG = _report(__name__)

# --html / --eml capture. Every line the run prints is kept here, with the
# tables rendered a SECOND time with colour forced on, so the mail carries the
# shading even when the terminal copy is plain because stdout is piped.
_CAP = None


def _emit(line):
    _LOG.info(line)
    if _CAP is not None:
        _CAP.append(line)


def _emit_table(df, **kw):
    kw.pop("color", None)
    _LOG.info(heat(df, color=COLOR, **kw))
    if _CAP is not None:
        _CAP.append(heat(df, color=True, **kw))


def _emit_legend():
    _LOG.info(heat_legend(color=COLOR))
    if _CAP is not None:
        _CAP.append(heat_legend(color=True))


def simulate(row, dt, horizon, paths, seed):
    """One parameter set -> `paths` horizon-day returns.

    Systematic block, so beta = gamma = 1 and kappa = rho = mu = 0, exactly as
    simulate_shock_returns_systematic_helper builds it in run_var_kimyi2025.
    """
    k = lambda v: ParametersConstant(np.array(float(v)))             # noqa: E731
    ret, _, _ = KimYiRiskEngine(
        mui=[k(0.)], kappai=[k(0.)], gammai=[k(1.)],
        betai=[k(1.)], rhoix=[k(0.)],
        alpha=k(row["dALPHA"]), sigma=k(row["dSIGMA"]),
        pprob=k(row["dPPROB"]), lamb=k(row["dLAMB"]),
        eta1=k(row["dETA1"]), eta2=k(row["dETA2"]),
        end_dt=np.array(dt),
    ).random(rng=RandomMT19937(np.int64(seed)), size=(1, paths, horizon))

    r = ret[0][:, 1:]                              # drop the identity column
    if horizon == 1:
        return r[:, 0]
    return np.expm1(np.log(np.clip(1.0 + r, 1e-300, None)).sum(axis=1))


def es(x, level, lower=True):
    """(quantile, ES, se, n), SIGNED, via Library.StatisticsMC.

    The house gatherers do the work: StatisticsMCConditionalQuantile for the
    ES, StatisticsMCQuantile for the quantile. The gatherer is fed the NEGATED
    sample for the upper tail, which turns its lower-tail ES into the upper
    one. It returns a positive magnitude either way, so the lower tail's sign
    is put back here: a down-side ES is a negative return and is reported as
    one.

    The tail is the gatherers' own int(n * level): TRUNCATED, and with no
    floor. So n is returned too, and 173 realised days at level 0.005 give an
    empty tail and a nan rather than a number resting on a single day.

    se is the Monte Carlo error of that same tail, sd(tail)/sqrt(n), computed
    here because the gatherers do not offer one - StatisticsMCConfidenceInterval
    is a CI on the MEAN of every path, not on a tail average.

    Used for the realised returns as well as the simulated paths, so both
    sides of every comparison are the same estimator.
    """
    v = np.asarray(x, dtype=float).ravel()
    m = int(v.size * level)
    if m < 1:
        return float("nan"), float("nan"), float("nan"), 0
    s = v if lower else -v

    g = StatisticsMCConditionalQuantile(level)
    g.dump_result(s.reshape(-1, 1))
    e = float(np.asarray(g.get_result_so_far()).item())
    if lower:
        e = -e                       # the gatherer returns +magnitude

    q = StatisticsMCQuantile(level if lower else 1.0 - level)
    q.dump_result(v.reshape(-1, 1))
    qv = float(np.asarray(q.get_result_so_far()).item())

    tail = np.sort(s)[:m]
    se = float(tail.std(ddof=1) / np.sqrt(m)) if m > 1 else float("nan")
    return qv, e, se, m


def main():
    global COLOR
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priors", default="alpha-pprob-eta-flat")
    ap.add_argument("--lookback", type=int, default=504)
    ap.add_argument("--levels", default="0.025,0.01,0.005")
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--beg", default="20070101")
    ap.add_argument("--end", default="20261231")
    ap.add_argument("--color", dest="color", action="store_true", default=None,
                    help="force heat shading on (default: on for a terminal)")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="plain numbers, no shading")
    ap.add_argument("--html", default=None, metavar="PATH",
                    help="also write the whole run as an HTML document that "
                         "keeps the shading in Outlook")
    ap.add_argument("--eml", default=None, metavar="PATH",
                    help="also write it as a message file Outlook can open")
    a = ap.parse_args()
    COLOR = a.color
    global _CAP
    if a.html or a.eml:
        _CAP = []

    levels = [float(v) for v in a.levels.split(",") if v.strip()]
    dt = 1.0 / BASE_DAYS

    # ---- realised ------------------------------------------------------
    panel, _ = get_aligned_price_panel([SYSTEMATIC_ID], reference=SYSTEMATIC_ID)
    r = panel[SYSTEMATIC_ID].pct_change().dropna()
    r = r[(r.index >= pd.to_datetime(a.beg, format="%Y%m%d")) &
          (r.index <= pd.to_datetime(a.end, format="%Y%m%d"))]
    if r.empty:
        raise SystemExit("no returns in %s -> %s" % (a.beg, a.end))
    ry = {y: v.values for y, v in r.groupby(r.index.year)}

    # ---- fitted --------------------------------------------------------
    drawer = store_id(a.priors, SYSTEMATIC_PRIOR_SETS[a.priors], a.lookback)
    files = sorted(glob.glob(os.path.join(PMLE_DIR, drawer, "*.csv")))
    if not files:
        raise SystemExit("no fitted drawer %s" % drawer)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["year"] = pd.to_datetime(df["dtVALUATION_DATE"]).dt.year
    n_dates = df.groupby("year").size()
    par = df.groupby("year")[COLS].mean()

    title = ("%s one-day tail, simulated vs realised, %d-%d"
             % (SYSTEMATIC_ID, r.index.min().year, r.index.max().year))
    _emit("=" * 78)
    _emit("EXPECTED SHORTFALL BY SIMULATION, AGAINST THE REALISED TAIL")
    _emit("%s   drawer %s" % (SYSTEMATIC_ID, drawer))
    _emit("realised %s -> %s, %d trading days"
              % (r.index.min().date(), r.index.max().date(), len(r)))
    _emit("%s paths per year   horizon %d day(s)   seed %d"
              % ("{:,}".format(a.paths), a.horizon, a.seed))
    _emit("=" * 78)
    _emit("")
    _emit("  Every ES and quantile below is SIGNED and in percent: a")
    _emit("  down-side ES of -7.617 is a day of -7.617%. Shading is on")
    _emit("  |v|, so the severest cell is the darkest either way.")

    _emit("\nRealised daily returns, pooled")
    _emit("  mean %+.4f%%   sd %.4f%%   annualised sd %.2f%%"
              % (100 * r.mean(), 100 * r.std(), 100 * r.std() * np.sqrt(BASE_DAYS)))
    _emit("  skew %+.3f   excess kurtosis %+.2f" % (r.skew(), r.kurtosis()))
    t = pd.DataFrame(index=["%.3f" % lv for lv in levels])
    sizes = []
    for lv, i in zip(levels, t.index):
        qd, ed, _, nd = es(r.values, lv, True)
        qu, eu, _, nu = es(r.values, lv, False)
        t.loc[i, "qDown"], t.loc[i, "ESdown"] = 100 * qd, 100 * ed
        t.loc[i, "qUp"], t.loc[i, "ESup"] = 100 * qu, 100 * eu
        sizes.append("%.3f: %d" % (lv, nd))
    _emit_table(t, decimals=3, abs_shade=True)
    _emit("  tail sizes (days each side)   %s" % "   ".join(sizes))
    _emit("  q is the ORDER STATISTIC - the m-th worst day itself, not an")
    _emit("  interpolated quantile - so it is a day that actually happened.")

    _emit("\nTen worst and ten best days")
    _emit("  %-12s %9s     %-12s %9s" % ("date", "worst", "date", "best"))
    for (dw, vw), (db, vb) in zip(r.nsmallest(10).items(), r.nlargest(10).items()):
        _emit("  %-12s %8.3f%%     %-12s %8.3f%%"
                  % (dw.date(), 100 * vw, db.date(), 100 * vb))

    _emit("\nHow often has a move of at least this size happened?")
    _emit("  %-9s %10s %10s %14s"
              % ("size", "down days", "up days", "1 in N days"))
    for x in (0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20):
        nd, nu = int((r <= -x).sum()), int((r >= x).sum())
        tot = nd + nu
        _emit("  %8.0f%% %10d %10d %14s"
                  % (100 * x, nd, nu, ("%d" % (len(r) / tot)) if tot else "never"))

    _emit("\nParameters simulated (mean of that year's valuation dates)")
    t = par.copy()
    t.columns = ["alpha", "sigma", "pprob", "lamb", "eta1", "eta2"]
    t.insert(0, "nDates", n_dates.reindex(par.index).values.astype(float))
    t.index = [str(i) for i in t.index]
    _emit_table(t, decimals=4)
    _emit_legend()

    t0 = time.time()
    sim = {y: simulate(p, dt, a.horizon, a.paths, a.seed)
           for y, p in par.iterrows()}
    _emit("\n  simulated %d x %s paths in %.1f s"
              % (len(par), "{:,}".format(a.paths), time.time() - t0))
    if a.horizon > 1:
        _emit("  horizon > 1: the realised columns are still ONE-DAY and")
        _emit("  are not comparable. Read the simulated side only.")

    # ---- simulated against realised, by year ---------------------------
    for lv in levels:
        t = pd.DataFrame(index=[str(y) for y in par.index])
        for y, i in zip(par.index, t.index):
            _, sd_, sdse, _ = es(sim[y], lv, True)
            _, su, suse, _ = es(sim[y], lv, False)
            t.loc[i, "simESdn"], t.loc[i, "seDn"] = 100 * sd_, 100 * sdse
            t.loc[i, "simESup"], t.loc[i, "seUp"] = 100 * su, 100 * suse
            if y in ry and int(len(ry[y]) * lv) >= 1:
                _, rd, _, _ = es(ry[y], lv, True)
                _, ru, _, _ = es(ry[y], lv, False)
                t.loc[i, "realESdn"] = 100 * rd
                t.loc[i, "realESup"] = 100 * ru
                t.loc[i, "ratioDn"] = sd_ / rd
                t.loc[i, "ratioUp"] = su / ru
            else:
                t.loc[i, "realESdn"] = t.loc[i, "realESup"] = np.nan
                t.loc[i, "ratioDn"] = t.loc[i, "ratioUp"] = np.nan
        t = t[["simESdn", "seDn", "realESdn", "ratioDn",
               "simESup", "seUp", "realESup", "ratioUp"]]
        _emit("\nLEVEL %.3f   simulated vs realised, by year" % lv)
        _emit_table(t, decimals=3, abs_shade=True)
        # How many realised days actually sit in a one-year tail at this
        # level. At 252 days a year, level 0.005 gives round(1.26) = 1, which
        # es() floors at 2 - so the realised column is the mean of the two
        # worst days of the year AND its effective level is 2/252 = 0.79%,
        # not 0.5%. It is being compared against a simulated column that is
        # genuinely at 0.5%, so the ratio is overstated. Say so rather than
        # let the column be read as if it meant what its header says.
        yrs = [y for y in par.index if y in ry]
        ns = [int(len(ry[y]) * lv) for y in yrs]
        eff = [n / float(len(ry[y])) for y, n in zip(yrs, ns)]
        _emit("  realised tail: %d-%d days per year (simulated: %s paths). "
                  % (min(ns), max(ns), "{:,}".format(max(int(round(lv * a.paths)), 2))))
        if min(ns) < 5:
            _emit("  TOO FEW to mean much. int() truncation puts the")
            _emit("  realised columns at an effective level of %.4f-%.4f,"
                      % (min(eff), max(eff)))
            _emit("  not %.3f, and a year with an empty tail is blank."
                      % lv)
        m = t.mean(numeric_only=True)
        _emit("  mean   simESdn %.3f  realESdn %.3f  ratio %.2f"
                  "   |   simESup %.3f  realESup %.3f  ratio %.2f"
                  % (m["simESdn"], m["realESdn"], m["ratioDn"],
                     m["simESup"], m["realESup"], m["ratioUp"]))
        _emit("  worst  simESdn %.3f (%s)  realESdn %.3f (%s)"
                  % (t["simESdn"].min(), t["simESdn"].idxmin(),
                     t["realESdn"].min(), t["realESdn"].idxmin()))
        _emit("  ratio = simulated / realised. Above 1 the fit is more")
        _emit("  severe than the year turned out; below 1 it is milder.")
    _emit_legend()
    _emit("")

    if a.html:
        with io.open(a.html, "w", encoding="utf-8") as fh:
            fh.write(to_html(_CAP, title=title))
        _LOG.info("  wrote %s" % a.html)
    if a.eml:
        with open(a.eml, "wb") as fh:
            fh.write(to_eml(_CAP, subject=title))
        _LOG.info("  wrote %s  (open it in Outlook, then Forward to edit "
                  "and send)" % a.eml)
    _LOG.info("")


if __name__ == "__main__":
    main()
