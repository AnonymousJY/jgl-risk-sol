"""Estimate the idiosyncratic block CONDITIONAL on the market, and recover rho_iX.

The published likelihood conditions on the name's own lagged state and not on
the market's contemporaneous increment, so rho_iX enters only through
phi_i^2 and is confounded with beta_i and kappa_i - one equation, three
unknowns, and whatever prior is placed on rho_iX comes straight back out.

Conditioning additionally on the market's increment adds exactly one moment,
Cov(U,V). This script runs that likelihood and then closes the last dimension:

  STAGE 2'  per valuation date, sample (b_i, kappa_perp, gamma_i, m_i) from
            KimYiLogLikeJoint. These are the identified quantities; note that
            b_i - the loading the paper's tables report - is estimated here
            rather than inherited from a prior.

  STAGE 3   b_i = beta_i + (kappa_i rho_iX) / sigma is LINEAR IN 1/sigma, and
            sigma moves by a factor of three across the windows. Regressing
            b_i on 1/sigma gives beta_i as the intercept and kappa_i rho_iX as
            the slope; with kappa_perp that closes to

                kappa_i = sqrt(kappa_perp^2 + pi^2),
                rho_iX  = pi / kappa_i,       pi := kappa_i rho_iX

            with NO prior on rho_iX anywhere.

Stage 1 is untouched: the systematic parameters arrive as fixed constants from
the same drawer estimate_idiosyncratic.py uses, and this script never refits
them. poc/joint_likelihood_probe.py verifies the whole chain on simulated data
where the answer is known - rho_iX recovered to 1.9%.

    python poc/estimate_idiosyncratic_joint.py --names C,BAC,JPM \\
        --priors alpha-pprob-eta-flat --lookback 504
"""
import argparse
import glob
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Library.DataAccess import PMLE_DIR, get_aligned_price_panel     # noqa: E402
from Library.Logging import report as _report                        # noqa: E402
from Library.RiskEngineKimYi2025 import (                            # noqa: E402
    SYSTEMATIC_PRIOR_SETS, pmle_kimyirisk_joint,
)
from Library.TableHeatmap import (                                   # noqa: E402
    render as heat, legend as heat_legend, to_html, to_eml,
)
from poc.estimate_systematic import store_id                         # noqa: E402

SYSTEMATIC_ID = "^SPX"
BASE_DAYS = 252
SEED = 20240114
N_MC_PATHS = 10_000
SYS_COLS = ["dALPHA", "dSIGMA", "dPPROB", "dLAMB", "dETA1", "dETA2"]

COLOR = None
_CAP = None
_LOG = _report(__name__)


def _emit(line):
    _LOG.info(line)
    if _CAP is not None:
        _CAP.append(line)


def _emit_table(df, **kw):
    kw.pop("color", None)
    _LOG.info(heat(df, color=COLOR, **kw))
    if _CAP is not None:
        _CAP.append(heat(df, color=True, **kw))


def _fit_one(task):
    """One (name, date). Module level so the process pool can pickle it."""
    name, dt, idi, sysr, params_sys, delta_t = task
    try:
        res = pmle_kimyirisk_joint(
            idi_returns=idi, sys_returns=sysr, params_sys=params_sys,
            delta_t=delta_t, seed_number=np.uint64(SEED),
            n_mc_paths=N_MC_PATHS, is_progress_bar=False)
        return name, dt, res, None
    except Exception as exc:                                   # noqa: BLE001
        return name, dt, None, "%s: %s" % (type(exc).__name__, exc)


def main():
    global COLOR, _CAP
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True)
    ap.add_argument("--priors", default="alpha-pprob-eta-flat",
                    help="SYSTEMATIC arm whose drawer supplies stage 1")
    ap.add_argument("--lookback", type=int, default=504)
    ap.add_argument("--step", type=int, default=1,
                    help="use every Nth valuation date in the drawer")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--color", dest="color", action="store_true", default=None)
    ap.add_argument("--no-color", dest="color", action="store_false")
    ap.add_argument("--html", default=None, metavar="PATH")
    ap.add_argument("--eml", default=None, metavar="PATH")
    a = ap.parse_args()
    COLOR = a.color
    if a.html or a.eml:
        _CAP = []
        COLOR = True if COLOR is None else COLOR

    names = [s.strip().upper() for s in a.names.split(",") if s.strip()]
    delta_t = np.array(1.0 / BASE_DAYS)

    drawer = store_id(a.priors, SYSTEMATIC_PRIOR_SETS[a.priors], a.lookback)
    files = sorted(glob.glob(os.path.join(PMLE_DIR, drawer, "*.csv")))
    if not files:
        raise SystemExit("no systematic drawer %s" % drawer)
    sysdf = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sysdf["dt"] = pd.to_datetime(sysdf["dtVALUATION_DATE"])
    sysdf = sysdf.sort_values("dt").reset_index(drop=True)
    if a.step > 1:
        sysdf = sysdf.iloc[::a.step].reset_index(drop=True)

    panel, _ = get_aligned_price_panel([SYSTEMATIC_ID] + names,
                                       reference=SYSTEMATIC_ID)
    rets = panel.pct_change().dropna()

    _emit("=" * 78)
    _emit("IDIOSYNCRATIC BLOCK, CONDITIONAL ON THE MARKET")
    _emit("stage 1 drawer %s   (read, never refitted)" % drawer)
    _emit("lookback %d   %d valuation date(s)   names %s"
          % (a.lookback, len(sysdf), ", ".join(names)))
    _emit("=" * 78)
    _emit("")
    _emit("  Sampled: b_i, kappa_perp, gamma_i, m_i - the parameters the")
    _emit("  conditional likelihood identifies. beta_i, kappa_i and rho_iX are")
    _emit("  NOT sampled; they are recovered in stage 3 below.")

    tasks = []
    for _, row in sysdf.iterrows():
        d = row["dt"]
        params_sys = {c: np.array(float(row[c])) for c in SYS_COLS}
        sysr = rets.loc[rets.index <= d, SYSTEMATIC_ID].dropna()
        if len(sysr) < a.lookback:
            continue
        sysr = sysr.iloc[-a.lookback:].to_numpy()
        for nm in names:
            idi = rets.loc[rets.index <= d, nm].dropna()
            if len(idi) < a.lookback:
                continue
            tasks.append((nm, d, idi.iloc[-a.lookback:].to_numpy(),
                          sysr, params_sys, delta_t))
    _emit("\n  %d fit(s) to run" % len(tasks))
    if not tasks:
        raise SystemExit("nothing to fit")

    t0 = time.perf_counter()
    rows, bad = [], 0
    workers = a.workers or max(1, (os.cpu_count() or 4) // 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nm, d, res, err in ex.map(_fit_one, tasks):
            if err:
                bad += 1
                _LOG.info("    FAIL %-6s %s  %s" % (nm, d.date(), err))
                continue
            rows.append(dict(name=nm, dt=d,
                             dBI=res["dBI"].dMEAN,
                             dBI_W=res["dBI"].dCI_UPPER - res["dBI"].dCI_LOWER,
                             dKPERP=res["dKPERP"].dMEAN,
                             dGAMMAI=res["dGAMMAI"].dMEAN,
                             dMI=res["dMI"].dMEAN))
    out = pd.DataFrame(rows)
    _emit("  %d fit(s) in %.1f s  (%d failed)"
          % (len(out), time.perf_counter() - t0, bad))
    if out.empty:
        raise SystemExit("every fit failed")
    out["year"] = out["dt"].dt.year
    out["sigma"] = out["dt"].map(dict(zip(sysdf["dt"], sysdf["dSIGMA"])))

    for nm in names:
        blk = out[out["name"] == nm]
        if blk.empty:
            continue
        _emit("\n" + "=" * 78)
        _emit("%s :: %d valuation dates" % (nm, len(blk)))
        _emit("=" * 78)
        t = blk[["dBI", "dKPERP", "dGAMMAI", "dMI"]].describe().T
        t = t[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(4)
        t.index = [str(i) for i in t.index]
        _emit("\nDistribution across valuation dates:")
        _emit_table(t, decimals=4)
        y = blk.groupby("year")[["dBI", "dKPERP", "dGAMMAI", "dMI"]].mean().round(4)
        y.index = [str(i) for i in y.index]
        _emit("\nBy year (mean):")
        _emit_table(y, decimals=4)
        path = os.path.join("poc", "idio_joint_%s__%s__lb%d.csv"
                            % (nm, a.priors.replace("-", ""), a.lookback))
        blk.to_csv(path, index=False)
        _emit("  written to %s" % path)

    # ---- stage 3: b_i on 1/sigma ---------------------------------------
    _emit("\n" + "=" * 78)
    _emit("STAGE 3 :: regress b_i on 1/sigma, recover rho_iX")
    _emit("=" * 78)
    _emit("  b_i = beta_i + (kappa_i rho_iX)/sigma, so the intercept is beta_i")
    _emit("  and the slope is kappa_i rho_iX. With kappa_perp from stage 2',")
    _emit("  kappa_i = sqrt(kappa_perp^2 + slope^2) and rho_iX = slope/kappa_i.")
    _emit("  No prior on rho_iX enters anywhere in this.")
    res = {}
    for nm in names:
        blk = out[out["name"] == nm].dropna(subset=["sigma"])
        if len(blk) < 8:
            continue
        x = 1.0 / blk["sigma"].to_numpy()
        b = blk["dBI"].to_numpy()
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, b, rcond=None)
        beta_i, pi = float(coef[0]), float(coef[1])
        resid = b - X @ coef
        dof = max(len(b) - 2, 1)
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (resid @ resid) / dof)
        kperp = float(blk["dKPERP"].mean())
        kappa_i = float(np.sqrt(kperp ** 2 + pi ** 2))
        rho = pi / kappa_i
        r2 = 1.0 - (resid @ resid) / (((b - b.mean()) ** 2).sum() + 1e-300)
        res[nm] = dict(beta_i=beta_i, se_beta=se[0], pi=pi, se_pi=se[1],
                       kperp=kperp, kappa_i=kappa_i, rho_iX=rho, r2=r2,
                       n=len(b), sig_lo=blk["sigma"].min(),
                       sig_hi=blk["sigma"].max())
    if res:
        t = pd.DataFrame(res).T
        t = t[["beta_i", "se_beta", "pi", "se_pi", "kperp", "kappa_i",
               "rho_iX", "r2"]].astype(float).round(4)
        _emit("")
        _emit_table(t, decimals=4)
        _emit(heat_legend(color=COLOR))
        _emit("")
        for nm, r in res.items():
            _emit("  %-5s slope %+.4f (se %.4f, t %+.1f)   sigma %.3f-%.3f"
                  "   R2 %.3f   n %d"
                  % (nm, r["pi"], r["se_pi"], r["pi"] / max(r["se_pi"], 1e-12),
                     r["sig_lo"], r["sig_hi"], r["r2"], r["n"]))
        _emit("")
        _emit("  The slope's t statistic is the whole question: if it is not")
        _emit("  distinguishable from zero then kappa_i rho_iX is not either,")
        _emit("  and rho_iX stays unidentified whatever the point estimate says.")

    if a.html:
        with io.open(a.html, "w", encoding="utf-8") as fh:
            fh.write(to_html(_CAP, title="Conditional idiosyncratic fit"))
        _LOG.info("  wrote %s" % a.html)
    if a.eml:
        with open(a.eml, "wb") as fh:
            fh.write(to_eml(_CAP, subject="Conditional idiosyncratic fit"))
        _LOG.info("  wrote %s" % a.eml)
    _LOG.info("")


if __name__ == "__main__":
    main()
