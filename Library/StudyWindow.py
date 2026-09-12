"""The study's sampling window: dates, lookback, and the systematic symbol.

These live here rather than in poc/estimate_systematic.py because steps that
need no sampler should not have to import one. estimate_systematic.py drags in
pymc, nutpie and the whole compile stack; a step that only reads prices and
takes a correlation would pay minutes of import time and would not run at all
in an environment without them.

poc/estimate_systematic.py still declares its own copies. Fold it onto this
module once no run is in flight - editing it mid-run risks a forkserver child
re-importing a half-written file.

THE STUDY WINDOW IS NOW 504 DAYS (two years), decided 11 September 2026 off
the window ladder. LOOKBACK below stays 252 ANYWAY, and the distinction
matters enough to state twice.

252 is the NAMING BASE, not the study window. estimate_systematic's
BASE_LOOKBACK is what lookback_suffix compares against: a run at the base
gets no suffix, anything else gets __lb<n>. Every drawer fitted at 252 to
date is therefore unsuffixed. Move the base to 504 and the next 504-day run
writes into those unsuffixed drawers - two window lengths interleaved inside
one series, with run() reporting each date as "already on disk". That is the
exact contamination this suffix scheme exists to prevent, and changing a
constant named "STUDY DEFAULT" is the obvious way to walk into it.

So: the study reads __lb504 drawers, the base stays 252 forever, and which
window a drawer holds is read from its own _priors.json rather than inferred
from any constant.

WHY 504. From the ladder on alpha-pprob-flat, as posterior 95% width over
prior 95% width:

                 252      504      756
    dSIGMA     0.056    0.046    0.040    identified throughout
    dPPROB     0.885    0.638    0.461    crosses at two years
    dALPHA     0.994    0.937    0.730    needs ~19 years; full sample only
    dLAMB      0.906    1.177    1.445    never - and worse with more data
    dETA1      0.989    0.947    0.866    never
    dETA2      1.017    1.026    1.030    never

Two years is the shortest window that identifies both sigma and the jump
sign. It costs some responsiveness in sigma - its cross-date sd falls from
2.11 of its own posterior width to 1.66 - and it dilutes the episode and
vintage contrasts, which are statements about what a window CONTAINS.
"""
import numpy as np
import pandas as pd

SYSTEMATIC_ID = "^SPX"
DATE_FMT = "%Y%m%d"
BEG = "20070101"
END = "20260831"
LOOKBACK = 252
BASE_DAYS = 252
SEED = np.uint64(20240114)


# Representative dates for the episodes the stress work uses. One date each,
# because what matters is whether the episode falls INSIDE a trailing window,
# not its exact span.
STRESS_EVENTS = {
    "GFC":     "2008-10-15",
    "Euro":    "2011-08-08",
    "Covid":   "2020-03-16",
    "FTX":     "2022-11-09",
    "SVB":     "2023-03-13",
    "Tariffs": "2025-04-07",
}


def name_coverage(returns, lookbacks=(252, 504, 756), beg=None, end=None,
                  step=21, events=None):
    """What a name's history can support at each window length.

    Ask this BEFORE choosing a window, not after. The window that best
    identifies the SYSTEMATIC parameters is chosen on ^SPX, which has fifty
    years of history; a name listed in 2021 has four, and the two answers are
    not the same. COIN is the case that made this function exist - at 504 days
    its first estimable date already has the FTX collapse inside its trailing
    window, so there is no date at which it is observed outside a crisis until
    FTX rolls out in late 2024.

    BASELINE COUNT IS THE NUMBER THAT MATTERS, not the total. Every stress
    result in this study is a CONTRAST - a parameter in stress against the
    same parameter in calm - so a name with plenty of valuation dates and zero
    of them outside a crisis window cannot support one. The total looks
    healthy in exactly the case that is unusable.

    `returns` is the name's own return Series, already dropna'd.
    """
    events = STRESS_EVENTS if events is None else events
    ev = [pd.to_datetime(v) for v in events.values()]
    dates = [pd.to_datetime(d, format=DATE_FMT)
             for d in valuation_dates(beg or BEG, end or END, step)]
    idx = returns.index
    rows = []
    for lb in lookbacks:
        if len(returns) < lb:
            rows.append(dict(lookback=lb, first=None, n=0, baseline=0,
                             short=True))
            continue
        first_ok = idx[lb - 1]
        usable = [d for d in dates if d >= first_ok]
        baseline = 0
        for d in usable:
            upto = idx[idx <= d]
            if len(upto) < lb:
                continue
            lo = upto[-lb]                      # first day inside the window
            if not any(lo <= e <= d for e in ev):
                baseline += 1
        rows.append(dict(lookback=lb, first=first_ok, n=len(usable),
                         baseline=baseline, short=False))
    return rows


def valuation_dates(beg, end, step):
    days = pd.bdate_range(pd.to_datetime(beg, format=DATE_FMT),
                          pd.to_datetime(end, format=DATE_FMT))
    return [d.strftime(DATE_FMT) for d in days[::step]]
