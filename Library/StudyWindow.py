"""The study's sampling window: dates, lookback, and the systematic symbol.

These live here rather than in poc/estimate_systematic.py because steps that
need no sampler should not have to import one. estimate_systematic.py drags in
pymc, nutpie and the whole compile stack; a step that only reads prices and
takes a correlation would pay minutes of import time and would not run at all
in an environment without them.

poc/estimate_systematic.py still declares its own copies. Fold it onto this
module once no run is in flight - editing it mid-run risks a forkserver child
re-importing a half-written file.

LOOKBACK here is the STUDY DEFAULT, one regulatory year. The two estimation
drivers now take --lookback and carry the window in the drawer name (suffix
__lb<n>, empty at 252), so a 3-year run is a separate store rather than an
edit to this constant. Change this only to move the study itself; nothing
that reads a drawer should infer its window from this value - read the
drawer's _priors.json, which records the lookback the fits were made with.
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


def valuation_dates(beg, end, step):
    days = pd.bdate_range(pd.to_datetime(beg, format=DATE_FMT),
                          pd.to_datetime(end, format=DATE_FMT))
    return [d.strftime(DATE_FMT) for d in days[::step]]
