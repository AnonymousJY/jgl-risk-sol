"""The study's sampling window: dates, lookback, and the systematic symbol.

These live here rather than in poc/estimate_systematic.py because steps that
need no sampler should not have to import one. estimate_systematic.py drags in
pymc, nutpie and the whole compile stack; a step that only reads prices and
takes a correlation would pay minutes of import time and would not run at all
in an environment without them.

poc/estimate_systematic.py still declares its own copies. Fold it onto this
module once no run is in flight - editing it mid-run risks a forkserver child
re-importing a half-written file.
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
