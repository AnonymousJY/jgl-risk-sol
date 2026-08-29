"""
Pull daily closes for the spot backfill PoC and write prices.csv.

    pip install yfinance pandas
    python fetch_prices.py

Writes a wide CSV: first column 'date', one column per ticker, from 2007-01-01.
SPX is written as 'SPX' (fetched as ^GSPC).

Note on adjusted closes: most free providers restate dividend adjustments
retroactively, so the series is not strictly point-in-time. Immaterial for
relative returns in a PoC; note it in the write-up. auto_adjust=True is used
here so returns are total-return-like and splits are handled.
"""

import pandas as pd
import yfinance as yf

START = "2007-01-01"
OUT = "prices.csv"

INDEX = {"^GSPC": "SPX"}

# Sector ETFs used as the incumbent proxy benchmark. All launched Dec 1998,
# so all have full 2007 history.
ETFS = ["XLK", "XLF", "XLP", "XLE", "XLI", "XLV", "XLY", "XLU"]

# Test names. Stratification is on POST-CRISIS estimated liquidity beta, which
# is information a bank would actually have at backfill time - not on how the
# name behaved in 2008, which would be selection leakage. The sector/size mix
# below is the ex-ante selection; measure beta afterwards and bucket then.
NAMES = [
    # Technology
    "MSFT", "ORCL", "INTC",
    # Financials - expected high liquidity sensitivity
    "JPM", "GS", "BAC",
    # Consumer staples - expected low
    "PG", "KO", "WMT",
    # Energy
    "XOM", "CVX",
    # Industrials
    "CAT", "GE",
    # Health care
    "JNJ", "PFE",
    # Consumer discretionary
    "HD", "SBUX",
    # Utilities - expected lowest
    "SO",
]

ALL = list(INDEX) + ETFS + NAMES


def main():
    raw = yf.download(ALL, start=START, auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns=INDEX)
    raw.index.name = "date"

    missing = [c for c in raw.columns if raw[c].first_valid_index() is None]
    if missing:
        print("no data for:", missing)

    late = {
        c: str(raw[c].first_valid_index().date())
        for c in raw.columns
        if raw[c].first_valid_index() is not None
        and raw[c].first_valid_index() > pd.Timestamp("2007-01-15")
    }
    if late:
        print("WARNING - history starts after 2007-01-15 for:")
        for k, v in late.items():
            print("   %-6s %s" % (k, v))
        print("   These cannot be used as test names. Drop or replace them.")

    raw.to_csv(OUT)
    print("\nwrote %s: %d rows, %d columns, %s to %s"
          % (OUT, len(raw), raw.shape[1], raw.index.min().date(), raw.index.max().date()))


if __name__ == "__main__":
    main()
