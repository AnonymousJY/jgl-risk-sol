# jgl-risk-sol

Risk management solutions built on the systematic liquidity risk framework.

## Current work: spot backfill study

Reconstruction of crisis-period spot returns for equity underlyings that did not
trade during a stress period, using the systematic liquidity risk framework of
Yi & Kim (2026).

Seeded from the paper repository `mkt-depth-n-resiliency` at commit `76e943d`.
History was not carried over; this repository starts fresh.

## Why

A name that listed after 2009 has no global-financial-crisis history at any data
vendor. The regulatory fallback is to map it onto a proxy — a sector ETF, or the
nearest reduced-set risk factor. This study tests whether a structural
reconstruction beats that proxy.

The regulatory standard being targeted is Basel MAR31.25 Principle six: where
"instruments that are currently traded did not exist during a period of
significant financial stress, banks must demonstrate that the prices used match
changes in prices or spreads of similar instruments during the stress period."

## Layout

    Library/   model implementation carried over from the paper repository
    Scripts/   paper reproduction and calibration scripts
    poc/       the spot backfill study

## Running the study

    cd poc
    pip install yfinance pandas numpy
    python fetch_prices.py            # writes prices.csv from 2007-01-01
    python backfill_poc.py prices.csv

`backfill_poc.py` prints the systematic factor parameters and a
jump-mass-by-episode table. **Read that table first.** The recovered jump mass
must concentrate on Sep-Oct 2008, Aug 2011, Aug 2015, Feb 2018, Mar 2020 and
Apr 2025. If it does not, the filter is wrong and nothing downstream is
meaningful.

## Design notes

- The object reconstructed is the **relative return series**, not a price level.
  A synthetic price level for a period the security did not trade is not
  defensible, and the return series is what a risk engine consumes.
- Systematic diffusion and common jump realisations are **historical fact**,
  recovered from the index and held fixed. Only the idiosyncratic component is
  simulated. The reconstruction is therefore mostly pinned, partly simulated.
- Day-by-day path accuracy is not testable and is not claimed. The tests are
  distributional, and the claim is **dominance over the incumbent proxy**, not
  accuracy against ground truth.
- Test names are selected ex ante on sector and size, never on 2008 behaviour,
  which would be selection leakage. GE and BAC are deliberate negative controls
  with large firm-specific 2008 events.
- Every test name survived to the present, so the left tail is absent by
  construction. This is a known optimistic bias, disclosed rather than hidden.

## Status

Step 1 (systematic factor filter) not yet validated. See the study specification
kept outside this repository.
