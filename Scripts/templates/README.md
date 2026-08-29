# Canonical option data schema

The two calibration scripts (`skew_calibrate_systematic.py` and
`skew_calibrate_idiosyncratic.py`) are vendor-agnostic: they consume a
canonical DataFrame whose columns are described below, and any raw input
is reshaped into that shape by a *loader adapter* dispatched off
`config_skew.DATA_SOURCE`.

- **`cboe`** — the loader in `skew_calibrate_systematic.load_cboe` reads
  daily CBOE CSV snapshots (`quote_date`, `expiration`, `strike`,
  `option_type`, `active_underlying_price_1545`, `implied_volatility_1545`,
  `trade_volume`, `underlying_symbol`) and reshapes them.
- **`canonical`** — the loader in `skew_calibrate_systematic.load_canonical`
  reads a CSV or Parquet whose columns already match the schema below, so
  the user (or an upstream ETL step) is responsible for producing that
  shape. This is the path for anything that isn't CBOE.

Adding an adapter for OptionMetrics, Databento, or ORATS is a matter of
writing a function `(underlying_meta: dict, ticker: str) -> pd.DataFrame`
that returns canonical data and registering it in
`skew_calibrate_systematic.LOADERS`.

## Columns

All rates and yields are decimals — `0.0125` means 1.25%, matching
`FLAT_RATE` in `config_skew.py`. No `/ 100` anywhere in the pipeline.

| Column               | Type      | Meaning                                                        |
| -------------------- | --------- | -------------------------------------------------------------- |
| `quote_date`         | Timestamp | Valuation date the smile is priced at.                         |
| `underlying_symbol`  | str       | Ticker (`"^SPX"`, `"COIN"`, ...).                              |
| `dUND_PRICE`         | float     | Spot price of the underlying.                                  |
| `dUND_STRIKE`        | float     | Strike price of the option.                                    |
| `iEXPIRY`            | int       | Days to expiry (integer calendar days).                        |
| `dEXPIRY`            | float     | Years to expiry (`iEXPIRY / 365`).                             |
| `bIS_CALL_OPTION`    | bool      | `True` for calls, `False` for puts.                            |
| `dMKT_IMP_VOL`       | float     | Market implied vol as decimal (0.20 = 20%).                    |
| `dRISK_FREE_RATE`    | float     | Risk-free rate as decimal (0.0430 = 4.30%).                    |
| `dDIVIDEND_YIELD`    | float     | Continuous dividend yield as decimal.                          |

The `cboe` loader does not carry `dRISK_FREE_RATE` — the pipeline calls
`attach_risk_free_rate` after loading to populate it from
`cfg.RATE_SOURCE`. The `canonical` loader expects the file to carry
`dRISK_FREE_RATE`; the downstream `attach_risk_free_rate` is idempotent
and skips itself when the column is already present with non-null values.

### Derived columns (never required in input)

The pipeline computes these automatically after loading, so they should
**not** be in the canonical file:

- `dMONEYNESS` — `dUND_STRIKE / dUND_PRICE * 100`, used for filtering
  against `cfg.MONEYNESS_PUT_RANGE` / `MONEYNESS_CALL_RANGE`.
- `dVEGA` — Black-Scholes-Merton put vega, computed by
  `attach_vega_weights`, used as the least-squares weight vector.

### Optional columns

- `trade_volume` — if present, `cfg.MIN_TRADE_VOLUME` filter applies. If
  absent, no volume filter (equivalent to +∞).

Any additional columns in the input are ignored.

## How to configure the pipeline to read a canonical file

`config_skew.py`:

```python
DATA_SOURCE = "canonical"

SYSTEMATIC_UNDERLYING = {
    "ticker": "^SPX",
    "dividend_yield": 0.0125,
    "data_path": "~/data/spx_option_chain_2007_2025.parquet",  # or .csv
    "display_name": "S&P 500 Index",
}

IDIOSYNCRATIC_UNDERLYINGS = {
    "COIN": {
        "dividend_yield": 0.0,
        "data_path": "~/data/coin_option_chain.parquet",
        "display_name": "Coinbase Global",
    },
}
```

Or override from the command line for a one-off run:

```
python Scripts/skew_calibrate_systematic.py \
    --data-source canonical \
    --data-path ~/data/spx_2007_2025.parquet \
    --tenor-mode list --tenors 30 90 --tenor-tolerance 5 \
    --n-jobs 8
```

## Format notes

- **CSV or Parquet** — the loader dispatches on file extension. Prefer
  Parquet for anything over ~100k rows: smaller on disk, faster to read,
  preserves the boolean and datetime dtypes without string round-tripping.
- **One file, multiple tickers is fine** — the loader filters by
  `underlying_symbol == ticker`. So a single canonical file mixing
  `^SPX` and `COIN` rows can back the whole pipeline; each stage picks up
  the rows it needs.
- **UTF-8, no BOM.**
- **Datetime format** — the CSV path uses `pd.read_csv(..., parse_dates=[
  "quote_date"])`, which accepts `YYYY-MM-DD` and `YYYY-MM-DDTHH:MM:SS`
  cleanly. Parquet files preserve dtype so anything Pandas-writable works.

See `canonical_option_data.csv` in this directory for a fully worked
12-row example with two tenors, both puts and calls, unit-consistent
yields, and an at-the-money pair on each tenor.
