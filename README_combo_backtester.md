# Combo Backtester — MOVED

This module has been migrated to the **Momentum-Trading** project where it belongs (it only operates on Momentum-Trading data).

## New Location

**Repo:** [Momentum-Trading](https://github.com/mcohenmcohen/Momentum-Trading)

```
/Users/mattcohen/Dev/Momentum-Trading/backtester/combo_backtester.py
/Users/mattcohen/Dev/Momentum-Trading/backtester/ohlcv_cache.py
/Users/mattcohen/Dev/Momentum-Trading/backtester/README.md
```

## Running It

```bash
cd /Users/mattcohen/Dev/Momentum-Trading
python backtester/combo_backtester.py                    # Full run
python backtester/combo_backtester.py --clear-cache      # Rebuild after indicator changes
python backtester/combo_backtester.py --list-indicators  # Show all 41 indicators
python backtester/combo_backtester.py --use-cache        # Use 20-year Parquet cache
python backtester/combo_backtester.py --update-cache     # Download/update OHLCV cache
```

See `backtester/README.md` in the Momentum-Trading project for full documentation.
