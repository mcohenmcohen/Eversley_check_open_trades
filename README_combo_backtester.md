# Combo Backtester

Tests permutations of binary indicator combinations to find which produce the best forward returns over 5 years of stock market history.

Reads pre-computed data from the [Momentum-Trading](https://github.com/mcohenmcohen/Momentum-Trading) pipeline — no need to run from that project.

## Quick Start

```bash
cd "/Users/mattcohen/Documents/Stocks-trading/Eversley/Python Find Open trades/currency_strategy_project_all_fixes"

# Full run (first time ~15 min to build cache, then <1s)
python combo_backtester.py

# Test a specific combo
python combo_backtester.py --test-combo has_candle_w has_close_MA200_w has_low_bband_w

# List all available indicators
python combo_backtester.py --list-indicators
```

## What It Does

1. Loads daily indicator data from `full.csv` (~1585 symbols, 5 years)
2. Computes weekly indicators from `ohlcv.pkl` via `trading_systems.py`
3. Computes Bayesian probability and curvature (saucer bottom) indicators
4. Tests all 2-4 indicator permutations (~92K combos)
5. For each combo, measures forward return win rates at 5, 10, 20, and 60 trading days
6. Outputs an Excel workbook with analysis, rankings, and confidence ratings

## Indicators (39 total)

| Category | Count | Examples |
|----------|-------|---------|
| Daily binary | 11 | `has_candle`, `has_volume`, `has_low_bband`, `has_20cross50` |
| Weekly binary | 11 | `has_candle_w`, `has_volume_w`, `has_low_bband_w` |
| Monthly | 1 | `has_close_MA40_m` (ATR-based proximity to 40-month MA) |
| Score Mean deciles | 10 | `score_00_10` through `score_90_100` |
| Bayesian probability | 4 | `bayes_up_5d`, `bayes_up_10d`, `bayes_up_20d`, `bayes_up_60d` |
| Curvature (saucer) | 2 | `has_curve_60d`, `has_curve_90d` |

### Bayesian Indicators

Rolling Beta-Binomial model on resolved historical outcomes. For each horizon N, looks at the last 50 resolved N-day trades and computes the posterior probability of price going up. Fires when P(up) > 0.6. Outcomes are lagged by N days to avoid look-ahead bias.

### Curvature Indicators

Fits a quadratic polynomial to rolling price windows (60 or 90 days). Detects saucer/rounding bottom patterns where price curves from downtrend to flat to uptrend. Fires when:
- Positive curvature (concave up)
- R-squared > 0.65 (clean curve fit)
- Bottom position in the "sweet spot" (50-80% of window) — not too early, not too late
- Current slope is positive (past the bottom)
- Prior decline confirmed

## CLI Options

```
python combo_backtester.py [options]

Options:
  --combo-size N        Max indicators per combo (default: 4)
  --test-combo IND...   Test one specific combo with detailed output
  --indicators IND...   Only test combos from these specific indicators
  --score-range LO HI   Additional Score Mean % filter (e.g., --score-range 70 90)
  --min-signals N       Minimum signals for valid combo (default: 30)
  --top N               Top results to display (default: 30)
  --output FILE         Custom output filename
  --no-save             Don't save xlsx output
  --clear-cache         Delete cache and rebuild from source data
  --daily-only          Skip weekly/monthly indicators
  --list-indicators     List all available indicators and exit
  --data-dir PATH       Path to symbol_data directory
  --date YYYY-MM-DD     Data date (default: latest available)
```

## Output

### Excel Workbook (`combo_backtest_results_YYYY-MM-DD.xlsx`)

| Tab | Contents |
|-----|----------|
| **Analysis** | Automated insights: indicator frequency, themes, best pairs, high-confidence combos, short opportunities, contrarian analysis, warnings |
| **Top 50** | Top 50 combos by 20d win rate with Confidence rating (High/Medium/Low), unique months, long and short win rates |
| **Baselines** | Each indicator tested individually for reference |
| **All Results** | Every valid combo with individual indicator flag columns (1/0) for Excel filtering |

### Key Columns

| Column | Description |
|--------|-------------|
| `win_Nd` | % of signals where price went UP over N days (long win rate) |
| `short_win_Nd` | % of signals where price went DOWN over N days (short win rate) |
| `avg_Nd` | Average forward return over N days |
| `unique_months` | Number of distinct calendar months with signals (detects clustering) |
| `Confidence` | High (200+ signals, 70%+ win, 60%+ at 60d), Medium (100+, 65%+), Low |

## Data Dependencies

Reads from the Momentum-Trading project's `symbol_data/` directory:

| File | Size | Purpose |
|------|------|---------|
| `{date}-full.csv` | ~360MB | Daily indicators, Score Mean, Close prices |
| `{date}-ohlcv.pkl` | ~91MB | Raw OHLCV for weekly indicator computation |

Also imports `trading_systems.get_technical_indicators_optimized()` at cache build time.

## Cache

First run builds a cache file (`{date}-combo_cache_v3.pkl`, ~650MB) in the `symbol_data/` directory. Subsequent runs load this in <1 second.

Clear the cache when:
- Indicator definitions change in `trading_systems.py`
- New data is available (new `full.csv` date)
- You modify the backtester's indicator computation

```bash
python combo_backtester.py --clear-cache
```

## Example Results

Top high-confidence combos (February 2026 data):

| Combo | Win 20d | Signals | Unique Months |
|-------|---------|---------|---------------|
| `has_candle_w + has_close_MA200_w + has_low_bband_w` | 96.8% | 346 | 4 |
| `has_candle_w + has_close_MA200_w + score_00_10` | 99.1% | 232 | 3 |
| `has_candle_w + has_close_MA40_m + has_low_bband_w` | 93.7% | 318 | — |
| `has_candle + has_close_MA200_w + has_low_bband + has_low_bband_w` | 94.7% | 306 | — |

Key insight: lowest-ranked stocks (score_00_10) paired with weekly candle/MA indicators produce the highest win rates — classic mean reversion on technically oversold names.
