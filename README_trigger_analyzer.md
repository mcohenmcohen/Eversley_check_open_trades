# ETF Strategy Trigger Analyzer

Automated signal detection system that scans ETF symbols for Eversley's strategy triggers on any given date. Implements the indicator calculations and setup/trigger rules from the strategy PDFs so you don't need to wait for manual signals.

## Quick Start

```bash
# Scan all 29 ETFs for today's triggers
python trigger_analyzer.py

# Scan specific symbols
python trigger_analyzer.py --symbols AAPL MSFT GOOGL

# Scan a historical date
python trigger_analyzer.py --date 2025-06-15

# Save results to CSV
python trigger_analyzer.py --output triggers.csv

# Include Put/Call and internals strategies
python trigger_analyzer.py --include-internals

# Verbose debug output (shows indicator values and rule evaluation)
python trigger_analyzer.py --debug
```

## Requirements

**Environment variable:**
```bash
export POLYGON_API_KEY="your_massive_api_key"  # Massive.com (formerly Polygon.io) API key
```

**Python packages:**
```
pandas
numpy
requests
polygon-api-client
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--symbols SYM1 SYM2 ...` | Eversley's 29 ETFs | Symbols to scan (any US equity/ETF) |
| `--date YYYY-MM-DD` | Today | Date to evaluate triggers for |
| `--include-internals` | Off | Enable Put/Call ratio and NYSE A/D strategies |
| `--output PATH` | stdout | Save results to CSV file |
| `--debug` | Off | Print indicator values and rule evaluation details |
| `--lookback-days N` | 730 | Days of historical data to fetch for indicator computation |
| `--config PATH` | etf_trigger_config.json | Custom strategy config file |

## Strategies (21 total)

### Daily Broad (9 strategies)

| Strategy | Direction | Symbols |
|----------|-----------|---------|
| Super Trend Sector Buy | Buy | DIA, QQQ, SPY, XLI, XLK |
| Donchian Crossover Buy | Buy | All 29 ETFs |
| Donchian T-Lines Buy | Buy | All 29 ETFs |
| Donchian T-Lines Sell | Sell | All 29 ETFs |
| Donchian Hook Buy | Buy | All 29 ETFs |
| Ichimoku Kijun 5 Hook Buy | Buy | All 29 ETFs |
| ETF Squeeze Sell | Sell | All 29 ETFs |
| Squeeze Play Sector Buy | Buy | DIA, QQQ, SPY, XLI, XLK |
| Squeeze Play Sector Sell | Sell | DIA, QQQ, SPY, XLI, XLK |

### Weekly (5 strategies)

| Strategy | Direction | Symbols |
|----------|-----------|---------|
| Weekly Swing Trend Buy (Setup 1 & 2) | Buy | All 29 ETFs |
| Weekly Turbo Hook Buy | Buy | All 29 ETFs |
| Weekly 10.5 Squeeze Buy | Buy | All 29 ETFs |
| Weekly Swing Sell | Sell | All 29 ETFs |
| Weekly Ichimoku Squeeze Sell | Sell | All 29 ETFs (requires `--include-internals`) |

### Daily Internals (6 strategies, require `--include-internals`)

All use CBOE Put/Call Ratio as a trigger condition.

| Strategy | Direction | Symbols |
|----------|-----------|---------|
| Squeeze Put/Call Buy | Buy | DIA, QQQ, SPY, XLI, XLK |
| Ichimoku Tenkan Hook Put/Call Buy | Buy | DIA, QQQ, SPY, XLI, XLK |
| Ichimoku Tenkan Flat Put/Call Buy | Buy | DIA, QQQ, SPY, XLI, XLK |
| Stochastics Hook Put/Call Buy | Buy | DIA, QQQ, SPY, XLI, XLK |
| 20.8 Trigger Lines Put/Call Buy | Buy | DIA, QQQ, SPY, XLI, XLK |
| Gap and Go Put/Call Buy | Buy | DIA, QQQ, SPY, XLI, XLK |

## Technical Indicators (15)

All implemented in pure pandas/numpy (no talib dependency). Calibrated against NinjaTrader source code.

| Indicator | Configurations |
|-----------|---------------|
| Donchian Channel | 7-period, 14-period |
| Trigger Lines | (20,8) and (10,5) — LinReg + EMA |
| SMA | 14, 40-period |
| Bollinger Bands | (20,2), (30,2), (30,1), (20,1.7), (20,1.8) |
| ATR (Wilder's) | 5, 14, 20-period |
| RSqueeze / BBSqueeze | 3 configs (momentum 4, 20, 23) |
| TSSuperTrend | Daily (14,0.78) and Weekly (14,0.38), (5,0.23) |
| TTM Squeeze | (20, 1.5, 2.0, 1.0) |
| Ichimoku Cloud | (9,26,52) and (9,3,52) |
| Stochastics Full | (K=10, D=10, Slowing=3) |
| Woodies CCI | (14, 6) dual-period |
| Price Action Swing Trend | Gann-style bar-by-bar swing detection |
| Volume | Day-over-day comparison |
| CBOE Put/Call Ratio | External fetch with local cache |
| NYSE Advance/Decline | External fetch with local cache |

## Output Format

```
symbol  strategy                  direction  timeframe  scan_date   atr_5   donchian_14_mean
SPY     Donchian Crossover Buy    buy        daily      2025-05-12  10.38   558.44
DIA     Ichimoku Kijun 5 Hook Buy buy        daily      2025-05-12   6.88   NaN
GDX     Weekly 10.5 Squeeze Buy   buy        weekly     2025-01-31   2.18   NaN
```

Each row is a triggered signal. Columns include the key signal info plus relevant indicator values for that strategy (other indicators show NaN).

## Programmatic Usage

The `TriggerAnalyzer` class can be used directly in Python:

```python
from datetime import date
from trigger_analyzer import TriggerAnalyzer

analyzer = TriggerAnalyzer(
    symbols=['SPY', 'QQQ', 'DIA'],
    scan_date=date(2025, 5, 12),
    lookback_days=730
)
results = analyzer.run()  # Returns a pandas DataFrame
```

This API is designed for composability — the planned backtester (Task 2) will call this in a loop across historical dates to generate win rate statistics for custom symbol sets.

## Architecture

```
trigger_analyzer.py          CLI entry point + TriggerAnalyzer orchestrator
    |
    +-- indicators.py        15 technical indicator implementations
    +-- strategy_rules.py    JSON-driven rule engine (8 condition types)
    +-- etf_trigger_config.json  21 strategy definitions
    +-- external_data.py     CBOE Put/Call ratio + NYSE A/D fetchers
    |
    +-- data_sources.py      OHLCV data from Massive.com API (existing)
    +-- utilities.py         Date/holiday helpers (existing)
```

**Separate from `trade_evaluator.py`** which evaluates known signals (entry/stop/target outcomes). This module *detects* the signals; the trade evaluator *evaluates* them.

## Validation

Tested against Eversley's actual 2025 signals across 7 dates (50 signals total):

| Date | Score | Strategies Tested |
|------|-------|-------------------|
| 1/17/25 | 3/3 | Weekly 10.5 Squeeze, Weekly Turbo Hook |
| 1/22/25 | 3/3 | Donchian Hook, Donchian T-Lines |
| 1/31/25 | 8/8 | Weekly 10.5 Squeeze, Weekly Swing Sell, Weekly Swing Trend |
| 2/7/25 | 5/7 | ETF Squeeze Sell, Weekly Swing Trend |
| 2/14/25 | 8/8 | Squeeze Play Sell, Weekly 10.5 Squeeze, Weekly Swing Trend, Weekly Turbo Hook |
| 4/3/25 | 6/11 | Donchian T-Lines Sell, ETF Squeeze Sell |
| 5/12/25 | 9/10 | Donchian Crossover, Ichimoku Kijun 5 Hook |

**Overall: 42/50 (84%)** — 92% excluding April 3 tariff-day anomaly where SMA(40) had already turned down.

## Default Symbols (29 ETFs)

DIA, EEM, EFA, EWG, EWH, FXI, GDX, GLD, IBB, IWM, IYR, QQQ, SLV, SPY, TLT, UNG, USO, VNQ, VWO, XHB, XLB, XLE, XLF, XLI, XLK, XLV, XME, XOP, XRT
