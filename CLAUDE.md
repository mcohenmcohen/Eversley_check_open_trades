# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Required environment variables in your `.zshrc` or `.bashrc`:
```bash
export POLYGON_API_KEY="your_polygon_api_key_here"
export INSIGHTSENTRY_API_KEY="your_insightsentry_api_key_here"  # Optional for futures
```

## Common Commands

Navigate to the main project directory:
```bash
cd "currency_strategy_project_all_fixes"
```

Run ETF backtesting:
```bash
python currency_strategy_backtester.py --mode etfs --debug
```

Run futures backtesting:
```bash
python currency_strategy_backtester.py --mode futures --debug
```

Launch Jupyter notebooks:
```bash
jupyter notebook currency_backtest_notebook.ipynb
jupyter notebook test_bench.ipynb
jupyter notebook symbol_source_tests.ipynb
```

Test data sources integration:
```bash
python test_data_sources.py
```

Test individual Polygon.io API:
```bash
python test_polygon_import.py --symbol QQQ --start_date 2023-10-27 --end_date 2023-10-30
```

## Project Architecture

This is a currency futures and ETF trading strategy backtesting system housed in the `currency_strategy_project_all_fixes/` directory.

### Core Components

**currency_strategy_backtester.py** - Main backtesting engine supporting both ETF and futures modes:
- Loads 11+ predefined trading strategies from JSON configuration
- Fetches historical price data via Polygon.io API and local CSV files
- Simulates trade execution with realistic entry/exit rules
- Outputs detailed trade results with P&L calculations

**strategies_complete.json** - Strategy configuration file defining:
- Entry trigger formulas (high/low offset calculations with tick adjustments)
- Stop loss rules (ATR-based or fixed offset calculations)
- Target price formulas
- Trigger window parameters (2-3 day windows)

**utilities.py** - Shared utility functions for:
- Date parsing and options expiration calculations  
- Strategy name resolution and fuzzy matching
- Local CSV data loading for futures contracts
- Holiday and trading calendar management

### Data Sources and Flow

**Price Data Sources:**
- ETFs: Downloaded via Polygon.io API
- Futures: InsightSentry API (primary) with fallback to local CSV files in `data/` directory (6A.csv, 6B.csv, 6C.csv, 6E.csv, 6S.csv, ES.csv, NQ.csv, RTY.csv, YM.csv)

**Input/Output Files:**
- `trade_signals_*.csv` - Input files containing ticker, strategy name, signal date
- `trade_results_*.csv` - Output files with comprehensive trade outcomes
- Historical OHLC data cached for performance

### Symbol and Market Support

**Currency Futures:** 6A (Australian Dollar), 6B (British Pound), 6C (Canadian Dollar), 6E (Euro), 6S (Swiss Franc)

**ETFs:** 20+ ETFs including SPY, QQQ, GLD, DIA, IWM with proper tick size mappings

### Data Sources Module

**data_sources.py** - Unified data source management:
- `DataSourceManager` class coordinates between different APIs
- `PolygonDataSource` handles ETF data via Polygon.io API
- `InsightSentryDataSource` handles futures data via InsightSentry API
- Automatic fallback to local CSV files for futures if API unavailable
- Concurrent data fetching for improved performance

### Key Dependencies

Required Python packages:
- pandas (data manipulation)
- polygon-api-client (Polygon.io data for ETFs)
- requests (InsightSentry API integration)
- rapidfuzz (fuzzy string matching for strategy names)
- dateutil (date calculations)
- holidays (trading calendar)

### Trading Logic Framework

- Entry triggers within configurable windows (2-3 days) after signal dates
- Dynamic entry pricing using high/low offsets with symbol-specific tick adjustments
- Multi-layered stop loss: ATR-based calculations with minimum offset floors
- Target calculations using ATR multiples or fixed ratios
- Position tracking until stop/target hit or options expiration
- Comprehensive trade metrics including candlestick pattern analysis

## Development Notes

The system uses Polygon.io API for ETF data and local CSV files for futures data, with built-in caching for performance. The backtesting engine handles different execution rules for futures vs ETF strategies, supporting both intraday and daily timeframes with realistic slippage and commission modeling.