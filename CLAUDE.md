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
- Target price formulas (including multi-target system for optimal target selection)
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
- **Multi-target system**: Evaluates multiple target options (ATR percentages, entry-stop percentages) and automatically selects the target with fewest ticks for optimal probability
- Position tracking until stop/target hit or options expiration
- Comprehensive trade metrics including candlestick pattern analysis

### Multi-Target System

The system supports sophisticated target selection for futures strategies:

**Target Types:**
- `atr_percentage`: ATR-based targets (e.g., ATR5 x 0.6, ATR5 x 0.7)
- `entry_stop_percentage`: Entry-stop difference percentages (e.g., Entry-Stop x 0.4, Entry-Stop x 0.45)
- `multi_target`: Evaluates multiple target options and selects by rank (1=closest target)

**Configuration Example:**
```json
"target": {
    "formula": {
        "type": "multi_target",
        "target_rank": 1,
        "direction": "buy",
        "target_options": [
            {"type": "atr_percentage", "atr_length": 5, "percentage": 0.6},
            {"type": "atr_percentage", "atr_length": 5, "percentage": 0.7},
            {"type": "entry_stop_percentage", "percentage": 0.4},
            {"type": "entry_stop_percentage", "percentage": 0.45}
        ]
    }
}
```

**Auto-Detection:**
- Index futures (ES, YM, NQ, RTY): Use points for calculations
- Currency futures (6A, 6B, 6C, 6E, 6S): Use tick-adjusted calculations
- Results show specific target type selected (e.g., "ATR5 x 0.6", "Entry-Stop x 0.4")

## Development Notes

The system uses Polygon.io API for ETF data and InsightSentry API for futures data, with built-in caching for performance. The backtesting engine handles different execution rules for futures vs ETF strategies, supporting both intraday and daily timeframes with realistic slippage and commission modeling.

### Recent Updates

**Weekly ATR Look-Ahead Bias Fix (December 2024):**
- **CRITICAL BUG FIX:** Fixed look-ahead bias in weekly ATR calculations
- Weekly strategies were using future data (most recent week's ATR) instead of ATR as of signal date
- Fixed hardcoded ETF target calculations that ignored strategy configuration
- Now correctly uses strategy JSON config for timeframe (weekly vs daily) and multiplier (e.g., 0.55)
- Results are now consistent and reproducible regardless of when backtest is run

**Impact:**
- Weekly strategies now use correct ATR values (e.g., GLD 11/28: $15.59 instead of $14.03)
- Target prices accurately reflect strategy config (e.g., GLD: $399.19 instead of $395.92)
- Eliminates look-ahead bias that made historical backtests unreliable

**Key Files Modified:**
- `currency_strategy_backtester.py` (lines 389, 1191): Fixed weekly ATR to use signal date
- `trading_strategies.py` (lines 53-87): Rewrote ETF evaluate_exit() to respect strategy config

**Multi-Target Implementation (2025):**
- Added sophisticated multi-target evaluation system for futures strategies
- Supports ATR-based and entry-stop percentage target calculations
- Automatic target selection based on tick count optimization (selects closest/most probable target)
- Generic formula naming system with auto-detection of points vs ticks based on symbol type
- Target type results show specific selected target (e.g., "ATR5 x 0.6") instead of generic labels

**Key Files Modified:**
- `currency_strategy_backtester.py`: Added multi-target calculation and selection logic
- `trading_strategies.py`: Enhanced target type detection and polymorphic formula handling
- `strategies_complete.json`: Updated to support flexible multi-target configurations per strategy