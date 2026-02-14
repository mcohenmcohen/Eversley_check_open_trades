# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Required environment variables in your `.zshrc` or `.bashrc`:
```bash
export POLYGON_API_KEY="your_massive_api_key_here"  # Massive.com API key (formerly Polygon.io) — used for both ETFs and futures
export INSIGHTSENTRY_API_KEY="your_insightsentry_api_key_here"  # Deprecated, kept for backward compatibility
```

## Common Commands

Navigate to the main project directory:
```bash
cd "currency_strategy_project_all_fixes"
```

Run ETF trade evaluation:
```bash
python trade_evaluator.py --mode etfs --debug
```

Run futures trade evaluation:
```bash
python trade_evaluator.py --mode futures --debug
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

Test individual Massive.com API (via legacy polygon SDK):
```bash
python test_polygon_import.py --symbol QQQ --start_date 2023-10-27 --end_date 2023-10-30
```

## Project Architecture

This is a currency futures and ETF trading strategy trade evaluation system housed in the `currency_strategy_project_all_fixes/` directory.

### Core Components

**trade_evaluator.py** - Main trade evaluation engine supporting both ETF and futures modes:
- Loads 11+ predefined trading strategies from JSON configuration
- Fetches historical price data via Massive.com API and local CSV files
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

**Price Data Sources (all via Massive.com, formerly Polygon.io):**
- ETFs: Massive.com API via legacy `polygon` Python SDK
- Futures: Massive.com REST API with fallback to local CSV files in `data/` directory (6A.csv, 6B.csv, 6C.csv, 6E.csv, 6S.csv, ES.csv, NQ.csv, RTY.csv, YM.csv)

**Input/Output Files:**
- `trade_signals_*.csv` - Input files containing ticker, strategy name, signal date
- `trade_results_*.csv` - Output files with comprehensive trade outcomes
- Historical OHLC data cached for performance

### Symbol and Market Support

**Currency Futures:** 6A (Australian Dollar), 6B (British Pound), 6C (Canadian Dollar), 6E (Euro), 6S (Swiss Franc)

**ETFs:** 20+ ETFs including SPY, QQQ, GLD, DIA, IWM with proper tick size mappings

### Data Sources Module

**data_sources.py** - Unified data source management (all via Massive.com, formerly Polygon.io):
- `DataSourceManager` class coordinates between data sources
- `PolygonDataSource` handles ETF data via legacy `polygon` Python SDK (Massive.com)
- `MassiveDataSource` handles futures data via Massive.com REST API
- `InsightSentryDataSource` deprecated (kept for backward compatibility)
- Automatic fallback to local CSV files for futures if API unavailable
- Concurrent data fetching for improved performance

### Key Dependencies

Required Python packages:
- pandas (data manipulation)
- polygon-api-client (Massive.com ETF data via legacy Polygon SDK)
- requests (Massive.com futures REST API, legacy InsightSentry)
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

The system uses Massive.com (formerly Polygon.io) for all market data — both ETFs and futures use the same API key. The trade evaluation engine handles different execution rules for futures vs ETF strategies, supporting both intraday and daily timeframes with realistic slippage and commission modeling.

### Recent Updates

**Weekly ATR Look-Ahead Bias Fix (December 2024):**
- **CRITICAL BUG FIX:** Fixed look-ahead bias in weekly ATR calculations
- Weekly strategies were using future data (most recent week's ATR) instead of ATR as of signal date
- Fixed hardcoded ETF target calculations that ignored strategy configuration
- Now correctly uses strategy JSON config for timeframe (weekly vs daily) and multiplier (e.g., 0.55)
- Results are now consistent and reproducible regardless of when evaluation is run

**Impact:**
- Weekly strategies now use correct ATR values (e.g., GLD 11/28: $15.59 instead of $14.03)
- Target prices accurately reflect strategy config (e.g., GLD: $399.19 instead of $395.92)
- Eliminates look-ahead bias that made historical evaluations unreliable

**Key Files Modified:**
- `trade_evaluator.py` (lines 389, 1191): Fixed weekly ATR to use signal date
- `trading_strategies.py` (lines 53-87): Rewrote ETF evaluate_exit() to respect strategy config

**Multi-Target Implementation (2025):**
- Added sophisticated multi-target evaluation system for futures strategies
- Supports ATR-based and entry-stop percentage target calculations
- Automatic target selection based on tick count optimization (selects closest/most probable target)
- Generic formula naming system with auto-detection of points vs ticks based on symbol type
- Target type results show specific selected target (e.g., "ATR5 x 0.6") instead of generic labels

**Key Files Modified:**
- `trade_evaluator.py`: Added multi-target calculation and selection logic
- `trading_strategies.py`: Enhanced target type detection and polymorphic formula handling
- `strategies_complete.json`: Updated to support flexible multi-target configurations per strategy

**Massive.com Data Migration (December 2024):**
- **CRITICAL DATA QUALITY FIX:** Migrated from InsightSentry to Massive.com (formerly Polygon.io) for futures data
- InsightSentry had severe data quality issues (42+ tick discrepancies vs ThinkorSwim)
- Massive.com data matches ThinkorSwim within 0-1 ticks (production quality)
- Added `MassiveDataSource` class with proper contract code mapping (6B→6BZ5, 6E→6EZ5, etc.)
- Same Massive.com API key (POLYGON_API_KEY env var) powers both ETF and futures data

**Impact:**
- 6B (12/3): Entry now 1.3369 (exactly matches manual calculations vs 1.3373 with InsightSentry)
- 6E (12/10): High 1.1703 matches ThinkorSwim perfectly (vs 1.16605, 42 ticks off with InsightSentry)
- All futures calculations now accurate and reliable for live trading decisions

**Non-Triggered Trade Improvements:**
- Added "Open - Entry Not Triggered" status for trades still in trigger window
- Added "Expired - Entry Not Triggered" for trades past trigger window
- Now shows calculated entry/stop/target values even for non-triggered trades (essential for analysis)
- `num_days_open` column shows "Expired" for expired non-triggered trades vs "open" for active ones

**Key Files Modified:**
- `data_sources.py` (lines 193-405): Added MassiveDataSource class with contract mappings
- `data_sources.py` (lines 913-1045): Updated DataSourceManager to prioritize Massive.com
- `trade_evaluator.py` (lines 938-1036): Enhanced non-triggered trade value calculation
- `trade_evaluator.py` (lines 1217-1222): Added Expired status for num_days_open column
