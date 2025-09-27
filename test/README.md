# Test Files Directory

This directory contains all test and utility scripts for the currency strategy backtesting system.

## Test Files

### Data Source Tests
- `test_data_sources.py` - Comprehensive test of both Polygon (ETF) and InsightSentry (futures) data sources
- `test_one_future.py` - Simple test of InsightSentry futures integration for one symbol
- `test_polygon.py` - Basic Polygon API test for individual symbols
- `test_polygon_import.py` - Advanced Polygon API test with date ranges

### Polygon API Tests
- `test_polygon_futures.py` - Test various Polygon futures symbol formats
- `test_polygon_symbol_search.py` - Search for symbols on Polygon
- `test_polygon_symbols.py` - Test Polygon symbol variations
- `check_polygon_contracts.py` - Fetch and analyze all available Polygon futures contracts

### Bulk Data Tests
- `test_bulk_aggs.py` - Test bulk aggregates retrieval without compression
- `test_compressed_data.py` - Test bulk download with LZ4 compression
- `test_9_symbols.py` - Test specific set of 9 symbols

### InsightSentry Tests
- `test_insightsentry.py` - Direct InsightSentry API testing

## Running Tests

From the main project directory:

```bash
# Test data sources integration
python test/test_data_sources.py

# Test individual futures symbol
python test/test_one_future.py

# Test Polygon API for specific symbol
python test/test_polygon.py QQQ

# Test with date range
python test/test_polygon_import.py --symbol QQQ --start_date 2023-10-27 --end_date 2023-10-30
```

## Notes

- All test files that import local modules (`data_sources`, `currency_strategy_backtester`) have been updated with proper path handling
- Tests require appropriate environment variables (`POLYGON_API_KEY`, `INSIGHTSENTRY_API_KEY`)
- Some tests may fail if API keys are not set or if data is not available for the requested symbols/dates