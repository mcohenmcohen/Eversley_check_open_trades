#!/usr/bin/env python3
"""
Test script for the new data_sources module.
Tests both Polygon (ETF) and InsightSentry (futures) data sources.
"""

import sys
from datetime import datetime, timedelta
from data_sources import DataSourceManager

import os

# API Keys from environment variables (same as main backtester)
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
INSIGHTSENTRY_API_KEY = os.getenv('INSIGHTSENTRY_API_KEY')

def test_etf_data():
    """Test ETF data fetching from Polygon."""
    print("\n🔍 Testing ETF Data Sources")
    print("=" * 50)
    
    # Initialize data manager
    data_manager = DataSourceManager(
        polygon_api_key=POLYGON_API_KEY,
        insightsentry_api_key=INSIGHTSENTRY_API_KEY
    )
    
    # Test dates
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 10)
    
    # Test symbols
    test_symbols = ['SPY', 'QQQ', 'GLD']
    
    print(f"Testing symbols: {test_symbols}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        # Test single symbol
        print(f"\n📊 Testing single ETF: {test_symbols[0]}")
        df = data_manager.get_etf_data(test_symbols[0], start_date, end_date)
        if not df.empty:
            print(f"✅ Success: Got {len(df)} rows for {test_symbols[0]}")
            print(f"   Data range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample: Close = {df['Close'].iloc[-1]:.2f}")
        else:
            print(f"❌ Failed: No data for {test_symbols[0]}")
        
        # Test multiple symbols
        print(f"\n📊 Testing multiple ETFs")
        data_dict = data_manager.get_etf_data(test_symbols, start_date, end_date)
        if data_dict:
            print(f"✅ Success: Got data for {len(data_dict)}/{len(test_symbols)} symbols")
            for symbol, df in data_dict.items():
                print(f"   {symbol}: {len(df)} rows, Close = {df['Close'].iloc[-1]:.2f}")
        else:
            print("❌ Failed: No data returned for multiple symbols")
            
    except Exception as e:
        print(f"❌ Error testing ETF data: {e}")

def test_futures_data():
    """Test futures data fetching from InsightSentry and fallback to CSV."""
    print("\n🔍 Testing Futures Data Sources")
    print("=" * 50)
    
    # Initialize data manager
    data_manager = DataSourceManager(
        polygon_api_key=POLYGON_API_KEY,
        insightsentry_api_key=INSIGHTSENTRY_API_KEY
    )
    
    # Test dates
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 10)
    
    # Test symbols (currency futures)
    test_symbols = ['6E', '6A', '6B']
    
    print(f"Testing symbols: {test_symbols}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    if INSIGHTSENTRY_API_KEY:
        print("🌐 InsightSentry API key available - testing API")
        try:
            # Test single symbol
            print(f"\n📊 Testing single futures: {test_symbols[0]}")
            df = data_manager.get_futures_data(test_symbols[0], start_date, end_date)
            if not df.empty:
                print(f"✅ Success: Got {len(df)} rows for {test_symbols[0]} from InsightSentry")
                print(f"   Data range: {df.index.min().date()} to {df.index.max().date()}")
                print(f"   Sample: Close = {df['Close'].iloc[-1]:.5f}")
            else:
                print(f"❌ Failed: No data for {test_symbols[0]} from InsightSentry")
                
        except Exception as e:
            print(f"❌ Error testing InsightSentry: {e}")
    else:
        print("⚠️ InsightSentry API key not set - testing CSV fallback only")
    
    # Test CSV fallback
    print(f"\n📂 Testing CSV fallback for futures")
    from data_sources import load_futures_data_from_csv
    
    for symbol in test_symbols:
        try:
            df = load_futures_data_from_csv(symbol)
            if not df.empty:
                print(f"✅ CSV Success: Got {len(df)} rows for {symbol}")
                print(f"   Data range: {df.index.min().date()} to {df.index.max().date()}")
                print(f"   Sample: Close = {df['Close'].iloc[-1]:.5f}")
            else:
                print(f"❌ CSV Failed: No local data for {symbol}")
        except Exception as e:
            print(f"❌ CSV Error for {symbol}: {e}")

def test_integration():
    """Test integration with the main backtester interface."""
    print("\n🔍 Testing Integration with Main Backtester")
    print("=" * 50)
    
    # Test the main interface used by currency_strategy_backtester.py
    from currency_strategy_backtester import get_price_data, fetched_data_cache
    
    # Test ETF mode
    print("📊 Testing ETF mode integration")
    try:
        df = get_price_data('SPY', 'etfs', fetched_data_cache, datetime(2024, 1, 1))
        if df is not None and not df.empty:
            print(f"✅ ETF Integration Success: Got {len(df)} rows for SPY")
        else:
            print("❌ ETF Integration Failed")
    except Exception as e:
        print(f"❌ ETF Integration Error: {e}")
    
    # Test futures mode
    print("📊 Testing futures mode integration")
    try:
        df = get_price_data('6E', 'futures', fetched_data_cache, datetime(2024, 1, 1))
        if df is not None and not df.empty:
            print(f"✅ Futures Integration Success: Got {len(df)} rows for 6E")
        else:
            print("❌ Futures Integration Failed")
    except Exception as e:
        print(f"❌ Futures Integration Error: {e}")

def main():
    """Run all data source tests."""
    print("🚀 Data Sources Test Suite")
    print("=" * 60)
    
    # Test ETF data sources
    if POLYGON_API_KEY:
        test_etf_data()
    else:
        print("⚠️ Polygon API key not set - skipping ETF tests")
    
    # Test futures data sources
    test_futures_data()
    
    # Test integration
    test_integration()
    
    print("\n📋 Test Summary")
    print("=" * 60)
    print("✅ Tests completed - check output above for results")
    print("💡 To add InsightSentry testing, set INSIGHTSENTRY_API_KEY in this file")

if __name__ == "__main__":
    main()