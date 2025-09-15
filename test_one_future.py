#!/usr/bin/env python3
"""
Test one futures symbol to verify the InsightSentry integration works.
"""

from datetime import datetime, timedelta
from data_sources import DataSourceManager
import os

# Test the updated futures mapping
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
INSIGHTSENTRY_API_KEY = os.getenv('INSIGHTSENTRY_API_KEY')

def test_futures_symbol():
    print("🧪 Testing InsightSentry Futures Integration")
    print("=" * 50)
    
    # Initialize data manager
    data_manager = DataSourceManager(
        polygon_api_key=POLYGON_API_KEY,
        insightsentry_api_key=INSIGHTSENTRY_API_KEY
    )
    
    # Test one futures symbol
    test_symbol = '6E'  # Euro futures
    start_date = datetime(2024, 8, 1)
    end_date = datetime(2024, 8, 10)
    
    print(f"Testing symbol: {test_symbol} (Euro futures)")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        df = data_manager.get_futures_data(test_symbol, start_date, end_date)
        
        if df.empty:
            print("❌ No data returned")
        else:
            print(f"✅ Success! Retrieved {len(df)} rows")
            print(f"📊 Data range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"📈 Sample prices:")
            print(f"   Open: {df['Open'].iloc[0]:.5f}")
            print(f"   High: {df['High'].iloc[0]:.5f}")
            print(f"   Low:  {df['Low'].iloc[0]:.5f}")
            print(f"   Close: {df['Close'].iloc[0]:.5f}")
            
            print(f"\n📋 Data structure:")
            print(df.head(3))
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if not INSIGHTSENTRY_API_KEY:
        print("❌ INSIGHTSENTRY_API_KEY not set")
        exit(1)
    
    test_futures_symbol()