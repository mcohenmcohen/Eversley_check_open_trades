#!/usr/bin/env python3

import sys
from datetime import datetime, timedelta
from polygon.rest import RESTClient

POLYGON_API_KEY = 'qFTWmhmyAj2pJqpx0Pwxp2AyShKFVPv9'
polygon_client = RESTClient(api_key=POLYGON_API_KEY)

# Symbol mapping from your code
polygon_symbol_map = {
    # Currency futures
    "6A": "C:AUDUSD",  # Australian Dollar
    "6B": "C:GBPUSD",  # British Pound  
    "6C": "C:USDCAD",  # Canadian Dollar
    "6E": "C:EURUSD",  # Euro
    "6S": "C:USDCHF",  # Swiss Franc
    # Equity index futures (using ETF proxies where direct futures unavailable)
    "ES": "ES",        # S&P 500 E-mini
    "NQ": "I:NDX",     # Nasdaq 100 Index  
    "RTY": "IWM",      # Russell 2000 ETF proxy
    "YM": "DIA",       # Dow Jones ETF proxy
}

def test_symbol(symbol, polygon_ticker, test_date="2023-10-27"):
    """Test a single Polygon symbol"""
    print(f"\n🧪 Testing {symbol} -> {polygon_ticker}")
    
    try:
        # Test date range - recent business day
        start_date = test_date
        end_date = test_date
        
        print(f"   Fetching data for {test_date}")
        
        # Try to fetch data
        bars = polygon_client.get_aggs(
            ticker=polygon_ticker,
            multiplier=1,
            timespan='day',
            from_=start_date,
            to=end_date,
            adjusted=True,
            sort="asc",
            limit=5000
        )
        
        if bars and len(bars) > 0:
            bar = bars[0]
            print(f"   ✅ SUCCESS: O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume}")
            return True
        else:
            print(f"   ❌ FAILED: No data returned")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False

def main():
    print("🚀 Testing Polygon Futures Symbols")
    print("=" * 50)
    
    results = {}
    
    for symbol, polygon_ticker in polygon_symbol_map.items():
        success = test_symbol(symbol, polygon_ticker)
        results[symbol] = success
    
    print("\n📊 SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for symbol, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{symbol:4} -> {polygon_symbol_map[symbol]:12} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(results)} total")
    
    if failed > 0:
        print("\n⚠️  Some symbols failed. You may need to update the symbol mapping.")
        sys.exit(1)
    else:
        print("\n🎉 All symbols passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()