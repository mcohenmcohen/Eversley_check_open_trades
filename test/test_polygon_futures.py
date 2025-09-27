#!/usr/bin/env python3

from polygon import RESTClient
import os
from datetime import datetime, timedelta

# Get API key from environment variable
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

def test_polygon_futures_symbol(symbol, test_date="2024-10-27"):
    """
    Test a Polygon futures symbol and return results or error

    Args:
        symbol (str): The futures symbol to test (e.g., '6A=F', 'ES=F', etc.)
        test_date (str): Date to test in YYYY-MM-DD format

    Returns:
        dict: Results containing success status, data, or error message
    """
    try:
        print(f"📡 Testing futures symbol: {symbol} for date: {test_date}")

        if not POLYGON_API_KEY:
            raise ValueError("POLYGON_API_KEY environment variable not set")

        client = RESTClient(api_key=POLYGON_API_KEY)

        # Get daily bars for the test date
        bars = client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan='day',
            from_=test_date,
            to=test_date,
            adjusted=True,
            sort='asc',
            limit=50
        )

        print(f"   Request: GET /v2/aggs/ticker/{symbol}/range/1/day/{test_date}/{test_date}")

        if bars and len(bars) > 0:
            bar = bars[0]
            result = {
                'success': True,
                'symbol': symbol,
                'date': test_date,
                'data': {
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'timestamp': bar.timestamp
                }
            }
            print(f"✅ SUCCESS: O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume}")
            return result
        else:
            result = {
                'success': False,
                'symbol': symbol,
                'date': test_date,
                'error': 'No data returned from Polygon API'
            }
            print(f"❌ FAILED: No data returned")
            return result

    except Exception as e:
        result = {
            'success': False,
            'symbol': symbol,
            'date': test_date,
            'error': str(e)
        }
        print(f"❌ ERROR: {str(e)}")
        return result

def main():
    """
    Test all 9 futures symbols used in your trading system

    Expected execution:
    python test_polygon_futures.py                    # Test all futures symbols
    python test_polygon_futures.py SYMBOL            # Test specific symbol with default date
    python test_polygon_futures.py SYMBOL DATE       # Test specific symbol with specific date
    """
    print("🚀 Polygon Futures Symbol Tester")
    print("=" * 50)

    if not POLYGON_API_KEY:
        print("❌ ERROR: POLYGON_API_KEY environment variable not set")
        print("Please set it with: export POLYGON_API_KEY=your_api_key_here")
        return

    # Test all 9 futures symbols from your trading system
    test_symbols = [
        # Currency Futures
        '6A=F',     # Australian Dollar futures
        '6B=F',     # British Pound futures
        '6C=F',     # Canadian Dollar futures
        '6E=F',     # Euro futures
        '6S=F',     # Swiss Franc futures

        # Index E-mini Futures
        'ES=F',     # S&P 500 E-mini futures
        'YM=F',     # Dow Jones E-mini futures
        'NQ=F',     # Nasdaq E-mini futures
        'RTY=F'     # Russell 2000 E-mini futures
    ]

    # Alternative symbol formats to test if =F doesn't work
    alternative_formats = {
        '6A=F': ['6A1!', 'C:6A1!', '6AZ24'],
        '6B=F': ['6B1!', 'C:6B1!', '6BZ24'],
        '6C=F': ['6C1!', 'C:6C1!', '6CZ24'],
        '6E=F': ['6E1!', 'C:6E1!', '6EZ24'],
        '6S=F': ['6S1!', 'C:6S1!', '6SZ24'],
        'ES=F': ['ES1!', 'C:ES1!', 'ESZ24'],
        'YM=F': ['YM1!', 'C:YM1!', 'YMZ24'],
        'NQ=F': ['NQ1!', 'C:NQ1!', 'NQZ24'],
        'RTY=F': ['RTY1!', 'C:RTY1!', 'RTYZ24']
    }

    results = []
    working_symbols = {}

    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol} and alternatives...")

        # Test primary symbol format
        result = test_polygon_futures_symbol(symbol)
        results.append(result)

        if result['success']:
            working_symbols[symbol.replace('=F', '')] = symbol
            print(f"   ✅ PRIMARY FORMAT WORKS: {symbol}")
        else:
            # Try alternative formats
            base_symbol = symbol.replace('=F', '')
            alternatives = alternative_formats.get(symbol, [])

            found_working = False
            for alt_symbol in alternatives:
                print(f"   🔄 Trying alternative: {alt_symbol}")
                alt_result = test_polygon_futures_symbol(alt_symbol)

                if alt_result['success']:
                    working_symbols[base_symbol] = alt_symbol
                    print(f"   ✅ ALTERNATIVE WORKS: {alt_symbol}")
                    found_working = True
                    break
                else:
                    print(f"   ❌ Failed: {alt_symbol}")

            if not found_working:
                print(f"   ❌ NO WORKING FORMAT FOUND for {base_symbol}")

        print()  # Empty line for readability

    # Summary
    print("📊 SUMMARY:")
    print("=" * 50)

    print("Working Symbol Mappings:")
    for base, working in working_symbols.items():
        print(f"  {base:3} -> {working}")

    passed = len(working_symbols)
    total = len(test_symbols)

    print(f"\nSuccess Rate: {passed}/{total} symbols working")

    if passed == total:
        print("🎉 ALL FUTURES SYMBOLS WORKING!")
        print("\nRecommended symbol mapping for your script:")
        print("symbols = [")
        for base in ['6A', '6B', '6C', '6E', '6S', 'ES', 'YM', 'NQ', 'RTY']:
            if base in working_symbols:
                print(f'    "{working_symbols[base]}",  # {base}')
        print("]")
    else:
        print("⚠️  Some symbols need investigation")

if __name__ == "__main__":
    import sys

    # If symbol provided as command line argument, test just that symbol
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        test_date = sys.argv[2] if len(sys.argv) > 2 else "2024-10-27"
        result = test_polygon_futures_symbol(symbol, test_date)

        if result['success']:
            print(f"\n📊 Data for {symbol}:")
            data = result['data']
            for key, value in data.items():
                print(f"   {key}: {value}")
        else:
            print(f"\n❌ Failed to get data for {symbol}: {result['error']}")
    else:
        main()