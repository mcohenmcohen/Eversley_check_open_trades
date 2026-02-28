#!/usr/bin/env python3

import requests
import os
from datetime import datetime, timedelta

# Get API key from environment variable
INSIGHTSENTRY_API_KEY = os.getenv('INSIGHTSENTRY_API_KEY')

def test_insightsentry_symbol(symbol, test_date="2024-10-27"):
    """
    Test an InsightSentry symbol and return results or error

    Args:
        symbol (str): The symbol to test (e.g., '6C', '6E', 'ES', etc.)
        test_date (str): Date to test in YYYY-MM-DD format

    Returns:
        dict: Results containing success status, data, or error message
    """
    try:
        print(f"📡 Testing symbol: {symbol} for date: {test_date}")

        if not INSIGHTSENTRY_API_KEY:
            raise ValueError("INSIGHTSENTRY_API_KEY environment variable not set")

        # Map symbols to InsightSentry format
        symbol_map = {
            '6C': 'CME:6C1!',
            '6A': 'CME:6A1!',
            '6B': 'CME:6B1!',
            '6E': 'CME:6E1!',
            '6S': 'CME:6S1!',
            'ES': 'CME_MINI:ES1!',
            'YM': 'CBOT_MINI:YM1!',
            'NQ': 'CME_MINI:NQ1!',
            'RTY': 'CME_MINI:RTY1!'
        }

        mapped_symbol = symbol_map.get(symbol, symbol)
        print(f"   Mapped to: {mapped_symbol}")

        # Calculate date range (get a few days of data around test date)
        test_dt = datetime.strptime(test_date, '%Y-%m-%d')
        start_date = test_dt - timedelta(days=5)
        end_date = test_dt + timedelta(days=1)

        # Build API request using correct InsightSentry format
        url = f"https://api.insightsentry.com/v2/symbols/{mapped_symbol}/history"
        params = {
            'bar_type': 'day',
            'bar_interval': 1,
            'dp': 10,  # Get 10 data points around the test date
            'extended': False,
            'dadj': True,   # dividend adjustment for stocks
            'badj': False,  # back adjustment
            'long_poll': False
        }

        headers = {
            'Authorization': f'Bearer {INSIGHTSENTRY_API_KEY}',
            'Content-Type': 'application/json'
        }

        print(f"   Request: {url}")
        print(f"   Params: {params}")

        response = requests.get(url, params=params, headers=headers, timeout=30)

        print(f"   Status Code: {response.status_code}")

        if response.status_code != 200:
            raise ValueError(f"API request failed with status {response.status_code}: {response.text}")

        data = response.json()
        print(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

        # Check if we have series data (InsightSentry format)
        if 'series' not in data or not data['series']:
            raise ValueError("No series data returned from InsightSentry API")

        series = data['series']
        print(f"   Retrieved {len(series)} data points")

        if not series:
            raise ValueError("Empty series data returned")

        # Use the most recent bar (series is usually in chronological order)
        target_bar = series[-1]
        print(f"   Using most recent bar from series")
        print(f"   Raw bar data: {target_bar}")
        print(f"   Bar length: {len(target_bar)}")

        # Extract OHLCV data - InsightSentry format: [timestamp, open, high, low, close, volume]
        if len(target_bar) >= 5:
            ohlc_data = {
                'timestamp': target_bar[0],
                'open': target_bar[1],
                'high': target_bar[2],
                'low': target_bar[3],
                'close': target_bar[4],
                'volume': target_bar[5] if len(target_bar) > 5 else 0
            }
        else:
            raise ValueError(f"Invalid bar format: expected at least 5 elements, got {len(target_bar)}")

        result = {
            'success': True,
            'symbol': symbol,
            'mapped_symbol': mapped_symbol,
            'date': test_date,
            'data': ohlc_data,
            'raw_bar': target_bar
        }

        print(f"✅ SUCCESS: O:{ohlc_data['open']:.4f} H:{ohlc_data['high']:.4f} L:{ohlc_data['low']:.4f} C:{ohlc_data['close']:.4f} V:{ohlc_data['volume']}")
        return result

    except Exception as e:
        import traceback
        error_msg = f"{str(e)} | {traceback.format_exc()}"
        result = {
            'success': False,
            'symbol': symbol,
            'mapped_symbol': symbol_map.get(symbol, symbol),
            'date': test_date,
            'error': error_msg
        }
        print(f"❌ ERROR: {error_msg}")
        return result

def main():
    """
    Main function to test InsightSentry symbols

    Expected execution string:
    python test_insightsentry.py                    # Test default symbols
    python test_insightsentry.py SYMBOL            # Test specific symbol with default date
    python test_insightsentry.py SYMBOL DATE       # Test specific symbol with specific date (YYYY-MM-DD)
    """
    print("🚀 InsightSentry Symbol Tester")
    print("=" * 40)

    if not INSIGHTSENTRY_API_KEY:
        print("❌ ERROR: INSIGHTSENTRY_API_KEY environment variable not set")
        print("Please set it with: export INSIGHTSENTRY_API_KEY=your_api_key_here")
        return

    # Test some default symbols
    test_symbols = [
        '6C',       # Canadian Dollar futures
        '6A',       # Australian Dollar futures
        '6E',       # Euro futures
        '6B',       # British Pound futures
        '6S',       # Swiss Franc futures
        'ES',       # S&P 500 E-mini futures
        'YM',       # Dow Jones E-mini futures
        'NQ',       # Nasdaq E-mini futures
        'RTY'       # Russell 2000 E-mini futures
    ]

    results = []

    for symbol in test_symbols:
        result = test_insightsentry_symbol(symbol)
        results.append(result)
        print()  # Empty line for readability

    # Summary
    print("📊 SUMMARY:")
    print("=" * 40)
    passed = sum(1 for r in results if r['success'])
    total = len(results)

    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        mapped = result.get('mapped_symbol', 'N/A')
        print(f"{result['symbol']:3} -> {mapped:15} {status}")
        if not result['success']:
            print(f"           Error: {result['error']}")

    print(f"\nTotal: {passed}/{total} symbols successful")

if __name__ == "__main__":
    import sys

    # If symbol provided as command line argument, test just that symbol
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        test_date = sys.argv[2] if len(sys.argv) > 2 else "2024-10-27"
        result = test_insightsentry_symbol(symbol, test_date)

        if result['success']:
            print(f"\n📊 Data for {symbol} ({result['mapped_symbol']}):")
            data = result['data']
            for key, value in data.items():
                if key != 'raw_bar':  # Skip the raw bar data in summary
                    print(f"   {key}: {value}")
        else:
            print(f"\n❌ Failed to get data for {symbol}: {result['error']}")
    else:
        main()