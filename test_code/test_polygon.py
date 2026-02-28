#!/usr/bin/env python3

from polygon.rest import RESTClient

POLYGON_API_KEY = 'qFTWmhmyAj2pJqpx0Pwxp2AyShKFVPv9'
polygon_client = RESTClient(api_key=POLYGON_API_KEY)

def test_polygon_symbol(symbol, test_date="2024-10-27"):
    """
    Test a Polygon symbol and return results or error
    
    Args:
        symbol (str): The symbol to test
        test_date (str): Date to test in YYYY-MM-DD format
    
    Returns:
        dict: Results containing success status, data, or error message
    """
    try:
        print(f"📡 Testing symbol: {symbol} for date: {test_date}")
        
        bars = polygon_client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan='day',
            from_=test_date,
            to=test_date,
            adjusted=True,
            sort='asc',
            limit=50
        )

        print("bars:", bars)
        
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
            print(f"✅ SUCCESS: O:{bar.open:.4f} H:{bar.high:.4f} L:{bar.low:.4f} C:{bar.close:.4f} V:{bar.volume}")
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
    Main function to test polygon symbols
    
    Expected execution string:
    python test_polygon.py                    # Test default symbols
    python test_polygon.py SYMBOL            # Test specific symbol with default date
    python test_polygon.py SYMBOL DATE       # Test specific symbol with specific date (YYYY-MM-DD)
    """
    print("🚀 Polygon Symbol Tester")
    print("=" * 40)
    
    # Test some default symbols
    test_symbols = [
        'SPY',      # ETF
        'ES',       # S&P 500 E-mini futures
        '6A',       # Australian Dollar futures
        'C:AUDUSD', # Currency pair
        'AAPL',     # Stock
        'I:NDX'     # Index
    ]
    
    results = []
    
    for symbol in test_symbols:
        result = test_polygon_symbol(symbol)
        results.append(result)
        print()  # Empty line for readability
    
    # Summary
    print("📊 SUMMARY:")
    print("=" * 40)
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{result['symbol']:10} {status}")
        if not result['success']:
            print(f"           Error: {result['error']}")
    
    print(f"\nTotal: {passed}/{total} symbols successful")

if __name__ == "__main__":
    import sys
    
    # If symbol provided as command line argument, test just that symbol
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        test_date = sys.argv[2] if len(sys.argv) > 2 else "2024-10-27"
        result = test_polygon_symbol(symbol, test_date)
        
        if result['success']:
            print(f"\n📊 Data for {symbol}:")
            data = result['data']
            for key, value in data.items():
                print(f"   {key}: {value}")
        else:
            print(f"\n❌ Failed to get data for {symbol}: {result['error']}")
    else:
        main()