#!/usr/bin/env python3

from polygon import RESTClient
import os

# Get API key from environment variable
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

def search_polygon_symbols(search_term, asset_class=None):
    """Search for symbols on Polygon"""
    try:
        if not POLYGON_API_KEY:
            raise ValueError("POLYGON_API_KEY environment variable not set")

        client = RESTClient(api_key=POLYGON_API_KEY)

        print(f"🔍 Searching for: '{search_term}'")
        if asset_class:
            print(f"   Asset class filter: {asset_class}")

        # Search for tickers
        params = {
            'search': search_term,
            'limit': 20
        }

        if asset_class:
            params['type'] = asset_class

        try:
            # Use the reference tickers endpoint
            tickers = client.list_tickers(**params)

            print(f"   Found {len(list(tickers))} results:")

            # Reset the iterator and display results
            tickers = client.list_tickers(**params)
            for ticker in tickers:
                print(f"     {ticker.ticker} - {getattr(ticker, 'name', 'N/A')} - Type: {getattr(ticker, 'type', 'N/A')}")

        except Exception as e:
            print(f"   Error with search: {e}")

    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_known_futures_formats():
    """Test some known futures contract formats"""
    test_symbols = [
        # Try different date codes and formats
        "6AH24",  # March 2024
        "6AH25",  # March 2025
        "6AM24",  # June 2024
        "6AU24",  # September 2024
        "6AZ24",  # December 2024
        "6AZ25",  # December 2025
        "ESH24",  # S&P March 2024
        "ESM24",  # S&P June 2024
        "ESU24",  # S&P September 2024
        "ESZ24",  # S&P December 2024
        "ESH25",  # S&P March 2025
        # Continuous contract formats
        "@6A",
        "@ES",
        "/6A",
        "/ES",
        "6A.FUT",
        "ES.FUT"
    ]

    client = RESTClient(api_key=POLYGON_API_KEY)

    print("🧪 Testing known futures formats...")
    working_symbols = []

    for symbol in test_symbols:
        try:
            print(f"   Testing: {symbol}", end=" ")
            bars = client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan='day',
                from_='2024-10-01',
                to='2024-10-01',
                limit=1
            )

            if bars and len(list(bars)) > 0:
                print("✅ WORKS!")
                working_symbols.append(symbol)
            else:
                print("❌ No data")

        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}...")

    if working_symbols:
        print(f"\n✅ Working symbols found: {working_symbols}")
    else:
        print("\n❌ No working futures symbols found")

def main():
    """Main function to search for futures symbols"""
    print("🚀 Polygon Symbol Search for Futures")
    print("=" * 50)

    if not POLYGON_API_KEY:
        print("❌ ERROR: POLYGON_API_KEY environment variable not set")
        return

    # Search for different futures-related terms
    search_terms = [
        ("6A", None),           # Australian Dollar
        ("6C", None),           # Canadian Dollar
        ("6E", None),           # Euro
        ("ES", None),           # S&P 500
        ("NQ", None),           # Nasdaq
        ("YM", None),           # Dow Jones
        ("Australian Dollar", "FX"),
        ("Euro", "FX"),
        ("S&P 500", "FX"),
        ("futures", None)
    ]

    for term, asset_class in search_terms:
        search_polygon_symbols(term, asset_class)
        print()

    # Test specific formats
    test_known_futures_formats()

if __name__ == "__main__":
    main()