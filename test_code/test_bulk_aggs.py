#!/usr/bin/env python3

import datetime
from polygon import RESTClient
import os

"""
Test version of your bulk aggs script to verify data is actually being retrieved
for NQ and 6A symbols before compression.
"""

def test_aggs_for_symbol_and_date(symbol, date):
    """Test retrieving aggs for a given symbol and date (without compression)"""
    print(f"\n🔍 Testing {symbol} for {date}")

    aggs = []
    client = RESTClient()  # Uses POLYGON_API_KEY environment variable

    try:
        print(f"   Making API request...")
        agg_iterator = client.list_aggs(
            symbol,
            1,
            "minute",
            date,
            date,
            limit=50000,
        )

        # Convert iterator to list to see actual data
        for a in agg_iterator:
            aggs.append(a)

        print(f"   📊 Retrieved {len(aggs)} minute bars")

        if len(aggs) > 0:
            # Show first few bars to verify data quality
            print(f"   📈 Sample data (first 3 bars):")
            for i, agg in enumerate(aggs[:3]):
                # Show the attributes of the agg object
                timestamp = getattr(agg, 'timestamp', 'N/A')
                open_price = getattr(agg, 'open', 'N/A')
                high_price = getattr(agg, 'high', 'N/A')
                low_price = getattr(agg, 'low', 'N/A')
                close_price = getattr(agg, 'close', 'N/A')
                volume = getattr(agg, 'volume', 'N/A')

                # Convert timestamp if it's a number
                if isinstance(timestamp, (int, float)):
                    time_str = datetime.datetime.fromtimestamp(timestamp/1000).strftime('%H:%M')
                else:
                    time_str = str(timestamp)

                print(f"     Bar {i+1}: {time_str} O:{open_price} H:{high_price} L:{low_price} C:{close_price} V:{volume}")

            if len(aggs) > 3:
                print(f"     ... and {len(aggs)-3} more bars")

            # Show last bar too
            if len(aggs) > 1:
                last_agg = aggs[-1]
                timestamp = getattr(last_agg, 'timestamp', 'N/A')
                if isinstance(timestamp, (int, float)):
                    time_str = datetime.datetime.fromtimestamp(timestamp/1000).strftime('%H:%M')
                else:
                    time_str = str(timestamp)
                print(f"     Last bar: {time_str} O:{getattr(last_agg, 'open', 'N/A')} H:{getattr(last_agg, 'high', 'N/A')} L:{getattr(last_agg, 'low', 'N/A')} C:{getattr(last_agg, 'close', 'N/A')} V:{getattr(last_agg, 'volume', 'N/A')}")

            return True, len(aggs)
        else:
            print(f"   ❌ No data returned")
            return False, 0

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0

def main():
    """Test the symbols from your original script"""
    print("🚀 Testing Bulk Aggs Data Retrieval")
    print("=" * 50)

    # Check API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("❌ ERROR: POLYGON_API_KEY environment variable not set")
        return

    print(f"✅ Using API key: {api_key[:10]}...")

    # Test the symbols you mentioned work: NQ and 6A
    test_symbols = ["6A=F", "NQ=F"]

    # Test a few recent weekdays
    test_dates = [
        datetime.date(2024, 12, 16),  # Monday
        datetime.date(2024, 12, 17),  # Tuesday
        datetime.date(2024, 12, 18),  # Wednesday
    ]

    results = {}

    for symbol in test_symbols:
        print(f"\n{'='*20} TESTING {symbol} {'='*20}")
        results[symbol] = []

        for date in test_dates:
            # Skip weekends
            if date.weekday() < 5:  # 0-4 = Monday-Friday
                success, count = test_aggs_for_symbol_and_date(symbol, date)
                results[symbol].append((date, success, count))

    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    for symbol, symbol_results in results.items():
        print(f"\n📊 {symbol}:")
        total_bars = 0
        successful_days = 0

        for date, success, count in symbol_results:
            status = "✅" if success else "❌"
            print(f"   {date}: {status} {count} bars")
            if success:
                successful_days += 1
                total_bars += count

        print(f"   Total: {successful_days}/{len(symbol_results)} days successful, {total_bars} total bars")

        if successful_days > 0:
            print(f"   ✅ {symbol} appears to be working with Polygon!")
        else:
            print(f"   ❌ {symbol} is not returning data from Polygon")

if __name__ == "__main__":
    main()