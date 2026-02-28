#!/usr/bin/env python3

import datetime
import concurrent.futures
import logging
from polygon import RESTClient
import signal
import sys
import pickle
import lz4.frame
import os

"""
Test script that:
1. Uses your exact bulk download code to create compressed files
2. Then decompresses and analyzes the data to see what was actually retrieved
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_aggs_for_symbol_and_date(symbol_date_pair):
    """Retrieve aggs for a given symbol and date (your exact code)"""
    symbol, date = symbol_date_pair
    aggs = []
    client = RESTClient(trace=True)  # Uses POLYGON_API_KEY environment variable

    for a in client.list_aggs(
        symbol,
        1,
        "minute",
        date,
        date,
        limit=50000,
    ):
        aggs.append(a)

    print(f"Downloaded {len(aggs)} aggs for {symbol} on {date}")

    filename = f"{symbol}-aggs-{date}.pickle.lz4"
    with open(filename, "wb") as file:
        try:
            compressed_data = lz4.frame.compress(pickle.dumps(aggs))
            file.write(compressed_data)
        except TypeError as e:
            print(f"Serialization Error: {e}")

    logging.info(f"Downloaded aggs for {symbol} on {date} and saved to {filename}")
    return filename, len(aggs)

def decompress_and_analyze_file(filename):
    """Decompress and analyze a single compressed file"""
    try:
        print(f"\n🔍 Analyzing {filename}")

        if not os.path.exists(filename):
            print(f"   ❌ File not found")
            return

        file_size = os.path.getsize(filename)
        print(f"   📁 File size: {file_size} bytes")

        # Read and decompress
        with open(filename, "rb") as file:
            compressed_data = file.read()

        try:
            decompressed_data = lz4.frame.decompress(compressed_data)
            aggs = pickle.loads(decompressed_data)

            print(f"   📊 Decompressed: {len(aggs)} bars")

            if len(aggs) > 0:
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

                return len(aggs)
            else:
                print(f"   ⚠️  File contains empty data")
                return 0

        except Exception as e:
            print(f"   ❌ Decompression error: {e}")
            return None

    except Exception as e:
        print(f"   ❌ File analysis error: {e}")
        return None

def weekdays_between(start_date, end_date):
    """Generate all weekdays between start_date and end_date"""
    day = start_date
    while day <= end_date:
        if day.weekday() < 5:  # 0-4 denotes Monday to Friday
            yield day
        day += datetime.timedelta(days=1)

def main():
    print("🚀 Testing Compressed Bulk Aggs Data")
    print("=" * 50)

    # Use a smaller date range for testing
    start_date = datetime.date(2024, 12, 16)
    end_date = datetime.date(2024, 12, 18)

    # Your exact symbols
    symbols = ["6A=F", "CME_MINI:NQU2025", "TSLA", "AAPL", "HCP", "GOOG"]

    dates = list(weekdays_between(start_date, end_date))
    print(f"Testing dates: {dates}")

    # Generate a list of (symbol, date) pairs
    symbol_date_pairs = [(symbol, date) for symbol in symbols for date in dates]

    print(f"\nDownloading data for {len(symbol_date_pairs)} symbol-date combinations...")

    # Use ThreadPoolExecutor to download data in parallel (reduced workers for testing)
    created_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(get_aggs_for_symbol_and_date, symbol_date_pairs))
        created_files.extend([result[0] for result in results])

    print(f"\n{'='*50}")
    print("DOWNLOAD COMPLETE - Now analyzing compressed files...")
    print(f"{'='*50}")

    # Analyze each created file
    analysis_results = {}
    for symbol in symbols:
        analysis_results[symbol] = []

        for date in dates:
            filename = f"{symbol}-aggs-{date}.pickle.lz4"
            bar_count = decompress_and_analyze_file(filename)
            analysis_results[symbol].append((date, bar_count))

    # Summary
    print(f"\n{'='*20} FINAL SUMMARY {'='*20}")
    for symbol, results in analysis_results.items():
        print(f"\n📊 {symbol}:")
        total_bars = 0
        successful_days = 0

        for date, count in results:
            if count is not None and count > 0:
                status = "✅"
                successful_days += 1
                total_bars += count
            elif count == 0:
                status = "⚠️  (empty)"
            else:
                status = "❌"

            print(f"   {date}: {status} {count if count is not None else 'error'} bars")

        print(f"   Total: {successful_days}/{len(results)} days successful, {total_bars} total bars")

        if successful_days > 0:
            print(f"   ✅ {symbol} is working with your bulk download method!")
        else:
            print(f"   ❌ {symbol} is not returning data")

    # Cleanup option
    print(f"\n🧹 Created {len(created_files)} test files")
    cleanup = input("Delete test files? (y/n): ").lower().strip()
    if cleanup == 'y':
        for filename in created_files:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    print(f"   Deleted {filename}")
            except Exception as e:
                print(f"   Error deleting {filename}: {e}")

if __name__ == "__main__":
    main()