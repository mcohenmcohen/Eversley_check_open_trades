#!/usr/bin/env python3
"""
Test script to fetch real-time OHLCV data from InsightSentry for 6B
Signal Date: 12/3/25
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
from data_sources import DataSourceManager

def test_insightsentry_6b():
    """Fetch and display OHLCV data for 6B from InsightSentry API"""

    symbol = '6B'
    signal_date_str = '2025-12-03'

    print(f"\n{'='*80}")
    print(f"INSIGHTSENTRY API TEST - 6B (British Pound)")
    print(f"Signal Date: {signal_date_str}")
    print(f"{'='*80}\n")

    # Check for InsightSentry API key
    api_key = os.environ.get('INSIGHTSENTRY_API_KEY')
    if not api_key:
        print("ERROR: INSIGHTSENTRY_API_KEY not found in environment variables")
        print("Please set it in your shell configuration (.zshrc or .bashrc)")
        return

    print(f"InsightSentry API Key found: {api_key[:10]}...")
    print()

    # Initialize data source manager
    polygon_api_key = os.environ.get('POLYGON_API_KEY')
    dsm = DataSourceManager(
        polygon_api_key=polygon_api_key,
        insightsentry_api_key=api_key
    )

    # Fetch data from a few days before signal to a few days after
    signal_date = datetime.strptime(signal_date_str, '%Y-%m-%d')
    start_date = signal_date - timedelta(days=5)
    end_date = signal_date + timedelta(days=5)

    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")
    print(f"Using InsightSentry API for futures data")
    print()

    try:
        # For futures, use get_futures_data directly instead of fetch_price_data
        # fetch_price_data routes to ETF data only
        price_data = dsm.get_futures_data([symbol], start_date, end_date)
        df = price_data.get(symbol)

        if df is None or df.empty:
            print("ERROR: No data returned from InsightSentry API")
            print("This could mean:")
            print("  1. Invalid API key")
            print("  2. Symbol not found (6B)")
            print("  3. Date range has no data")
            print("  4. API connectivity issue")
            return

        print(f"SUCCESS: Fetched {len(df)} days of data")
        print(f"Data range: {df.index.min().date()} to {df.index.max().date()}")
        print()

        # Display all OHLCV data
        print("OHLCV Data from InsightSentry:")
        print("-" * 80)
        print(df[['Open', 'High', 'Low', 'Close', 'Volume']])
        print()

        # Highlight key dates for the strategy
        print("KEY DATES FOR STRATEGY CALCULATION:")
        print("=" * 80)

        # Signal day (12/3)
        if signal_date in df.index:
            signal_row = df.loc[signal_date]
            print(f"\nSignal Day (12/3/25):")
            print(f"  Open:   {signal_row['Open']:.5f}")
            print(f"  High:   {signal_row['High']:.5f}")
            print(f"  Low:    {signal_row['Low']:.5f}")
            print(f"  Close:  {signal_row['Close']:.5f}")
            print(f"  Volume: {signal_row['Volume']:.0f}")

            # Entry calculation: High + 15 ticks (tick size = 0.0001)
            tick_size = 0.0001
            entry_high = signal_row['High']
            entry_price = entry_high + (15 * tick_size)
            print(f"\n  Entry Calculation:")
            print(f"    High: {entry_high:.5f}")
            print(f"    + 15 ticks (15 × {tick_size}): {15 * tick_size:.5f}")
            print(f"    Entry Threshold: {entry_price:.5f}")
        else:
            print(f"\nWARNING: Signal day {signal_date_str} not in dataset")

        # Previous day for stop calculation (12/2)
        prev_day = signal_date - timedelta(days=1)
        if prev_day in df.index:
            prev_row = df.loc[prev_day]
            print(f"\nPrevious Day (12/2/25) - for stop calculation:")
            print(f"  Low:    {prev_row['Low']:.5f}")

        # Stop calculation: min(Low[12/2], Low[12/3]) - 10 ticks
        lookback_start = signal_date - timedelta(days=2)
        lookback_data = df[(df.index >= lookback_start) & (df.index <= signal_date)]

        if len(lookback_data) > 0:
            print(f"\nStop Calculation (lookback 2 days including signal day):")
            print(f"  Lookback period: {lookback_data.index.min().date()} to {lookback_data.index.max().date()}")
            print(f"  Lows in period:")
            for date, row in lookback_data.iterrows():
                print(f"    {date.date()}: {row['Low']:.5f}")

            lowest_low = lookback_data['Low'].min()
            stop_price = lowest_low - (10 * tick_size)
            print(f"\n  Lowest Low: {lowest_low:.5f}")
            print(f"  - 10 ticks (10 × {tick_size}): {10 * tick_size:.5f}")
            print(f"  Stop Price: {stop_price:.5f}")

        print("\n" + "=" * 80)
        print("COMPARISON WITH USER'S VALUES:")
        print("=" * 80)
        print(f"User's High (12/3):  1.3354")
        if signal_date in df.index:
            print(f"InsightSentry High:  {df.loc[signal_date]['High']:.5f}")
            print(f"Difference:          {df.loc[signal_date]['High'] - 1.3354:.5f}")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"ERROR: Exception occurred while fetching data")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_insightsentry_6b()
