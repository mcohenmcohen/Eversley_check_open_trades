#!/usr/bin/env python3
"""
Test 6E Euro Futures Trade for 12/10/25
Verify entry and stop calculations
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from data_sources import DataSourceManager

def test_6e_trade():
    """Fetch and calculate 6E trade for 12/10/25"""

    symbol = '6E'
    signal_date_str = '2025-12-10'
    signal_date = datetime.strptime(signal_date_str, '%Y-%m-%d')

    print(f"\n{'='*80}")
    print(f"6E (Euro Futures) Trade Analysis")
    print(f"Signal Date: {signal_date_str}")
    print(f"{'='*80}\n")

    # Check for API keys
    api_key = os.environ.get('INSIGHTSENTRY_API_KEY')
    if not api_key:
        print("ERROR: INSIGHTSENTRY_API_KEY not found")
        return

    # Initialize data source manager
    polygon_api_key = os.environ.get('POLYGON_API_KEY')
    dsm = DataSourceManager(
        polygon_api_key=polygon_api_key,
        insightsentry_api_key=api_key
    )

    # Fetch data
    start_date = signal_date - timedelta(days=5)
    end_date = signal_date + timedelta(days=5)

    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")
    print()

    try:
        price_data = dsm.get_futures_data([symbol], start_date, end_date)
        df = price_data.get(symbol)

        if df is None or df.empty:
            print("ERROR: No data returned from InsightSentry API")
            return

        print(f"✅ Fetched {len(df)} days of data")
        print(f"Data range: {df.index.min().date()} to {df.index.max().date()}")
        print()

        # Display data around signal date
        mask = (df.index >= start_date) & (df.index <= end_date)
        recent_data = df[mask]

        print("OHLCV Data from InsightSentry:")
        print("-" * 80)
        print(recent_data[['Open', 'High', 'Low', 'Close', 'Volume']])
        print()

        # Check if signal date exists in data
        if signal_date not in df.index:
            print(f"⚠️ Signal date {signal_date_str} not in dataset")
            # Find closest date
            closest_dates = df.index[df.index >= signal_date][:2]
            if len(closest_dates) > 0:
                print(f"Closest dates available: {[d.date() for d in closest_dates]}")
            return

        # Signal day data
        signal_row = df.loc[signal_date]
        print(f"\nSignal Day ({signal_date.date()}):")
        print(f"  Open:   {signal_row['Open']:.5f}")
        print(f"  High:   {signal_row['High']:.5f}")
        print(f"  Low:    {signal_row['Low']:.5f}")
        print(f"  Close:  {signal_row['Close']:.5f}")
        print(f"  Volume: {signal_row['Volume']:.0f}")
        print()

        # For 6E, need to check what strategy is being used
        # Assuming similar to 6B: entry based on high + offset, stop based on low - offset

        # Check the signal file to see what strategy
        print("Checking trade signals file for strategy...")
        try:
            signals_df = pd.read_csv('trade_signals_futures.csv')
            signals_df.columns = signals_df.columns.str.strip()
            signals_df['date'] = pd.to_datetime(signals_df['date'])

            # Find the 6E signal for 12/10
            signal_match = signals_df[
                (signals_df['date'] == signal_date) &
                (signals_df['ticker'].str.strip().isin(['6E', 'EC']))  # EC is the symbol code for Euro
            ]

            if len(signal_match) > 0:
                strategy_name = signal_match.iloc[0]['strategy']
                direction = signal_match.iloc[0]['direction']
                print(f"Strategy: {strategy_name}")
                print(f"Direction: {direction}")
            else:
                print("⚠️ No matching signal found in trade_signals_futures.csv")
                strategy_name = "Unknown"
                direction = "Unknown"
        except Exception as e:
            print(f"Could not read signals file: {e}")
            strategy_name = "Unknown"
            direction = "Unknown"
        print()

        # Tick size for 6E (Euro) is 0.0001
        tick_size = 0.0001

        # ENTRY CALCULATION
        # Assuming high + offset (need to check strategy JSON for exact offset)
        # Common offsets are 15 ticks for currency futures
        print("ENTRY CALCULATION:")
        print("-" * 80)

        # Try different offset values to match user's 1.1718
        user_entry = 1.1718
        calculated_offset_ticks = round((user_entry - signal_row['High']) / tick_size)

        print(f"Signal day High: {signal_row['High']:.5f}")
        print(f"Your entry: {user_entry:.4f}")
        print(f"Implied offset: {calculated_offset_ticks} ticks")
        print()

        # Check if it's High + 15 ticks (common for currency futures)
        for offset in [10, 15, 20]:
            entry_calc = signal_row['High'] + (offset * tick_size)
            print(f"  High + {offset} ticks: {entry_calc:.5f}")
            if abs(entry_calc - user_entry) < 0.0001:
                print(f"  ✓ MATCHES your entry!")
        print()

        # STOP CALCULATION
        print("STOP CALCULATION:")
        print("-" * 80)

        # Common stop: Low of lookback period - offset ticks
        # Try 2-day lookback including signal day
        lookback_days = 2
        lookback_start = signal_date - timedelta(days=lookback_days)
        lookback_data = df[(df.index >= lookback_start) & (df.index <= signal_date)]

        print(f"Lookback period ({lookback_days} days including signal day):")
        print(f"  Period: {lookback_data.index.min().date()} to {lookback_data.index.max().date()}")
        print(f"  Lows in period:")
        for date, row in lookback_data.iterrows():
            print(f"    {date.date()}: {row['Low']:.5f}")
        print()

        lowest_low = lookback_data['Low'].min()
        user_stop = 1.1609
        calculated_stop_offset_ticks = round((lowest_low - user_stop) / tick_size)

        print(f"Lowest Low in lookback: {lowest_low:.5f}")
        print(f"Your stop: {user_stop:.4f}")
        print(f"Implied offset: {calculated_stop_offset_ticks} ticks")
        print()

        # Check common stop offsets
        for offset in [5, 10, 15, 20]:
            stop_calc = lowest_low - (offset * tick_size)
            print(f"  Lowest Low - {offset} ticks: {stop_calc:.5f}")
            if abs(stop_calc - user_stop) < 0.0001:
                print(f"  ✓ MATCHES your stop!")
        print()

        print("=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        print(f"Your values:")
        print(f"  Entry: {user_entry:.4f}")
        print(f"  Stop:  {user_stop:.4f}")
        print()
        print(f"InsightSentry data:")
        print(f"  Signal day High: {signal_row['High']:.5f}")
        print(f"  Lookback Low:    {lowest_low:.5f}")
        print("=" * 80)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_6e_trade()
