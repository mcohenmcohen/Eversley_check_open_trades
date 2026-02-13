#!/usr/bin/env python3
"""
Test to exactly replicate the trade evaluator's ATR calculation logic
"""

import pandas as pd
from datetime import datetime, timedelta
from data_sources import DataSourceManager, fetch_price_data
import os

def wilders_atr(df, period):
    """Exact copy of trade evaluator's wilders_atr function"""
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()

def test_exact_evaluator_logic(symbol='XLE', signal_date_str='2025-11-21', expected_atr='1.7736'):
    """Replicate exact trade evaluator logic"""

    signal_date = pd.to_datetime(signal_date_str).normalize()

    print(f"\n{'='*80}")
    print(f"EXACT TRADE EVALUATOR LOGIC REPLICATION")
    print(f"Symbol: {symbol}")
    print(f"Signal Date: {signal_date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}\n")

    # Initialize data source manager (matching trade evaluator)
    polygon_api_key = os.environ.get('POLYGON_API_KEY')
    dsm = DataSourceManager(polygon_api_key=polygon_api_key)

    # Fetch data with EXACT same date range as trade evaluator
    start_date = datetime(2025, 1, 2)  # ETFs start from Jan 2, 2025
    end_date = datetime.today() + timedelta(days=1)  # Same as trade evaluator

    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")
    print(f"(This matches the trade evaluator's date range)\n")

    price_data = fetch_price_data([symbol], start_date, end_date, dsm)
    df = price_data.get(symbol)

    if df is None or df.empty:
        print("ERROR: Could not fetch data")
        return

    print(f"Fetched {len(df)} days of data")
    print(f"Data range: {df.index.min().date()} to {df.index.max().date()}\n")

    # EXACT TRADE EVALUATOR CODE for weekly ATR (from lines 1183-1190):
    atr_length = 5
    multiplier = 0.55

    print("Resampling to weekly data (W-FRI)...")
    df_weekly = df.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    print(f"Weekly dataframe has {len(df_weekly)} weeks")
    print(f"Weekly range: {df_weekly.index.min().date()} to {df_weekly.index.max().date()}\n")

    print("Last 5 weekly bars:")
    print(df_weekly[['Open', 'High', 'Low', 'Close']].tail(5))
    print()

    print("Calculating Wilder's ATR on weekly data...")
    atr_series = wilders_atr(df_weekly, atr_length)

    print(f"\nLast 5 weekly ATR values:")
    print(atr_series.tail(5))
    print()

    # THIS IS THE KEY LINE (line 1189 in trade evaluator):
    atr_raw = atr_series.iloc[-1] if len(atr_series) > 0 else 0

    print(f"*** TRADE EVALUATOR USES .iloc[-1] ***")
    print(f"ATR value from .iloc[-1]: ${atr_raw:.4f}")
    print(f"This is the ATR for week ending: {df_weekly.index[-1].date()}")
    print()

    # Now check what the ATR was for the signal date's week
    weekly_up_to_signal = df_weekly[df_weekly.index <= signal_date]

    if len(weekly_up_to_signal) > 0:
        atr_at_signal = atr_series[atr_series.index <= signal_date].iloc[-1]
        print(f"ATR for week of signal date ({signal_date.strftime('%m/%d')}): ${atr_at_signal:.4f}")
        print(f"This is week ending: {weekly_up_to_signal.index[-1].date()}")
    print()

    atr_target = round(atr_raw * multiplier, 4) if atr_raw else ""

    print(f"={'='*80}")
    print(f"RESULTS:")
    print(f"={'='*80}")
    print(f"Evaluator's ATR (using .iloc[-1]):    ${atr_raw:.4f}")
    print(f"Evaluator's Target (ATR * 0.55):      ${atr_target:.4f}")
    print(f"")
    print(f"Trade Results File shows:")
    print(f"  ATR:    ${expected_atr}")
    print(f"={'='*80}\n")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("TEST 1: XLE on 11/21/25")
    print("="*80)
    test_exact_evaluator_logic(symbol='XLE', signal_date_str='2025-11-21', expected_atr='1.7736')

    print("\n" + "="*80)
    print("TEST 2: GLD on 11/28/25")
    print("="*80)
    test_exact_evaluator_logic(symbol='GLD', signal_date_str='2025-11-28', expected_atr='7.718')
