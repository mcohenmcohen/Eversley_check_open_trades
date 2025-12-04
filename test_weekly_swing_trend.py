#!/usr/bin/env python3
"""
Test script to verify calculations for Weekly Swing Trend Buy strategy
Symbol: XLE
Signal Date: 11/28/25
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_sources import DataSourceManager, fetch_price_data
import os

def calculate_atr(df, period=5, timeframe='weekly'):
    """Calculate ATR for the given timeframe"""
    if timeframe == 'weekly':
        # Resample to weekly data
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }
        if 'Volume' in df.columns:
            agg_dict['Volume'] = 'sum'

        weekly = df.resample('W-FRI').agg(agg_dict).dropna()
        df_calc = weekly
    else:
        df_calc = df

    # Calculate True Range
    tr = pd.concat([
        df_calc['High'] - df_calc['Low'],
        abs(df_calc['High'] - df_calc['Close'].shift(1)),
        abs(df_calc['Low'] - df_calc['Close'].shift(1))
    ], axis=1).max(axis=1)

    # Calculate Wilder's ATR (exponential weighted moving average)
    df_calc['ATR'] = tr.ewm(alpha=1/period, adjust=False).mean()

    return df_calc

def test_weekly_swing_trend_buy():
    """Test the Weekly Swing Trend Buy strategy for XLE on 11/21/25"""

    symbol = 'XLE'
    signal_date = '2025-11-21'

    print(f"\n{'='*80}")
    print(f"Testing Weekly Swing Trend Buy Strategy")
    print(f"Symbol: {symbol}")
    print(f"Signal Date: {signal_date}")
    print(f"{'='*80}\n")

    # Initialize data source manager
    polygon_api_key = os.environ.get('POLYGON_API_KEY')
    dsm = DataSourceManager(polygon_api_key=polygon_api_key)

    # Fetch data from well before signal date to after
    start_date = datetime(2025, 10, 1)
    end_date = datetime(2025, 12, 5)

    print(f"Fetching price data from {start_date.date()} to {end_date.date()}...")
    price_data = fetch_price_data([symbol], start_date, end_date, dsm)
    df = price_data.get(symbol)

    if df is None or df.empty:
        print("ERROR: Could not fetch data")
        return

    print(f"Data fetched successfully: {len(df)} days of data\n")

    # Show daily data around signal date
    signal_dt = pd.to_datetime(signal_date)
    mask = (df.index >= signal_dt - timedelta(days=5)) & (df.index <= signal_dt + timedelta(days=10))
    daily_around_signal = df[mask]

    print("Daily OHLC around signal date:")
    print(daily_around_signal[['Open', 'High', 'Low', 'Close']])
    print()

    # Calculate both daily and weekly ATR for comparison
    print("Calculating Daily ATR(5)...")
    daily_df = calculate_atr(df, period=5, timeframe='daily')

    print("Calculating Weekly ATR(5)...")
    weekly_df = calculate_atr(df, period=5, timeframe='weekly')

    # Find the weekly bar containing the signal date
    signal_week_end = signal_dt + timedelta(days=(4 - signal_dt.weekday()))  # Friday

    print("\nWeekly data (last 10 weeks):")
    print(weekly_df[['Open', 'High', 'Low', 'Close', 'ATR']].tail(10))
    print()

    # Get ATR as of signal date
    # We need the ATR calculated up to the week containing the signal
    weekly_up_to_signal = weekly_df[weekly_df.index <= signal_week_end]

    if len(weekly_up_to_signal) == 0:
        print("ERROR: No weekly data available")
        return

    atr_value = weekly_up_to_signal['ATR'].iloc[-1]

    # Also get daily ATR for comparison
    daily_up_to_signal = daily_df[daily_df.index <= signal_dt]
    daily_atr_value = daily_up_to_signal['ATR'].iloc[-1] if len(daily_up_to_signal) > 0 else None

    print(f"\n*** ATR COMPARISON ***")
    print(f"Daily ATR(5) as of signal date:  ${daily_atr_value:.4f}")
    print(f"Weekly ATR(5) as of week containing signal: ${atr_value:.4f}")

    # Check previous week's ATR
    if len(weekly_up_to_signal) >= 2:
        prev_week_atr = weekly_up_to_signal['ATR'].iloc[-2]
        print(f"Weekly ATR(5) of PREVIOUS week:  ${prev_week_atr:.4f}")

    # For comparison with trade results
    if symbol == 'XLE':
        print(f"Trade Results File shows:         $1.7736")
    elif symbol == 'GLD':
        print(f"Trade Results File shows:         $7.718")
    print()

    # Strategy configuration: Weekly Swing Trend Buy
    # Entry: "open_or_better" (buy direction) = enter at open or lower
    # Target: ATR(5) * 0.55 (weekly timeframe)

    # Find entry day (first trading day after signal date)
    entry_candidates = df[df.index > signal_dt].head(5)

    if len(entry_candidates) == 0:
        print("ERROR: No trading days after signal date")
        return

    entry_date = entry_candidates.index[0]
    entry_open = entry_candidates['Open'].iloc[0]

    print(f"Strategy: Weekly Swing Trend Buy")
    print(f"Entry Formula: 'open_or_better' (direction: buy)")
    print(f"  -> Entry at open or lower on first trading day after signal")
    print(f"  -> Entry Date: {entry_date.strftime('%Y-%m-%d')}")
    print(f"  -> Entry Price: ${entry_open:.2f} (open price)")
    print()

    # Calculate target
    target_multiplier = 0.55
    target_offset = atr_value * target_multiplier
    target_price = entry_open + target_offset

    print(f"Target Formula: ATR(5) * {target_multiplier} (weekly timeframe)")
    print(f"  -> ATR(5) = ${atr_value:.4f}")
    print(f"  -> Target Offset = ${atr_value:.4f} * {target_multiplier} = ${target_offset:.4f}")
    print(f"  -> Target Price = ${entry_open:.2f} + ${target_offset:.4f} = ${target_price:.2f}")
    print()

    # Stop loss
    print(f"Stop Loss: null (no stop loss for this strategy)")
    print()

    # Summary
    print(f"\n{'='*80}")
    print(f"CALCULATION SUMMARY")
    print(f"{'='*80}")
    print(f"Signal Date:      {signal_date}")
    print(f"Symbol:           {symbol}")
    print(f"Strategy:         Weekly Swing Trend Buy")
    print(f"")
    print(f"Weekly ATR(5):    ${atr_value:.4f}")
    print(f"Entry Date:       {entry_date.strftime('%Y-%m-%d')}")
    print(f"Entry Price:      ${entry_open:.2f}")
    print(f"Target Price:     ${target_price:.2f}")
    print(f"Target Formula:   Entry + (Weekly ATR5 * 0.55)")
    print(f"Stop Loss:        None")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    test_weekly_swing_trend_buy()
