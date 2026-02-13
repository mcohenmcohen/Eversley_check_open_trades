#!/usr/bin/env python3
"""
Test 6B TXD Trendlines Currency Futures Buy on 12/3/25
Verify entry, stop, and target calculations

Strategy:
- Entry: High + 15 ticks (for symbol B)
- Stop: Low - 10 ticks (lookback 2 days, include signal day)
- Target: Multi-target, rank 1 (closest)
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

# Load 6B data
data_file = Path("data/6B.csv")
df = pd.read_csv(data_file, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Signal date
signal_date = pd.to_datetime('2025-12-03')

print(f"\n{'='*80}")
print(f"6B TXD Trendlines Currency Futures Buy - 12/3/25")
print(f"{'='*80}\n")

# Show data around signal date
print("Price data around signal date:")
mask = (df.index >= signal_date - pd.Timedelta(days=5)) & (df.index <= signal_date + pd.Timedelta(days=5))
print(df[mask][['Open', 'High', 'Low', 'Close']])
print()

# Get signal day data
signal_day = df.loc[signal_date]
print(f"Signal day (12/3/25):")
print(f"  Open:  {signal_day['Open']:.5f}")
print(f"  High:  {signal_day['High']:.5f}")
print(f"  Low:   {signal_day['Low']:.5f}")
print(f"  Close: {signal_day['Close']:.5f}")
print()

# Entry calculation: High + 15 ticks
# For 6B, tick size is 0.0001
tick_size = 0.0001
offset_ticks = 15

# Find the trigger window (2 days)
trigger_window = df[(df.index >= signal_date) & (df.index <= signal_date + pd.Timedelta(days=2))]
print(f"Trigger window (2 days from signal):")
print(trigger_window[['Open', 'High', 'Low', 'Close']])
print()

# Entry is highest high in trigger window + 15 ticks
trigger_high = trigger_window['High'].max()
entry_price = trigger_high + (offset_ticks * tick_size)

print(f"ENTRY CALCULATION:")
print(f"  Highest high in trigger window: {trigger_high:.5f}")
print(f"  + 15 ticks (15 × {tick_size}): {offset_ticks * tick_size:.5f}")
print(f"  Entry Price: {entry_price:.5f}")
print()

# Stop calculation: Low - 10 ticks (lookback 2 days, include signal day)
# Lookback includes signal day + 2 days before
lookback_start = signal_date - pd.Timedelta(days=2)
lookback_data = df[(df.index >= lookback_start) & (df.index <= signal_date)]

print(f"STOP CALCULATION:")
print(f"Lookback period (signal day + 2 days before):")
print(lookback_data[['Open', 'High', 'Low', 'Close']])
print()

lookback_low = lookback_data['Low'].min()
stop_price = lookback_low - (10 * tick_size)

print(f"  Lowest low in lookback: {lookback_low:.5f}")
print(f"  - 10 ticks (10 × {tick_size}): {10 * tick_size:.5f}")
print(f"  Stop Price: {stop_price:.5f}")
print()

# Target calculation: Multi-target with ATR
# Calculate ATR(5)
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR5'] = df['TR'].ewm(alpha=1/5, adjust=False).mean()

atr_value = df.loc[signal_date, 'ATR5']

print(f"TARGET CALCULATION:")
print(f"  ATR(5) on signal date: {atr_value:.5f}")
print()

# Calculate all target options
entry_stop_diff = entry_price - stop_price
print(f"  Entry - Stop difference: {entry_stop_diff:.5f}")
print()

targets = []

# ATR percentage targets
for pct in [0.6, 0.7]:
    target_price = entry_price + (atr_value * pct)
    ticks = round((target_price - entry_price) / tick_size)
    targets.append({'type': f'ATR5 x {pct}', 'price': target_price, 'ticks': ticks})
    print(f"  ATR5 × {pct}: Entry + (ATR × {pct}) = {entry_price:.5f} + {atr_value * pct:.5f} = {target_price:.5f} ({ticks} ticks)")

# Entry-stop percentage targets
for pct in [0.4, 0.45, 0.5]:
    target_price = entry_price + (entry_stop_diff * pct)
    ticks = round((target_price - entry_price) / tick_size)
    targets.append({'type': f'Entry-Stop x {pct}', 'price': target_price, 'ticks': ticks})
    print(f"  Entry-Stop × {pct}: Entry + (E-S × {pct}) = {entry_price:.5f} + {entry_stop_diff * pct:.5f} = {target_price:.5f} ({ticks} ticks)")

print()

# Sort by ticks (rank 1 = fewest ticks)
targets.sort(key=lambda x: x['ticks'])
selected_target = targets[0]

print(f"SELECTED TARGET (Rank 1 - fewest ticks):")
print(f"  {selected_target['type']}: {selected_target['price']:.5f} ({selected_target['ticks']} ticks)")
print()

print(f"{'='*80}")
print(f"SUMMARY:")
print(f"{'='*80}")
print(f"Your calculations:")
print(f"  Entry:  {1.3369:.4f}")
print(f"  Target: {1.3428:.4f}")
print(f"  Stop:   {1.3169:.4f}")
print()
print(f"Trade evaluator results:")
print(f"  Entry:  {1.3373:.4f}")
print(f"  Target: {1.34216:.5f}")
print(f"  Stop:   {1.3246:.4f}")
print()
print(f"My calculations:")
print(f"  Entry:  {entry_price:.4f}")
print(f"  Target: {selected_target['price']:.5f}")
print(f"  Stop:   {stop_price:.4f}")
print(f"{'='*80}\n")
