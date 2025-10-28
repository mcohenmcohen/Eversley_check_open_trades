#!/usr/bin/env python
"""Direct test of Polygon API for ES futures."""

import os
from polygon import RESTClient
from datetime import datetime

# Get API key
api_key = os.environ.get('POLYGON_API_KEY')
if not api_key:
    print("POLYGON_API_KEY not set")
    exit(1)

client = RESTClient(api_key=api_key)

# Futures month codes: H=Mar, M=Jun, U=Sep, Z=Dec
# Format: Product + Month + Year (single digit for year)
# ESZ5 = ES December 2025
tickers_to_try = [
    'ESZ4',         # December 2024 contract
    'ESZ5',         # December 2025 contract
    'ESU5',         # September 2025 contract
    'ESM5',         # June 2025 contract
]

print("Testing ES futures ticker formats on Polygon using list_aggs:")
print("=" * 80)

for ticker in tickers_to_try:
    print(f"\nTrying ticker: {ticker}")
    try:
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_="2025-10-20",
            to="2025-10-24",
            limit=10
        ):
            aggs.append(a)

        if aggs and len(aggs) > 0:
            print(f"  ✅ SUCCESS! Got {len(aggs)} bars")
            for agg in aggs:
                date = datetime.fromtimestamp(agg.timestamp / 1000)
                print(f"    {date.date()}: O={agg.open:.2f} H={agg.high:.2f} L={agg.low:.2f} C={agg.close:.2f}")
        else:
            print(f"  ❌ No data returned")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 80)
print("Expected for 2025-10-24: Open=6778.25, High=6841.25, Low=6776.5, Close=6825.25")
