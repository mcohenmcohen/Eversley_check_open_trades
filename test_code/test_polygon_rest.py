#!/usr/bin/env python
"""Test Polygon futures API using direct REST calls."""

import os
import requests
from datetime import datetime

# Get API key
api_key = os.environ.get('POLYGON_API_KEY')
if not api_key:
    print("POLYGON_API_KEY not set")
    exit(1)

print("Testing Polygon futures REST API:")
print("=" * 80)

# Try the aggregates endpoint for futures
# Based on Polygon docs: /v2/aggs/ticker/{futuresTicker}/range/{multiplier}/{timespan}/{from}/{to}

tickers = ['ESZ5', 'ESZ4', 'ESU5']

for ticker in tickers:
    print(f"\nTrying {ticker}:")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2025-10-20/2025-10-24"
    params = {'apiKey': api_key}

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {data}")

            if data.get('resultsCount', 0) > 0:
                print(f"  ✅ SUCCESS! Got {data['resultsCount']} bars")
                for bar in data.get('results', []):
                    ts = bar['t'] / 1000
                    date = datetime.fromtimestamp(ts)
                    print(f"    {date.date()}: O={bar['o']:.2f} H={bar['h']:.2f} L={bar['l']:.2f} C={bar['c']:.2f}")
            else:
                print(f"  ❌ No results")
        elif response.status_code == 403:
            print(f"  ❌ Forbidden - futures data may not be included in your subscription")
        elif response.status_code == 404:
            print(f"  ❌ Not found - ticker may not exist or wrong format")
        else:
            print(f"  ❌ Error: {response.text}")

    except Exception as e:
        print(f"  ❌ Exception: {e}")

print("\n" + "=" * 80)
print("Expected for 2025-10-24: Open=6778.25, High=6841.25, Low=6776.5, Close=6825.25")
