#!/usr/bin/env python
"""
Test script to fetch futures data from InsightSentry API.
Usage: python test_insightsentry_fetch.py --symbol ES --start 2025-10-01 --end 2025-10-25
"""

import argparse
import os
from datetime import datetime
import pandas as pd
from data_sources import DataSourceManager

def fetch_futures_data(symbol, start_date, end_date):
    """
    Fetch futures data from InsightSentry for a given symbol and date range.

    Args:
        symbol: Futures symbol (e.g., 'ES', '6E', 'NQ')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLC data
    """
    # Check if API key is set
    api_key = os.environ.get('INSIGHTSENTRY_API_KEY')
    if not api_key:
        print("⚠️  INSIGHTSENTRY_API_KEY not set in environment variables")
        print("   Set it in your ~/.zshrc or ~/.bashrc:")
        print('   export INSIGHTSENTRY_API_KEY="your_api_key_here"')
        return None

    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    print(f"📊 Fetching {symbol} data from {start_date} to {end_date}")

    # Create data manager
    dm = DataSourceManager()

    try:
        # Fetch data
        df = dm.get_price_data(symbol, 'futures', start, end)

        if df is not None and not df.empty:
            print(f"✅ Successfully fetched {len(df)} records")
            print(f"\nData range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"\nLast 10 records:")
            print(df.tail(10))

            # Calculate ATR5 for reference
            from trade_evaluator import wilders_atr
            atr5 = wilders_atr(df, 5)
            print(f"\nATR5 on {df.index[-1].date()}: {atr5.iloc[-1]:.4f}")

            return df
        else:
            print("❌ No data returned")
            return None

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Fetch futures data from InsightSentry')
    parser.add_argument('--symbol', required=True, help='Futures symbol (e.g., ES, 6E, NQ)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--save', help='Save to CSV file (optional)')

    args = parser.parse_args()

    # Fetch data
    df = fetch_futures_data(args.symbol, args.start, args.end)

    # Save if requested
    if df is not None and args.save:
        df.to_csv(args.save)
        print(f"\n💾 Saved to {args.save}")


if __name__ == "__main__":
    main()
