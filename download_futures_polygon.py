#!/usr/bin/env python3
"""
Polygon.io Futures Data Downloader
Downloads OHLC data for futures symbols from Polygon.io and saves to CSV files
in the same format as NinjaTrader exports.

Ready for when Polygon.io $29/month Futures Starter plan becomes available.
"""

from polygon.rest import RESTClient
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import json
from pathlib import Path

# Use existing API key - will work when futures access is available
POLYGON_API_KEY = 'qFTWmhmyAj2pJqpx0Pwxp2AyShKFVPv9'

# Futures symbols for download
FUTURES_SYMBOLS = {
    # Currency futures
    "6A": "Australian Dollar futures",
    "6B": "British Pound futures", 
    "6C": "Canadian Dollar futures",
    "6E": "Euro futures",
    "6S": "Swiss Franc futures",
    
    # Index futures
    "ES": "E-mini S&P 500 futures",
    "NQ": "E-mini Nasdaq futures", 
    "RTY": "E-mini Russell 2000 futures",
    "YM": "E-mini Dow Jones futures"
}

def download_futures_data(symbol, start_date, end_date, data_dir="data", api_key=None, max_retries=3):
    """
    Download futures data from Polygon.io and save as CSV in NinjaTrader format.
    
    Args:
        symbol: Futures symbol (e.g., "6A", "ES")
        start_date: Start date as string "YYYY-MM-DD" or datetime
        end_date: End date as string "YYYY-MM-DD" or datetime  
        data_dir: Directory to save CSV files
        api_key: Polygon.io API key (uses global if not provided)
        max_retries: Maximum number of retry attempts
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    if api_key is None:
        api_key = POLYGON_API_KEY
        
    client = RESTClient(api_key=api_key)
    
    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {symbol} ({FUTURES_SYMBOLS.get(symbol, 'Unknown')})... (attempt {attempt + 1})")
            
            # Add delay between retries
            if attempt > 0:
                delay = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                print(f"  Waiting {delay} seconds...")
                time.sleep(delay)
            
            # Get aggregate bars (OHLC data) from Polygon.io
            aggs = client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan='day',
                from_=start_date,
                to=end_date,
                limit=50000  # Get up to 50k bars (should be plenty for most ranges)
            )
            
            if not aggs or len(aggs) == 0:
                if attempt == max_retries - 1:
                    print(f"❌ No data returned for {symbol} after {max_retries} attempts")
                    return False
                print(f"  No data returned, retrying...")
                continue
            
            # Convert to NinjaTrader CSV format
            df = pd.DataFrame()
            
            for bar in aggs:
                # Convert timestamp to datetime
                dt = datetime.fromtimestamp(bar.timestamp / 1000)
                
                # Create row matching NinjaTrader format
                row = {
                    'Date': dt.strftime('%Y%m%d'),  # YYYYMMDD format
                    'Open': round(bar.open, 4),
                    'High': round(bar.high, 4),
                    'Low': round(bar.low, 4),
                    'Close': round(bar.close, 4),
                    'Volume': int(bar.volume) if bar.volume else 0
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            
            # Sort by date to ensure proper chronological order
            df = df.sort_values('Date')
            
            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Save to CSV file (same format as NinjaTrader)
            csv_path = os.path.join(data_dir, f"{symbol}.csv")
            df.to_csv(csv_path, index=False)
            
            print(f"✅ Saved {len(df)} rows to {csv_path}")
            print(f"   Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
            print(f"   Latest OHLC: O={df['Open'].iloc[-1]} H={df['High'].iloc[-1]} L={df['Low'].iloc[-1]} C={df['Close'].iloc[-1]}")
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            if attempt == max_retries - 1:
                print(f"❌ Error downloading {symbol} after {max_retries} attempts:")
                print(f"   {error_msg}")
                
                # Check if it's an access/permission issue
                if "unauthorized" in error_msg.lower() or "access" in error_msg.lower():
                    print(f"   💡 This may indicate that futures access is not yet available on your plan.")
                    print(f"   💡 Check if the $29/month Futures Starter plan is active on your account.")
                return False
            else:
                print(f"  Attempt {attempt + 1} failed: {error_msg[:80]}...")
                continue
    
    return False

def download_all_futures(days_back=365, data_dir="data", api_key=None):
    """
    Download data for all futures symbols.
    
    Args:
        days_back: Number of days of historical data to download
        data_dir: Directory to save CSV files
        api_key: Polygon.io API key (uses global if not provided)
    """
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Polygon.io Futures Data Downloader")
    print(f"{'=' * 60}")
    print(f"Date range: {start_date} to {end_date} ({days_back} days)")
    print(f"Symbols: {', '.join(FUTURES_SYMBOLS.keys())}")
    print(f"Output directory: {data_dir}")
    print(f"{'=' * 60}")
    print()
    
    success_count = 0
    total_count = len(FUTURES_SYMBOLS)
    
    for i, symbol in enumerate(FUTURES_SYMBOLS.keys()):
        if download_futures_data(symbol, start_date, end_date, data_dir, api_key):
            success_count += 1
        
        # Add delay between downloads to be respectful to API
        if i < len(FUTURES_SYMBOLS) - 1:  # Don't delay after the last one
            delay = 1  # 1 second between requests
            print(f"Waiting {delay} seconds before next download...")
            time.sleep(delay)
        
        print()  # Add spacing between symbols
    
    print("=" * 60)
    print(f"Download complete: {success_count}/{total_count} symbols successful")
    
    if success_count < total_count:
        print("⚠️  Some downloads failed. Check the error messages above.")
        if success_count == 0:
            print("💡 If all downloads failed, this may indicate:")
            print("   - Futures access is not yet available on your Polygon.io plan")
            print("   - The $29/month Futures Starter plan is not yet active")
            print("   - API key needs to be updated")
    else:
        print("🎉 All futures data downloaded successfully!")
        print(f"📁 CSV files saved to: {os.path.abspath(data_dir)}/")

def verify_csv_format(data_dir="data"):
    """
    Verify that downloaded CSV files match the NinjaTrader format.
    """
    print("Verifying CSV file formats...")
    print("=" * 40)
    
    for symbol in FUTURES_SYMBOLS.keys():
        csv_path = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                if list(df.columns) == expected_columns:
                    print(f"✅ {symbol}.csv: Format correct ({len(df)} rows)")
                    # Show sample of data
                    if len(df) > 0:
                        sample = df.iloc[-1]  # Show latest row
                        print(f"   Latest: {sample['Date']} OHLC=({sample['Open']},{sample['High']},{sample['Low']},{sample['Close']}) Vol={sample['Volume']}")
                else:
                    print(f"❌ {symbol}.csv: Wrong columns. Expected {expected_columns}, got {list(df.columns)}")
                    
            except Exception as e:
                print(f"❌ {symbol}.csv: Error reading file - {str(e)}")
        else:
            print(f"❌ {symbol}.csv: File not found")
    
    print()

def main():
    """Main function with command line options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download futures data from Polygon.io")
    parser.add_argument("--symbol", type=str, help=f"Download specific symbol. Options: {', '.join(FUTURES_SYMBOLS.keys())}")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data (default: 365)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (default: 'data')")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--verify", action="store_true", help="Verify existing CSV file formats")
    parser.add_argument("--api-key", type=str, help="Polygon.io API key (uses default if not specified)")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_csv_format(args.data_dir)
        return
    
    # Determine date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        end_date = datetime.now().date()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    else:
        start_date = end_date - timedelta(days=args.days)
    
    if args.symbol:
        # Download single symbol
        if args.symbol in FUTURES_SYMBOLS:
            download_futures_data(args.symbol, start_date, end_date, args.data_dir, args.api_key)
        else:
            print(f"❌ Unknown symbol: {args.symbol}")
            print(f"Available symbols: {', '.join(FUTURES_SYMBOLS.keys())}")
    else:
        # Download all symbols
        days_back = (end_date - start_date).days
        download_all_futures(days_back, args.data_dir, args.api_key)

if __name__ == "__main__":
    main()