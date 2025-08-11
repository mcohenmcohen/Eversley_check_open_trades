#!/usr/bin/env python3
"""
Futures Data Downloader
Downloads OHLC data for futures symbols from yfinance and saves to CSV files
in the same format as NinjaTrader exports.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import random
from pathlib import Path

# Futures symbols mapping (local symbol -> yfinance symbol)
FUTURES_SYMBOLS = {
    # Currency futures
    "6A": "6A=F",  # Australian Dollar
    "6B": "6B=F",  # British Pound
    "6C": "6C=F",  # Canadian Dollar
    "6E": "6E=F",  # Euro
    "6S": "6S=F",  # Swiss Franc
    
    # Index futures
    "ES": "ES=F",  # E-mini S&P 500
    "NQ": "NQ=F",  # E-mini Nasdaq
    "RTY": "RTY=F", # E-mini Russell 2000
    "YM": "YM=F"   # E-mini Dow Jones
}

def download_futures_data(symbol, yf_symbol, start_date, end_date, data_dir="data", max_retries=3):
    """
    Download futures data from yfinance and save as CSV in NinjaTrader format.
    
    Args:
        symbol: Local symbol (e.g., "6A")
        yf_symbol: Yahoo Finance symbol (e.g., "6A=F")
        start_date: Start date as string "YYYY-MM-DD"
        end_date: End date as string "YYYY-MM-DD"
        data_dir: Directory to save CSV files
        max_retries: Maximum number of retry attempts
    
    Returns:
        bool: True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            print(f"Downloading {symbol} ({yf_symbol})... (attempt {attempt + 1})")
            
            # Add random delay to avoid rate limiting
            if attempt > 0:
                delay = random.uniform(2, 5)
                print(f"  Waiting {delay:.1f} seconds...")
                time.sleep(delay)
            
            # Download data from yfinance
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(start=start_date, end=end_date, interval="1d", timeout=30)
            
            if data.empty:
                if attempt == max_retries - 1:
                    print(f"❌ No data returned for {yf_symbol} after {max_retries} attempts")
                    return False
                continue
            
            # Convert to NinjaTrader CSV format
            # Date format: YYYYMMDD (no separators)
            df = pd.DataFrame()
            df['Date'] = data.index.strftime('%Y%m%d')
            df['Open'] = data['Open'].round(4)
            df['High'] = data['High'].round(4) 
            df['Low'] = data['Low'].round(4)
            df['Close'] = data['Close'].round(4)
            df['Volume'] = data['Volume'].astype(int)
            
            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Save to CSV file
            csv_path = os.path.join(data_dir, f"{symbol}.csv")
            df.to_csv(csv_path, index=False)
            
            print(f"✅ Saved {len(df)} rows to {csv_path}")
            print(f"   Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
            
            return True
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Error downloading {symbol} after {max_retries} attempts: {str(e)}")
                return False
            else:
                print(f"  Attempt {attempt + 1} failed: {str(e)[:50]}...")
                continue
    
    return False

def download_all_futures(days_back=365, data_dir="data"):
    """
    Download data for all futures symbols.
    
    Args:
        days_back: Number of days of historical data to download
        data_dir: Directory to save CSV files
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Downloading futures data from {start_str} to {end_str}")
    print("=" * 60)
    
    success_count = 0
    total_count = len(FUTURES_SYMBOLS)
    
    for i, (symbol, yf_symbol) in enumerate(FUTURES_SYMBOLS.items()):
        if download_futures_data(symbol, yf_symbol, start_str, end_str, data_dir):
            success_count += 1
        
        # Add delay between downloads to avoid rate limiting
        if i < len(FUTURES_SYMBOLS) - 1:  # Don't delay after the last one
            delay = random.uniform(1, 3)
            print(f"Waiting {delay:.1f} seconds before next download...")
            time.sleep(delay)
        
        print()  # Add spacing between symbols
    
    print("=" * 60)
    print(f"Download complete: {success_count}/{total_count} symbols successful")
    
    if success_count < total_count:
        print("⚠️  Some downloads failed. Check the error messages above.")
    else:
        print("🎉 All futures data downloaded successfully!")

def main():
    """Main function with command line options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download futures data from yfinance")
    parser.add_argument("--symbol", type=str, help="Download specific symbol (e.g., 'ES')")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data (default: 365)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (default: 'data')")
    
    args = parser.parse_args()
    
    if args.symbol:
        # Download single symbol
        if args.symbol in FUTURES_SYMBOLS:
            yf_symbol = FUTURES_SYMBOLS[args.symbol]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            download_futures_data(args.symbol, yf_symbol, start_str, end_str, args.data_dir)
        else:
            print(f"❌ Unknown symbol: {args.symbol}")
            print(f"Available symbols: {', '.join(FUTURES_SYMBOLS.keys())}")
    else:
        # Download all symbols
        download_all_futures(args.days, args.data_dir)

if __name__ == "__main__":
    main()