#!/usr/bin/env python3
"""
InsightSentry WebSocket client for bulk historical data fetching.
Optimized for Ultra plan performance with multiple symbol subscriptions.
"""

import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os


class InsightSentryWebSocket:
    """WebSocket client for bulk InsightSentry data fetching."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = "wss://realtime.insightsentry.com/live"
        self.connection = None
        self.data_cache = {}
        self.pending_symbols = set()
        self.completed_symbols = set()
        
    def _get_symbol_mapping(self, symbol: str) -> str:
        """Map internal symbols to InsightSentry WebSocket format."""
        futures_symbols = {'6A', '6B', '6C', '6E', '6S', 'ES', 'NQ', 'RTY', 'YM'}
        
        if symbol in futures_symbols:
            # Currency futures mapping
            futures_mappings = {
                '6A': 'CME:6A1!', '6B': 'CME:6B1!', '6C': 'CME:6C1!',
                '6E': 'CME:6E1!', '6S': 'CME:6S1!',
                'ES': 'CME_MINI:ES1!', 'NQ': 'CME_MINI:NQ1!',
                'RTY': 'CME_MINI:RTY1!', 'YM': 'CBOT_MINI:YM1!'
            }
            return futures_mappings.get(symbol, f"CME:{symbol}")
        else:
            # ETF/Stock mappings
            etf_mappings = {
                'SPY': 'AMEX:SPY', 'QQQ': 'NASDAQ:QQQ', 'DIA': 'AMEX:DIA',
                'IWM': 'AMEX:IWM', 'GLD': 'AMEX:GLD', 'SLV': 'AMEX:SLV',
                'XLE': 'AMEX:XLE', 'XLF': 'AMEX:XLF', 'XLK': 'AMEX:XLK',
                'XLV': 'AMEX:XLV', 'XLI': 'AMEX:XLI', 'XLB': 'AMEX:XLB',
                'XRT': 'AMEX:XRT', 'XOP': 'AMEX:XOP', 'XME': 'AMEX:XME',
                'VNQ': 'AMEX:VNQ', 'IYR': 'AMEX:IYR', 'XHB': 'AMEX:XHB',
                'IBB': 'NASDAQ:IBB', 'GDX': 'AMEX:GDX', 'TLT': 'NASDAQ:TLT',
                'EEM': 'AMEX:EEM', 'EFA': 'AMEX:EFA', 'VWO': 'AMEX:VWO',
                'FXI': 'AMEX:FXI', 'EWG': 'AMEX:EWG', 'EWH': 'AMEX:EWH',
                'USO': 'AMEX:USO', 'UNG': 'AMEX:UNG'
            }
            return etf_mappings.get(symbol, f'AMEX:{symbol}')
    
    async def connect(self):
        """Establish WebSocket connection with authentication."""
        try:
            print("🔌 Connecting to InsightSentry WebSocket...")
            self.connection = await websockets.connect(self.ws_url)
            print("✅ WebSocket connected")
            return True
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def subscribe_symbols(self, symbols: List[str], bar_type: str = "day") -> None:
        """Subscribe to multiple symbols for historical data."""
        if not self.connection:
            raise RuntimeError("WebSocket not connected")
        
        # Map symbols to InsightSentry format
        mapped_symbols = [self._get_symbol_mapping(sym) for sym in symbols]
        self.pending_symbols = set(symbols)
        
        subscription_message = {
            "api_key": self.api_key,
            "action": "subscribe",
            "codes": mapped_symbols,
            "data": {
                "series": {
                    "bar_type": bar_type,
                    "bar_interval": 1,
                    "recent_bars": True,  # Get up to 100 recent bars (3-4 months)
                    "extended": False,
                    "dividend_adjustment": True if bar_type == "day" else False,
                    "back_adjustment": False
                }
            }
        }
        
        print(f"📡 Subscribing to {len(symbols)} symbols via WebSocket...")
        await self.connection.send(json.dumps(subscription_message))
    
    async def handle_messages(self, timeout_seconds: int = 30) -> Dict[str, pd.DataFrame]:
        """Handle incoming WebSocket messages and collect data."""
        if not self.connection:
            raise RuntimeError("WebSocket not connected")
        
        print("📥 Collecting historical data...")
        start_time = asyncio.get_event_loop().time()
        
        while self.pending_symbols and (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.connection.recv(), 
                    timeout=5.0
                )
                
                data = json.loads(message)
                await self._process_message(data)
                
            except asyncio.TimeoutError:
                print("⏰ Message timeout, continuing...")
                continue
            except websockets.exceptions.ConnectionClosed:
                print("🔌 WebSocket connection closed")
                break
            except Exception as e:
                print(f"⚠️ Message handling error: {e}")
                continue
        
        print(f"✅ Collected data for {len(self.completed_symbols)} symbols")
        return self._format_dataframes()
    
    async def _process_message(self, data: dict) -> None:
        """Process incoming WebSocket message and extract historical data."""
        if data.get("type") != "series":
            return
        
        # Extract symbol and series data
        symbol_code = data.get("code", "")
        series_data = data.get("series", [])
        
        if not symbol_code or not series_data:
            return
        
        # Reverse map from InsightSentry format to internal format
        internal_symbol = self._reverse_map_symbol(symbol_code)
        if not internal_symbol or internal_symbol not in self.pending_symbols:
            return
        
        print(f"📊 Received {len(series_data)} bars for {internal_symbol}")
        
        # Store the data
        self.data_cache[internal_symbol] = series_data
        self.pending_symbols.discard(internal_symbol)
        self.completed_symbols.add(internal_symbol)
    
    def _reverse_map_symbol(self, insightsentry_code: str) -> Optional[str]:
        """Reverse map InsightSentry symbol back to internal format."""
        # Create reverse mapping
        reverse_map = {
            'CME:6A1!': '6A', 'CME:6B1!': '6B', 'CME:6C1!': '6C',
            'CME:6E1!': '6E', 'CME:6S1!': '6S',
            'CME_MINI:ES1!': 'ES', 'CME_MINI:NQ1!': 'NQ',
            'CME_MINI:RTY1!': 'RTY', 'CBOT_MINI:YM1!': 'YM',
            'AMEX:SPY': 'SPY', 'NASDAQ:QQQ': 'QQQ', 'AMEX:DIA': 'DIA',
            'AMEX:IWM': 'IWM', 'AMEX:GLD': 'GLD', 'AMEX:SLV': 'SLV',
            'AMEX:XLE': 'XLE', 'AMEX:XLF': 'XLF', 'AMEX:XLK': 'XLK',
            'AMEX:XLV': 'XLV', 'AMEX:XLI': 'XLI', 'AMEX:XLB': 'XLB',
            'AMEX:XRT': 'XRT', 'AMEX:XOP': 'XOP', 'AMEX:XME': 'XME',
            'AMEX:VNQ': 'VNQ', 'AMEX:IYR': 'IYR', 'AMEX:XHB': 'XHB',
            'NASDAQ:IBB': 'IBB', 'AMEX:GDX': 'GDX', 'NASDAQ:TLT': 'TLT',
            'AMEX:EEM': 'EEM', 'AMEX:EFA': 'EFA', 'AMEX:VWO': 'VWO',
            'AMEX:FXI': 'FXI', 'AMEX:EWG': 'EWG', 'AMEX:EWH': 'EWH',
            'AMEX:USO': 'USO', 'AMEX:UNG': 'UNG'
        }
        return reverse_map.get(insightsentry_code)
    
    def _format_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Convert cached data to pandas DataFrames."""
        formatted_data = {}
        
        for symbol, bars in self.data_cache.items():
            try:
                df_data = []
                for bar in bars:
                    # Parse timestamp
                    if 'time' in bar:
                        timestamp = pd.to_datetime(bar['time'], unit='s')
                    elif 'timestamp' in bar:
                        timestamp = pd.to_datetime(bar['timestamp'], unit='s')
                    else:
                        continue
                    
                    df_data.append({
                        'Date': timestamp,
                        'Open': float(bar.get('open', 0)),
                        'High': float(bar.get('high', 0)),
                        'Low': float(bar.get('low', 0)),
                        'Close': float(bar.get('close', 0)),
                        'Volume': int(bar.get('volume', 0))
                    })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)
                    formatted_data[symbol] = df
                    print(f"📈 {symbol}: {len(df)} bars from {df.index.min().date()} to {df.index.max().date()}")
                    
            except Exception as e:
                print(f"❌ Error formatting data for {symbol}: {e}")
        
        return formatted_data
    
    async def close(self):
        """Close WebSocket connection."""
        if self.connection:
            await self.connection.close()
            print("🔌 WebSocket connection closed")


async def bulk_fetch_historical_data(symbols: List[str], api_key: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple symbols using WebSocket.
    
    Args:
        symbols: List of symbols to fetch
        api_key: InsightSentry API key
    
    Returns:
        Dictionary of {symbol: DataFrame} with historical data
    """
    ws_client = InsightSentryWebSocket(api_key)
    
    try:
        # Connect to WebSocket
        if not await ws_client.connect():
            return {}
        
        # Subscribe to all symbols
        await ws_client.subscribe_symbols(symbols, bar_type="day")
        
        # Collect data with timeout
        return await ws_client.handle_messages(timeout_seconds=60)
        
    except Exception as e:
        print(f"❌ WebSocket bulk fetch failed: {e}")
        return {}
    finally:
        await ws_client.close()


# Synchronous wrapper for existing code
def fetch_bulk_historical_data_sync(symbols: List[str], api_key: str) -> Dict[str, pd.DataFrame]:
    """Synchronous wrapper for WebSocket bulk data fetching."""
    return asyncio.run(bulk_fetch_historical_data(symbols, api_key))


if __name__ == "__main__":
    # Test with a few symbols
    test_symbols = ['SPY', 'QQQ', 'XLE', 'GLD']
    api_key = os.getenv('INSIGHTSENTRY_API_KEY')
    
    if not api_key:
        print("❌ INSIGHTSENTRY_API_KEY not found")
        exit(1)
    
    print("🧪 Testing WebSocket bulk data fetching...")
    start_time = datetime.now()
    
    results = fetch_bulk_historical_data_sync(test_symbols, api_key)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n📊 Results:")
    print(f"⏱️ Total time: {duration:.2f} seconds")
    print(f"📈 Symbols fetched: {len(results)}/{len(test_symbols)}")
    print(f"🚀 Average time per symbol: {duration/len(test_symbols):.2f} seconds")
    
    for symbol, df in results.items():
        print(f"  {symbol}: {len(df)} rows")