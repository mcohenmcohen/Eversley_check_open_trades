"""
Data sources module for currency strategy backtesting.
Supports Polygon.io for ETFs and InsightSentry for futures data.
"""
import pandas as pd
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Polygon imports
from polygon.rest import RESTClient
from polygon.exceptions import BadResponse, AuthError


class DataSourceError(Exception):
    """Custom exception for data source errors."""
    pass


class PolygonDataSource:
    """Polygon.io data source implementation for ETF data."""
    
    def __init__(self, api_key: str):
        """Initialize Polygon client."""
        self.api_key = api_key
        if not self.api_key:
            raise DataSourceError("Polygon API key required")
        
        self.client = RESTClient(api_key=self.api_key)
        print("✅ Polygon client initialized")
    
    def _is_futures_symbol(self, symbol: str) -> bool:
        """Determine if symbol is a futures symbol based on the symbol itself."""
        futures_symbols = {
            # Currency futures
            '6A', '6B', '6C', '6E', '6S',
            # Index E-mini futures  
            'ES', 'NQ', 'RTY', 'YM'
        }
        return symbol in futures_symbols
    
    def _get_symbol_mapping(self, symbol: str) -> str:
        """
        Map internal symbols to Polygon format.
        Symbol type (futures vs stock/ETF) is inferred from the symbol itself.
        
        Args:
            symbol: Internal symbol (e.g., 'FXI', '6E')
            
        Returns:
            Polygon formatted symbol
        """
        if self._is_futures_symbol(symbol):
            # Polygon futures symbols need =F suffix
            futures_mappings = {
                '6A': '6A=F',   # Australian Dollar
                '6B': '6B=F',   # British Pound  
                '6C': '6C=F',   # Canadian Dollar
                '6E': '6E=F',   # Euro
                '6S': '6S=F',   # Swiss Franc
                'ES': 'ES=F',   # S&P 500 E-mini
                'NQ': 'NQ=F',   # Nasdaq 100 E-mini
                'RTY': 'RTY=F', # Russell 2000 E-mini
                'YM': 'YM=F',   # Dow Jones E-mini
            }
            return futures_mappings.get(symbol, f"{symbol}=F")
        else:
            # Polygon stocks/ETFs use the symbol as-is
            return symbol
    
    def test_authentication(self) -> bool:
        """
        Test if Polygon API key is valid.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Make a simple API call to test authentication
            aggs = self.client.get_aggs(
                ticker="SPY",
                multiplier=1,
                timespan="day",
                from_="2024-01-01",
                to="2024-01-02",
                limit=1
            )
            print("✅ Polygon authentication successful")
            return True
        except AuthError:
            print("❌ Polygon authentication failed - invalid API key")
            return False
        except Exception as e:
            print(f"⚠️ Polygon authentication test error: {e}")
            return False
    
    def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical stock data from Polygon for ETFs.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            start_date: Start date as datetime
            end_date: End date as datetime
            
        Returns:
            DataFrame with OHLCV data indexed by Date
        """
        try:
            # Map symbol to Polygon format
            polygon_symbol = self._get_symbol_mapping(symbol)
            print(f"📡 Fetching {symbol} ({polygon_symbol}) from Polygon")
            
            aggs = self.client.get_aggs(
                ticker=polygon_symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                limit=50000
            )
            
            if not aggs or len(aggs) == 0:
                print(f"⚠️ No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    "Date": pd.to_datetime(agg.timestamp, unit='ms'),
                    "Open": agg.open,
                    "High": agg.high,
                    "Low": agg.low,
                    "Close": agg.close
                })
            
            df = pd.DataFrame(data)
            df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index).normalize()
            df.sort_index(inplace=True)
            
            print(f"{symbol}: fetched {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
            return df
            
        except (BadResponse, AuthError) as e:
            print(f"❌ Polygon API error for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Unexpected error getting {symbol} from Polygon: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime, max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks with concurrent requests.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date as datetime
            end_date: End date as datetime
            max_workers: Max concurrent requests
            
        Returns:
            Dictionary of {symbol: DataFrame}
        """
        results = {}
        
        print(f"🚀 Getting Polygon data for {len(symbols)} symbols...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_stock_data, symbol, start_date, end_date): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
        
        print(f"✅ Retrieved Polygon data for {len(results)}/{len(symbols)} symbols")
        return results


class InsightSentryDataSource:
    """InsightSentry data source implementation for futures data."""
    
    def __init__(self, api_key: str):
        """Initialize InsightSentry client."""
        self.api_key = api_key
        if not self.api_key:
            raise DataSourceError("InsightSentry API key required")
        
        self.base_url = "https://api.insightsentry.com"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        print("✅ InsightSentry client initialized")
    
    def test_authentication(self) -> bool:
        """
        Test if InsightSentry API key is valid.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            import requests
            
            # Make a simple API call to test authentication using a known working symbol
            test_symbol = self._get_symbol_mapping('QQQ')
            url = f"{self.base_url}/v2/symbols/{test_symbol}/history"
            params = {
                'bar_type': 'day',
                'bar_interval': 1,
                'data_points': 1
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 401:
                print("❌ InsightSentry authentication failed - invalid API key")
                return False
            elif response.status_code == 200:
                print("✅ InsightSentry authentication successful")
                return True
            else:
                print(f"⚠️ InsightSentry authentication test returned status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"⚠️ InsightSentry authentication test error: {e}")
            return False
    
    def _is_futures_symbol(self, symbol: str) -> bool:
        """Determine if symbol is a futures symbol based on the symbol itself."""
        futures_symbols = {
            # Currency futures
            '6A', '6B', '6C', '6E', '6S',
            # Index E-mini futures  
            'ES', 'NQ', 'RTY', 'YM'
        }
        return symbol in futures_symbols
    
    def _get_symbol_mapping(self, symbol: str) -> str:
        """
        Map internal symbols to InsightSentry format.
        Symbol type (futures vs stock/ETF) is inferred from the symbol itself.
        
        Args:
            symbol: Internal symbol (e.g., '6A', 'SPY')
            
        Returns:
            InsightSentry formatted symbol
        """
        if self._is_futures_symbol(symbol):
            # Currency futures mapping for InsightSentry (continuous contracts with exchange)
            futures_mappings = {
                # Currency futures - CME exchange
                '6A': 'CME:6A1!',      # Australian Dollar continuous
                '6B': 'CME:6B1!',      # British Pound continuous  
                '6C': 'CME:6C1!',      # Canadian Dollar continuous
                '6E': 'CME:6E1!',      # Euro continuous
                '6S': 'CME:6S1!',      # Swiss Franc continuous
                # Index E-mini futures - CME_MINI/CBOT_MINI exchanges
                'ES': 'CME_MINI:ES1!', # S&P 500 E-mini continuous
                'NQ': 'CME_MINI:NQ1!', # Nasdaq 100 E-mini continuous
                'RTY': 'CME_MINI:RTY1!', # Russell 2000 E-mini continuous
                'YM': 'CBOT_MINI:YM1!', # Dow Jones E-mini continuous
            }
            return futures_mappings.get(symbol, f"CME:{symbol}")
        else:
            # For ETFs/stocks, InsightSentry typically uses exchange prefixes
            # Common ETF mappings for InsightSentry
            etf_mappings = {
                # Major ETFs with their InsightSentry exchange prefixes
                # Based on InsightSentry search results - most ETFs use AMEX: prefix
                'SPY': 'AMEX:SPY',      # SPDR S&P 500 ETF
                'QQQ': 'NASDAQ:QQQ',    # Invesco QQQ (confirmed working)
                'IWM': 'AMEX:IWM',      # iShares Russell 2000
                'DIA': 'AMEX:DIA',      # SPDR Dow Jones
                'GLD': 'AMEX:GLD',      # SPDR Gold Shares
                'SLV': 'AMEX:SLV',      # iShares Silver Trust
                'XLE': 'AMEX:XLE',      # Energy Select Sector SPDR
                'XLF': 'AMEX:XLF',      # Financial Select Sector SPDR
                'XLK': 'AMEX:XLK',      # Technology Select Sector SPDR
                'XLV': 'AMEX:XLV',      # Health Care Select Sector SPDR
                'XLI': 'AMEX:XLI',      # Industrial Select Sector SPDR
                'XLB': 'AMEX:XLB',      # Materials Select Sector SPDR
                'XRT': 'AMEX:XRT',      # SPDR S&P Retail ETF
                'XOP': 'AMEX:XOP',      # SPDR S&P Oil & Gas E&P ETF
                'XME': 'AMEX:XME',      # SPDR S&P Metals & Mining ETF
                'VNQ': 'AMEX:VNQ',      # Vanguard Real Estate ETF
                'IYR': 'AMEX:IYR',      # iShares U.S. Real Estate ETF
                'XHB': 'AMEX:XHB',      # SPDR S&P Homebuilders ETF
                'IBB': 'NASDAQ:IBB',    # iShares Biotechnology ETF
                'GDX': 'AMEX:GDX',      # VanEck Gold Miners ETF
                'TLT': 'NASDAQ:TLT',    # iShares 20+ Year Treasury Bond ETF
                'EEM': 'AMEX:EEM',      # iShares MSCI Emerging Markets ETF
                'EFA': 'AMEX:EFA',      # iShares MSCI EAFE ETF
                'VWO': 'AMEX:VWO',      # Vanguard Emerging Markets ETF
                'FXI': 'AMEX:FXI',      # iShares China Large-Cap ETF (confirmed from search)
                'EWG': 'AMEX:EWG',      # iShares MSCI Germany ETF
                'EWH': 'AMEX:EWH',      # iShares MSCI Hong Kong ETF
                'USO': 'AMEX:USO',      # United States Oil Fund
                'UNG': 'AMEX:UNG',      # United States Natural Gas Fund
            }
            
            # Return mapped symbol or fallback to AMEX prefix (most ETFs use AMEX in InsightSentry)
            return etf_mappings.get(symbol, f'AMEX:{symbol}')
    
    def _convert_hourly_to_daily_futures(self, hourly_df):
        """
        Convert hourly futures data to daily OHLC using 3pm-3pm cycle.
        
        Futures trading day runs from 3pm ET previous day to 3pm ET current day.
        This creates daily bars that align with futures market sessions.
        
        Args:
            hourly_df: DataFrame with hourly OHLCV data
            
        Returns:
            DataFrame with daily OHLC data indexed by trading date
        """
        if hourly_df.empty:
            return pd.DataFrame()
        
        try:
            # Convert to EST/EDT timezone for proper 3pm alignment
            hourly_df.index = pd.to_datetime(hourly_df.index)
            
            # Create a column for the futures trading date
            # Trading day starts at 3pm ET, so hours 15:00-23:59 belong to next calendar day
            # Hours 00:00-14:59 belong to current calendar day
            hourly_df = hourly_df.copy()
            
            # Shift timestamps by 9 hours to align 3pm with midnight
            # This way we can group by date and get proper 3pm-3pm sessions
            shifted_index = hourly_df.index - pd.Timedelta(hours=9)
            hourly_df['trading_date'] = shifted_index.normalize()
            
            # Group by trading date and aggregate to daily OHLC
            daily_data = []
            
            for trading_date, group in hourly_df.groupby('trading_date'):
                if len(group) == 0:
                    continue
                
                # Sort by time to ensure proper OHLC calculation
                group = group.sort_index()
                
                daily_bar = {
                    'Open': group['Open'].iloc[0],      # First hour's open
                    'High': group['High'].max(),         # Highest high of all hours
                    'Low': group['Low'].min(),           # Lowest low of all hours  
                    'Close': group['Close'].iloc[-1],    # Last hour's close
                    'Volume': group['Volume'].sum() if 'Volume' in group.columns else 0
                }
                
                daily_data.append({
                    'Date': trading_date,
                    **daily_bar
                })
            
            if not daily_data:
                print("⚠️ No daily data created from hourly bars")
                return pd.DataFrame()
            
            # Create daily DataFrame
            daily_df = pd.DataFrame(daily_data)
            daily_df.set_index('Date', inplace=True)
            daily_df.sort_index(inplace=True)
            
            print(f"Converted {len(hourly_df)} hourly bars to {len(daily_df)} daily bars")
            return daily_df
            
        except Exception as e:
            print(f"❌ Error converting hourly to daily: {e}")
            return pd.DataFrame()
    
    def get_futures_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical futures data from InsightSentry.
        
        Args:
            symbol: Futures symbol (e.g., '6A')
            start_date: Start date as datetime
            end_date: End date as datetime
            
        Returns:
            DataFrame with OHLCV data indexed by Date
        """
        try:
            import requests
            
            # Map symbol to InsightSentry format
            formatted_symbol = self._get_symbol_mapping(symbol)
            
            print(f"📡 Fetching {symbol} ({formatted_symbol}) from InsightSentry")
            
            # Use the v3 series endpoint for historical data (supports dp parameter)
            url = f"{self.base_url}/v3/symbols/{formatted_symbol}/series"
            
            # Parameters based on symbol type
            # Use dp (data points) parameter for better performance
            if self._is_futures_symbol(symbol):
                # Futures: hourly data (aggregated to daily using 3pm-3pm cycle)
                params = {
                    'bar_type': 'hour',
                    'bar_interval': 1,
                    'dp': 12000,  # ~500 days * 24 hours (more than a year to ensure current data)
                    'extended': False,
                    'dadj': False,  # dividend adjustment - not relevant for futures
                    'badj': True,   # back adjustment
                    'long_poll': False
                }
            else:
                # Stocks/ETFs: daily data with dividend adjustment  
                params = {
                    'bar_type': 'day',
                    'bar_interval': 1,
                    'dp': 500,  # ~500 days (more than a year to ensure current data)
                    'extended': False,
                    'dadj': True,   # dividend adjustment for stocks
                    'badj': False,  # back adjustment
                    'long_poll': False
                }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 401:
                raise DataSourceError(f"InsightSentry API authentication failed for {symbol}")
            elif response.status_code != 200:
                print(f"❌ InsightSentry API error for {symbol}: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            # InsightSentry uses 'series' field
            if not data or 'series' not in data or not data['series']:
                print(f"⚠️ No data returned from InsightSentry for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            series_data = data['series']
            df_data = []
            
            for bar in series_data:
                # Parse timestamp - InsightSentry uses 'time' field with epoch seconds
                if 'time' in bar:
                    try:
                        timestamp = pd.to_datetime(bar['time'], unit='s')
                    except:
                        continue
                elif 'timestamp' in bar:
                    try:
                        timestamp = pd.to_datetime(bar['timestamp'], unit='s')
                    except:
                        continue
                elif 'date' in bar:
                    timestamp = pd.to_datetime(bar['date'])
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
            
            if not df_data:
                print(f"⚠️ No valid bars found for {symbol}")
                return pd.DataFrame()
            
            # Create DataFrame with hourly data
            df = pd.DataFrame(df_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            total_bars = len(df)
            
            # Process based on symbol type
            if self._is_futures_symbol(symbol):
                print(f"Raw hourly data: {total_bars} bars")
                # Convert hourly data to daily OHLC (3pm-3pm cycle)
                df = self._convert_hourly_to_daily_futures(df)
                
                if df.empty:
                    print(f"⚠️ No daily bars created from hourly data for {symbol}")
                    return pd.DataFrame()
            else:
                print(f"Raw daily data: {total_bars} bars")
            
            # Don't filter by date range - return all historical data for calendar year backtesting
            # The dp parameter already limits us to the desired amount of historical data
            print(f"{symbol}: retrieved {len(df)} days from InsightSentry")
            return df
            
        except requests.RequestException as e:
            print(f"❌ Network error fetching {symbol} from InsightSentry: {e}")
        except Exception as e:
            print(f"❌ Error fetching {symbol} from InsightSentry: {e}")
        
        return pd.DataFrame()
    
    def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical stock/ETF data from InsightSentry.
        
        Args:
            symbol: Stock/ETF symbol (e.g., 'SPY')
            start_date: Start date as datetime
            end_date: End date as datetime
            
        Returns:
            DataFrame with OHLCV data indexed by Date
        """
        try:
            import requests
            
            # Map symbol to InsightSentry format
            formatted_symbol = self._get_symbol_mapping(symbol)
            
            print(f"📡 Fetching {symbol} ({formatted_symbol}) from InsightSentry")
            
            # Use the v3 series endpoint for historical data (supports dp parameter)
            url = f"{self.base_url}/v3/symbols/{formatted_symbol}/series"
            
            # Parameters for daily stock data  
            params = {
                'bar_type': 'day',
                'bar_interval': 1,   # 1 day bars
                'dp': 250,  # ~250 days (full year of trading days)
                'extended': False,
                'dadj': True,   # dividend adjustment for stocks
                'badj': False,  # back adjustment
                'long_poll': False
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 401:
                raise DataSourceError(f"InsightSentry API authentication failed for {symbol}")
            elif response.status_code != 200:
                print(f"❌ InsightSentry API error for {symbol}: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            # InsightSentry uses 'series' field
            if not data or 'series' not in data or not data['series']:
                print(f"⚠️ No data returned from InsightSentry for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            series_data = data['series']
            df_data = []
            
            for bar in series_data:
                # Parse timestamp - InsightSentry uses 'time' field with epoch seconds
                if 'time' in bar:
                    try:
                        timestamp = pd.to_datetime(bar['time'], unit='s')
                    except:
                        continue
                elif 'timestamp' in bar:
                    try:
                        timestamp = pd.to_datetime(bar['timestamp'], unit='s')
                    except:
                        continue
                elif 'date' in bar:
                    timestamp = pd.to_datetime(bar['date'])
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
            
            if not df_data:
                print(f"⚠️ No valid bars found for {symbol}")
                return pd.DataFrame()
            
            # Create DataFrame with daily data
            df = pd.DataFrame(df_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Don't filter by date range - return all historical data for calendar year backtesting
            # The dp parameter already limits us to the desired amount of historical data
            print(f"{symbol}: retrieved {len(df)} days from InsightSentry")
            return df
            
        except requests.RequestException as e:
            print(f"❌ Network error fetching {symbol} from InsightSentry: {e}")
        except Exception as e:
            print(f"❌ Error fetching {symbol} from InsightSentry: {e}")
        
        return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], start_date: datetime, 
                           end_date: datetime, max_workers: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple stock/ETF symbols from InsightSentry.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date as datetime
            end_date: End date as datetime
            max_workers: Max concurrent requests
            
        Returns:
            Dictionary of {symbol: DataFrame}
        """
        results = {}
        
        print(f"🚀 Getting InsightSentry stock data for {len(symbols)} symbols...")
        
        # Use ThreadPoolExecutor for parallel requests (lower concurrency for API)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_stock_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
        
        print(f"✅ Retrieved InsightSentry stock data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_multiple_futures(self, symbols: List[str], start_date: datetime, 
                           end_date: datetime, max_workers: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple futures symbols from InsightSentry.
        
        Args:
            symbols: List of futures symbols
            start_date: Start date as datetime
            end_date: End date as datetime
            max_workers: Max concurrent requests
            
        Returns:
            Dictionary of {symbol: DataFrame}
        """
        results = {}
        
        print(f"🚀 Getting InsightSentry futures data for {len(symbols)} symbols...")
        
        # Use ThreadPoolExecutor for parallel requests (lower concurrency for futures API)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_futures_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
        
        print(f"✅ Retrieved InsightSentry futures data for {len(results)}/{len(symbols)} symbols")
        return results


class DataSourceManager:
    """
    Manager class to coordinate between different data sources.
    Routes ETFs to Polygon and futures to InsightSentry.
    """
    
    def __init__(self, polygon_api_key: str = None, insightsentry_api_key: str = None):
        """
        Initialize data source manager.
        
        Args:
            polygon_api_key: Polygon.io API key for ETF data
            insightsentry_api_key: InsightSentry API key for futures data
        """
        self.sources = {}
        
        # Initialize Polygon for ETFs
        if polygon_api_key:
            try:
                self.sources['polygon'] = PolygonDataSource(polygon_api_key)
            except DataSourceError as e:
                print(f"⚠️ Polygon initialization failed: {e}")
            except Exception as e:
                print(f"⚠️ Unexpected error initializing Polygon: {e}")
        
        # Initialize InsightSentry for futures
        if insightsentry_api_key:
            try:
                self.sources['insightsentry'] = InsightSentryDataSource(insightsentry_api_key)
            except DataSourceError as e:
                print(f"⚠️ InsightSentry initialization failed: {e}")
        
        print(f"📊 Initialized data sources: {list(self.sources.keys())}")
    
    def test_authentication(self, source_name: str = None) -> Dict[str, bool]:
        """
        Test authentication for specified data source or all available sources.
        
        Args:
            source_name: Specific source to test ('polygon', 'insightsentry'), or None for all
            
        Returns:
            Dictionary of {source_name: auth_success}
        """
        results = {}
        
        sources_to_test = [source_name] if source_name else list(self.sources.keys())
        
        for source in sources_to_test:
            if source not in self.sources:
                print(f"⚠️ Data source '{source}' not available")
                results[source] = False
                continue
                
            print(f"🔐 Testing {source} authentication...")
            try:
                results[source] = self.sources[source].test_authentication()
            except Exception as e:
                print(f"❌ Error testing {source} authentication: {e}")
                results[source] = False
        
        return results
    
    def get_etf_data(self, symbols: Union[str, List[str]], start_date: datetime, 
                     end_date: datetime, source: str = 'polygon') -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get ETF data from specified data source.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date as datetime
            end_date: End date as datetime
            source: Data source to use ('polygon' or 'insightsentry')
            
        Returns:
            DataFrame for single symbol, Dict for multiple symbols
        """
        if source not in self.sources:
            raise DataSourceError(f"{source} data source not available")
        
        data_source = self.sources[source]
        
        if isinstance(symbols, str):
            return data_source.get_stock_data(symbols, start_date, end_date)
        else:
            if source == 'polygon':
                return data_source.get_multiple_stocks(symbols, start_date, end_date)
            else:  # insightsentry
                return data_source.get_multiple_stocks(symbols, start_date, end_date)
    
    def get_futures_data(self, symbols: Union[str, List[str]], start_date: datetime, 
                        end_date: datetime) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get futures data from InsightSentry.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date as datetime
            end_date: End date as datetime
            
        Returns:
            DataFrame for single symbol, Dict for multiple symbols
        """
        if 'insightsentry' not in self.sources:
            raise DataSourceError("InsightSentry data source not available")
        
        insightsentry_source = self.sources['insightsentry']
        
        if isinstance(symbols, str):
            return insightsentry_source.get_futures_data(symbols, start_date, end_date)
        else:
            return insightsentry_source.get_multiple_futures(symbols, start_date, end_date)
    
    def get_price_data(self, symbol: str, mode: str, start_date: datetime, 
                      end_date: datetime = None) -> pd.DataFrame:
        """
        Get price data for a symbol based on mode (etfs or futures).
        This is the main interface for the backtesting system.
        
        Args:
            symbol: Symbol to fetch
            mode: 'etfs' or 'futures'
            start_date: Start date as datetime
            end_date: End date as datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
        
        if mode == "etfs":
            return self.get_etf_data(symbol, start_date, end_date)
        elif mode == "futures":
            return self.get_futures_data(symbol, start_date, end_date)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'etfs' or 'futures'")
    
    def is_source_available(self, source: str) -> bool:
        """Check if a data source is available."""
        return source in self.sources


# Helper function to load futures data from local CSV (fallback)
def load_futures_data_from_csv(symbol: str) -> pd.DataFrame:
    """
    Load futures data from local CSV file as fallback.
    
    Args:
        symbol: Futures symbol (e.g., '6A')
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        file_path = f"./data/{symbol}.csv"
        df = pd.read_csv(file_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index).normalize()
        print(f"📂 Loaded local CSV data for {symbol}")
        return df
    except Exception as e:
        print(f"❌ Error loading local CSV for {symbol}: {e}")
        return pd.DataFrame()


# Backwards compatibility function
def fetch_price_data(symbols: List[str], start_date: datetime, end_date: datetime, 
                    data_manager: DataSourceManager) -> Dict[str, pd.DataFrame]:
    """
    Backwards compatible function for the existing backtesting system.
    
    Args:
        symbols: List of symbols to fetch
        start_date: Start date as datetime
        end_date: End date as datetime
        data_manager: DataSourceManager instance
        
    Returns:
        Dictionary of {symbol: DataFrame}
    """
    # This function will be used by ETF mode, so route to ETF data source
    return data_manager.get_etf_data(symbols, start_date, end_date)