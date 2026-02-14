"""
External data fetchers for market internals strategies.

Provides CBOE Total Exchange Put/Call Ratio and NYSE Advance/Decline data
with local CSV caching. Used only when --include-internals flag is set.

Data sources:
  - CBOE Put/Call: CBOE CDN API → local CSV fallback (data/cboe_putcall.csv)
  - NYSE A/D: Polygon indices API → local CSV fallback (data/nyse_ad.csv)
"""
import os
import json
import pandas as pd
import requests
from datetime import date, datetime, timedelta
from typing import Optional, Dict


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class CBOEPutCallFetcher:
    """
    Fetches CBOE Total Exchange Put/Call Ratio.

    Tries multiple sources in order:
      1. CBOE CDN API (JSON)
      2. Local CSV cache (data/cboe_putcall.csv)

    The local CSV can be populated manually from:
      https://www.barchart.com/stocks/quotes/$CPC/price-history/historical
    Format: Date,Open,High,Low,Close (where Close = the P/C ratio)
    """

    CBOE_CDN_URL = "https://cdn.cboe.com/api/global/us_options/market_statistics/daily/"
    CACHE_FILE = os.path.join(DATA_DIR, 'cboe_putcall.csv')
    JSON_CACHE = os.path.join(DATA_DIR, 'cboe_putcall_cache.json')

    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._load_json_cache()

    def _load_json_cache(self):
        """Load previously fetched values from JSON cache."""
        if os.path.exists(self.JSON_CACHE):
            try:
                with open(self.JSON_CACHE, 'r') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save_json_cache(self):
        """Save fetched values to JSON cache."""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(self.JSON_CACHE, 'w') as f:
            json.dump(self._cache, f, indent=2)

    def fetch(self, target_date: date) -> Optional[float]:
        """
        Get the Total Exchange Put/Call Ratio for a given date.

        Returns:
            Float ratio (e.g., 0.92) or None if unavailable.
        """
        date_str = target_date.strftime('%Y-%m-%d')

        # Check in-memory cache
        if date_str in self._cache:
            return self._cache[date_str]

        # Try CBOE CDN API
        ratio = self._try_cboe_api(target_date)
        if ratio is not None:
            self._cache[date_str] = ratio
            self._save_json_cache()
            return ratio

        # Try local CSV
        ratio = self._try_local_csv(target_date)
        if ratio is not None:
            self._cache[date_str] = ratio
            self._save_json_cache()
            return ratio

        return None

    def fetch_range(self, start_date: date, end_date: date) -> Dict[str, float]:
        """Fetch P/C ratios for a date range. Returns {date_str: ratio}."""
        result = {}
        # Try local CSV first for bulk data
        csv_data = self._load_csv_data()
        if csv_data is not None:
            for d, ratio in csv_data.items():
                dt = datetime.strptime(d, '%Y-%m-%d').date()
                if start_date <= dt <= end_date:
                    result[d] = ratio
                    self._cache[d] = ratio

        # Fill gaps from API
        current = start_date
        while current <= end_date:
            ds = current.strftime('%Y-%m-%d')
            if ds not in result:
                ratio = self._try_cboe_api(current)
                if ratio is not None:
                    result[ds] = ratio
                    self._cache[ds] = ratio
            current += timedelta(days=1)

        self._save_json_cache()
        return result

    def _try_cboe_api(self, target_date: date) -> Optional[float]:
        """Try fetching from CBOE CDN API."""
        try:
            url = f"{self.CBOE_CDN_URL}?dt={target_date.strftime('%Y-%m-%d')}"
            resp = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json'
            })
            if resp.status_code == 200:
                data = resp.json()
                # Parse CBOE JSON response for Total P/C ratio
                if isinstance(data, dict) and 'data' in data:
                    for row in data['data']:
                        if row.get('PRODUCT_TYPE', '').lower() == 'total':
                            return float(row.get('P_C_RATIO', 0))
                # Alternative format
                if isinstance(data, list):
                    for row in data:
                        if 'total' in str(row).lower():
                            for key in ['P_C_RATIO', 'pc_ratio', 'ratio']:
                                if key in row:
                                    return float(row[key])
        except Exception:
            pass
        return None

    def _try_local_csv(self, target_date: date) -> Optional[float]:
        """Try reading from local CSV file."""
        csv_data = self._load_csv_data()
        if csv_data is not None:
            date_str = target_date.strftime('%Y-%m-%d')
            return csv_data.get(date_str)
        return None

    def _load_csv_data(self) -> Optional[Dict[str, float]]:
        """Load and parse local CSV file."""
        if not os.path.exists(self.CACHE_FILE):
            return None
        try:
            df = pd.read_csv(self.CACHE_FILE)
            # Try common column name patterns
            date_col = None
            ratio_col = None
            for col in df.columns:
                cl = col.lower().strip()
                if cl in ('date', 'time', 'datetime'):
                    date_col = col
                if cl in ('close', 'last', 'p/c ratio', 'pc_ratio', 'ratio', 'value'):
                    ratio_col = col

            if date_col is None:
                date_col = df.columns[0]
            if ratio_col is None:
                ratio_col = df.columns[-1]

            result = {}
            for _, row in df.iterrows():
                try:
                    d = pd.Timestamp(row[date_col]).strftime('%Y-%m-%d')
                    result[d] = float(row[ratio_col])
                except (ValueError, TypeError):
                    continue
            return result
        except Exception:
            return None


class NYSEAdvanceDeclineFetcher:
    """
    Fetches NYSE Advance/Decline data for computing cumulative A/D line.

    Used by Weekly Ichimoku Squeeze Sell strategy:
      Advance_Decline_Cumulative_Average(NYSE, 5)
      Signal: CumulAD crosses below AvgCumulAD (bearish cross)

    Sources:
      1. Polygon indices API (I:ADV, I:DEC tickers)
      2. Local CSV cache (data/nyse_ad.csv)

    The local CSV can be populated from Barchart $ADDN or Yahoo Finance C:ISSU.
    Format: Date,Advances,Declines (or Date,Value where Value = advances - declines)
    """

    CACHE_FILE = os.path.join(DATA_DIR, 'nyse_ad.csv')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

    def __init__(self):
        self._ad_series: Optional[pd.Series] = None

    def fetch_ad_line(self, start_date: date, end_date: date,
                      avg_period: int = 5) -> Optional[Dict]:
        """
        Compute cumulative A/D line and its moving average.

        Returns:
            Dict with 'cumul_ad' Series, 'avg_cumul_ad' Series,
            and point-in-time values for the end_date.
        """
        # Try Polygon API
        ad_diff = self._try_polygon(start_date, end_date)

        # Fall back to local CSV
        if ad_diff is None or ad_diff.empty:
            ad_diff = self._try_local_csv(start_date, end_date)

        if ad_diff is None or ad_diff.empty:
            return None

        # Compute cumulative A/D line
        cumul_ad = ad_diff.cumsum()
        avg_cumul_ad = cumul_ad.rolling(avg_period).mean()

        # Get values at end_date
        target = pd.Timestamp(end_date)
        mask = cumul_ad.index <= target
        if not mask.any():
            return None

        idx = mask.values.nonzero()[0][-1]
        prev_idx = idx - 1 if idx > 0 else None

        cumul_val = cumul_ad.iloc[idx]
        avg_val = avg_cumul_ad.iloc[idx] if not pd.isna(avg_cumul_ad.iloc[idx]) else None
        cumul_prev = cumul_ad.iloc[prev_idx] if prev_idx is not None else None
        avg_prev = avg_cumul_ad.iloc[prev_idx] if prev_idx is not None and not pd.isna(avg_cumul_ad.iloc[prev_idx]) else None

        # Bearish cross: CumulAD was above AvgCumulAD, now below
        bearish_cross = False
        if (cumul_val is not None and avg_val is not None and
                cumul_prev is not None and avg_prev is not None):
            bearish_cross = (cumul_val < avg_val and cumul_prev >= avg_prev)

        return {
            'cumul_ad': cumul_val,
            'avg_cumul_ad': avg_val,
            'cumul_ad_prev': cumul_prev,
            'avg_cumul_ad_prev': avg_prev,
            'bearish_cross': bearish_cross,
            'cumul_ad_series': cumul_ad,
            'avg_cumul_ad_series': avg_cumul_ad,
        }

    def _try_polygon(self, start_date: date, end_date: date) -> Optional[pd.Series]:
        """Try fetching from Polygon/Massive indices API."""
        if not self.POLYGON_API_KEY:
            return None
        try:
            # Try common NYSE A/D tickers in Polygon
            for ticker in ['I:NYAD', 'I:ADV', 'I:ADVN']:
                url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                       f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                       f"?adjusted=true&apiKey={self.POLYGON_API_KEY}")
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('results'):
                        # Found data, return as Series
                        dates = [pd.Timestamp(r['t'], unit='ms') for r in data['results']]
                        values = [r['c'] for r in data['results']]
                        return pd.Series(values, index=dates)
        except Exception:
            pass
        return None

    def _try_local_csv(self, start_date: date, end_date: date) -> Optional[pd.Series]:
        """Load from local CSV. Expects Date,Advances,Declines or Date,Value columns."""
        if not os.path.exists(self.CACHE_FILE):
            return None
        try:
            df = pd.read_csv(self.CACHE_FILE, parse_dates=[0], index_col=0)
            df = df.sort_index()

            if 'Advances' in df.columns and 'Declines' in df.columns:
                ad_diff = df['Advances'] - df['Declines']
            elif 'Value' in df.columns:
                ad_diff = df['Value']
            elif len(df.columns) == 1:
                ad_diff = df.iloc[:, 0]
            else:
                return None

            # Filter to date range
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            ad_diff = ad_diff[(ad_diff.index >= start_ts) & (ad_diff.index <= end_ts)]
            return ad_diff
        except Exception:
            return None


def fetch_external_data(scan_date: date, lookback_days: int = 365) -> Dict:
    """
    Fetch all external data needed for internals strategies.

    Returns dict with:
      'put_call_ratio': {date_str: ratio} for recent dates
      'nyse_ad': dict with A/D line values
    """
    result = {}
    start_date = scan_date - timedelta(days=lookback_days)

    # CBOE Put/Call Ratio
    pc_fetcher = CBOEPutCallFetcher()
    pc_data = {}
    for offset in range(5):  # Fetch signal day + 2 previous days + buffer
        d = scan_date - timedelta(days=offset)
        ratio = pc_fetcher.fetch(d)
        if ratio is not None:
            pc_data[d.strftime('%Y-%m-%d')] = ratio

    result['put_call_ratio'] = pc_data

    # NYSE A/D Line
    ad_fetcher = NYSEAdvanceDeclineFetcher()
    ad_data = ad_fetcher.fetch_ad_line(start_date, scan_date, avg_period=5)
    result['nyse_ad'] = ad_data or {}

    return result
