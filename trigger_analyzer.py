"""
ETF Strategy Trigger Analyzer

Scans symbols for strategy triggers on a given date by computing technical
indicators and evaluating setup/trigger conditions from Eversley's strategy PDFs.

Separate from trade_evaluator.py which evaluates known signals from input CSVs.
This module DETECTS the signals by implementing the indicator logic.

Usage:
  python trigger_analyzer.py                              # Scan 29 ETFs for today
  python trigger_analyzer.py --symbols AAPL MSFT GOOGL    # Scan custom symbols
  python trigger_analyzer.py --date 2025-06-15            # Scan for historical date
  python trigger_analyzer.py --include-internals          # Include Put/Call strategies
  python trigger_analyzer.py --output triggers.csv        # Save to CSV
  python trigger_analyzer.py --debug                      # Verbose indicator output
"""
import argparse
import os
import sys
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional

from data_sources import DataSourceManager
from indicators import IndicatorEngine
from strategy_rules import StrategyRuleEngine, load_trigger_config


# Eversley's 29 ETF symbols
DEFAULT_ETF_SYMBOLS = [
    "DIA", "EEM", "EFA", "EWG", "EWH", "FXI", "GDX", "GLD", "IBB", "IWM",
    "IYR", "QQQ", "SLV", "SPY", "TLT", "UNG", "USO", "VNQ", "VWO", "XHB",
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLV", "XME", "XOP", "XRT"
]

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')


class TriggerAnalyzer:
    """
    Main orchestrator for ETF trigger scanning.

    Designed for composability: Task 2 backtester can instantiate this class
    and call run() programmatically with different dates/symbols.
    """

    def __init__(self, symbols: List[str], scan_date: date,
                 include_internals: bool = False,
                 lookback_days: int = 365,
                 debug: bool = False,
                 config_path: str = None):
        self.symbols = symbols
        self.scan_date = scan_date
        self.include_internals = include_internals
        self.lookback_days = lookback_days
        self.debug = debug

        # Initialize components
        self.data_manager = DataSourceManager(polygon_api_key=POLYGON_API_KEY)
        self.indicator_engine = IndicatorEngine()
        self.rule_engine = StrategyRuleEngine(load_trigger_config(config_path))

        # External data (lazy-loaded only if internals enabled)
        self.external_data = {}

        # Data caches
        self.daily_ohlcv: Dict[str, pd.DataFrame] = {}
        self.weekly_ohlcv: Dict[str, pd.DataFrame] = {}
        self.indicators: Dict[str, Dict] = {}

    def run(self) -> pd.DataFrame:
        """
        Execute full scan pipeline.

        Returns:
            DataFrame of triggered signals with indicator values.
        """
        print(f"\nETF Strategy Trigger Analyzer")
        print(f"Scan date: {self.scan_date}")
        print(f"Symbols: {len(self.symbols)}")
        print(f"Include internals: {self.include_internals}")
        print(f"{'='*60}")

        # Step 1: Fetch OHLCV data
        print(f"\n[1/4] Fetching price data...")
        self._fetch_price_data()

        if not self.daily_ohlcv:
            print("No price data available. Check API key and symbol names.")
            return pd.DataFrame()

        # Step 2: Compute indicators
        print(f"\n[2/4] Computing indicators for {len(self.daily_ohlcv)} symbols...")
        self._compute_all_indicators()

        # Step 3: Fetch external data if internals enabled
        if self.include_internals:
            print(f"\n[3/4] Fetching external data (CBOE, NYSE)...")
            self._fetch_external_data()
        else:
            print(f"\n[3/4] Skipping external data (use --include-internals to enable)")

        # Step 4: Evaluate strategy rules
        print(f"\n[4/4] Evaluating strategy rules...")
        triggers = self.rule_engine.evaluate_all(
            symbols=self.symbols,
            scan_date=self.scan_date,
            indicators=self.indicators,
            external_data=self.external_data,
            include_internals=self.include_internals,
            debug=self.debug
        )

        # Format output
        return self._format_output(triggers)

    def _fetch_price_data(self):
        """Fetch daily OHLCV and resample to weekly for all symbols."""
        start_date = datetime.combine(
            self.scan_date - timedelta(days=self.lookback_days),
            datetime.min.time()
        )
        end_date = datetime.combine(
            self.scan_date + timedelta(days=1),
            datetime.min.time()
        )

        for symbol in self.symbols:
            try:
                df = self.data_manager.get_etf_data(
                    symbol, start_date, end_date, source='polygon'
                )

                if isinstance(df, dict):
                    df = df.get(symbol, pd.DataFrame())

                if df is None or df.empty:
                    print(f"  WARNING: No data for {symbol}, skipping")
                    continue

                self.daily_ohlcv[symbol] = df

                # Resample to weekly (Friday close)
                agg_dict = {
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }
                if 'Volume' in df.columns:
                    agg_dict['Volume'] = 'sum'

                df_weekly = df.resample('W-FRI').agg(agg_dict).dropna(
                    subset=['Open', 'High', 'Low', 'Close']
                )
                self.weekly_ohlcv[symbol] = df_weekly

            except Exception as e:
                print(f"  ERROR fetching {symbol}: {e}")

        print(f"  Fetched data for {len(self.daily_ohlcv)}/{len(self.symbols)} symbols")

    def _compute_all_indicators(self):
        """Compute all needed indicators for each symbol."""
        for symbol in self.daily_ohlcv:
            daily = self.daily_ohlcv[symbol]
            weekly = self.weekly_ohlcv.get(symbol, pd.DataFrame())

            if self.debug:
                print(f"\n  Computing indicators for {symbol}...")
                print(f"    Daily: {len(daily)} bars, {daily.index.min().date()} to {daily.index.max().date()}")
                if not weekly.empty:
                    print(f"    Weekly: {len(weekly)} bars")

            self.indicators[symbol] = self.indicator_engine.compute_all(
                daily_df=daily,
                weekly_df=weekly,
                scan_date=self.scan_date
            )

        print(f"  Computed indicators for {len(self.indicators)} symbols")

    def _fetch_external_data(self):
        """Fetch CBOE Put/Call ratio and NYSE A/D data."""
        from external_data import fetch_external_data
        self.external_data = fetch_external_data(self.scan_date)

        pc = self.external_data.get('put_call_ratio', {})
        if pc:
            print(f"  Put/Call ratios: {len(pc)} dates fetched")
            for d in sorted(pc.keys(), reverse=True)[:3]:
                print(f"    {d}: {pc[d]}")
        else:
            print("  Put/Call ratios: No data available")

        ad = self.external_data.get('nyse_ad', {})
        if ad and 'bearish_cross' in ad:
            print(f"  NYSE A/D: cumul={ad.get('cumul_ad'):.1f}, "
                  f"avg={ad.get('avg_cumul_ad'):.1f}, "
                  f"bearish_cross={ad.get('bearish_cross')}")
        else:
            print("  NYSE A/D: No data available")

    def _format_output(self, triggers: List[dict]) -> pd.DataFrame:
        """Format trigger results into output DataFrame."""
        if not triggers:
            print(f"\n{'='*60}")
            print(f"No triggers found for {self.scan_date}")
            return pd.DataFrame()

        df = pd.DataFrame(triggers)

        # Reorder columns: key columns first, then display indicators
        key_cols = ['symbol', 'strategy', 'direction', 'timeframe', 'scan_date']
        other_cols = [c for c in df.columns if c not in key_cols and c != 'is_internals']
        df = df[key_cols + other_cols]

        df = df.sort_values(['strategy', 'symbol']).reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"Found {len(df)} triggers for {self.scan_date}:")
        print(f"{'='*60}")

        return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="ETF Strategy Trigger Analyzer — scans symbols for strategy triggers"
    )
    parser.add_argument(
        '--symbols', nargs='+', default=None,
        help='Symbols to scan (default: Eversley 29 ETFs)'
    )
    parser.add_argument(
        '--date', type=str, default=None,
        help='Scan date YYYY-MM-DD (default: today)'
    )
    parser.add_argument(
        '--include-internals', action='store_true',
        help='Include Put/Call and internals strategies (requires CBOE data)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Print detailed indicator and rule evaluation output'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path (default: print to stdout)'
    )
    parser.add_argument(
        '--lookback-days', type=int, default=730,
        help='Days of historical data to fetch for indicator computation (default: 730)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to trigger config JSON (default: etf_trigger_config.json)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    symbols = args.symbols or DEFAULT_ETF_SYMBOLS
    scan_date = (datetime.strptime(args.date, '%Y-%m-%d').date()
                 if args.date else date.today())

    analyzer = TriggerAnalyzer(
        symbols=symbols,
        scan_date=scan_date,
        include_internals=args.include_internals,
        debug=args.debug,
        lookback_days=args.lookback_days,
        config_path=args.config
    )

    results = analyzer.run()

    if results.empty:
        return

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    else:
        print(results.to_string(index=False))


if __name__ == '__main__':
    main()
