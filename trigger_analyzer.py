"""
ETF Strategy Trigger Analyzer

Scans symbols for strategy triggers on a given date by computing technical
indicators and evaluating setup/trigger conditions from Eversley's strategy PDFs.

Separate from trade_evaluator.py which evaluates known signals from input CSVs.
This module DETECTS the signals by implementing the indicator logic.

Usage:
  python trigger_analyzer.py                              # Scan 29 ETFs for today, save CSV
  python trigger_analyzer.py --symbols AAPL MSFT GOOGL    # Scan custom symbols
  python trigger_analyzer.py --date 2025-06-15            # Scan for historical date
  python trigger_analyzer.py --no-internals               # Exclude Put/Call & internals strategies
  python trigger_analyzer.py --output triggers.csv        # Custom output path
  python trigger_analyzer.py --no-save                    # Print to stdout only
  python trigger_analyzer.py --debug                      # Verbose indicator output
"""
import argparse
import json
import os
import sys
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional

from data_sources import DataSourceManager
from indicators import IndicatorEngine
from strategy_rules import StrategyRuleEngine, load_trigger_config
from trade_evaluator import lookup_win_rate, load_win_rates
from utilities import get_final_expiration_date


# Futures symbols (stop_price applicable; ETFs/stocks get "n/a")
FUTURES_SYMBOLS = {"6A", "6B", "6C", "6E", "6S", "ES", "NQ", "RTY", "YM"}

# Map trigger config strategy names → win_rates.json keys
WIN_RATE_STRATEGY_MAP = {
    "Donchian T-Lines Buy": "Donchian T-Lines Buy/Sell",
    "Donchian T-Lines Sell": "Donchian T-Lines Buy/Sell",
    "Weekly Swing Trend Buy Setup 1": "Weekly Swing Trend Buy",
    "Weekly Swing Trend Buy Setup 2": "Weekly Swing Trend Buy",
    "Squeeze Put/Call Buy": "Squeeze Put/Call Buy (Setup 1)",
    "Ichimoku Tenkan Hook Put/Call Buy": "Ichimoku Tenkan Hook Put/Call Buy (Setup 2)",
    "Ichimoku Tenkan Flat Put/Call Buy": "Ichimoku Tenkan Flat Put/Call Buy (Setup 3)",
    "Stochastics Hook Put/Call Buy": "Stochastics Hook Put/Call Buy (Setup 4)",
    "20.8 Trigger Lines Put/Call Buy": "20.8 Trigger Lines Put/Call Buy (Setup 5)",
    "Gap and Go Put/Call Buy": "Gap and Go Put/Call Buy (Setup 6)",
}

# Number of recent trading days to scan for triggers
SCAN_WINDOW_DAYS = 21

# Trade results columns matching trade_results_futures.csv column order
TRADE_RESULT_COLUMNS = [
    "symbol", "strategy", "direction", "win_rate", "signal_date", "status",
    "expiration", "target_type", "atr", "entry_price", "target_price",
    "stop_price", "last_close_price", "diff_from_entry", "entry_date",
    "exit_date", "num_days_open"
]

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

        # Load strategy formulas for target calculation (ATR length, multiplier, timeframe)
        strategies_path = os.path.join(os.path.dirname(__file__), 'strategies_complete.json')
        with open(strategies_path) as f:
            self._strategy_formulas = json.load(f)

        # External data (lazy-loaded only if internals enabled)
        self.external_data = {}

        # Data caches
        self.daily_ohlcv: Dict[str, pd.DataFrame] = {}
        self.weekly_ohlcv: Dict[str, pd.DataFrame] = {}
        self.last_trading_day: Optional[date] = None

        # Win rates (lazy-loaded on first lookup)
        self._win_rates = None

    def run(self) -> pd.DataFrame:
        """
        Execute full scan pipeline.

        Scans the last 21 trading days for triggers, then evaluates each
        triggered signal as a trade (entry, target, status tracking).

        Returns:
            DataFrame of triggered signals with trade evaluation columns.
        """
        print(f"\nETF Strategy Trigger Analyzer")
        print(f"{'='*60}")
        print(f"  Scan date:         {self.scan_date}")
        print(f"  Symbols:           {len(self.symbols)} ({', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''})")
        print(f"  Include internals: {self.include_internals}")
        print(f"  Lookback days:     {self.lookback_days}")
        print(f"  Debug:             {self.debug}")
        print(f"{'='*60}")

        # Step 1: Fetch OHLCV data
        print(f"\n[1/5] Fetching price data...")
        self._fetch_price_data()

        if not self.daily_ohlcv:
            print("No price data available. Check API key and symbol names.")
            return pd.DataFrame()

        # Step 2: Fetch external data if internals enabled
        if self.include_internals:
            print(f"\n[2/5] Fetching external data (CBOE, NYSE)...")
            self._fetch_external_data()
        else:
            print(f"\n[2/5] Skipping external data (use --include-internals to enable)")

        # Step 3: Scan last 21 trading days for triggers
        recent_dates = self._get_recent_trading_days(SCAN_WINDOW_DAYS)
        print(f"\n[3/5] Scanning {len(recent_dates)} trading days "
              f"({recent_dates[0]} to {recent_dates[-1]})...")

        all_triggers = []
        total_days = len(recent_dates)
        for day_idx, eval_date in enumerate(recent_dates, 1):
            print(f"  Day {day_idx}/{total_days}: {eval_date} "
                  f"({len(self.daily_ohlcv)} symbols)...", end='\r')

            # Compute indicators for this date
            indicators_for_date = {}
            for symbol in self.daily_ohlcv:
                daily = self.daily_ohlcv[symbol]
                weekly = self.weekly_ohlcv.get(symbol, pd.DataFrame())
                indicators_for_date[symbol] = self.indicator_engine.compute_all(
                    daily_df=daily,
                    weekly_df=weekly,
                    scan_date=eval_date
                )

            # Evaluate rules for this date
            triggers = self.rule_engine.evaluate_all(
                symbols=self.symbols,
                scan_date=eval_date,
                indicators=indicators_for_date,
                external_data=self.external_data,
                include_internals=self.include_internals,
                debug=(self.debug and eval_date == recent_dates[-1])
            )

            # Save ATR values with each trigger (indicators get overwritten per date)
            for t in triggers:
                sym = t['symbol']
                params = self._get_target_params(t['strategy'], t['direction'])
                atr_key = params['atr_key']
                atr_ind = indicators_for_date.get(sym, {}).get(atr_key)
                t['_atr_value'] = atr_ind.values.get('value') if atr_ind else None

            if triggers and self.debug:
                print(f"  {eval_date}: {len(triggers)} triggers")

            all_triggers.extend(triggers)

        print()  # Clear carriage return line
        print(f"  Raw triggers found: {len(all_triggers)}")

        # Deduplicate: same symbol+strategy on consecutive trading days = same signal
        all_triggers = self._dedup_consecutive(all_triggers, recent_dates)
        print(f"  After dedup: {len(all_triggers)} unique signals")

        # Step 4: Evaluate trades (entry, target, status)
        print(f"\n[4/5] Evaluating {len(all_triggers)} triggered trades...")
        self._evaluate_trades(all_triggers)

        # Step 5: Format output
        print(f"\n[5/5] Formatting results...")
        return self._format_output(all_triggers)

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

        # Determine the most recent trading day from fetched data
        if self.daily_ohlcv:
            last_dates = [df.index.max().date() for df in self.daily_ohlcv.values()]
            self.last_trading_day = max(last_dates)

        print(f"  Fetched data for {len(self.daily_ohlcv)}/{len(self.symbols)} symbols")
        if self.last_trading_day:
            print(f"  Most recent trading day: {self.last_trading_day}")

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

    def _get_recent_trading_days(self, num_days: int = SCAN_WINDOW_DAYS) -> List[date]:
        """Get the most recent N trading days from cached OHLCV data."""
        all_dates = set()
        for df in self.daily_ohlcv.values():
            all_dates.update(d.date() for d in df.index)
        return sorted(all_dates)[-num_days:]

    def _dedup_consecutive(self, triggers: List[dict],
                           trading_days: List[date]) -> List[dict]:
        """Remove consecutive-day duplicates for the same symbol+strategy.

        If the same (symbol, strategy, direction) triggers on consecutive
        trading days, keep only the first occurrence.  A gap of 2+ trading
        days resets and starts a new signal.
        """
        if not triggers:
            return triggers

        # Build a set for O(1) next-day lookup
        day_set = set(trading_days)
        day_to_next = {}
        for i in range(len(trading_days) - 1):
            day_to_next[trading_days[i]] = trading_days[i + 1]

        # Sort by date so we process in chronological order
        triggers.sort(key=lambda t: t['scan_date'])

        # Track last kept date per (symbol, strategy, direction)
        last_kept: Dict[tuple, date] = {}
        kept = []

        for t in triggers:
            key = (t['symbol'], t['strategy'], t['direction'])
            scan = date.fromisoformat(t['scan_date'])
            prev = last_kept.get(key)

            if prev is not None and day_to_next.get(prev) == scan:
                # Consecutive trading day — skip (same signal)
                continue

            last_kept[key] = scan
            kept.append(t)

        return kept

    def _evaluate_trades(self, triggers: List[dict]):
        """Evaluate entry, target, and status for each triggered signal."""
        for trigger in triggers:
            symbol = trigger['symbol']
            direction = trigger['direction']
            signal_date_val = date.fromisoformat(trigger['scan_date'])

            df = self.daily_ohlcv.get(symbol)
            if df is None or df.empty:
                self._set_na_trade_fields(trigger)
                continue

            # Get target params from strategies_complete.json
            strategy_name = trigger['strategy']
            params = self._get_target_params(strategy_name, direction)
            multiplier = params['multiplier']
            target_type = params['target_type']

            # ATR value saved during scanning loop
            atr_value = trigger.get('_atr_value')

            trigger['target_type'] = target_type
            trigger['atr'] = round(atr_value, 4) if atr_value else ''

            # Expiration: 3rd Friday, 2 months out
            expiration = get_final_expiration_date(signal_date_val, months_out=2)
            trigger['expiration'] = expiration.strftime('%m/%d/%y')

            # Stop price
            trigger['stop_price'] = '' if symbol in FUTURES_SYMBOLS else 'n/a'

            # Last close (most recent bar in data)
            last_close = round(float(df['Close'].iloc[-1]), 2)
            trigger['last_close_price'] = last_close

            # Win rate lookup
            trigger['win_rate'] = self._lookup_win_rate(
                trigger['strategy'], symbol, direction, target_type
            )

            # Find next trading day after signal for entry
            future_bars = df[df.index > pd.Timestamp(signal_date_val)]
            if future_bars.empty:
                # Most recent day trigger — no forward data
                trigger['signal_date'] = trigger['scan_date']
                trigger['entry_price'] = 'n/a'
                trigger['target_price'] = 'n/a'
                trigger['entry_date'] = 'n/a'
                trigger['exit_date'] = ''
                trigger['status'] = 'n/a'
                trigger['num_days_open'] = 'n/a'
                trigger['diff_from_entry'] = ''
                continue

            # Entry at next day's Open
            entry_idx = future_bars.index[0]
            entry_price = float(df.loc[entry_idx, 'Open'])
            entry_date_val = entry_idx.date()

            trigger['signal_date'] = trigger['scan_date']
            trigger['entry_price'] = round(entry_price, 2)
            trigger['entry_date'] = entry_date_val.isoformat()

            # Target price
            if atr_value and atr_value > 0:
                if direction == 'buy':
                    target_price = entry_price + (atr_value * multiplier)
                else:
                    target_price = entry_price - (atr_value * multiplier)
                trigger['target_price'] = round(target_price, 2)
            else:
                target_price = None
                trigger['target_price'] = ''

            # Track forward from entry date (inclusive) for target/expiration
            tracking_bars = df[df.index >= entry_idx]
            status = 'Open'
            exit_date_str = ''
            num_days_open = 'open'

            for i, (bar_ts, row) in enumerate(tracking_bars.iterrows()):
                bar_date = bar_ts.date()

                # Check expiration
                if bar_date >= expiration:
                    status = 'Expired'
                    exit_date_str = bar_date.isoformat()
                    num_days_open = i
                    break

                # Check target hit
                if target_price is not None:
                    if direction == 'buy' and row['High'] >= target_price:
                        status = 'Target Hit'
                        exit_date_str = bar_date.isoformat()
                        num_days_open = i
                        break
                    elif direction == 'sell' and row['Low'] <= target_price:
                        status = 'Target Hit'
                        exit_date_str = bar_date.isoformat()
                        num_days_open = i
                        break

            trigger['status'] = status
            trigger['exit_date'] = exit_date_str
            trigger['num_days_open'] = num_days_open

            # Diff from entry
            if direction == 'buy':
                trigger['diff_from_entry'] = round(last_close - entry_price, 2)
            else:
                trigger['diff_from_entry'] = round(entry_price - last_close, 2)

    def _set_na_trade_fields(self, trigger: dict):
        """Set trade fields to n/a when no OHLCV data available."""
        params = self._get_target_params(trigger['strategy'], trigger['direction'])
        trigger['signal_date'] = trigger['scan_date']
        trigger['target_type'] = params['target_type']
        trigger['atr'] = 'n/a'
        trigger['entry_price'] = 'n/a'
        trigger['target_price'] = 'n/a'
        trigger['stop_price'] = '' if trigger['symbol'] in FUTURES_SYMBOLS else 'n/a'
        trigger['entry_date'] = 'n/a'
        trigger['exit_date'] = ''
        trigger['status'] = 'n/a'
        trigger['num_days_open'] = 'n/a'
        trigger['expiration'] = ''
        trigger['last_close_price'] = ''
        trigger['diff_from_entry'] = ''
        trigger['win_rate'] = ''

    def _get_target_params(self, strategy_name: str, direction: str):
        """Resolve trigger config strategy name to target params from strategies_complete.json.

        Returns dict with keys: atr_length, multiplier, timeframe, target_type, atr_key.
        Uses same logic as trade_evaluator: 'Weekly' in name → exact match,
        otherwise → 'Daily ETF Options Buy/Sell'.
        """
        if "Weekly" in strategy_name:
            # Weekly strategies have exact matches (or map via WIN_RATE_STRATEGY_MAP)
            # Try exact match first, then mapped name
            formula_key = strategy_name
            if formula_key not in self._strategy_formulas:
                mapped = WIN_RATE_STRATEGY_MAP.get(strategy_name, strategy_name)
                if mapped in self._strategy_formulas:
                    formula_key = mapped
        else:
            formula_key = f"Daily ETF Options {direction.capitalize()}"

        strategy_def = self._strategy_formulas.get(formula_key, {})
        target_formula = strategy_def.get('target', {}).get('formula', {})

        atr_length = target_formula.get('atr_length', 5)
        multiplier = target_formula.get('multiplier', 1.0)
        timeframe = target_formula.get('timeframe', 'daily')

        # Build target type label matching _map_target_type_to_key expectations
        if timeframe == 'weekly':
            target_type = f"Weekly ATR{atr_length} x {multiplier}"
        else:
            target_type = f"ATR{atr_length} x {multiplier}"

        # ATR indicator key used during scanning
        if timeframe == 'weekly':
            atr_key = f"atr_{atr_length}_w"
        else:
            atr_key = f"atr_{atr_length}_d"

        return {
            'atr_length': atr_length,
            'multiplier': multiplier,
            'timeframe': timeframe,
            'target_type': target_type,
            'atr_key': atr_key,
        }

    def _lookup_win_rate(self, strategy_name: str, symbol: str,
                         direction: str, target_type: str) -> str:
        """Look up historical win rate from win_rates.json."""
        if self._win_rates is None:
            self._win_rates = load_win_rates()

        lookup_name = WIN_RATE_STRATEGY_MAP.get(strategy_name, strategy_name)
        return lookup_win_rate(
            self._win_rates, lookup_name, symbol, direction, target_type,
            mode='etfs'
        )

    def _format_output(self, triggers: List[dict]) -> pd.DataFrame:
        """Format trigger results into output DataFrame.

        Column order matches trade_results_futures.csv, with trigger-analyzer-
        specific columns (timeframe, scan_date, display indicators) appended.
        Trade columns are pre-populated by _evaluate_trades().
        """
        if not triggers:
            print(f"\n{'='*60}")
            print(f"No triggers found in last {SCAN_WINDOW_DAYS} trading days")
            return pd.DataFrame()

        df = pd.DataFrame(triggers)

        # Drop internal fields
        df = df.drop(columns=['_atr_value'], errors='ignore')

        # Ensure all trade-result columns exist (fallback for any missed fields)
        for col in TRADE_RESULT_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # Trigger-specific columns appended after trade-result columns
        trigger_cols = ['timeframe', 'scan_date']
        display_cols = [c for c in df.columns
                        if c not in TRADE_RESULT_COLUMNS
                        and c not in trigger_cols
                        and c != 'is_internals']

        df = df[TRADE_RESULT_COLUMNS + trigger_cols + display_cols]

        df = df.sort_values(['scan_date', 'strategy', 'symbol']).reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"Found {len(df)} triggers across last {SCAN_WINDOW_DAYS} trading days:")
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
        '--no-internals', action='store_true',
        help='Exclude Put/Call and internals strategies (enabled by default)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Print detailed indicator and rule evaluation output'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path (default: trigger_analyzer_results_<date>_ETFs.csv)'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Print to stdout instead of saving to CSV'
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
    import time
    t0 = time.time()

    args = parse_args()

    symbols = args.symbols or DEFAULT_ETF_SYMBOLS
    scan_date = (datetime.strptime(args.date, '%Y-%m-%d').date()
                 if args.date else date.today())

    include_internals = not args.no_internals

    analyzer = TriggerAnalyzer(
        symbols=symbols,
        scan_date=scan_date,
        include_internals=include_internals,
        debug=args.debug,
        lookback_days=args.lookback_days,
        config_path=args.config
    )

    results = analyzer.run()

    if results.empty:
        elapsed = time.time() - t0
        print(f"\nTotal execution time: {elapsed / 60:.1f} min")
        return

    if args.no_save:
        print(results.to_string(index=False))
    else:
        if args.output:
            output_path = args.output
        else:
            # Default filename using most recent trading day
            trade_date = analyzer.last_trading_day or scan_date
            suffix = "custom" if args.symbols else "ETFs"
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trigger_analyzer_results")
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, f"trigger_analyzer_results_{trade_date}_{suffix}.csv")
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

    elapsed = time.time() - t0
    print(f"\nTotal execution time: {elapsed / 60:.1f} min")


if __name__ == '__main__':
    main()
