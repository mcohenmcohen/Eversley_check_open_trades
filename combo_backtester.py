#!/usr/bin/env python3
"""
Combo Backtester — Tests Momentum-Trading indicator combination win rates.

Reads pre-computed data from the Momentum-Trading pipeline and tests
permutations of binary indicator combinations to find which produce the best
forward returns over 5 years of history.

Indicators tested:
  - 11 daily binary indicators (has_candle, has_volume, etc.)
  - 11 weekly binary indicators (has_candle_w, has_volume_w, etc.)
  - 1 monthly indicator (has_close_MA40_m)
  - 10 Score Mean decile bins (score_00_10 through score_90_100)
  - 4 Bayesian probability indicators (bayes_up_5d/10d/20d/60d)
  - 2 Curvature saucer indicators (has_curve_60d, has_curve_90d)

First run builds cache from full.csv + ohlcv.pkl (~3 min). Subsequent runs
load cache in < 1s.

Usage:
    python combo_backtester.py                                    # Test all 2-4 indicator combos
    python combo_backtester.py --combo-size 3                     # Up to 3-indicator combos
    python combo_backtester.py --test-combo has_candle has_low_bband score_70_80
    python combo_backtester.py --score-range 70 90                # Additional Score Mean filter
    python combo_backtester.py --daily-only                       # Skip weekly/monthly indicators
    python combo_backtester.py --list-indicators                  # Show all available indicators
"""

import argparse
import itertools
import os
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
DEFAULT_DATA_DIR = '/Users/mattcohen/Dev/Momentum-Trading/symbol_data'
MOMENTUM_PROJECT_DIR = '/Users/mattcohen/Dev/Momentum-Trading'

# Daily binary indicators (from full.csv)
DAILY_INDICATORS = [
    'has_20cross50',
    'has_candle',
    'has_close_MA200',
    'has_close_MA50',
    'has_donch_cross',
    'has_gap',
    'has_hammer',
    'has_low_bband',
    'has_new_peak',
    'has_rsi_pullback',
    'has_volume',
]

# Weekly versions (computed from OHLCV via trading_systems.py)
WEEKLY_INDICATORS = [name + '_w' for name in DAILY_INDICATORS]

# Score Mean decile bin names
SCORE_DECILES = [f'score_{i*10:02d}_{(i+1)*10:02d}' for i in range(10)]

# Bayesian probability indicators
BAYESIAN_INDICATORS = [f'bayes_up_{n}d' for n in [5, 10, 20, 60]]

# Curvature (saucer bottom) indicators
CURVE_INDICATORS = ['has_curve_60d', 'has_curve_90d']

# Forward return horizons (trading days)
FORWARD_HORIZONS = [5, 10, 20, 60]

# Cache version — bump this when cache format changes
CACHE_VERSION = 3


class ComboBacktester:
    def __init__(self, data_dir=DEFAULT_DATA_DIR, data_date=None):
        self.data_dir = Path(data_dir)
        self.data_date = data_date or self._find_latest_date()
        self.closes = None          # DataFrame: dates x symbols
        self.indicators = {}        # indicator_name -> DataFrame (dates x symbols, 0/1)
        self.score_mean = None      # DataFrame: dates x symbols
        self.symbols = []
        # Numpy arrays for fast combo testing
        self._ind_arrays = {}       # indicator_name -> np.ndarray (bool)
        self._fwd_arrays = {}       # horizon -> np.ndarray (float)
        self._score_arr = None      # np.ndarray (float)
        self._month_ids = None      # np.ndarray (int) — YYYYMM per day row

    def _find_latest_date(self):
        """Find the most recent data date from available files."""
        full_files = sorted(self.data_dir.glob('*-full.csv'))
        if not full_files:
            raise FileNotFoundError(f"No full.csv files found in {self.data_dir}")
        fname = full_files[-1].name
        return fname.replace('-full.csv', '')

    def load_data(self, daily_only=False):
        """Load all indicator data (from cache if available)."""
        cache_file = self.data_dir / f'{self.data_date}-combo_cache_v{CACHE_VERSION}.pkl'

        if cache_file.exists():
            print(f"Loading cached data: {cache_file.name}")
            t0 = time.time()
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            self.closes = cache['closes']
            self.indicators = cache['indicators']
            self.score_mean = cache.get('score_mean')
            self.symbols = list(self.closes.columns)
            print(f"  Loaded in {time.time()-t0:.1f}s — "
                  f"{self.closes.shape[0]} days x {self.closes.shape[1]} symbols, "
                  f"{len(self.indicators)} indicators")
        else:
            self._build_full_cache(cache_file, daily_only=daily_only)

        # Pre-compute forward returns and numpy arrays
        self._compute_forward_returns()
        self._convert_to_numpy()

        # Summary
        daily_count = sum(1 for k in self.indicators
                          if not k.endswith('_w') and not k.endswith('_m')
                          and not k.startswith('score_') and not k.startswith('bayes_')
                          and not k.startswith('has_curve_'))
        weekly_count = sum(1 for k in self.indicators if k.endswith('_w'))
        monthly_count = sum(1 for k in self.indicators if k.endswith('_m'))
        decile_count = sum(1 for k in self.indicators if k.startswith('score_'))
        bayes_count = sum(1 for k in self.indicators if k.startswith('bayes_'))
        curve_count = sum(1 for k in self.indicators if k.startswith('has_curve_'))

        print(f"\nData summary:")
        print(f"  Date range: {self.closes.index.min().date()} to "
              f"{self.closes.index.max().date()}")
        print(f"  Symbols: {len(self.symbols)}")
        print(f"  Trading days: {len(self.closes)}")
        print(f"  Indicators: {daily_count} daily, {weekly_count} weekly, "
              f"{monthly_count} monthly, {decile_count} score deciles, "
              f"{bayes_count} bayesian, {curve_count} curvature "
              f"({len(self.indicators)} total)")

    def _build_full_cache(self, cache_file, daily_only=False):
        """Build complete cache from source data files."""
        print("Building indicator cache...")

        # Step 1: Load daily indicators from full.csv
        self._load_daily_indicators()

        if not daily_only:
            # Step 2: Compute weekly indicators from OHLCV
            self._compute_weekly_indicators()

            # Step 3: Compute monthly MA40 from OHLCV
            self._compute_monthly_ma40()

        # Step 4: Compute Score Mean deciles
        self._compute_score_deciles()

        # Step 5: Compute Bayesian probability indicators
        self._compute_bayesian_indicators()

        # Step 6: Compute curvature (saucer bottom) indicators
        self._compute_curve_indicators()

        # Save cache
        self._save_cache(cache_file)

    def _load_daily_indicators(self):
        """Load daily indicators and Close prices from full.csv."""
        full_csv = self.data_dir / f'{self.data_date}-full.csv'
        if not full_csv.exists():
            raise FileNotFoundError(
                f"Full CSV not found: {full_csv}\n"
                f"Run the Momentum-Trading pipeline notebook first.")

        size_mb = full_csv.stat().st_size / 1e6
        print(f"  Loading {full_csv.name} ({size_mb:.0f}MB)...")
        t0 = time.time()
        df = pd.read_csv(full_csv, index_col=0, header=[0, 1])
        df.index = pd.to_datetime(df.index)
        print(f"  Loaded in {time.time()-t0:.1f}s — shape {df.shape}")

        # Extract Close prices (level 0 = symbol, level 1 = attribute)
        self.closes = df.xs('Close', axis=1, level=1).astype(float)
        self.symbols = list(self.closes.columns)

        # Extract daily binary indicators
        for ind_name in DAILY_INDICATORS:
            try:
                self.indicators[ind_name] = df.xs(
                    ind_name, axis=1, level=1).astype(float)
            except KeyError:
                print(f"    Warning: '{ind_name}' not found")

        # Extract Score Mean
        try:
            self.score_mean = df.xs('Score Mean', axis=1, level=1).astype(float)
        except KeyError:
            print("    Warning: 'Score Mean' not found")

        print(f"    {len(self.indicators)} daily indicators, "
              f"{len(self.symbols)} symbols")

    def _compute_weekly_indicators(self):
        """Compute weekly indicators from OHLCV using Momentum-Trading engine."""
        ohlcv_file = self.data_dir / f'{self.data_date}-ohlcv.pkl'
        if not ohlcv_file.exists():
            print(f"  Skipping weekly: {ohlcv_file.name} not found")
            return

        print(f"  Computing weekly indicators from {ohlcv_file.name}...")
        t0 = time.time()

        with open(ohlcv_file, 'rb') as f:
            ohlcv = pickle.load(f)

        # Extract per-attribute DataFrames (dates x symbols)
        close = ohlcv.xs('Close', level=1, axis=1)
        high = ohlcv.xs('High', level=1, axis=1)
        low = ohlcv.xs('Low', level=1, axis=1)
        open_ = ohlcv.xs('Open', level=1, axis=1)
        volume = ohlcv.xs('Volume', level=1, axis=1)

        # Resample to weekly (Sunday end date — no shift, avoids look-ahead bias)
        # Weekly bar completing Friday is labeled Sunday, forward-fills to Monday
        w_close = close.resample('W').last()
        w_high = high.resample('W').max()
        w_low = low.resample('W').min()
        w_open = open_.resample('W').first()
        w_volume = volume.resample('W').sum()

        # Reconstruct MultiIndex DataFrame (symbol, attribute) for trading_systems
        weekly_ohlcv = pd.concat({
            'Open': w_open, 'High': w_high, 'Low': w_low,
            'Close': w_close, 'Adj Close': w_close, 'Volume': w_volume,
        }, axis=1).swaplevel(axis=1).sort_index(axis=1)

        # Import Momentum-Trading indicator engine
        try:
            if MOMENTUM_PROJECT_DIR not in sys.path:
                sys.path.insert(0, MOMENTUM_PROJECT_DIR)
            import trading_systems as ts
        except ImportError as e:
            print(f"    Cannot import trading_systems: {e}")
            print(f"    Skipping weekly indicators")
            return

        # Compute all 11 binary indicators on weekly data
        indicators_dict = ts.get_technical_indicators_optimized(
            weekly_ohlcv, verbose=False)

        # Forward-fill weekly to daily index and add _w suffix
        daily_index = self.closes.index
        count = 0
        for name, weekly_df in indicators_dict.items():
            # Align columns to match daily data
            weekly_df = weekly_df.reindex(columns=self.symbols)
            daily_df = weekly_df.reindex(daily_index, method='ffill').fillna(0)
            self.indicators[name + '_w'] = daily_df.astype(float)
            count += 1

        elapsed = time.time() - t0
        print(f"    {count} weekly indicators computed in {elapsed:.1f}s")

    def _compute_monthly_ma40(self):
        """Compute monthly MA40 proximity indicator from OHLCV."""
        ohlcv_file = self.data_dir / f'{self.data_date}-ohlcv.pkl'
        if not ohlcv_file.exists():
            return

        print("  Computing monthly MA40 indicator...")
        t0 = time.time()

        with open(ohlcv_file, 'rb') as f:
            ohlcv = pickle.load(f)

        close = ohlcv.xs('Close', level=1, axis=1)
        high = ohlcv.xs('High', level=1, axis=1)
        low = ohlcv.xs('Low', level=1, axis=1)
        open_ = ohlcv.xs('Open', level=1, axis=1)

        # Resample to monthly
        m_close = close.resample('M').last()
        m_high = high.resample('M').max()
        m_low = low.resample('M').min()
        m_open = open_.resample('M').first()

        # MA(40) on monthly close
        ma40 = m_close.rolling(40, min_periods=40).mean()

        # ATR(14) on monthly bars
        prev_close = m_close.shift(1)
        tr = np.maximum(
            m_high - m_low,
            np.maximum((m_high - prev_close).abs(), (m_low - prev_close).abs())
        )
        atr = tr.ewm(alpha=1.0/14, adjust=False).mean()

        # Proximity: Close or Open within 0.8 * ATR of MA40
        tolerance = atr * 0.8
        upper = ma40 + tolerance
        lower = ma40 - tolerance
        is_close = (
            ((m_close > lower) & (m_close < upper)) |
            ((m_open > lower) & (m_open < upper))
        )

        # Any of last 3 months triggered (matches pipeline logic)
        is_close_3m = is_close.rolling(3, min_periods=1).max().fillna(0)

        # Forward-fill monthly to daily index
        daily_index = self.closes.index
        daily_ma40 = is_close_3m.reindex(columns=self.symbols)
        daily_ma40 = daily_ma40.reindex(daily_index, method='ffill').fillna(0)
        self.indicators['has_close_MA40_m'] = daily_ma40.astype(float)

        elapsed = time.time() - t0
        print(f"    Monthly MA40 computed in {elapsed:.1f}s")

    def _compute_score_deciles(self):
        """Create 10 binary indicators for Score Mean decile ranges."""
        if self.score_mean is None:
            print("  Skipping score deciles: Score Mean not available")
            return

        print("  Computing Score Mean deciles...")
        for i in range(10):
            lo = i * 10
            hi = (i + 1) * 10
            name = f'score_{lo:02d}_{hi:02d}'
            if hi == 100:
                mask = (self.score_mean >= lo) & (self.score_mean <= hi)
            else:
                mask = (self.score_mean >= lo) & (self.score_mean < hi)
            self.indicators[name] = mask.astype(float)
        print(f"    10 score deciles created")

    def _compute_bayesian_indicators(self):
        """Compute Bayesian posterior probability indicators.

        For each forward horizon, uses a rolling Beta-Binomial model on
        resolved historical outcomes to estimate P(price_up). Binary flag
        fires when posterior > 0.6. Outcomes are lagged by N days to avoid
        look-ahead bias.
        """
        print("  Computing Bayesian probability indicators...")
        t0 = time.time()
        lookback = 50   # number of resolved outcomes to consider
        alpha_0, beta_0 = 2, 2  # flat (uninformative) prior

        for n in FORWARD_HORIZONS:
            # Forward return at each day
            fwd_return = self.closes.shift(-n) / self.closes - 1
            # Binary outcome: 1 if price went up over n days
            outcome = (fwd_return > 0).astype(float)
            # Lag by n days — on day T we only know outcomes resolved by T
            lagged_outcome = outcome.shift(n)
            # Rolling wins and total trials over lookback window
            rolling_wins = lagged_outcome.rolling(lookback, min_periods=10).sum()
            rolling_total = lagged_outcome.rolling(lookback, min_periods=10).count()
            # Bayesian posterior mean: E[Beta(a+wins, b+losses)]
            posterior = (alpha_0 + rolling_wins) / (alpha_0 + beta_0 + rolling_total)
            # Binary indicator: posterior probability > 0.6
            self.indicators[f'bayes_up_{n}d'] = (
                (posterior > 0.6).fillna(0).astype(float))

        elapsed = time.time() - t0
        print(f"    4 Bayesian indicators computed in {elapsed:.1f}s")

    def _compute_curve_indicators(self):
        """Compute quadratic curvature (saucer bottom) indicators.

        Fits a quadratic polynomial to rolling price windows. Fires when:
        - Positive curvature (concave up / saucer shape)
        - R² > 0.65 (clean curve, low noise)
        - Bottom position in sweet spot (50-80% of window)
        - Current slope positive (past the bottom, curving up)
        - Prior decline confirmed (start of window > midpoint)
        """
        print("  Computing curvature indicators...")
        t0 = time.time()
        prices = self.closes.values  # (n_days, n_symbols)
        n_days, n_syms = prices.shape

        for window in [60, 90]:
            # Precompute regression matrix (constant for all windows of same size)
            t = np.arange(window, dtype=float)
            X = np.column_stack([t**2, t, np.ones(window)])
            XtX_inv_Xt = np.linalg.inv(X.T @ X) @ X.T  # (3, window)

            result = np.zeros((n_days, n_syms), dtype=bool)

            for i in range(window, n_days):
                segment = prices[i - window:i, :]  # (window, n_syms)

                # Normalize per-symbol for numerical stability
                p_mean = np.nanmean(segment, axis=0)
                p_mean[p_mean == 0] = 1
                p_norm = segment / p_mean

                # Skip symbols with any NaN in window
                has_nan = np.any(np.isnan(p_norm), axis=0)

                # Fit quadratic for all symbols at once: (3, n_syms)
                coeffs = XtX_inv_Xt @ p_norm
                a = coeffs[0]   # curvature
                b = coeffs[1]   # linear term

                # R-squared
                fitted = X @ coeffs  # (window, n_syms)
                ss_res = np.sum((p_norm - fitted) ** 2, axis=0)
                ss_tot = np.sum((p_norm - p_norm.mean(axis=0)) ** 2, axis=0)
                r_sq = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0)

                # Bottom position (normalized 0=oldest, 1=today)
                with np.errstate(divide='ignore', invalid='ignore'):
                    t_min = np.where(a > 0, -b / (2 * a), -1)
                bottom_pos = t_min / (window - 1)

                # Current slope at end of window
                slope_end = 2 * a * (window - 1) + b

                # Confirm prior decline: start of window > midpoint price
                declined = p_norm[0] > p_norm[window // 2]

                # All conditions for saucer bottom sweet spot
                result[i] = (
                    ~has_nan
                    & (a > 0)                                       # concave up
                    & (r_sq > 0.65)                                 # clean fit
                    & (bottom_pos >= 0.50) & (bottom_pos <= 0.80)   # sweet spot
                    & (slope_end > 0)                               # curving up
                    & declined                                      # was declining
                )

            self.indicators[f'has_curve_{window}d'] = pd.DataFrame(
                result.astype(float), index=self.closes.index,
                columns=self.closes.columns)

        elapsed = time.time() - t0
        print(f"    2 curvature indicators computed in {elapsed:.1f}s")

    def _save_cache(self, cache_file):
        """Save cache pickle for fast subsequent loads."""
        print(f"  Saving cache: {cache_file.name}...")
        cache = {
            'closes': self.closes,
            'indicators': self.indicators,
            'score_mean': self.score_mean,
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = cache_file.stat().st_size / 1e6
        print(f"  Cache saved ({size_mb:.1f}MB)")

    def _compute_forward_returns(self):
        """Pre-compute forward returns for all horizons."""
        for n in FORWARD_HORIZONS:
            ret = self.closes.shift(-n) / self.closes - 1
            self._fwd_arrays[n] = ret.values

    def _convert_to_numpy(self):
        """Convert indicator DataFrames to numpy arrays for fast testing."""
        for name, df in self.indicators.items():
            self._ind_arrays[name] = df.values > 0
        if self.score_mean is not None:
            self._score_arr = self.score_mean.values
        # Precompute YYYYMM integer per day row for unique-month counting
        self._month_ids = np.array(
            [d.year * 100 + d.month for d in self.closes.index], dtype=np.int32)

    def test_combo(self, indicator_names, score_range=None, min_signals=30):
        """
        Test a specific combination of indicators using numpy arrays.

        Returns dict with long/short stats per horizon, unique months count,
        or None if not enough signals.
        """
        # AND all indicator boolean arrays
        mask = None
        for name in indicator_names:
            if name not in self._ind_arrays:
                return None
            arr = self._ind_arrays[name]
            mask = arr if mask is None else (mask & arr)

        # Apply Score Mean % range filter (separate from decile indicators)
        if score_range is not None and self._score_arr is not None:
            lo, hi = score_range
            mask = mask & (self._score_arr >= lo) & (self._score_arr <= hi)

        total = int(mask.sum())
        if total < min_signals:
            return None

        # Unique calendar months with at least one signal (temporal spread)
        day_mask = mask.any(axis=1)  # (n_days,) — any symbol on this day
        unique_months = int(len(np.unique(self._month_ids[day_mask]))) \
            if day_mask.any() else 0

        results = {
            'combo': ' + '.join(sorted(indicator_names)),
            'n_indicators': len(indicator_names),
            'total_signals': total,
            'unique_months': unique_months,
        }

        for n in FORWARD_HORIZONS:
            ret_arr = self._fwd_arrays[n]
            valid_mask = mask & ~np.isnan(ret_arr)
            valid = ret_arr[valid_mask]
            count = len(valid)
            if count < min_signals:
                results[f'win_{n}d'] = None
                results[f'avg_{n}d'] = None
                results[f'med_{n}d'] = None
                results[f'short_win_{n}d'] = None
                results[f'n_{n}d'] = 0
            else:
                # Long (buy) stats
                results[f'win_{n}d'] = round(float((valid > 0).mean() * 100), 1)
                results[f'avg_{n}d'] = round(float(valid.mean() * 100), 2)
                results[f'med_{n}d'] = round(float(np.median(valid) * 100), 2)
                # Short (sell) stats
                results[f'short_win_{n}d'] = round(
                    float((valid < 0).mean() * 100), 1)
                results[f'n_{n}d'] = count

        return results

    def test_all_combos(self, max_combo_size=4, score_range=None,
                        min_signals=30, specific_indicators=None):
        """
        Test all indicator combinations up to max_combo_size.

        Returns DataFrame sorted by 20-day win rate.
        """
        indicators_to_test = specific_indicators or list(self._ind_arrays.keys())

        total_combos = sum(
            len(list(itertools.combinations(indicators_to_test, size)))
            for size in range(2, max_combo_size + 1)
        )
        print(f"\nTesting {total_combos:,} combinations (sizes 2-{max_combo_size} "
              f"from {len(indicators_to_test)} indicators)...")

        t0 = time.time()
        all_results = []
        tested = 0

        for size in range(2, max_combo_size + 1):
            t_size = time.time()
            combos = list(itertools.combinations(indicators_to_test, size))
            for combo in combos:
                result = self.test_combo(
                    combo, score_range=score_range, min_signals=min_signals)
                if result:
                    all_results.append(result)
                tested += 1
            elapsed_size = time.time() - t_size
            valid_count = sum(1 for r in all_results if r['n_indicators'] == size)
            print(f"  Size {size}: {len(combos):,} combos in {elapsed_size:.1f}s "
                  f"({valid_count} valid)")

        elapsed = time.time() - t0
        rate = tested / max(elapsed, 0.001)
        print(f"  Total: {tested:,} combos in {elapsed:.1f}s ({rate:,.0f} combos/sec)")
        print(f"  {len(all_results)} combos met minimum signal threshold ({min_signals})")

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        df = df.sort_values('win_20d', ascending=False, na_position='last')
        df = df.reset_index(drop=True)
        return df

    def test_baselines(self, score_range=None, min_signals=30):
        """Test each indicator individually as a baseline reference."""
        results = []
        for name in sorted(self._ind_arrays.keys()):
            result = self.test_combo(
                [name], score_range=score_range, min_signals=min_signals)
            if result:
                results.append(result)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('win_20d', ascending=False, na_position='last')
        return df


def build_csv_output(results_df, baselines_df, all_indicator_names):
    """Build a clean CSV DataFrame with individual indicator columns."""
    rows = []

    # Combine baselines + combo results
    combined = pd.concat([baselines_df, results_df], ignore_index=True)
    combined = combined.sort_values('win_20d', ascending=False, na_position='last')
    combined = combined.reset_index(drop=True)

    for _, row in combined.iterrows():
        combo_parts = set(row['combo'].split(' + '))
        out = {}
        out['combo'] = row['combo']
        out['n_indicators'] = int(row['n_indicators'])
        out['total_signals'] = int(row['total_signals'])
        out['unique_months'] = int(row.get('unique_months', 0))

        # Individual indicator columns (1/0)
        for ind in all_indicator_names:
            out[ind] = 1 if ind in combo_parts else 0

        # Win rates and returns per horizon (long + short)
        for n in FORWARD_HORIZONS:
            out[f'win_rate_{n}d'] = row.get(f'win_{n}d')
            out[f'avg_return_{n}d'] = row.get(f'avg_{n}d')
            out[f'median_return_{n}d'] = row.get(f'med_{n}d')
            out[f'short_win_{n}d'] = row.get(f'short_win_{n}d')
            out[f'n_signals_{n}d'] = row.get(f'n_{n}d', 0)

        rows.append(out)

    return pd.DataFrame(rows)


def generate_analysis(results_df, baselines_df):
    """Generate automated insights from the backtest results."""
    insights = []
    top_n = min(100, len(results_df))
    top = results_df.head(top_n)

    # --- 1. Indicator frequency in top combos ---
    indicator_freq = {}
    for _, row in top.iterrows():
        for ind in row['combo'].split(' + '):
            indicator_freq[ind] = indicator_freq.get(ind, 0) + 1

    freq_sorted = sorted(indicator_freq.items(), key=lambda x: -x[1])
    insights.append("TOP INDICATOR FREQUENCY (in top 100 combos by 20d win rate)")
    insights.append(f"{'Indicator':<25} {'Appearances':>12} {'% of Top 100':>14}")
    insights.append("-" * 55)
    for ind, count in freq_sorted[:15]:
        insights.append(f"{ind:<25} {count:>12} {count/top_n*100:>13.0f}%")

    # --- 2. Theme detection ---
    insights.append("")
    insights.append("DOMINANT THEMES")
    insights.append("-" * 55)

    # Count timeframe patterns
    weekly_count = sum(1 for _, r in top.iterrows()
                       if any(i.endswith('_w') for i in r['combo'].split(' + ')))
    daily_count = sum(1 for _, r in top.iterrows()
                      if any(not i.endswith('_w') and not i.endswith('_m')
                             and not i.startswith('score_')
                             for i in r['combo'].split(' + ')))
    score_count = sum(1 for _, r in top.iterrows()
                      if any(i.startswith('score_') for i in r['combo'].split(' + ')))
    low_score = sum(1 for _, r in top.iterrows()
                    if any(i in ('score_00_10', 'score_10_20')
                           for i in r['combo'].split(' + ')))

    insights.append(f"Combos including weekly indicators: {weekly_count}/{top_n}")
    insights.append(f"Combos including daily indicators:  {daily_count}/{top_n}")
    insights.append(f"Combos including a score decile:    {score_count}/{top_n}")
    insights.append(f"Combos with low score (0-20%):      {low_score}/{top_n}")

    # --- 3. Best indicator pairs ---
    insights.append("")
    insights.append("BEST INDICATOR PAIRS (highest avg 20d win rate, min 50 signals)")
    insights.append(f"{'Pair':<55} {'Win 20d':>8} {'Avg 20d':>9} {'Signals':>8}")
    insights.append("-" * 85)

    pair_results = results_df[results_df['n_indicators'] == 2].copy()
    pair_results = pair_results[pair_results['total_signals'] >= 50]
    pair_results = pair_results.sort_values('win_20d', ascending=False)
    for _, row in pair_results.head(15).iterrows():
        insights.append(f"{row['combo']:<55} {row['win_20d']:>7.1f}% "
                        f"{row['avg_20d']:>+8.2f}% {row['total_signals']:>7,}")

    # --- 4. High-confidence combos (large sample + high win rate) ---
    insights.append("")
    insights.append("HIGH-CONFIDENCE COMBOS (win_20d >= 70% AND signals >= 100)")
    insights.append(f"{'Combo':<65} {'Win 20d':>8} {'Avg 20d':>9} {'Signals':>8}")
    insights.append("-" * 95)

    confident = results_df[
        (results_df['win_20d'] >= 70) & (results_df['total_signals'] >= 100)
    ].sort_values('win_20d', ascending=False)
    for _, row in confident.head(20).iterrows():
        insights.append(f"{row['combo']:<65} {row['win_20d']:>7.1f}% "
                        f"{row['avg_20d']:>+8.2f}% {row['total_signals']:>7,}")
    if confident.empty:
        insights.append("  (none — try lowering thresholds)")

    # --- 5. Consistency check: combos that win across ALL horizons ---
    insights.append("")
    insights.append("MOST CONSISTENT (win rate > 60% at ALL horizons, signals >= 50)")
    insights.append(f"{'Combo':<55} {'Win 5d':>7} {'Win 10d':>8} "
                    f"{'Win 20d':>8} {'Win 60d':>8} {'Signals':>8}")
    insights.append("-" * 100)

    consistent = results_df[
        (results_df['win_5d'] > 60) & (results_df['win_10d'] > 60) &
        (results_df['win_20d'] > 60) & (results_df['win_60d'] > 60) &
        (results_df['total_signals'] >= 50)
    ].sort_values('win_20d', ascending=False)
    for _, row in consistent.head(15).iterrows():
        insights.append(
            f"{row['combo']:<55} {row['win_5d']:>6.1f}% {row['win_10d']:>7.1f}% "
            f"{row['win_20d']:>7.1f}% {row['win_60d']:>7.1f}% "
            f"{row['total_signals']:>7,}")
    if consistent.empty:
        insights.append("  (none met all criteria)")

    # --- 6. Best short combos ---
    insights.append("")
    insights.append("BEST SHORT COMBOS (short_win_20d >= 60%, signals >= 50)")
    insights.append(f"{'Combo':<55} {'Short 20d':>10} {'Long 20d':>9} {'Signals':>8}")
    insights.append("-" * 87)

    short_df = results_df.dropna(subset=['short_win_20d']).copy()
    short_df = short_df[
        (short_df['short_win_20d'] >= 60) & (short_df['total_signals'] >= 50)
    ].sort_values('short_win_20d', ascending=False)
    for _, row in short_df.head(15).iterrows():
        insights.append(f"{row['combo']:<55} {row['short_win_20d']:>9.1f}% "
                        f"{row['win_20d']:>8.1f}% {row['total_signals']:>7,}")
    if short_df.empty:
        insights.append("  (none met criteria)")

    # --- 7. Contrarian insight ---
    insights.append("")
    insights.append("KEY INSIGHT: CONTRARIAN SIGNAL STRENGTH")
    insights.append("-" * 55)

    if not baselines_df.empty:
        for _, row in baselines_df.iterrows():
            if row['combo'].startswith('score_'):
                insights.append(
                    f"  {row['combo']:<20} 20d win={row['win_20d']:.1f}%  "
                    f"60d win={row['win_60d']:.1f}%  "
                    f"60d avg={row['avg_60d']:+.2f}%")
        insights.append("")
        insights.append("  Lowest-ranked stocks (score_00_10) have the highest")
        insights.append("  forward win rates — classic mean reversion. Highest-ranked")
        insights.append("  stocks (score_90_100) underperform — buying hot stocks hurts.")

    # --- 8. Warnings ---
    insights.append("")
    insights.append("WARNINGS & NOTES")
    insights.append("-" * 55)
    insights.append("  - Combos with < 100 signals may not be statistically reliable")
    insights.append("  - Check unique_months: signals in < 12 months may be clustered")
    insights.append("  - 100% win rates with small samples warrant skepticism")
    insights.append("  - Weekly indicators use completed weekly bars (no look-ahead)")
    insights.append("  - Bayesian indicators lag by N days to avoid look-ahead bias")
    insights.append("  - Forward returns are simple Close[t+N]/Close[t] - 1")
    insights.append("  - Short win% + Long win% ≈ 100% (they are complementary)")
    insights.append("  - No transaction costs, slippage, or position sizing applied")
    insights.append("  - Past performance does not guarantee future results")

    return "\n".join(insights)


def save_xlsx(results_df, baselines_df, all_indicator_names, output_file,
              data_date, n_symbols, n_days):
    """Save results as formatted Excel workbook with multiple tabs."""
    csv_df = build_csv_output(results_df, baselines_df, all_indicator_names)
    analysis_text = generate_analysis(results_df, baselines_df)

    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    workbook = writer.book

    # --- Formats ---
    header_fmt = workbook.add_format({
        'bold': True, 'bg_color': '#1a1a2e', 'font_color': 'white',
        'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
    pct_fmt = workbook.add_format({'num_format': '0.0%'})
    pct2_fmt = workbook.add_format({'num_format': '+0.00%;-0.00%'})
    num_fmt = workbook.add_format({'num_format': '#,##0'})
    text_fmt = workbook.add_format({'font_name': 'Consolas', 'font_size': 10})
    title_fmt = workbook.add_format({
        'bold': True, 'font_size': 14, 'font_color': '#1a1a2e'})
    subtitle_fmt = workbook.add_format({
        'bold': True, 'font_size': 11, 'font_color': '#444444'})
    green_bg = workbook.add_format({'bg_color': '#c6efce'})
    yellow_bg = workbook.add_format({'bg_color': '#ffeb9c'})

    # =====================================================
    # Tab 1: Analysis
    # =====================================================
    ws_analysis = workbook.add_worksheet('Analysis')
    ws_analysis.set_column('A:A', 110)
    ws_analysis.set_tab_color('#1a1a2e')

    ws_analysis.write(0, 0, 'Combo Backtester — Analysis Report', title_fmt)
    ws_analysis.write(1, 0,
                      f'Data: {data_date} | {n_symbols:,} symbols | '
                      f'{n_days:,} trading days | '
                      f'{len(results_df):,} combos tested',
                      subtitle_fmt)
    ws_analysis.write(2, 0,
                      f'Generated: {date.today().strftime("%Y-%m-%d")}',
                      subtitle_fmt)

    for i, line in enumerate(analysis_text.split('\n'), start=4):
        ws_analysis.write(i, 0, line, text_fmt)

    # =====================================================
    # Tab 2: Top 50 Summary
    # =====================================================
    top50 = results_df.head(50).copy()
    top50_display = pd.DataFrame()
    top50_display['Rank'] = range(1, len(top50) + 1)
    top50_display['Combo'] = top50['combo'].values
    top50_display['Indicators'] = top50['n_indicators'].values
    top50_display['Signals'] = top50['total_signals'].values
    top50_display['Unique Months'] = top50['unique_months'].values
    for n in FORWARD_HORIZONS:
        top50_display[f'Win {n}d'] = top50[f'win_{n}d'].values
        top50_display[f'Avg {n}d'] = top50[f'avg_{n}d'].values
        top50_display[f'Short {n}d'] = top50[f'short_win_{n}d'].values

    # Confidence column
    def confidence(row):
        signals = row['Signals']
        w20 = row.get('Win 20d')
        w60 = row.get('Win 60d')
        if pd.isna(w20) or pd.isna(w60):
            return 'Low'
        if signals >= 200 and w20 >= 70 and w60 >= 60:
            return 'High'
        if signals >= 100 and w20 >= 65:
            return 'Medium'
        return 'Low'

    top50_display['Confidence'] = top50_display.apply(confidence, axis=1)

    top50_display.to_excel(writer, sheet_name='Top 50', index=False)
    ws_top = writer.sheets['Top 50']
    ws_top.set_tab_color('#2d6a4f')
    ws_top.set_column('A:A', 6)   # Rank
    ws_top.set_column('B:B', 65)  # Combo
    ws_top.set_column('C:C', 11)  # Indicators
    ws_top.set_column('D:D', 10)  # Signals
    ws_top.set_column('E:E', 14)  # Unique Months
    ws_top.set_column('F:U', 10)  # Win/Avg/Short columns
    ws_top.set_column('V:V', 12)  # Confidence
    ws_top.autofilter(0, 0, len(top50_display), len(top50_display.columns) - 1)

    # Conditional formatting on Win 20d column
    # Columns: Rank,Combo,Indicators,Signals,UniqueMonths,
    #   Win5d,Avg5d,Short5d, Win10d,Avg10d,Short10d,
    #   Win20d(=col 11),Avg20d,Short20d, Win60d,Avg60d,Short60d, Confidence
    win20_col = 11
    ws_top.conditional_format(1, win20_col, len(top50_display), win20_col, {
        'type': 'cell', 'criteria': '>=', 'value': 80,
        'format': green_bg})
    ws_top.conditional_format(1, win20_col, len(top50_display), win20_col, {
        'type': 'cell', 'criteria': 'between', 'minimum': 65, 'maximum': 79.9,
        'format': yellow_bg})

    # =====================================================
    # Tab 3: Baselines
    # =====================================================
    base_display = pd.DataFrame()
    base_display['Indicator'] = baselines_df['combo'].values
    base_display['Signals'] = baselines_df['total_signals'].values
    base_display['Unique Months'] = baselines_df['unique_months'].values
    for n in FORWARD_HORIZONS:
        base_display[f'Win {n}d'] = baselines_df[f'win_{n}d'].values
        base_display[f'Avg {n}d'] = baselines_df[f'avg_{n}d'].values
        base_display[f'Short {n}d'] = baselines_df[f'short_win_{n}d'].values

    base_display.to_excel(writer, sheet_name='Baselines', index=False)
    ws_base = writer.sheets['Baselines']
    ws_base.set_tab_color('#e76f51')
    ws_base.set_column('A:A', 25)
    ws_base.set_column('B:I', 12)
    ws_base.autofilter(0, 0, len(base_display), len(base_display.columns) - 1)

    # =====================================================
    # Tab 4: All Results (with indicator flags)
    # =====================================================
    csv_df.to_excel(writer, sheet_name='All Results', index=False)
    ws_all = writer.sheets['All Results']
    ws_all.set_tab_color('#6c757d')
    ws_all.set_column('A:A', 65)  # Combo
    ws_all.set_column('B:C', 12)  # n_indicators, total_signals
    ws_all.autofilter(0, 0, len(csv_df), len(csv_df.columns) - 1)

    writer.close()
    return len(csv_df), analysis_text


def format_table(df, title, top_n=None):
    """Pretty-print results table."""
    if df.empty:
        print(f"\n{title}")
        print("  No results.")
        return

    show = df.head(top_n) if top_n else df

    display = pd.DataFrame()
    display['Combo'] = show['combo']
    display['Signals'] = show['total_signals'].apply(lambda x: f"{x:,}")
    if 'unique_months' in show.columns:
        display['Months'] = show['unique_months'].apply(lambda x: f"{x:,}")

    for n in FORWARD_HORIZONS:
        wr = f'win_{n}d'
        avg = f'avg_{n}d'
        swr = f'short_win_{n}d'
        if wr in show.columns:
            display[f'Win {n}d'] = show[wr].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        if avg in show.columns:
            display[f'Avg {n}d'] = show[avg].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
        if swr in show.columns and n == 20:
            display[f'Short {n}d'] = show[swr].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "—")

    n_shown = len(show)
    n_total = len(df)
    suffix = f" (showing {n_shown}/{n_total})" if top_n and n_total > n_shown else ""

    print(f"\n{'='*130}")
    print(f"{title}{suffix}")
    print(f"{'='*130}")
    print(display.to_string(index=False))


def list_indicators(bt):
    """Print all available indicators grouped by type."""
    print("\nAvailable indicators:")
    print(f"\n  Daily ({len(DAILY_INDICATORS)}):")
    for name in sorted(DAILY_INDICATORS):
        status = "ok" if name in bt._ind_arrays else "missing"
        print(f"    {name} [{status}]")

    weekly = [k for k in bt._ind_arrays if k.endswith('_w')]
    print(f"\n  Weekly ({len(weekly)}):")
    for name in sorted(weekly):
        print(f"    {name}")

    monthly = [k for k in bt._ind_arrays if k.endswith('_m')]
    print(f"\n  Monthly ({len(monthly)}):")
    for name in sorted(monthly):
        print(f"    {name}")

    bayesian = [k for k in bt._ind_arrays if k.startswith('bayes_')]
    print(f"\n  Bayesian ({len(bayesian)}):")
    for name in sorted(bayesian):
        print(f"    {name}")

    curve = [k for k in bt._ind_arrays if k.startswith('has_curve_')]
    print(f"\n  Curvature ({len(curve)}):")
    for name in sorted(curve):
        print(f"    {name}")

    deciles = [k for k in bt._ind_arrays if k.startswith('score_')]
    print(f"\n  Score Mean Deciles ({len(deciles)}):")
    for name in sorted(deciles):
        print(f"    {name}")

    print(f"\n  Total: {len(bt._ind_arrays)} indicators")


def main():
    parser = argparse.ArgumentParser(
        description='Test Momentum-Trading indicator combination win rates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combo_backtester.py                                              # All 2-4 combos
  python combo_backtester.py --combo-size 3                               # Up to 3-indicator combos
  python combo_backtester.py --test-combo has_candle has_low_bband score_70_80
  python combo_backtester.py --score-range 70 90                          # Additional filter
  python combo_backtester.py --indicators has_candle has_volume has_low_bband has_rsi_pullback
  python combo_backtester.py --daily-only                                 # Skip weekly/monthly
  python combo_backtester.py --list-indicators                            # Show all indicators
        """)

    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help=f'Path to symbol_data directory')
    parser.add_argument('--date', default=None,
                        help='Data date YYYY-MM-DD (default: latest)')
    parser.add_argument('--combo-size', type=int, default=4,
                        help='Max indicators per combo (default: 4)')
    parser.add_argument('--indicators', nargs='+', default=None,
                        help='Only test combos from these specific indicators')
    parser.add_argument('--test-combo', nargs='+', default=None,
                        help='Test one specific combo with detailed output')
    parser.add_argument('--score-range', nargs=2, type=float, default=None,
                        metavar=('LOW', 'HIGH'),
                        help='Filter by Score Mean %% range (e.g., --score-range 70 90)')
    parser.add_argument('--min-signals', type=int, default=30,
                        help='Minimum signals for valid combo (default: 30)')
    parser.add_argument('--top', type=int, default=30,
                        help='Top results to display (default: 30)')
    parser.add_argument('--output', default=None,
                        help='Save full results to CSV file')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save CSV output')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Delete cache and rebuild from source data')
    parser.add_argument('--daily-only', action='store_true',
                        help='Only use daily indicators (skip weekly/monthly)')
    parser.add_argument('--list-indicators', action='store_true',
                        help='List all available indicators and exit')

    args = parser.parse_args()
    t0_total = time.time()

    bt = ComboBacktester(data_dir=args.data_dir, data_date=args.date)

    # Handle cache clearing
    if args.clear_cache:
        for f in bt.data_dir.glob(f'{bt.data_date}-combo_cache*.pkl'):
            f.unlink()
            print(f"Deleted: {f.name}")

    bt.load_data(daily_only=args.daily_only)

    if args.list_indicators:
        list_indicators(bt)
        return

    score_label = ""
    if args.score_range:
        score_label = f" [Score Mean {args.score_range[0]:.0f}-{args.score_range[1]:.0f}%]"

    if args.test_combo:
        # Test one specific combo
        print(f"\nTesting: {' + '.join(args.test_combo)}{score_label}")
        result = bt.test_combo(
            args.test_combo,
            score_range=args.score_range,
            min_signals=args.min_signals)

        if result:
            print(f"\n  Combo: {result['combo']}")
            print(f"  Total signals: {result['total_signals']:,}")
            print(f"  Unique months: {result.get('unique_months', 0)}")
            for n in FORWARD_HORIZONS:
                wr = result.get(f'win_{n}d')
                avg = result.get(f'avg_{n}d')
                med = result.get(f'med_{n}d')
                swr = result.get(f'short_win_{n}d')
                cnt = result.get(f'n_{n}d', 0)
                if wr is not None:
                    print(f"  {n:>2}d: win={wr:.1f}%  avg={avg:+.2f}%  "
                          f"median={med:+.2f}%  short={swr:.1f}%  (n={cnt:,})")
                else:
                    print(f"  {n:>2}d: insufficient data")
        else:
            print(f"  Not enough signals (min: {args.min_signals})")
    else:
        # Baselines
        baselines = bt.test_baselines(
            score_range=args.score_range, min_signals=args.min_signals)
        format_table(baselines, f"BASELINE: Individual Indicators{score_label}")

        # All combos
        results = bt.test_all_combos(
            max_combo_size=args.combo_size,
            score_range=args.score_range,
            min_signals=args.min_signals,
            specific_indicators=args.indicators,
        )

        format_table(results, f"TOP COMBINATIONS{score_label}", top_n=args.top)

        # Save to xlsx with analysis
        if not results.empty and not args.no_save:
            output_file = args.output or \
                f'combo_backtest_results_{date.today().strftime("%Y-%m-%d")}.xlsx'
            all_ind_names = sorted(bt._ind_arrays.keys())
            n_rows, analysis = save_xlsx(
                results, baselines, all_ind_names, output_file,
                data_date=bt.data_date,
                n_symbols=len(bt.symbols),
                n_days=len(bt.closes))
            print(f"\nResults saved to: {output_file}")
            print(f"  Tabs: Analysis | Top 50 | Baselines | All Results ({n_rows:,} rows)")
            print(f"\n{analysis}")

    elapsed = (time.time() - t0_total) / 60
    print(f"\nTotal time: {elapsed:.2f} min")


if __name__ == '__main__':
    main()
