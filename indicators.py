"""
Technical indicator calculation engine for ETF strategy trigger detection.

Pure pandas/numpy implementations — no talib or ta library dependency.
Each indicator function takes an OHLCV DataFrame and returns an IndicatorResult
with point-in-time values and full series for multi-bar lookback.

Adapted from Momentum-Trading/ta_utils.py indicator logic where applicable.
"""
import numpy as np
import pandas as pd
from datetime import date
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class IndicatorResult:
    """Container for a single indicator's computed values."""
    name: str
    timeframe: str  # 'daily' or 'weekly'
    values: dict = field(default_factory=dict)


class IndicatorEngine:
    """Computes all technical indicators needed by strategy rules."""

    def compute_all(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame,
                    scan_date: date) -> Dict[str, IndicatorResult]:
        """
        Compute all indicators for a single symbol.

        Args:
            daily_df: Daily OHLCV DataFrame (index=Date, cols=Open,High,Low,Close[,Volume])
            weekly_df: Weekly OHLCV DataFrame (resampled W-FRI)
            scan_date: The date to evaluate

        Returns:
            Dict mapping indicator_key -> IndicatorResult
        """
        results = {}

        # --- DAILY INDICATORS ---
        results['donchian_7_d'] = self.donchian_channel(daily_df, period=7,
                                                         scan_date=scan_date, tf='daily')
        results['donchian_14_d'] = self.donchian_channel(daily_df, period=14,
                                                          scan_date=scan_date, tf='daily')

        results['trig_20_8_d'] = self.trigger_lines(daily_df, period=20, trig_avg=8,
                                                      scan_date=scan_date, tf='daily')
        results['trig_10_5_d'] = self.trigger_lines(daily_df, period=10, trig_avg=5,
                                                      scan_date=scan_date, tf='daily')

        results['sma_14_d'] = self.sma(daily_df, period=14, scan_date=scan_date, tf='daily')
        results['sma_40_d'] = self.sma(daily_df, period=40, scan_date=scan_date, tf='daily')

        for period, std_dev in [(20, 2.0), (30, 2.0), (30, 1.0), (20, 1.7), (20, 1.8)]:
            key = f'bb_{period}_{str(std_dev).replace(".", "")}_d'
            results[key] = self.bollinger_bands(daily_df, period=period, std_dev=std_dev,
                                                 scan_date=scan_date, tf='daily')

        for atr_period in [5, 14, 20]:
            results[f'atr_{atr_period}_d'] = self.atr_wilders(daily_df, period=atr_period,
                                                                scan_date=scan_date, tf='daily')

        results['rsqueeze_20_d'] = self.rsqueeze(daily_df, momentum_length=20,
                                                   scan_date=scan_date, tf='daily')
        results['rsqueeze_4_d'] = self.rsqueeze(daily_df, momentum_length=4,
                                                  scan_date=scan_date, tf='daily')
        results['rsqueeze_23_d'] = self.rsqueeze(daily_df, momentum_length=23,
                                                   scan_date=scan_date, tf='daily')

        # TSSuperTrend daily (14, EMA, 0.78, 5)
        results['supertrend_14_078_d'] = self.ts_supertrend(
            daily_df, atr_period=14, multiplier=0.78, ma_type='ema', ma_period=5,
            scan_date=scan_date, tf='daily')

        # TTM Squeeze daily
        results['ttm_squeeze_d'] = self.ttm_squeeze(daily_df, length=20, bb_mult=1.5,
                                                      kc_mult_high=2.0, kc_mult_low=1.0,
                                                      scan_date=scan_date, tf='daily')

        # Ichimoku daily (9,26,52) and (9,3,52)
        results['ichimoku_9_26_52_d'] = self.ichimoku(daily_df, tenkan_period=9,
                                                        kijun_period=26, senkou_b_period=52,
                                                        scan_date=scan_date, tf='daily')
        results['ichimoku_9_3_52_d'] = self.ichimoku(daily_df, tenkan_period=9,
                                                       kijun_period=3, senkou_b_period=52,
                                                       scan_date=scan_date, tf='daily')

        # Stochastics Full daily (10, 10, 3)
        results['stoch_10_10_3_d'] = self.stochastics_full(daily_df, k_period=10,
                                                             d_period=10, slowing=3,
                                                             scan_date=scan_date, tf='daily')

        # Woodies CCI daily (14, 6)
        results['cci_14_6_d'] = self.woodies_cci(daily_df, long_period=14, short_period=6,
                                                    scan_date=scan_date, tf='daily')

        # OHLCV price data for price_comparison conditions
        results['price_d'] = self._price_data(daily_df, scan_date=scan_date, tf='daily')

        # Volume
        if 'Volume' in daily_df.columns:
            results['volume_d'] = self._volume_data(daily_df, scan_date=scan_date, tf='daily')

        # --- WEEKLY INDICATORS ---
        if weekly_df is not None and not weekly_df.empty:
            results['trig_20_8_w'] = self.trigger_lines(weekly_df, period=20, trig_avg=8,
                                                          scan_date=scan_date, tf='weekly')
            results['trig_10_5_w'] = self.trigger_lines(weekly_df, period=10, trig_avg=5,
                                                          scan_date=scan_date, tf='weekly')

            results['sma_40_w'] = self.sma(weekly_df, period=40, scan_date=scan_date, tf='weekly')
            results['sma_14_w'] = self.sma(weekly_df, period=14, scan_date=scan_date, tf='weekly')

            results['supertrend_14_038_w'] = self.ts_supertrend(
                weekly_df, atr_period=14, multiplier=0.38, ma_type='ema', ma_period=5,
                scan_date=scan_date, tf='weekly')
            results['supertrend_5_023_w'] = self.ts_supertrend(
                weekly_df, atr_period=5, multiplier=0.23, ma_type='ema', ma_period=5,
                scan_date=scan_date, tf='weekly')

            for atr_period in [5, 14]:
                results[f'atr_{atr_period}_w'] = self.atr_wilders(weekly_df, period=atr_period,
                                                                    scan_date=scan_date, tf='weekly')

            results['ichimoku_9_26_52_w'] = self.ichimoku(weekly_df, tenkan_period=9,
                                                            kijun_period=26, senkou_b_period=52,
                                                            scan_date=scan_date, tf='weekly')
            results['ichimoku_9_3_52_w'] = self.ichimoku(weekly_df, tenkan_period=9,
                                                           kijun_period=3, senkou_b_period=52,
                                                           scan_date=scan_date, tf='weekly')

            results['bb_20_17_w'] = self.bollinger_bands(weekly_df, period=20, std_dev=1.7,
                                                          scan_date=scan_date, tf='weekly')

            results['ttm_squeeze_w'] = self.ttm_squeeze(weekly_df, length=20, bb_mult=1.5,
                                                          kc_mult_high=2.0, kc_mult_low=1.0,
                                                          scan_date=scan_date, tf='weekly')

            results['swing_trend_w'] = self.price_action_swing_trend(
                weekly_df, strength=15, scan_date=scan_date, tf='weekly')

            results['cci_14_6_w'] = self.woodies_cci(weekly_df, long_period=14, short_period=6,
                                                       scan_date=scan_date, tf='weekly')

            # RSqueeze weekly with default params (bb_length=20, kc_length=20)
            results['rsqueeze_20_w'] = self.rsqueeze(weekly_df, momentum_length=20,
                                                       scan_date=scan_date, tf='weekly')
            # RSqueeze weekly with ATR_14 params (as specified in Weekly ETF PDFs)
            # ATR StdDev Length=14, BB=2, KC=1.5, Momentum=20
            results['rsqueeze_atr14_w'] = self.rsqueeze(weekly_df,
                                                          bb_length=14, kc_length=14,
                                                          momentum_length=20,
                                                          scan_date=scan_date, tf='weekly')

            results['price_w'] = self._price_data(weekly_df, scan_date=scan_date, tf='weekly')

        return results

    # ====================================================================
    # INDICATOR IMPLEMENTATIONS
    # ====================================================================

    def donchian_channel(self, df: pd.DataFrame, period: int,
                         scan_date: date, tf: str) -> IndicatorResult:
        """
        Donchian Channel.
        Upper = max(High, N), Lower = min(Low, N), Mean = (Upper + Lower) / 2
        """
        upper = df['High'].rolling(period).max()
        lower = df['Low'].rolling(period).min()
        mean = (upper + lower) / 2

        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)
        prev2_idx = prev_idx - 1 if prev_idx is not None and prev_idx > 0 else None

        return IndicatorResult(
            name=f'donchian_{period}', timeframe=tf,
            values={
                'upper': self._val(upper, idx),
                'lower': self._val(lower, idx),
                'mean': self._val(mean, idx),
                'mean_prev': self._val(mean, prev_idx),
                'mean_prev2': self._val(mean, prev2_idx),
                'mean_increasing': self._increasing(mean, idx, prev_idx),
                'mean_decreasing': self._decreasing(mean, idx, prev_idx),
                'mean_series': mean,
                'upper_series': upper,
                'lower_series': lower,
            }
        )

    def trigger_lines(self, df: pd.DataFrame, period: int, trig_avg: int,
                      scan_date: date, tf: str) -> IndicatorResult:
        """
        Trigger Lines (NinjaTrader/ThinkOrSwim study).
        Plot1 (value1) = LinReg(Close, period) — linear regression endpoint value
        Plot2 (value2) = EMA(Plot1, trig_avg) — EMA of the LinReg line
        Bullish when Plot1 > Plot2, Bearish when Plot2 >= Plot1.
        Source: TriggerLines.cs lines 59-60.
        """
        close = df['Close']
        # value1 = Linear Regression endpoint value of Close over period bars
        plot1 = close.rolling(period).apply(
            lambda x: np.polyval(np.polyfit(range(len(x)), x, 1), len(x) - 1)
            if len(x) == period else np.nan,
            raw=True
        )
        # value2 = EMA of the LinReg values over trig_avg bars
        plot2 = plot1.ewm(span=trig_avg, adjust=False).mean()

        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)
        prev2_idx = prev_idx - 1 if prev_idx is not None and prev_idx > 0 else None

        p1 = self._val(plot1, idx)
        p2 = self._val(plot2, idx)
        p1_prev = self._val(plot1, prev_idx)
        p2_prev = self._val(plot2, prev_idx)
        p1_prev2 = self._val(plot1, prev2_idx)
        p2_prev2 = self._val(plot2, prev2_idx)

        # Crossover detection
        bullish_cross = (p1 is not None and p2 is not None and
                         p1_prev is not None and p2_prev is not None and
                         p1 > p2 and p1_prev < p2_prev)
        bearish_cross = (p1 is not None and p2 is not None and
                         p1_prev is not None and p2_prev is not None and
                         p1 < p2 and p1_prev > p2_prev)

        # Plot1_Up / Plot1_Dn markers (ThinkOrSwim: N/A means no directional signal)
        # Plot1_Up is set when Plot1 > Plot2 AND Plot1 is increasing
        p1_up = (p1 is not None and p2 is not None and p1_prev is not None and
                 p1 > p2 and p1 > p1_prev)
        p1_dn = (p1 is not None and p2 is not None and p1_prev is not None and
                 p1 < p2 and p1 < p1_prev)
        p2_up = (p2 is not None and p1 is not None and p2_prev is not None and
                 p1 > p2 and p2 > p2_prev)
        p2_dn = (p2 is not None and p1 is not None and p2_prev is not None and
                 p1 < p2 and p2 < p2_prev)

        # Previous day markers
        p1_up_prev = (p1_prev is not None and p2_prev is not None and p1_prev2 is not None and
                      p1_prev > p2_prev and p1_prev > p1_prev2)
        p1_dn_prev = (p1_prev is not None and p2_prev is not None and p1_prev2 is not None and
                      p1_prev < p2_prev and p1_prev < p1_prev2)
        p2_up_prev = (p2_prev is not None and p1_prev is not None and p2_prev2 is not None and
                      p1_prev > p2_prev and p2_prev > p2_prev2)
        p2_dn_prev = (p2_prev is not None and p1_prev is not None and p2_prev2 is not None and
                      p1_prev < p2_prev and p2_prev < p2_prev2)

        return IndicatorResult(
            name=f'trigger_lines_{period}_{trig_avg}', timeframe=tf,
            values={
                'plot1': p1,
                'plot2': p2,
                'plot1_prev': p1_prev,
                'plot2_prev': p2_prev,
                'plot1_prev2': p1_prev2,
                'plot2_prev2': p2_prev2,
                'plot1_increasing': self._increasing(plot1, idx, prev_idx),
                'plot2_increasing': self._increasing(plot2, idx, prev_idx),
                'plot1_decreasing': self._decreasing(plot1, idx, prev_idx),
                'plot2_decreasing': self._decreasing(plot2, idx, prev_idx),
                'bullish_cross': bullish_cross,
                'bearish_cross': bearish_cross,
                'is_bullish': p1 is not None and p2 is not None and p1 > p2,
                'is_bearish': p1 is not None and p2 is not None and p1 < p2,
                'plot1_up': p1_up,
                'plot1_dn': p1_dn,
                'plot2_up': p2_up,
                'plot2_dn': p2_dn,
                'plot1_up_prev': p1_up_prev,
                'plot1_dn_prev': p1_dn_prev,
                'plot2_up_prev': p2_up_prev,
                'plot2_dn_prev': p2_dn_prev,
                # Price vs trigger line comparisons (for Weekly Swing Sell)
                'close_lte_plot2': (self._val(df['Close'], idx) is not None and
                                    p2 is not None and
                                    self._val(df['Close'], idx) <= p2),
                'close_gt_plot1': (self._val(df['Close'], idx) is not None and
                                   p1 is not None and
                                   self._val(df['Close'], idx) > p1),
                # For Weekly Swing Sell: check if close was above Plot1 during
                # the current bullish cross period, then closed <= Plot2
                'swing_sell_pattern': self._detect_swing_sell_pattern(
                    df, plot1, plot2, idx),
                'plot1_series': plot1,
                'plot2_series': plot2,
            }
        )

    def _detect_swing_sell_pattern(self, df, plot1, plot2, idx):
        """
        Weekly Swing Sell setup: during a bullish TriggerLines cross (Plot1 > Plot2),
        price first closes above Plot1, then later closes at or below Plot2.
        Returns True if on the current bar, close <= Plot2 AND there was a prior bar
        during this bullish cross where close > Plot1.
        """
        if idx is None or idx < 2:
            return False
        close = df['Close']
        # Current bar must have close <= Plot2 and be in bullish cross
        if (pd.isna(plot1.iloc[idx]) or pd.isna(plot2.iloc[idx]) or
                pd.isna(close.iloc[idx])):
            return False
        if plot1.iloc[idx] <= plot2.iloc[idx]:  # not in bullish cross
            return False
        if close.iloc[idx] > plot2.iloc[idx]:  # close not <= Plot2
            return False
        # Scan back through the bullish cross to find close > Plot1
        for j in range(idx - 1, max(idx - 30, 0) - 1, -1):
            if pd.isna(plot1.iloc[j]) or pd.isna(plot2.iloc[j]):
                break
            if plot1.iloc[j] <= plot2.iloc[j]:  # bullish cross ended
                break
            if close.iloc[j] > plot1.iloc[j]:
                return True
        return False

    def sma(self, df: pd.DataFrame, period: int,
            scan_date: date, tf: str) -> IndicatorResult:
        """Simple Moving Average."""
        sma_val = df['Close'].rolling(period).mean()
        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)
        prev2_idx = prev_idx - 1 if prev_idx is not None and prev_idx > 0 else None

        prev3_idx = prev2_idx - 1 if prev2_idx is not None and prev2_idx > 0 else None

        return IndicatorResult(
            name=f'sma_{period}', timeframe=tf,
            values={
                'value': self._val(sma_val, idx),
                'value_prev': self._val(sma_val, prev_idx),
                'value_prev2': self._val(sma_val, prev2_idx),
                'value_prev3': self._val(sma_val, prev3_idx),
                'increasing': self._increasing(sma_val, idx, prev_idx),
                'decreasing': self._decreasing(sma_val, idx, prev_idx),
                'increasing_2d': (self._increasing(sma_val, idx, prev_idx) and
                                  self._increasing(sma_val, prev_idx, prev2_idx)),
                'decreasing_2d': (self._decreasing(sma_val, idx, prev_idx) and
                                  self._decreasing(sma_val, prev_idx, prev2_idx)),
                'increasing_3d': (self._increasing(sma_val, idx, prev_idx) and
                                  self._increasing(sma_val, prev_idx, prev2_idx) and
                                  self._increasing(sma_val, prev2_idx, prev3_idx)),
                'equal_or_increasing': (self._val(sma_val, idx) is not None and
                                        self._val(sma_val, prev_idx) is not None and
                                        self._val(sma_val, idx) >= self._val(sma_val, prev_idx)),
                'equal_or_decreasing': (self._val(sma_val, idx) is not None and
                                        self._val(sma_val, prev_idx) is not None and
                                        self._val(sma_val, idx) <= self._val(sma_val, prev_idx)),
                'series': sma_val,
            }
        )

    def bollinger_bands(self, df: pd.DataFrame, period: int, std_dev: float,
                        scan_date: date, tf: str) -> IndicatorResult:
        """Bollinger Bands: middle = SMA(Close, N), upper/lower = middle +/- std_dev * std."""
        middle = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std

        idx = self._get_index(df, scan_date)

        return IndicatorResult(
            name=f'bb_{period}_{std_dev}', timeframe=tf,
            values={
                'upper': self._val(upper, idx),
                'middle': self._val(middle, idx),
                'lower': self._val(lower, idx),
                'upper_series': upper,
                'middle_series': middle,
                'lower_series': lower,
            }
        )

    def atr_wilders(self, df: pd.DataFrame, period: int,
                    scan_date: date, tf: str) -> IndicatorResult:
        """Wilder's ATR (exponentially weighted true range)."""
        tr = self._true_range(df)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

        idx = self._get_index(df, scan_date)
        return IndicatorResult(
            name=f'atr_{period}', timeframe=tf,
            values={
                'value': self._val(atr, idx),
                'series': atr,
            }
        )

    def rsqueeze(self, df: pd.DataFrame, scan_date: date, tf: str,
                 bb_length: int = 20, bb_mult: float = 2.0,
                 kc_length: int = 20, kc_mult: float = 1.5,
                 momentum_length: int = 20,
                 cci_period: int = 13,
                 gauss_period: int = 21, gauss_poles: int = 3) -> IndicatorResult:
        """
        RSqueeze / BBSqueeze indicator.
        Squeeze state: Bollinger Bands inside Keltner Channels.
        Momentum: linear regression of (close - midline) over N bars.
        """
        # Squeeze detection: BB width vs KC width
        # bbsInd = (nBb * StdDev) / (nK * ATR) — squeeze when <= 1
        # Source: RSqueeze.cs lines 81-83
        bb_std = df['Close'].rolling(bb_length).std()
        tr = self._true_range(df)
        kc_atr = tr.ewm(alpha=1/kc_length, adjust=False).mean()
        bbs_ind = pd.Series(np.nan, index=df.index)
        denom = kc_mult * kc_atr
        bbs_ind = np.where(denom != 0, bb_mult * bb_std / denom, 1.0)
        bbs_ind = pd.Series(bbs_ind, index=df.index)
        squeeze_on = bbs_ind <= 1.0

        # Momentum: LinReg of (Close - midpoint) where midpoint = (DonchianMean + EMA) / 2
        # Source: RSqueeze.cs lines 85-86
        # _myValue2 = Input[0] - ((DonchianChannel(Input, mLength).Mean + EMA(Input, mLength)) / 2)
        # Mom = LinReg(_myValue2, mLength)   [LinReg returns fitted value at current bar]
        highest = df['High'].rolling(momentum_length).max()
        lowest = df['Low'].rolling(momentum_length).min()
        donchian_mean = (highest + lowest) / 2
        ema_line = df['Close'].ewm(span=momentum_length, adjust=False).mean()
        delta = df['Close'] - (donchian_mean + ema_line) / 2

        # LinReg endpoint value = np.polyval(coeffs, N-1) = slope*(N-1) + intercept
        momentum = delta.rolling(momentum_length).apply(
            lambda x: np.polyval(np.polyfit(range(len(x)), x, 1), len(x) - 1)
            if len(x) == momentum_length else np.nan,
            raw=True
        )

        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)
        prev2_idx = prev_idx - 1 if prev_idx is not None and prev_idx > 0 else None
        prev3_idx = prev2_idx - 1 if prev2_idx is not None and prev2_idx > 0 else None

        mom = self._val(momentum, idx)
        mom_prev = self._val(momentum, prev_idx)
        mom_prev2 = self._val(momentum, prev2_idx)
        mom_prev3 = self._val(momentum, prev3_idx)

        return IndicatorResult(
            name=f'rsqueeze_{momentum_length}', timeframe=tf,
            values={
                'squeeze_on': self._val(squeeze_on, idx),
                'squeeze_on_prev': self._val(squeeze_on, prev_idx),
                'momentum': mom,
                'momentum_prev': mom_prev,
                'momentum_prev2': mom_prev2,
                'momentum_prev3': mom_prev3,
                'momentum_increasing': (mom is not None and mom_prev is not None and
                                        mom > mom_prev),
                'momentum_decreasing': (mom is not None and mom_prev is not None and
                                        mom < mom_prev),
                # For ETF Squeeze Sell: prev 2 days > 0 and increasing
                'prev_2_positive_increasing': (
                    mom_prev is not None and mom_prev2 is not None and
                    mom_prev3 is not None and
                    mom_prev > 0 and mom_prev2 > 0 and
                    mom_prev > mom_prev2
                ),
                # For squeeze buy setups: prev 2 days < 0 and decreasing
                'prev_2_negative_decreasing': (
                    mom_prev is not None and mom_prev2 is not None and
                    mom_prev3 is not None and
                    mom_prev < 0 and mom_prev2 < 0 and
                    mom_prev < mom_prev2
                ),
                'squeeze_on_series': squeeze_on,
                'momentum_series': momentum,
            }
        )

    def ts_supertrend(self, df: pd.DataFrame, atr_period: int, multiplier: float,
                      ma_type: str, ma_period: int,
                      scan_date: date, tf: str) -> IndicatorResult:
        """
        TSSuperTrend: ATR-based trend indicator with trailing bands.
        Source: TSSuperTrend.cs lines 51-166.

        Key NinjaTrader logic:
        - avg = MA(Close, smooth) — smoothing is applied to Close, NOT ATR
        - offset = ATR(length) * multiplier
        - Bands: avg +/- offset
        - Trend: Close > DownTrend[prev] → bullish, Close < UpTrend[prev] → bearish
        - Tracks th (highest High in uptrend) and tl (lowest Low in downtrend)
          as constraints on initial band placement after flip

        Direction: True = bullish (green), False = bearish (red).
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        tr = self._true_range(df)

        # ATR uses Wilder's smoothing (alpha=1/period)
        atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

        # Smoothing is applied to Close via MA type (not to ATR!)
        if ma_period > 1:
            if ma_type == 'ema':
                avg = close.ewm(span=ma_period, adjust=False).mean()
            elif ma_type == 'sma':
                avg = close.rolling(ma_period).mean()
            else:
                avg = close.ewm(span=ma_period, adjust=False).mean()
        else:
            avg = close

        offset = atr * multiplier

        n = len(df)
        up_trend = pd.Series(np.nan, index=df.index, dtype=float)  # green line
        down_trend = pd.Series(np.nan, index=df.index, dtype=float)  # red line
        trend = pd.Series(True, index=df.index)  # True=bullish, False=bearish
        th = high.iloc[0] if n > 0 else 0.0  # highest high during uptrend
        tl = low.iloc[0] if n > 0 else float('inf')  # lowest low during downtrend

        if n > 0:
            up_trend.iloc[0] = close.iloc[0]
            down_trend.iloc[0] = close.iloc[0]

        for i in range(1, n):
            # Determine trend: Close > DownTrend → bull, Close < UpTrend → bear
            if close.iloc[i] > down_trend.iloc[i-1]:
                trend.iloc[i] = True
            elif close.iloc[i] < up_trend.iloc[i-1]:
                trend.iloc[i] = False
            else:
                trend.iloc[i] = trend.iloc[i-1]

            if trend.iloc[i] and not trend.iloc[i-1]:
                # Flip to bullish: constrain UpTrend by tl (lowest low during prior downtrend)
                th = high.iloc[i]
                up_trend.iloc[i] = max(avg.iloc[i] - offset.iloc[i], tl)
                down_trend.iloc[i] = down_trend.iloc[i-1]
            elif not trend.iloc[i] and trend.iloc[i-1]:
                # Flip to bearish: constrain DownTrend by th (highest high during prior uptrend)
                tl = low.iloc[i]
                down_trend.iloc[i] = min(avg.iloc[i] + offset.iloc[i], th)
                up_trend.iloc[i] = up_trend.iloc[i-1]
            else:
                if trend.iloc[i]:
                    # Continue bullish: ratchet UpTrend up only
                    up_trend.iloc[i] = max(avg.iloc[i] - offset.iloc[i], up_trend.iloc[i-1])
                    th = max(th, high.iloc[i])
                    down_trend.iloc[i] = down_trend.iloc[i-1]
                else:
                    # Continue bearish: ratchet DownTrend down only
                    down_trend.iloc[i] = min(avg.iloc[i] + offset.iloc[i], down_trend.iloc[i-1])
                    tl = min(tl, low.iloc[i])
                    up_trend.iloc[i] = up_trend.iloc[i-1]

        # Convert trend (bool) to direction int: -1=bullish, 1=bearish (matches old convention)
        direction = pd.Series(np.where(trend, -1, 1), index=df.index, dtype=int)
        # Supertrend line = UpTrend when bullish, DownTrend when bearish
        supertrend = pd.Series(np.where(trend, up_trend, down_trend), index=df.index)

        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)

        # Count consecutive bearish (red) candles before current
        consecutive_red = 0
        if idx is not None:
            for j in range(idx - 1, max(idx - 20, 0) - 1, -1):
                if direction.iloc[j] == 1:  # bearish
                    consecutive_red += 1
                else:
                    break

        return IndicatorResult(
            name=f'supertrend_{atr_period}_{multiplier}', timeframe=tf,
            values={
                'value': self._val(supertrend, idx),
                'direction': self._val(direction, idx),
                'direction_prev': self._val(direction, prev_idx),
                'is_bullish': (self._val(direction, idx) == -1) if idx is not None else None,
                'is_bearish': (self._val(direction, idx) == 1) if idx is not None else None,
                'just_turned_bullish': (
                    self._val(direction, idx) == -1 and
                    self._val(direction, prev_idx) == 1
                ) if idx is not None and prev_idx is not None else None,
                'consecutive_red_before': consecutive_red,
                'series': supertrend,
                'direction_series': direction,
            }
        )

    def ttm_squeeze(self, df: pd.DataFrame, length: int, bb_mult: float,
                    kc_mult_high: float, kc_mult_low: float,
                    scan_date: date, tf: str) -> IndicatorResult:
        """
        TTM Squeeze: BB vs KC squeeze with momentum histogram.
        Uses linear regression of (close - midpoint) for momentum.
        """
        # TTM Squeeze uses same momentum formula as RSqueeze (BBSqueeze mode)
        # Squeeze detection: bbsInd = (bb_mult * StdDev) / (kc_mult_low * ATR) <= 1
        bb_std = df['Close'].rolling(length).std()
        tr = self._true_range(df)
        kc_atr = tr.ewm(alpha=1/length, adjust=False).mean()
        denom = kc_mult_low * kc_atr
        bbs_ind = np.where(denom != 0, bb_mult * bb_std / denom, 1.0)
        bbs_ind = pd.Series(bbs_ind, index=df.index)
        squeeze_on = bbs_ind <= 1.0

        # Momentum: LinReg of (Close - (DonchianMean + EMA) / 2)
        # Same formula as RSqueeze BBSqueeze mode
        highest = df['High'].rolling(length).max()
        lowest = df['Low'].rolling(length).min()
        donchian_mean = (highest + lowest) / 2
        ema_line = df['Close'].ewm(span=length, adjust=False).mean()
        delta = df['Close'] - (donchian_mean + ema_line) / 2

        # LinReg endpoint value (fitted value at current bar)
        momentum = delta.rolling(length).apply(
            lambda x: np.polyval(np.polyfit(range(len(x)), x, 1), len(x) - 1)
            if len(x) == length else np.nan,
            raw=True
        )

        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)
        prev2_idx = prev_idx - 1 if prev_idx is not None and prev_idx > 0 else None
        prev3_idx = prev2_idx - 1 if prev2_idx is not None and prev2_idx > 0 else None

        mom = self._val(momentum, idx)
        mom_prev = self._val(momentum, prev_idx)
        mom_prev2 = self._val(momentum, prev2_idx)
        mom_prev3 = self._val(momentum, prev3_idx)

        return IndicatorResult(
            name='ttm_squeeze', timeframe=tf,
            values={
                'squeeze_on': self._val(squeeze_on, idx),
                'momentum': mom,
                'momentum_prev': mom_prev,
                'momentum_prev2': mom_prev2,
                'momentum_increasing': (mom is not None and mom_prev is not None and
                                        mom > mom_prev),
                'momentum_decreasing': (mom is not None and mom_prev is not None and
                                        mom < mom_prev),
                # ETF Squeeze Sell setup: prev 2 days > 0 and increasing, signal day decreasing
                'prev_2_positive_increasing': (
                    mom_prev is not None and mom_prev2 is not None and
                    mom_prev3 is not None and
                    mom_prev > 0 and mom_prev2 > 0 and
                    mom_prev > mom_prev2
                ),
                # Squeeze buy setups: prev 2 days < 0 and decreasing, signal day increasing
                'prev_2_negative_decreasing': (
                    mom_prev is not None and mom_prev2 is not None and
                    mom_prev3 is not None and
                    mom_prev < 0 and mom_prev2 < 0 and
                    mom_prev < mom_prev2
                ),
                'momentum_series': momentum,
                'squeeze_on_series': squeeze_on,
            }
        )

    def ichimoku(self, df: pd.DataFrame, tenkan_period: int, kijun_period: int,
                 senkou_b_period: int, scan_date: date, tf: str) -> IndicatorResult:
        """
        Ichimoku Cloud components: Tenkan-sen, Kijun-sen, Senkou Span A/B.
        """
        tenkan = (df['High'].rolling(tenkan_period).max() +
                  df['Low'].rolling(tenkan_period).min()) / 2
        kijun = (df['High'].rolling(kijun_period).max() +
                 df['Low'].rolling(kijun_period).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
        senkou_b = ((df['High'].rolling(senkou_b_period).max() +
                     df['Low'].rolling(senkou_b_period).min()) / 2).shift(kijun_period)

        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)
        prev2_idx = prev_idx - 1 if prev_idx is not None and prev_idx > 0 else None

        # Count kijun decreasing days before the hook (non-consecutive: flat days skipped)
        # Start from prev_idx since signal day itself is the hook-up day
        kijun_decrease_count = 0
        start_j = prev_idx if prev_idx is not None else (idx - 1 if idx is not None and idx > 0 else None)
        if start_j is not None:
            for j in range(start_j, max(start_j - 60, 0), -1):
                if j > 0 and kijun.iloc[j] < kijun.iloc[j-1]:
                    kijun_decrease_count += 1
                elif j > 0 and kijun.iloc[j] > kijun.iloc[j-1]:
                    break  # actual increase breaks the streak
                # flat (equal) days are skipped — don't break, don't count

        # Kijun hook up: was decreasing, now increasing
        kijun_hook_up = (
            self._increasing(kijun, idx, prev_idx) and
            self._decreasing(kijun, prev_idx, prev2_idx)
        ) if idx is not None and prev_idx is not None and prev2_idx is not None else False

        return IndicatorResult(
            name=f'ichimoku_{tenkan_period}_{kijun_period}_{senkou_b_period}', timeframe=tf,
            values={
                'tenkan': self._val(tenkan, idx),
                'kijun': self._val(kijun, idx),
                'tenkan_prev': self._val(tenkan, prev_idx),
                'kijun_prev': self._val(kijun, prev_idx),
                'tenkan_prev2': self._val(tenkan, prev2_idx),
                'kijun_prev2': self._val(kijun, prev2_idx),
                'senkou_a': self._val(senkou_a, idx),
                'senkou_b': self._val(senkou_b, idx),
                'tenkan_increasing': self._increasing(tenkan, idx, prev_idx),
                'kijun_increasing': self._increasing(kijun, idx, prev_idx),
                'kijun_not_decreasing': not self._decreasing(kijun, idx, prev_idx),
                'kijun_decreasing': self._decreasing(kijun, idx, prev_idx),
                'kijun_hook_up': kijun_hook_up,
                'kijun_decrease_count': kijun_decrease_count,
                'tenkan_below_kijun_prev': (
                    self._val(tenkan, prev_idx) is not None and
                    self._val(kijun, prev_idx) is not None and
                    self._val(tenkan, prev_idx) < self._val(kijun, prev_idx)
                ),
                'tenkan_below_kijun_prev2': (
                    self._val(tenkan, prev2_idx) is not None and
                    self._val(kijun, prev2_idx) is not None and
                    self._val(tenkan, prev2_idx) < self._val(kijun, prev2_idx)
                ),
                # Tenkan hook up (Setup 2): was declining, now increasing, was below Kijun
                'tenkan_hook_up': (
                    self._increasing(tenkan, idx, prev_idx) and
                    self._decreasing(tenkan, prev_idx, prev2_idx) and
                    self._val(tenkan, prev_idx) is not None and
                    self._val(kijun, prev_idx) is not None and
                    self._val(tenkan, prev_idx) < self._val(kijun, prev_idx)
                ) if idx is not None and prev_idx is not None and prev2_idx is not None else False,
                # Tenkan flat then hook up (Setup 3): prev 2 days flat, now increasing
                'tenkan_flat_hook_up': (
                    self._increasing(tenkan, idx, prev_idx) and
                    self._val(tenkan, prev_idx) is not None and
                    self._val(tenkan, prev2_idx) is not None and
                    self._val(tenkan, prev_idx) == self._val(tenkan, prev2_idx)
                ) if idx is not None and prev_idx is not None and prev2_idx is not None else False,
                # Kijun equal or increasing (Setup 3, 4 trigger)
                'kijun_equal_or_increasing': (
                    self._val(kijun, idx) is not None and
                    self._val(kijun, prev_idx) is not None and
                    self._val(kijun, idx) >= self._val(kijun, prev_idx)
                ),
                'tenkan_series': tenkan,
                'kijun_series': kijun,
            }
        )

    def stochastics_full(self, df: pd.DataFrame, k_period: int, d_period: int,
                         slowing: int, scan_date: date, tf: str) -> IndicatorResult:
        """Full Stochastics (%K with slowing, %D)."""
        lowest_low = df['Low'].rolling(k_period).min()
        highest_high = df['High'].rolling(k_period).max()
        raw_k = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        k = raw_k.rolling(slowing).mean()
        d = k.rolling(d_period).mean()

        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)
        prev2_idx = prev_idx - 1 if prev_idx is not None and prev_idx > 0 else None

        return IndicatorResult(
            name=f'stoch_{k_period}_{d_period}_{slowing}', timeframe=tf,
            values={
                'k': self._val(k, idx),
                'd': self._val(d, idx),
                'k_prev': self._val(k, prev_idx),
                'k_prev2': self._val(k, prev2_idx),
                'k_hooking_up': self._increasing(k, idx, prev_idx),
                'k_was_decreasing': self._decreasing(k, prev_idx, prev2_idx),
                'k_prev_below_20': (self._val(k, prev_idx) is not None and
                                     self._val(k, prev_idx) < 20),
                'k_above_80': (self._val(k, idx) is not None and
                                self._val(k, idx) > 80),
                # Setup 4: %K hooks up from below 20 (oversold)
                # %K increasing, was decreasing, and prev was below 20
                'k_hook_from_oversold': (
                    self._increasing(k, idx, prev_idx) and
                    self._decreasing(k, prev_idx, prev2_idx) and
                    self._val(k, prev_idx) is not None and
                    self._val(k, prev_idx) < 20
                ),
            }
        )

    def woodies_cci(self, df: pd.DataFrame, long_period: int, short_period: int,
                    scan_date: date, tf: str) -> IndicatorResult:
        """
        Woodies CCI: dual-period CCI for trend/entry signals.
        Long period (14) = trend, Short period (6) = turbo/entry.
        """
        tp = (df['High'] + df['Low'] + df['Close']) / 3

        def _cci(series, period):
            ma = series.rolling(period).mean()
            mad = series.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            return (series - ma) / (0.015 * mad)

        cci_long = _cci(tp, long_period)
        cci_short = _cci(tp, short_period)

        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)

        # Track turbo (short CCI) behavior for hook detection
        turbo = self._val(cci_short, idx)
        turbo_prev = self._val(cci_short, prev_idx)
        cci_l = self._val(cci_long, idx)
        cci_l_prev = self._val(cci_long, prev_idx)

        # For Weekly Turbo Hook: turbo drops below 0 then hooks up
        turbo_hooking_up = (turbo is not None and turbo_prev is not None and
                            turbo > turbo_prev)
        turbo_prev_below_zero = turbo_prev is not None and turbo_prev < 0

        # Weekly Turbo Hook Buy pattern: turbo was below 0, now hooking up
        turbo_hook_from_negative = (turbo_hooking_up and turbo_prev_below_zero)

        # Additional Weekly Turbo Hook conditions:
        # CCI long must be > 0 when turbo hooks up
        # CCI long must have been > 0 before turbo dropped below 0
        # CCI long cannot be < 0 for more than 1 bar while turbo was below 0
        cci_long_valid_during_turbo_dip = False
        if turbo_hook_from_negative and idx is not None:
            # Scan backward to find when turbo first dropped below 0
            turbo_dip_start = None
            for j in range(idx - 1, max(idx - 20, 0) - 1, -1):
                if cci_short.iloc[j] >= 0:
                    turbo_dip_start = j
                    break
            if turbo_dip_start is not None:
                # CCI long was > 0 before the turbo dip
                long_positive_before = cci_long.iloc[turbo_dip_start] > 0
                # Count weeks CCI long was < 0 during the dip
                weeks_long_negative = 0
                for j in range(turbo_dip_start + 1, idx):
                    if cci_long.iloc[j] < 0:
                        weeks_long_negative += 1
                cci_long_valid_during_turbo_dip = (
                    long_positive_before and weeks_long_negative <= 1
                )

        return IndicatorResult(
            name=f'cci_{long_period}_{short_period}', timeframe=tf,
            values={
                'cci_long': cci_l,
                'cci_short': turbo,
                'cci_long_prev': cci_l_prev,
                'cci_short_prev': turbo_prev,
                'turbo_hooking_up': turbo_hooking_up,
                'turbo_below_zero': turbo is not None and turbo < 0,
                'turbo_prev_below_zero': turbo_prev_below_zero,
                'turbo_hook_from_negative': turbo_hook_from_negative,
                'cci_long_positive': cci_l is not None and cci_l > 0,
                'cci_long_positive_prev': cci_l_prev is not None and cci_l_prev > 0,
                'cci_long_valid_during_turbo_dip': cci_long_valid_during_turbo_dip,
                'cci_long_series': cci_long,
                'cci_short_series': cci_short,
            }
        )

    def price_action_swing_trend(self, df: pd.DataFrame, strength: int,
                                 scan_date: date, tf: str) -> IndicatorResult:
        """
        Price Action Swing Trend — Gann-style bar-by-bar swing detection.

        Classifies each bar as up/down/inside based on H/L vs previous bar,
        then tracks swing direction and running extremes. Swings are confirmed
        when bar classification reverses (e.g., a down bar after an upswing).

        Minimum spacing of 3 bars between swing confirmations prevents noise
        from rapid bar-to-bar oscillations on weekly data.

        Colors based on swing relationships (including pending running extreme):
        - Blue: Higher high AND higher low (bullish uptrend)
        - Red: Lower high AND lower low (bearish downtrend)
        - Gold: Mixed signals (transitional — triggers Weekly Swing Trend Buy)
        """
        n = len(df)
        if n < 5:
            color_series = pd.Series('gold', index=df.index)
            return IndicatorResult(
                name='swing_trend', timeframe=tf,
                values={
                    'color': 'gold', 'color_prev': 'gold', 'color_prev2': 'gold',
                    'is_blue': False, 'is_red': False, 'is_gold': True,
                    'red_to_gold': False, 'blue_gold_red_gold': False,
                    'color_series': color_series,
                }
            )

        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        min_bars = 3  # minimum spacing between swing confirmations

        # Phase 1: Gann bar-by-bar swing detection
        swing_dir = None  # 'up' or 'down'
        run_high = high[0]
        run_high_bar = 0
        run_low = low[0]
        run_low_bar = 0
        last_confirm_bar = -min_bars  # allow first swing immediately

        confirmed_highs = []  # list of swing high prices
        confirmed_lows = []   # list of swing low prices
        # Track which bar each confirmed swing occurs at
        confirmed_high_bars = []
        confirmed_low_bars = []

        colors = pd.Series('gold', index=df.index)

        for i in range(1, n):
            # Classify bar: up, down, or inside
            bar_type = self._classify_gann_bar(
                high[i], low[i], close[i],
                high[i-1], low[i-1], close[i-1]
            )

            if swing_dir is None:
                if bar_type == 'up':
                    swing_dir = 'up'
                    run_high = high[i]
                    run_high_bar = i
                elif bar_type == 'down':
                    swing_dir = 'down'
                    run_low = low[i]
                    run_low_bar = i
            elif swing_dir == 'up':
                if high[i] >= run_high:
                    run_high = high[i]
                    run_high_bar = i
                if bar_type == 'down' and (i - last_confirm_bar) >= min_bars:
                    confirmed_highs.append(run_high)
                    confirmed_high_bars.append(run_high_bar)
                    last_confirm_bar = i
                    swing_dir = 'down'
                    run_low = low[i]
                    run_low_bar = i
            elif swing_dir == 'down':
                if low[i] <= run_low:
                    run_low = low[i]
                    run_low_bar = i
                if bar_type == 'up' and (i - last_confirm_bar) >= min_bars:
                    confirmed_lows.append(run_low)
                    confirmed_low_bars.append(run_low_bar)
                    last_confirm_bar = i
                    swing_dir = 'up'
                    run_high = high[i]
                    run_high_bar = i

            # Determine color: use confirmed swings + running extreme
            eval_highs = list(confirmed_highs[-2:])
            eval_lows = list(confirmed_lows[-2:])

            # Include current running extreme as pending swing
            if swing_dir == 'up' and (not eval_highs or run_high != eval_highs[-1]):
                eval_highs.append(run_high)
            elif swing_dir == 'down' and (not eval_lows or run_low != eval_lows[-1]):
                eval_lows.append(run_low)

            eval_highs = eval_highs[-2:]
            eval_lows = eval_lows[-2:]

            if len(eval_highs) >= 2 and len(eval_lows) >= 2:
                hh = eval_highs[-1] > eval_highs[-2]
                hl = eval_lows[-1] > eval_lows[-2]
                if hh and hl:
                    colors.iloc[i] = 'blue'
                elif not hh and not hl:
                    colors.iloc[i] = 'red'
                else:
                    colors.iloc[i] = 'gold'

        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)
        prev2_idx = prev_idx - 1 if prev_idx is not None and prev_idx > 0 else None

        color = self._val(colors, idx)
        color_prev = self._val(colors, prev_idx)
        color_prev2 = self._val(colors, prev2_idx)

        # Detect color transition patterns
        red_to_gold = (color == 'gold' and color_prev is not None and
                       self._find_recent_color(colors, idx, 'red', lookback=10))
        blue_gold_red_gold = self._detect_swing_buy_pattern(colors, idx)

        return IndicatorResult(
            name='swing_trend', timeframe=tf,
            values={
                'color': color,
                'color_prev': color_prev,
                'color_prev2': color_prev2,
                'is_blue': color == 'blue',
                'is_red': color == 'red',
                'is_gold': color == 'gold',
                'red_to_gold': red_to_gold,
                'blue_gold_red_gold': blue_gold_red_gold,
                'color_series': colors,
            }
        )

    @staticmethod
    def _classify_gann_bar(h, l, c, h_prev, l_prev, c_prev) -> str:
        """Classify a bar as up/down/inside using Gann rules."""
        if h > h_prev:
            if l > l_prev:
                return 'up'
            elif l < l_prev:
                # Outside bar: use close to decide
                return 'up' if c > c_prev else 'down'
            else:
                return 'up'  # High higher, low same
        elif h < h_prev:
            if l < l_prev:
                return 'down'
            elif l > l_prev:
                return 'inside'
            else:
                return 'down'  # High lower, low same
        else:  # High same
            return 'down' if l < l_prev else 'inside'

    def _detect_swing_buy_pattern(self, colors: pd.Series, idx: int,
                                   max_lookback: int = 52) -> bool:
        """
        Detect Blue → Gold → Red → Gold pattern in PriceActionSwingTrend.
        Current bar must be Gold. Looking back, must find Red, then Gold, then Blue.
        Changes don't have to be consecutive weeks.
        """
        if idx is None or idx < 3:
            return False
        if colors.iloc[idx] != 'gold':
            return False

        start = max(0, idx - max_lookback)
        # Walk backwards looking for distinct color phases
        # Phase 0: skip current gold period
        # Phase 1: find red
        # Phase 2: find gold (between blue and red)
        # Phase 3: find blue
        phase = 0
        for j in range(idx - 1, start - 1, -1):
            c = colors.iloc[j]
            if phase == 0:  # Looking for red (skip any initial gold)
                if c == 'red':
                    phase = 1
                elif c == 'blue':
                    return False  # Blue before Red — wrong order
            elif phase == 1:  # In red phase, looking for transition to gold
                if c == 'gold':
                    phase = 2
                elif c == 'blue':
                    # Blue directly after red with no gold in between
                    # This can still count — the "gold" between might be brief
                    return True
            elif phase == 2:  # In gold phase between red and blue, looking for blue
                if c == 'blue':
                    return True
                elif c == 'red':
                    return False  # Another red before blue — pattern broken
        return False

    # ====================================================================
    # PRICE / VOLUME DATA HELPERS
    # ====================================================================

    def _price_data(self, df: pd.DataFrame, scan_date: date, tf: str) -> IndicatorResult:
        """Extract OHLC price data for the scan date."""
        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)
        prev2_idx = prev_idx - 1 if prev_idx is not None and prev_idx > 0 else None

        return IndicatorResult(
            name='price', timeframe=tf,
            values={
                'open': self._val(df['Open'], idx),
                'high': self._val(df['High'], idx),
                'low': self._val(df['Low'], idx),
                'close': self._val(df['Close'], idx),
                'open_prev': self._val(df['Open'], prev_idx),
                'high_prev': self._val(df['High'], prev_idx),
                'low_prev': self._val(df['Low'], prev_idx),
                'close_prev': self._val(df['Close'], prev_idx),
                'open_prev2': self._val(df['Open'], prev2_idx),
                'high_prev2': self._val(df['High'], prev2_idx),
                'low_prev2': self._val(df['Low'], prev2_idx),
                'close_prev2': self._val(df['Close'], prev2_idx),
                'is_up_day': (self._val(df['Close'], idx) is not None and
                              self._val(df['Open'], idx) is not None and
                              self._val(df['Close'], idx) > self._val(df['Open'], idx)),
                'is_down_day': (self._val(df['Close'], idx) is not None and
                                self._val(df['Open'], idx) is not None and
                                self._val(df['Close'], idx) < self._val(df['Open'], idx)),
                'high_series': df['High'],
                'low_series': df['Low'],
                'close_series': df['Close'],
                'open_series': df['Open'],
            }
        )

    def _volume_data(self, df: pd.DataFrame, scan_date: date, tf: str) -> IndicatorResult:
        """Extract volume data with comparisons."""
        vol = df['Volume']
        idx = self._get_index(df, scan_date)
        prev_idx = self._get_prev_index(df, scan_date)

        # Max volume in previous 5 days
        vol_max_5 = vol.rolling(5).max()

        return IndicatorResult(
            name='volume', timeframe=tf,
            values={
                'value': self._val(vol, idx),
                'value_prev': self._val(vol, prev_idx),
                'increasing': self._increasing(vol, idx, prev_idx),
                'prev_is_5d_max': (
                    self._val(vol, prev_idx) is not None and
                    self._val(vol_max_5, prev_idx) is not None and
                    self._val(vol, prev_idx) == self._val(vol_max_5, prev_idx)
                ),
            }
        )

    # ====================================================================
    # UTILITY METHODS
    # ====================================================================

    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def _get_index(self, df: pd.DataFrame, scan_date: date) -> Optional[int]:
        """Get positional index for scan_date or closest prior trading day."""
        if df.empty:
            return None
        target = pd.Timestamp(scan_date)
        mask = df.index <= target
        if not mask.any():
            return None
        mask_arr = np.asarray(mask)
        indices = mask_arr.nonzero()[0]
        return int(indices[-1])

    def _get_prev_index(self, df: pd.DataFrame, scan_date: date) -> Optional[int]:
        """Get positional index for the bar before scan_date."""
        idx = self._get_index(df, scan_date)
        if idx is not None and idx > 0:
            return idx - 1
        return None

    def _val(self, series, idx) -> Optional[float]:
        """Safely get a value from a series by positional index."""
        if idx is None or series is None:
            return None
        try:
            v = series.iloc[idx]
            if pd.isna(v):
                return None
            return v
        except (IndexError, KeyError):
            return None

    def _increasing(self, series, idx, prev_idx) -> bool:
        """Check if value at idx > value at prev_idx."""
        curr = self._val(series, idx)
        prev = self._val(series, prev_idx)
        if curr is None or prev is None:
            return False
        return curr > prev

    def _decreasing(self, series, idx, prev_idx) -> bool:
        """Check if value at idx < value at prev_idx."""
        curr = self._val(series, idx)
        prev = self._val(series, prev_idx)
        if curr is None or prev is None:
            return False
        return curr < prev

    def _find_recent_color(self, colors: pd.Series, idx: int,
                           target_color: str, lookback: int = 10) -> bool:
        """Check if a specific color appeared within the lookback window."""
        if idx is None:
            return False
        start = max(0, idx - lookback)
        for i in range(idx - 1, start - 1, -1):
            if colors.iloc[i] == target_color:
                return True
        return False
