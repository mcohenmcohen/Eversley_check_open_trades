"""
Strategy rule engine for ETF trigger detection.

Evaluates JSON-defined setup and trigger conditions against computed indicators.
Each strategy has setup conditions (AND) and trigger conditions (AND).
A signal fires when ALL conditions pass.

Condition types:
- indicator_comparison: compare two indicator values
- indicator_state: check boolean/value state
- cross_above / cross_below: crossover detection within lookback
- hook_up / hook_down: value increasing/decreasing from prior bar
- consecutive_condition: N bars meeting criteria
- price_comparison: compare OHLC price to indicator
- external_threshold: check external data (CBOE, NYSE A/D)
- any_of: OR logic (at least one sub-condition must pass)
"""
import json
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class TriggerResult:
    """Result of a strategy trigger evaluation."""
    symbol: str
    strategy: str
    direction: str
    triggered: bool
    scan_date: date
    indicator_values: Dict[str, Any]
    is_internals: bool = False


def load_trigger_config(path: str = None) -> dict:
    """Load strategy trigger configuration from JSON."""
    if path is None:
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'etf_trigger_config.json')
    with open(path) as f:
        return json.load(f)


class StrategyRuleEngine:
    """Evaluates strategy setup and trigger conditions against computed indicators."""

    def __init__(self, config: dict):
        self.strategies = config.get('strategies', {})

    def evaluate_all(self, symbols: List[str], scan_date: date,
                     indicators: Dict[str, Dict], external_data: dict = None,
                     include_internals: bool = False,
                     debug: bool = False) -> List[dict]:
        """
        Evaluate all strategies against all symbols.

        Args:
            symbols: List of symbols to scan
            scan_date: Date to evaluate
            indicators: Dict[symbol -> Dict[indicator_key -> IndicatorResult]]
            external_data: Optional dict of external data (cboe_put_call, nyse_ad, etc.)
            include_internals: Whether to evaluate internals strategies
            debug: Print debug output

        Returns:
            List of trigger result dicts for output
        """
        if external_data is None:
            external_data = {}

        triggers = []

        for strategy_name, strategy_config in self.strategies.items():
            # Skip comment keys
            if strategy_name.startswith('_'):
                continue
            # Skip internals strategies unless flag is set
            if strategy_config.get('is_internals', False) and not include_internals:
                continue

            applicable_symbols = strategy_config.get('symbols', 'all')
            direction = strategy_config['direction']
            timeframe = strategy_config.get('timeframe', 'daily')

            for symbol in symbols:
                # Check if symbol is applicable for this strategy
                if applicable_symbols != 'all' and symbol not in applicable_symbols:
                    continue
                if symbol not in indicators:
                    continue

                # Proxy symbol support: XLI uses DIA indicators, XLK uses QQQ
                proxy_symbols = strategy_config.get('proxy_symbols', {})
                indicator_symbol = proxy_symbols.get(symbol, symbol)
                if indicator_symbol not in indicators:
                    continue

                symbol_indicators = indicators[indicator_symbol]

                if debug:
                    print(f"\n  Evaluating: {strategy_name} for {symbol}")

                # Evaluate setup conditions
                setup_conditions = strategy_config.get('setup_conditions', [])
                setup_met = self._evaluate_conditions(
                    setup_conditions, symbol_indicators, external_data,
                    symbol, scan_date, debug=debug, phase='setup'
                )

                if not setup_met:
                    if debug:
                        print(f"    Setup NOT met")
                    continue

                if debug:
                    print(f"    Setup met, checking triggers...")

                # Evaluate trigger conditions
                trigger_conditions = strategy_config.get('trigger_conditions', [])
                trigger_met = self._evaluate_conditions(
                    trigger_conditions, symbol_indicators, external_data,
                    symbol, scan_date, debug=debug, phase='trigger'
                )

                if trigger_met:
                    display_values = self._collect_display_values(
                        strategy_config.get('display_indicators', []),
                        symbol_indicators
                    )

                    if debug:
                        print(f"    TRIGGERED! {symbol} {strategy_name} {direction}")

                    triggers.append({
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'direction': direction,
                        'timeframe': timeframe,
                        'scan_date': scan_date.isoformat(),
                        'is_internals': strategy_config.get('is_internals', False),
                        **display_values
                    })
                elif debug:
                    print(f"    Trigger NOT met")

        return triggers

    def _evaluate_conditions(self, conditions: List[dict],
                             indicators: Dict, external_data: dict,
                             symbol: str, scan_date: date,
                             debug: bool = False, phase: str = '') -> bool:
        """
        Evaluate a list of conditions (AND logic — all must be true).
        """
        if not conditions:
            return True

        for i, condition in enumerate(conditions):
            cond_type = condition.get('type', '')
            try:
                result = self._dispatch_condition(
                    condition, indicators, external_data, symbol, scan_date
                )

                if debug:
                    desc = condition.get('description', f"{cond_type}")
                    status = "PASS" if result else "FAIL"
                    print(f"      [{phase}][{i}] {desc}: {status}")

                if not result:
                    return False

            except (KeyError, TypeError, IndexError, ValueError) as e:
                if debug:
                    print(f"      [{phase}][{i}] ERROR: {e}")
                return False

        return True

    def _dispatch_condition(self, condition: dict, indicators: Dict,
                            external_data: dict, symbol: str,
                            scan_date: date) -> bool:
        """Route condition to the appropriate evaluator."""
        cond_type = condition['type']

        # Most evaluators take (cond, indicators, external_data)
        # Some need scan_date for series-based lookups
        evaluators = {
            'indicator_comparison': lambda c, ind, ext: self._eval_indicator_comparison(c, ind, ext),
            'indicator_state': lambda c, ind, ext: self._eval_indicator_state(c, ind, ext),
            'price_comparison': lambda c, ind, ext: self._eval_price_comparison(c, ind, ext),
            'cross_above': lambda c, ind, ext: self._eval_cross(c, ind, 'above', scan_date),
            'cross_below': lambda c, ind, ext: self._eval_cross(c, ind, 'below', scan_date),
            'hook_up': lambda c, ind, ext: self._eval_hook(c, ind, 'up'),
            'hook_down': lambda c, ind, ext: self._eval_hook(c, ind, 'down'),
            'consecutive_condition': lambda c, ind, ext: self._eval_consecutive(c, ind, ext, scan_date),
            'external_threshold': lambda c, ind, ext: self._eval_external(c, ext),
            'any_of': lambda c, ind, ext: self._eval_any_of(
                c, ind, ext, symbol, scan_date),
            'lookback_any': lambda c, ind, ext: self._eval_lookback_any(c, ind, ext, scan_date),
            'series_condition': lambda c, ind, ext: self._eval_series_condition(c, ind, ext, scan_date),
            'put_call_threshold': lambda c, ind, ext: self._eval_put_call_threshold(c, ext, scan_date),
        }

        evaluator = evaluators.get(cond_type)
        if evaluator is None:
            return False

        return evaluator(condition, indicators, external_data)

    def _eval_indicator_comparison(self, cond: dict, indicators: Dict,
                                   external_data: dict) -> bool:
        """Compare two indicator values with an operator."""
        left = self._resolve_value(cond, indicators, external_data)
        right = self._resolve_compare_to(cond, indicators, external_data)

        if left is None or right is None:
            return False

        return self._compare(left, cond['operator'], right)

    def _eval_indicator_state(self, cond: dict, indicators: Dict,
                              external_data: dict) -> bool:
        """Check if an indicator field matches expected value."""
        ind = indicators.get(cond['indicator'])
        if not ind:
            return False
        actual = ind.values.get(cond['field'])
        expected = cond['expected']

        if actual is None:
            return False

        return actual == expected

    def _eval_price_comparison(self, cond: dict, indicators: Dict,
                               external_data: dict) -> bool:
        """Compare a price field against an indicator value."""
        # Get price from the price indicator
        tf_suffix = cond.get('timeframe_suffix', 'd')
        price_ind = indicators.get(f'price_{tf_suffix}')
        if not price_ind:
            return False

        price_field = cond.get('price_field', 'close')
        left = price_ind.values.get(price_field)

        right = self._resolve_compare_to(cond, indicators, external_data)

        if left is None or right is None:
            return False

        return self._compare(left, cond['operator'], right)

    def _eval_cross(self, cond: dict, indicators: Dict,
                    direction: str, scan_date: date = None) -> bool:
        """Check if series A crossed above/below series B within lookback of scan_date."""
        ind = indicators.get(cond['indicator'])
        ref = indicators.get(cond['compare_to']['indicator'])
        if not ind or not ref:
            return False

        series_a = ind.values.get(cond['field'])
        series_b = ref.values.get(cond['compare_to']['field'])
        if series_a is None or series_b is None:
            return False

        if not isinstance(series_a, pd.Series) or not isinstance(series_b, pd.Series):
            return False

        # Find scan_date position in series
        end_idx = self._find_scan_date_idx(series_a, scan_date)
        if end_idx is None:
            end_idx = min(len(series_a), len(series_b)) - 1

        lookback = cond.get('lookback', 1)

        for offset in range(lookback):
            i = end_idx - offset
            if i < 1:
                continue
            try:
                curr_above = series_a.iloc[i] > series_b.iloc[i]
                prev_above = series_a.iloc[i-1] > series_b.iloc[i-1]
                if direction == 'above' and curr_above and not prev_above:
                    return True
                if direction == 'below' and not curr_above and prev_above:
                    return True
            except IndexError:
                continue
        return False

    def _eval_hook(self, cond: dict, indicators: Dict, direction: str) -> bool:
        """Check if an indicator value is hooking up/down."""
        ind = indicators.get(cond['indicator'])
        if not ind:
            return False

        field = cond['field']
        prev_field = cond.get('prev_field', field + '_prev')

        current = ind.values.get(field)
        previous = ind.values.get(prev_field)

        if current is None or previous is None:
            return False

        if direction == 'up':
            return current > previous
        else:
            return current < previous

    def _eval_consecutive(self, cond: dict, indicators: Dict,
                          external_data: dict, scan_date: date = None) -> bool:
        """
        Check if a condition held for N consecutive bars.
        Uses series data for lookback.

        Config example:
        {
            "type": "consecutive_condition",
            "indicator": "donchian_14_d",
            "field": "mean_series",
            "compare_to": {"indicator": "sma_40_d", "field": "series"},
            "operator": "<",
            "min_bars": 8,
            "end_offset": 1  // how many bars back from scan date to start checking
        }
        """
        ind = indicators.get(cond['indicator'])
        if not ind:
            return False

        series_a = ind.values.get(cond['field'])
        if series_a is None or not isinstance(series_a, pd.Series):
            return False

        # Get comparison value (can be series or static)
        compare = cond.get('compare_to', {})
        if 'indicator' in compare:
            ref = indicators.get(compare['indicator'])
            if not ref:
                return False
            series_b = ref.values.get(compare['field'])
        elif 'value' in compare:
            series_b = compare['value']
        else:
            return False

        operator = cond['operator']
        min_bars = cond.get('min_bars', 1)
        end_offset = cond.get('end_offset', 1)

        # Find scan_date position, then apply end_offset
        scan_idx = self._find_scan_date_idx(series_a, scan_date)
        if scan_idx is None:
            scan_idx = len(series_a) - 1
        end_idx = scan_idx - end_offset + 1
        count = 0

        for i in range(end_idx - 1, max(end_idx - min_bars - 5, 0) - 1, -1):
            try:
                val_a = series_a.iloc[i]
                if isinstance(series_b, pd.Series):
                    val_b = series_b.iloc[i]
                else:
                    val_b = series_b

                if pd.isna(val_a) or (isinstance(val_b, float) and pd.isna(val_b)):
                    break

                if self._compare(val_a, operator, val_b):
                    count += 1
                else:
                    break
            except IndexError:
                break

        return count >= min_bars

    def _eval_external(self, cond: dict, external_data: dict) -> bool:
        """Check external data against threshold."""
        value = external_data.get(cond['source'])
        if value is None:
            return False

        compare = cond.get('compare_to', {})
        if 'value' in compare:
            right = compare['value']
        elif 'value' in cond:
            right = cond['value']
        else:
            return False

        return self._compare(value, cond['operator'], right)

    def _eval_any_of(self, cond: dict, indicators: Dict, external_data: dict,
                     symbol: str, scan_date: date) -> bool:
        """OR logic: at least one sub-condition must pass."""
        sub_conditions = cond.get('conditions', [])
        for sub in sub_conditions:
            if self._dispatch_condition(sub, indicators, external_data,
                                        symbol, scan_date):
                return True
        return False

    def _eval_lookback_any(self, cond: dict, indicators: Dict,
                           external_data: dict, scan_date: date = None) -> bool:
        """
        Check if a condition was true for any bar within a lookback window.

        Config example:
        {
            "type": "lookback_any",
            "indicator": "price_d",
            "field_series": "high_series",  // or use the parent df
            "compare_to": {"indicator": "bb_30_20_d", "field": "upper_series"},
            "operator": ">",
            "lookback": 5
        }
        """
        ind = indicators.get(cond['indicator'])
        if not ind:
            return False

        # Get series data
        series_key = cond.get('field_series')
        if series_key:
            series_a = ind.values.get(series_key)
        else:
            return False

        compare = cond.get('compare_to', {})
        ref = indicators.get(compare.get('indicator', ''))
        if not ref:
            return False
        series_b = ref.values.get(compare.get('field', ''))

        if not isinstance(series_a, pd.Series) or not isinstance(series_b, pd.Series):
            return False

        lookback = cond.get('lookback', 5)
        scan_idx = self._find_scan_date_idx(series_a, scan_date)
        if scan_idx is None:
            scan_idx = min(len(series_a), len(series_b)) - 1

        for i in range(scan_idx - lookback + 1, scan_idx + 1):
            if i < 0:
                continue
            try:
                val_a = series_a.iloc[i]
                val_b = series_b.iloc[i]
                if pd.notna(val_a) and pd.notna(val_b):
                    if self._compare(val_a, cond['operator'], val_b):
                        return True
            except IndexError:
                continue
        return False

    def _eval_series_condition(self, cond: dict, indicators: Dict,
                               external_data: dict, scan_date: date = None) -> bool:
        """
        Evaluate a condition on series data at specific offsets.
        Useful for checking values N bars ago.

        Config example:
        {
            "type": "series_condition",
            "indicator": "ichimoku_9_26_52_d",
            "field": "tenkan_series",
            "offset": -2,  // 2 bars before scan date
            "operator": "<",
            "compare_to": {
                "indicator": "ichimoku_9_26_52_d",
                "field": "kijun_series",
                "offset": -2
            }
        }
        """
        ind = indicators.get(cond['indicator'])
        if not ind:
            return False

        series_a = ind.values.get(cond['field'])
        offset_a = cond.get('offset', 0)

        if not isinstance(series_a, pd.Series):
            return False

        compare = cond.get('compare_to', {})
        if 'indicator' in compare:
            ref = indicators.get(compare['indicator'])
            if not ref:
                return False
            series_b = ref.values.get(compare.get('field', ''))
            offset_b = compare.get('offset', 0)
        elif 'value' in compare:
            series_b = None
            val_b = compare['value']
            offset_b = 0
        else:
            return False

        try:
            base_idx = self._find_scan_date_idx(series_a, scan_date)
            if base_idx is None:
                base_idx = len(series_a) - 1
            idx_a = base_idx + offset_a
            val_a = series_a.iloc[idx_a]

            if series_b is not None and isinstance(series_b, pd.Series):
                base_idx_b = self._find_scan_date_idx(series_b, scan_date)
                if base_idx_b is None:
                    base_idx_b = len(series_b) - 1
                idx_b = base_idx_b + offset_b
                val_b = series_b.iloc[idx_b]
            elif series_b is None:
                pass  # val_b already set from compare['value']
            else:
                return False

            if pd.isna(val_a) or (isinstance(val_b, float) and pd.isna(val_b)):
                return False

            return self._compare(val_a, cond['operator'], val_b)
        except (IndexError, KeyError):
            return False

    def _eval_put_call_threshold(self, cond: dict, external_data: dict,
                                 scan_date: date) -> bool:
        """
        Check if CBOE Put/Call ratio meets threshold within a window.

        Evaluates: ratio <= low_threshold OR ratio >= high_threshold
        over a window of N most recent trading days up to scan_date.

        Config:
            source: key in external_data (default: 'put_call_ratio')
            low_threshold: upper bound for low side (default: 0.85)
            high_threshold: lower bound for high side (default: 1.0)
            window: number of trading days to check (1=signal day, 3=signal+2 prev)
        """
        pc_data = external_data.get(cond.get('source', 'put_call_ratio'), {})
        if not pc_data:
            return False

        low = cond.get('low_threshold', 0.85)
        high = cond.get('high_threshold', 1.0)
        window = cond.get('window', 1)

        # Get dates with data that are on or before scan_date, sorted descending
        scan_str = scan_date.strftime('%Y-%m-%d')
        valid_dates = sorted(
            [d for d in pc_data if d <= scan_str],
            reverse=True
        )

        # Check the most recent N trading days
        for d in valid_dates[:window]:
            ratio = pc_data[d]
            if ratio <= low or ratio >= high:
                return True

        return False

    # ====================================================================
    # HELPER METHODS
    # ====================================================================

    @staticmethod
    def _find_scan_date_idx(series: pd.Series, scan_date: date) -> Optional[int]:
        """Find the positional index of scan_date (or nearest prior date) in a series."""
        if scan_date is None or series is None or series.empty:
            return None
        import numpy as np
        target = pd.Timestamp(scan_date)
        mask = series.index <= target
        if not mask.any():
            return None
        indices = np.asarray(mask).nonzero()[0]
        return int(indices[-1])

    def _resolve_value(self, cond: dict, indicators: Dict,
                       external_data: dict) -> Optional[Any]:
        """Resolve the left-hand value from a condition."""
        ind = indicators.get(cond.get('indicator', ''))
        if not ind:
            return None
        return ind.values.get(cond.get('field', ''))

    def _resolve_compare_to(self, cond: dict, indicators: Dict,
                            external_data: dict) -> Optional[Any]:
        """Resolve the right-hand value from a condition's compare_to."""
        compare = cond.get('compare_to', {})

        if 'value' in compare:
            return compare['value']

        if 'indicator' in compare:
            ref = indicators.get(compare['indicator'])
            if not ref:
                return None
            return ref.values.get(compare.get('field', ''))

        if 'external' in compare:
            return external_data.get(compare['external'])

        return None

    @staticmethod
    def _compare(left, operator: str, right) -> bool:
        """Compare two values with the given operator."""
        ops = {
            '>': lambda a, b: a > b,
            '<': lambda a, b: a < b,
            '>=': lambda a, b: a >= b,
            '<=': lambda a, b: a <= b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
        }
        op_fn = ops.get(operator)
        if op_fn is None:
            return False
        try:
            return op_fn(left, right)
        except (TypeError, ValueError):
            return False

    def _collect_display_values(self, display_keys: List[str],
                                indicators: Dict) -> dict:
        """Extract indicator values for display in output."""
        result = {}
        for key in display_keys:
            parts = key.split('.')
            if len(parts) == 2:
                ind = indicators.get(parts[0])
                if ind:
                    val = ind.values.get(parts[1])
                    if val is not None and not isinstance(val, pd.Series):
                        if isinstance(val, float):
                            result[key] = round(val, 4)
                        else:
                            result[key] = val
        return result
