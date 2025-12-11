"""
Trading strategy classes for ETF and Futures backtesting.
Provides clean separation between different trading logic types.
"""

from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, date
import utilities as util


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, tick_sizes=None):
        self.tick_sizes = tick_sizes or {}
    
    @abstractmethod
    def evaluate_exit(self, strategy, formulas, df, signal_date, symbol, entry_price, strategy_name, direction):
        """Calculate stop and target prices for a trade."""
        pass
    
    @abstractmethod
    def get_target_type(self, strategy):
        """Get target type description for display."""
        pass
    
    @abstractmethod
    def check_stop_hit(self, row, stop_price, direction, has_stop, i):
        """Check if stop loss was hit."""
        pass
    
    @abstractmethod
    def check_target_hit(self, row, target_price, direction):
        """Check if target was hit."""
        pass
    
    def wilders_atr(self, df, period):
        """Compute Wilder's ATR based on historical OHLC data."""
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        return tr.ewm(alpha=1/period, adjust=False).mean()


class ETFTradingStrategy(TradingStrategy):
    """Trading strategy for ETF options."""
    
    def evaluate_exit(self, strategy, formulas, df, signal_date, symbol, entry_price, strategy_name, direction):
        """Calculate ETF options exit conditions using ATR targets and expiration dates."""
        # Get target configuration from strategy JSON
        target_formula = strategy.get("target", {}).get("formula", {})
        atr_length = target_formula.get("atr_length", 5)
        multiplier = target_formula.get("multiplier", 1.0)
        timeframe = target_formula.get("timeframe", "daily").lower()

        # Calculate ATR based on timeframe (weekly or daily)
        if timeframe == "weekly":
            # Resample to weekly and calculate weekly ATR
            df_weekly = df.resample('W-FRI').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            atr = self.wilders_atr(df_weekly, atr_length)
            # Use ATR as of signal date, not future data
            atr_at_signal = atr[atr.index <= signal_date]
            atr_value = atr_at_signal.iloc[-1] if len(atr_at_signal) > 0 else 0
        else:
            # Use daily ATR
            atr = self.wilders_atr(df, atr_length)
            atr_value = atr.loc[signal_date]

        # Apply the multiplier from config (e.g., 0.55 for Weekly strategies)
        atr_offset = atr_value * multiplier

        # Calculate target price based on direction
        if direction == "sell":
            target_price = entry_price - atr_offset
        else:
            target_price = entry_price + atr_offset

        # ETF options expire 2 months out
        stop_price = util.get_final_expiration_date(signal_date, months_out=2)
        return stop_price, round(target_price, 2)
    
    def get_target_type(self, strategy):
        """Generate target type based on actual strategy configuration."""
        target_formula = strategy.get("target", {}).get("formula", {})

        if target_formula.get("type") == "atr_multiple":
            atr_length = target_formula.get("atr_length", 5)
            multiplier = target_formula.get("multiplier", 1.0)
            timeframe = target_formula.get("timeframe", "daily")

            # Show weekly vs daily timeframe
            timeframe_prefix = "Weekly " if timeframe.lower() == "weekly" else ""
            return f"{timeframe_prefix}ATR{atr_length} x {multiplier}"

        # Default fallback
        return "ATR5 x 1.0"
    
    def check_stop_hit(self, row, stop_price, direction, has_stop, i):
        """Check ETF options expiration date."""
        if isinstance(stop_price, date):
            if i.date() >= stop_price:
                return {
                    "hit": True,
                    "status": "Expired",
                    "stop_display": "Expired"
                }
        return {"hit": False}
    
    def check_target_hit(self, row, target_price, direction):
        """Check if ETF target price was hit."""
        if direction == "buy" and row["High"] >= target_price:
            return True
        elif direction == "sell" and row["Low"] <= target_price:
            return True
        return False


class FuturesTradingStrategy(TradingStrategy):
    """Trading strategy for futures contracts."""

    def __init__(self, tick_sizes=None):
        super().__init__(tick_sizes)
        self.evaluate_formula_func = None  # Will be set by main module
    
    def set_evaluate_formula_function(self, func):
        """Set the evaluate_formula function from main module."""
        self.evaluate_formula_func = func
    
    def evaluate_exit(self, strategy, formulas, df, signal_date, symbol, entry_price, strategy_name, direction):
        """Calculate futures exit conditions using JSON strategy configuration."""
        if not self.evaluate_formula_func:
            raise ValueError("evaluate_formula function not set")
        
        # Use JSON strategy configuration for stops and targets
        stop_formula = strategy["stop"]["formula"]
        stop_price = self.evaluate_formula_func(stop_formula, df, signal_date, symbol, entry_price=entry_price)
        
        target_formula = strategy["target"]["formula"]
        target_result = self.evaluate_formula_func(target_formula, df, signal_date, symbol, entry_price=entry_price, stop_price=stop_price)
        
        # Handle different target result types
        if isinstance(target_result, dict) and "target_price" in target_result and "target_type" in target_result:
            # Multi-target result with specific target type
            target_price = target_result["target_price"]
            target_type = target_result["target_type"]
            return stop_price, target_price, target_type
        elif isinstance(target_result, list):
            target_price = max(target_result)
        else:
            target_price = target_result
        
        return stop_price, target_price
    
    def get_target_type(self, strategy):
        """Extract target type from futures strategy configuration."""
        try:
            target_formula = strategy.get("target", {}).get("formula", {})
            if target_formula.get("type") == "atr_multiple":
                atr_length = target_formula.get("atr_length", 5)
                multiplier = target_formula.get("multiplier", 1.0)
                return f"ATR{atr_length} x {multiplier}"
            elif target_formula.get("type") == "fixed_atr_target":
                atr_length = target_formula.get("atr_length", 5)
                return f"ATR{atr_length} x 0.6"
            elif target_formula.get("type") == "atr_percentage":
                atr_length = target_formula.get("atr_length", 5)
                percentage = target_formula.get("percentage", 0.6)
                return f"ATR{atr_length} x {percentage}"
            elif target_formula.get("type") == "entry_stop_percentage":
                percentage = target_formula.get("percentage", 0.4)
                return f"Entry-Stop x {percentage}"
            elif target_formula.get("type") == "multi_target":
                target_rank = target_formula.get("target_rank", 1)
                num_options = len(target_formula.get("target_options", []))
                return f"Multi-Target Rank {target_rank}/{num_options}"
            else:
                return target_formula.get("type", "Unknown")
        except:
            return "Unknown"
    
    def check_stop_hit(self, row, stop_price, direction, has_stop, i):
        """Check futures stop loss price hit."""
        if has_stop and isinstance(stop_price, (float, int)):
            if direction == "buy" and row["Low"] <= stop_price:
                return {
                    "hit": True,
                    "status": "Stopped out",
                    "stop_display": round(stop_price, 5)
                }
            elif direction == "sell" and row["High"] >= stop_price:
                return {
                    "hit": True,
                    "status": "Stopped out", 
                    "stop_display": round(stop_price, 5)
                }
        return {"hit": False}
    
    def check_target_hit(self, row, target_price, direction):
        """Check if futures target price was hit."""
        if direction == "buy" and row["High"] >= target_price:
            return True
        elif direction == "sell" and row["Low"] <= target_price:
            return True
        return False


class StrategyFactory:
    """Factory to create appropriate trading strategy based on mode."""
    
    @staticmethod
    def create_strategy(mode: str, **kwargs) -> TradingStrategy:
        """
        Create trading strategy instance based on mode.
        
        Args:
            mode: 'etfs' or 'futures'
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            TradingStrategy instance
        """
        if mode == "etfs":
            return ETFTradingStrategy(**kwargs)
        elif mode == "futures":
            return FuturesTradingStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown trading mode: {mode}")