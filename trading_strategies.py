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
    
    def __init__(self, symbol_mappings=None, tick_sizes=None):
        self.symbol_mappings = symbol_mappings or {}
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
        atr = self.wilders_atr(df, 5)
        atr_value = atr.loc[signal_date]
        
        # Target is 1x ATR(5)
        if direction == "sell":
            target_price = entry_price - atr_value
        else:
            target_price = entry_price + atr_value
        
        # ETF options expire 2 months out
        stop_price = util.get_final_expiration_date(signal_date, months_out=2)
        return stop_price, round(target_price, 2)
    
    def get_target_type(self, strategy):
        """ETF options always use ATR5 x 1.0 target."""
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
    
    def __init__(self, symbol_mappings=None, tick_sizes=None):
        super().__init__(symbol_mappings, tick_sizes)
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
        
        if isinstance(target_result, list):
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