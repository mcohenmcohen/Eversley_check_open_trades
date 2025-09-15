import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from pathlib import Path
import time
import os
from collections import defaultdict
from rapidfuzz import process
from alpha_vantage.timeseries import TimeSeries
import utilities as util

from data_sources import DataSourceManager, load_futures_data_from_csv
from trading_strategies import StrategyFactory

# API Keys
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
INSIGHTSENTRY_API_KEY = os.getenv('INSIGHTSENTRY_API_KEY')  # Add to your environment variables


# Initialize data source manager
data_manager = DataSourceManager(
    polygon_api_key=POLYGON_API_KEY,
    insightsentry_api_key=INSIGHTSENTRY_API_KEY
)
print("Data sources initialized!")
fetched_data_cache = {}  # cache already-fetched symbols

# AV_API_KEY = 'IN8A22IKNXHRN9P5'  # Replace with your Alpha Vantage key
TICK_SIZE = {
    # Futures
    "6A": 0.0001, "6B": 0.0001, "6C": 0.0001, "6E": 0.0001, "6S": 0.0001,
    
    # Index futures
    "ES": 0.25, "NQ": 0.25, "RTY": 0.10, "YM": 1.00,

    # Common ETFs
    "DIA": 0.01, "EEM": 0.01, "EFA": 0.01, "EWG": 0.01, "EWH": 0.01,
    "FXI": 0.01, "GDX": 0.01, "GLD": 0.01, "IBB": 0.01, "IWM": 0.01,
    "IYR": 0.01, "QQQ": 0.01, "SLV": 0.01, "SPY": 0.01, "TLT": 0.01,
    "UNG": 0.01, "USO": 0.01, "VNQ": 0.01, "VWO": 0.01, "XHB": 0.01,
    "XLB": 0.01, "XLE": 0.01, "XLF": 0.01, "XLI": 0.01, "XLK": 0.01,
    "XLV": 0.01, "XME": 0.01, "XOP": 0.01, "XRT": 0.01
}

# # Maps internal CME futures symbols to 
# Yahoo Finance ticker symbols (for futures, not needed now since i'm downloading them from ninjatrader)
# ETF tickers from Alpha Vantage
symbol_map = {
    # Futures
    "6A": "6A=F",
    "6B": "6B=F",
    "6C": "6C=F",
    "6E": "6E=F",
    "6S": "6S=F",
    
    # Index futures
    "ES": "ES=F",
    "NQ": "NQ=F",
    "RTY": "RTY=F",
    "YM": "YM=F",

    # ETFs
    "DIA": "DIA",
    "EEM": "EEM",
    "EFA": "EFA",
    "EWG": "EWG",
    "EWH": "EWH",
    "FXI": "FXI",
    "GDX": "GDX",
    "GLD": "GLD",
    "IBB": "IBB",
    "IWM": "IWM",
    "IYR": "IYR",
    "QQQ": "QQQ",
    "SLV": "SLV",
    "SPY": "SPY",
    "TLT": "TLT",
    "UNG": "UNG",
    "USO": "USO",
    "VNQ": "VNQ",
    "VWO": "VWO",
    "XHB": "XHB",
    "XLB": "XLB",
    "XLE": "XLE",
    "XLF": "XLF",
    "XLI": "XLI",
    "XLK": "XLK",
    "XLV": "XLV",
    "XME": "XME",
    "XOP": "XOP",
    "XRT": "XRT"
}

def resolve_strategy_name(input_name, strategy_names):
    match, score, _ = process.extractOne(input_name, strategy_names)
    return match if score > 80 else None

# Load strategy definitions from the given JSON path
def load_strategies(path):
    with open(path) as f:
        return json.load(f)

# Load configuration settings
def load_config(path="config.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Config file {path} not found, using defaults")
        return {
            "data_sources": {
                "etf_data_source": "insightsentry",
                "futures_data_source": "insightsentry"
            }
        }

# Downloads yfinance historical OHLC data for all required symbols
# def fetch_price_data(symbols, start_date, end_date):
#     data = {}
#     for sym in symbols:
#         local_path = Path(f"data/{sym}.csv")
#         if local_path.exists():
#             print(f"📂 Loading local data for {sym} from {local_path}")
#             try:
#                 df = pd.read_csv(local_path)
#                 df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
#                 df.set_index("Date", inplace=True)
#                 df = df[["Open", "High", "Low", "Close", "Volume"]]
#                 df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
#                 data[sym] = df
#                 continue
#             except Exception as e:
#                 print(f"❌ Failed to load {sym} from local file: {e}")

#         yf_sym = symbol_map.get(sym)
#         if not yf_sym:
#             print(f"⚠️ No Yahoo symbol mapped for {sym}, skipping.")
#             continue

#         try:
#             df = yf.download(yf_sym, start=start_date.strftime('%Y-%m-%d'),
#                              end=end_date.strftime('%Y-%m-%d'), interval="1d", progress=False)
#             if df.empty:
#                 print(f"⚠️ No data returned for {yf_sym}. Skipping.")
#                 continue
#             df.index = pd.to_datetime(df.index)
#             data[sym] = df
#         except Exception as e:
#             print(f"❌ Error fetching {yf_sym}: {e}")
#     return data


# Call the right source for getting price data, depending on if etfs or futures
def get_price_data(symbol, mode, cache, start_date=None, etf_source="insightsentry"):
    # For ETFs, only fetch 3 months of data (max 2 expiration cycles + current month)
    if start_date is None:
        if mode == "etfs":
            start_date = datetime.now() - timedelta(days=90)  # 3 months
        else:
            start_date = datetime(2025, 1, 2)  # Futures need more history
    if symbol in cache:
        return cache[symbol]

    try:
        end_date = datetime.today() + timedelta(days=1)  # buffer in case you're running same-day
        
        if mode == "futures":
            # Try InsightSentry first, fallback to local CSV
            if data_manager.is_source_available('insightsentry'):
                df = data_manager.get_futures_data(symbol, start_date, end_date)
                if df.empty:
                    print(f"⚠️ No InsightSentry data for {symbol}, trying local CSV")
                    df = load_futures_data_from_csv(symbol)
            else:
                print(f"📂 InsightSentry not available, using local CSV for {symbol}")
                df = load_futures_data_from_csv(symbol)
            
            if df.empty:
                print(f"❌ Could not load data for futures symbol {symbol}")
                return None
                
        else:  # ETF mode
            df = data_manager.get_etf_data(symbol, start_date, end_date, source=etf_source)
            if df.empty:
                print(f"❌ No ETF data returned for {symbol}")
                return None
                
            print(f"{symbol}: price range {df.index.min().date()} to {df.index.max().date()}")

    except Exception as e:
        print(f"❌ Failed to fetch data for {symbol}: {e}")
        return None

    cache[symbol] = df
    return df

# Get the ticker data from Polygon (now handled by data_sources module)
def fetch_price_data(symbols, start_date=None, end_date=None):
    # Optimize date range for ETFs (3 months max)
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)  # 3 months for ETFs
    if end_date is None:
        end_date = datetime.now() + timedelta(days=1)
    """Backwards compatible wrapper for existing ETF data fetching."""
    global fetched_data_cache
    data = {}

    # At the moment, this array will only be 1 symbol
    for sym in symbols:
        clean_sym = sym.strip().upper()  # Normalize symbol for consistent caching

        if clean_sym in fetched_data_cache:
            print(f"✅ Using cached data for {clean_sym}")
            data[clean_sym] = fetched_data_cache[clean_sym]
            continue
        
        try:
            df = data_manager.get_etf_data(clean_sym, start_date, end_date)
            
            if df.empty:
                print(f"⚠️ No price data available for {sym}. Skipping.")
                continue
                
            data[sym] = df
            fetched_data_cache[sym] = df
            
        except Exception as e:
            print(f"❌ Error fetching {sym} from data sources: {e}")

    return data

# Load futures data from local files from ninja trader (now handled by data_sources module)
def load_futures_data(symbol):
    """Backwards compatible wrapper for local futures data loading."""
    return load_futures_data_from_csv(symbol)


 
# Computes Wilder's ATR based on historical OHLC data
def wilders_atr(df, period):
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()


# Evaluates a calculation rule such as entry thresholds, ATRs, or stop offsets
def evaluate_formula(formula, df, signal_date, symbol, entry_price=None, stop_price=None):
    # Symbol map and tick size
    symbol_code_map = {
        "6A": "A", "6B": "B", "6C": "C", "6E": "E", "6S": "S",
        "ES": "ES", "NQ": "NQ", "RTY": "RTY", "YM": "YM"
    }
    code = symbol_code_map.get(symbol, symbol)
    tick_size = TICK_SIZE.get(symbol, 0.01)

    # Auto-detect if this is an index future (uses points) vs currency future (uses ticks)
    is_index_future = symbol in ["ES", "YM", "NQ", "RTY"]
    offset_multiplier = 1.0 if is_index_future else tick_size

    # Resolve offset_ticks safely
    offset_ticks = None
    needs_offset = formula["type"] in {
        "lookback_extreme_plus_offset",
        "high_offset",
        "low_offset",
        "max_of_atr_or_low_offset",
    }

    if "offset_value" in formula:
        offset_ticks = formula["offset_value"]
    elif "offset_by_symbol" in formula and code in formula["offset_by_symbol"]:
        offset_ticks = formula["offset_by_symbol"][code]
    elif "offset_ticks" in formula:
        offset_ticks = formula["offset_ticks"]
    elif needs_offset:
        raise ValueError(f"Missing offset_value, offset_by_symbol, or offset_ticks for {symbol} in formula: {formula}")

    # === Formula Logic ===
    if formula["type"] == "lookback_extreme_plus_offset":
        end_idx = df.index.get_loc(signal_date)
        start_idx = max(0, end_idx - formula["lookback_days"] + 1)
        sub_df = df.iloc[start_idx : end_idx + 1]
        base = (
            sub_df[formula["price_field"]].max()
            if formula["agg_func"] == "max"
            else sub_df[formula["price_field"]].min()
        )
        return base + offset_ticks * formula["tick_size"]

    elif formula["type"] == "high_offset":
        days = formula.get("lookback_days", 1)
        include_signal = formula.get("include_signal_day", True)
        end_idx = df.index.get_loc(signal_date)
        start_idx = end_idx - days + (1 if include_signal else 0)

        sub_df = df.iloc[start_idx:end_idx + 1] if include_signal else df.iloc[start_idx:end_idx]
        base_high = sub_df["High"].max()

        threshold = base_high + offset_ticks * offset_multiplier
        return threshold

    elif formula["type"] == "low_offset":
        days = formula.get("lookback_days", 1)
        include_signal = formula.get("include_signal_day", True)
        end_idx = df.index.get_loc(signal_date)
        start_idx = end_idx - days + (1 if include_signal else 0)

        sub_df = df.iloc[start_idx:end_idx + 1] if include_signal else df.iloc[start_idx:end_idx]
        base_low = sub_df["Low"].min()

        threshold = base_low - offset_ticks * offset_multiplier
        return threshold
    
    elif formula["type"] == "max_of_atr_or_high_offset":
        if entry_price is None:
            raise ValueError("entry_price is required for max_of_atr_or_high_offset")

        atr_length = formula.get("atr_length", 5)
        atr_mult = formula.get("atr_multiplier", 1.4)

        base_high = df.loc[signal_date]["High"]
        offset_stop = base_high + offset_ticks * offset_multiplier

        atr = wilders_atr(df, atr_length).loc[signal_date]
        atr_stop = entry_price + atr * atr_mult

        # For short trades: use the *higher* stop to give more room above
        return max(offset_stop, atr_stop)

    elif formula["type"] == "max_of_atr_or_low_offset":
        if entry_price is None:
            raise ValueError("entry_price is required for max_of_atr_or_low_offset")

        atr_length = formula.get("atr_length", 5)
        atr_mult = formula.get("atr_multiplier", 1.4)

        base_low = df.loc[signal_date]["Low"]
        offset_stop = base_low - offset_ticks * offset_multiplier

        atr = wilders_atr(df, atr_length).loc[signal_date]
        atr_stop = entry_price - atr * atr_mult

        # For long trades: use the *lower* stop to give more room below
        return min(offset_stop, atr_stop)


    elif formula["type"] == "atr_multiple":
        atr = wilders_atr(df, formula["atr_length"])
        return atr.loc[signal_date] * formula["multiplier"]

    elif formula["type"] == "atr_offset_range":
        atr = wilders_atr(df, formula["atr_length"])
        return [atr.loc[signal_date] * m for m in formula["multipliers"]]
    
    elif formula["type"] == "fixed_atr_target":
        if entry_price is None:
            raise ValueError("entry_price is required for fixed_atr_target")

        atr_length = formula.get("atr_length", 5)
        atr = wilders_atr(df, atr_length).loc[signal_date]
        direction = formula.get("direction", "buy").lower()

        if direction == "sell":
            return entry_price - atr * 0.6
        else:
            return entry_price + atr * 0.6

    elif formula["type"] == "risk_ratio_target":
        if entry_price is None or stop_price is None:
            raise ValueError("entry_price and stop_price required for risk_ratio_target")
        risk = entry_price - stop_price
        return [entry_price + risk * m for m in formula["multipliers"]]
    
    elif formula["type"] == "atr_percentage":
        # ATR-based targets: configurable percentage of ATR (60%, 70%) with configurable ATR length (5, 20)
        atr_length = formula.get("atr_length", 5)
        percentage = formula.get("percentage", 0.6)  # default 60%
        direction = formula.get("direction", "buy")
        
        atr_values = wilders_atr(df, atr_length)
        signal_idx = df.index.get_loc(signal_date)
        atr_value = atr_values.iloc[signal_idx]
        
        if direction == "buy":
            return entry_price + (atr_value * percentage)
        else:  # sell
            return entry_price - (atr_value * percentage)
    
    elif formula["type"] == "entry_stop_percentage":
        # Percentage of entry-to-stop difference: configurable % (40%, 45%, 50%)
        if entry_price is None or stop_price is None:
            raise ValueError("entry_price and stop_price required for entry_stop_percentage")
        
        percentage = formula.get("percentage", 0.4)  # default 40%
        direction = formula.get("direction", "buy")
        
        entry_stop_diff = abs(entry_price - stop_price)
        
        if direction == "buy":
            return entry_price + (entry_stop_diff * percentage)
        else:  # sell
            return entry_price - (entry_stop_diff * percentage)
    
    elif formula["type"] == "multi_target":
        # Calculate specified target options, rank by ticks, and select by rank
        if entry_price is None or stop_price is None:
            raise ValueError("entry_price and stop_price required for multi_target")
        
        target_rank = formula.get("target_rank", 1)  # default to smallest target
        direction = formula.get("direction", "buy")
        target_options = formula.get("target_options", [])
        
        if not target_options:
            raise ValueError("target_options array required for multi_target")
        
        result = calculate_multi_targets_custom(df, signal_date, symbol, entry_price, stop_price, direction, target_rank, target_options)
        return result["target_price"]
    
    elif formula["type"] == "open_or_better":
        # Entry trigger is simply the open price of the next trading day
        next_day_idx = df.index.get_loc(signal_date) + 1
        if next_day_idx >= len(df):
            raise ValueError(f"No trading data available after {signal_date} for {symbol}")
        next_day = df.index[next_day_idx]
        open_price = df.loc[next_day]["Open"]
        return open_price  # used as threshold


    else:
        raise ValueError(f"Unknown formula type: {formula['type']}")


def calculate_multi_targets(df, signal_date, symbol, entry_price, stop_price, direction="buy", target_rank=1):
    """
    Calculate all 5 target options, rank by tick count, and return selected target.
    
    Args:
        df: Price dataframe
        signal_date: Signal date
        symbol: Trading symbol
        entry_price: Entry price
        stop_price: Stop loss price  
        direction: "buy" or "sell"
        target_rank: Which target to select (1=smallest ticks, 2=2nd smallest, etc.)
    
    Returns:
        dict: {"target_price": float, "target_type": str, "tick_count": int}
    """
    tick_size = TICK_SIZE.get(symbol, 0.0001)
    
    # Define all 5 target calculations
    target_configs = [
        {"type": "atr_percentage", "atr_length": 5, "percentage": 0.6, "label": "ATR5 x 0.6"},
        {"type": "atr_percentage", "atr_length": 5, "percentage": 0.7, "label": "ATR5 x 0.7"},
        {"type": "atr_percentage", "atr_length": 20, "percentage": 0.6, "label": "ATR20 x 0.6"},
        {"type": "atr_percentage", "atr_length": 20, "percentage": 0.7, "label": "ATR20 x 0.7"},
        {"type": "entry_stop_percentage", "percentage": 0.4, "label": "Entry-Stop x 0.4"},
        {"type": "entry_stop_percentage", "percentage": 0.45, "label": "Entry-Stop x 0.45"},
        {"type": "entry_stop_percentage", "percentage": 0.5, "label": "Entry-Stop x 0.5"}
    ]
    
    targets = []
    
    for config in target_configs:
        try:
            # Create formula dict for evaluate_formula
            formula = {
                "type": config["type"],
                "direction": direction
            }
            
            if config["type"] == "atr_percentage":
                formula["atr_length"] = config["atr_length"]
                formula["percentage"] = config["percentage"]
            elif config["type"] == "entry_stop_percentage":
                formula["percentage"] = config["percentage"]
            
            # Calculate target price
            target_price = evaluate_formula(formula, df, signal_date, symbol, 
                                          entry_price=entry_price, stop_price=stop_price)
            
            # Calculate tick count from entry to target
            price_diff = abs(target_price - entry_price)
            tick_count = round(price_diff / tick_size)
            
            targets.append({
                "target_price": target_price,
                "target_type": config["label"],
                "tick_count": tick_count,
                "config": config
            })
            
        except Exception as e:
            print(f"⚠️  Error calculating {config['label']}: {e}")
            continue
    
    if not targets:
        raise ValueError("No valid targets could be calculated")
    
    # Sort by tick count (ascending - smallest first)
    targets.sort(key=lambda x: x["tick_count"])
    
    # Select the target by rank (1-based indexing)
    if target_rank < 1 or target_rank > len(targets):
        print(f"⚠️  Target rank {target_rank} out of range (1-{len(targets)}), using rank 1")
        target_rank = 1
    
    selected_target = targets[target_rank - 1]
    
    # Debug output - will be controlled by main debug flag
    print(f"🎯 Multi-target analysis for {symbol}:")
    for i, target in enumerate(targets, 1):
        marker = "👈 SELECTED" if i == target_rank else ""
        print(f"   {i}. {target['target_type']}: ${target['target_price']:.5f} ({target['tick_count']} ticks) {marker}")
    
    return selected_target


def calculate_multi_targets_custom(df, signal_date, symbol, entry_price, stop_price, direction="buy", target_rank=1, target_options=None):
    """
    Calculate custom target options, rank by tick count, and return selected target.
    
    Args:
        df: Price dataframe
        signal_date: Signal date
        symbol: Trading symbol
        entry_price: Entry price
        stop_price: Stop loss price  
        direction: "buy" or "sell"
        target_rank: Which target to select (1=smallest ticks, 2=2nd smallest, etc.)
        target_options: Array of target option dicts from JSON config
    
    Returns:
        dict: {"target_price": float, "target_type": str, "tick_count": int}
    """
    if not target_options:
        raise ValueError("target_options required")
    
    tick_size = TICK_SIZE.get(symbol, 0.0001)
    targets = []
    
    for i, option in enumerate(target_options):
        try:
            # Create formula dict for evaluate_formula
            formula = {
                "type": option["type"],
                "direction": direction
            }
            
            # Copy all option parameters to formula
            for key, value in option.items():
                if key != "type":
                    formula[key] = value
            
            # Calculate target price
            target_price = evaluate_formula(formula, df, signal_date, symbol, 
                                          entry_price=entry_price, stop_price=stop_price)
            
            # Calculate tick count from entry to target
            price_diff = abs(target_price - entry_price)
            tick_count = round(price_diff / tick_size)
            
            # Generate label for display
            if option["type"] == "atr_percentage":
                atr_length = option.get("atr_length", 5)
                percentage = option.get("percentage", 0.6)
                label = f"ATR{atr_length} x {percentage}"
            elif option["type"] == "entry_stop_percentage":
                percentage = option.get("percentage", 0.4)
                label = f"Entry-Stop x {percentage}"
            else:
                label = f"{option['type']} #{i+1}"
            
            targets.append({
                "target_price": target_price,
                "target_type": label,
                "tick_count": tick_count,
                "config": option
            })
            
        except Exception as e:
            print(f"⚠️  Error calculating target option {i+1}: {e}")
            continue
    
    if not targets:
        raise ValueError("No valid targets could be calculated from target_options")
    
    # Sort by tick count (ascending - smallest first)
    targets.sort(key=lambda x: x["tick_count"])
    
    # Select the target by rank (1-based indexing)
    if target_rank < 1 or target_rank > len(targets):
        print(f"⚠️  Target rank {target_rank} out of range (1-{len(targets)}), using rank 1")
        target_rank = 1
    
    selected_target = targets[target_rank - 1]
    
    # Debug output showing all calculated targets
    print(f"🎯 Multi-target analysis for {symbol} ({len(targets)} options):")
    for i, target in enumerate(targets, 1):
        marker = "👈 SELECTED" if i == target_rank else ""
        print(f"   {i}. {target['target_type']}: ${target['target_price']:.5f} ({target['tick_count']} ticks) {marker}")
    
    return selected_target


# Checks if an entry condition is met on a given signal date
def evaluate_entry(entry_conf, df, test_date, symbol, strategy_name, signal_date):
    formula = entry_conf["formula"]
    formula_type = formula.get("type")
    direction = formula.get("direction", "buy").lower()
    is_sell = "sell" in direction or "Sell" in strategy_name

    # print('signal_date', signal_date)
    # print('sig = date', )
    # temp_date = datetime.strptime("2025-05-30", "%Y-%m-%d").date()
    # if signal_date == temp_date:
    #     import pdb; pdb.set_trace()

    # type is in the json.  "open_or_better" is only for ETFs
    if formula_type == "open_or_better":
        try:
            next_day_idx = df.index.get_loc(signal_date) + 1
            next_day = df.index[next_day_idx]
        except IndexError:
            # print('evaluate_entry: Next day open has not occured yet.')
            return False, None
        
        open_price = df.loc[next_day]["Open"]

        is_etf_strategy = "etf" in strategy_name.lower()
        if is_etf_strategy:
            print(f"[{symbol}] {strategy_name} ETF entry on {next_day.strftime('%Y-%m-%d')} at open {open_price:.5f}")
            return True, open_price  
        else:
            # Standard: only enter if price goes beyond open
            price = df.loc[test_date]["Low"] if direction == "buy" else df.loc[test_date]["High"]
            triggered = price <= open_price if direction == "buy" else price >= open_price

            print(f"[{symbol}] {strategy_name} on {test_date.strftime('%Y-%m-%d (%a)')} - Price: {price:.5f}, Open: {open_price:.5f}, Triggered: {triggered}")
            return triggered, price if triggered else None

    else:
        threshold = evaluate_formula(formula, df, signal_date, symbol)
        price = df.loc[test_date]["Low"] if is_sell else df.loc[test_date]["High"]
        triggered = price <= threshold if is_sell else price >= threshold

        print(f"[{symbol}] {strategy_name} on {test_date.strftime('%Y-%m-%d (%a)')} - Price: {price:.5f}, Threshold: {threshold:.5f}, Triggered: {triggered}")
        return triggered, price if triggered else None


# This function is now handled by trading strategy classes
# Kept for backwards compatibility during refactoring
def evaluate_exit(strategy, formulas, df, signal_date, symbol, entry_price, strategy_name, direction, trading_strategy=None):
    """Legacy function - now delegated to trading strategy classes."""
    if trading_strategy:
        return trading_strategy.evaluate_exit(strategy, formulas, df, signal_date, symbol, entry_price, strategy_name, direction)
    
    # Fallback to original logic if no trading_strategy provided
    stop_formula = strategy["stop"]["formula"]
    stop_price = evaluate_formula(stop_formula, df, signal_date, symbol, entry_price=entry_price)

    target_formula = strategy["target"]["formula"]
    target_result = evaluate_formula(target_formula, df, signal_date, symbol, entry_price=entry_price, stop_price=stop_price)

    if isinstance(target_result, list):
        target_price = max(target_result)
    else:
        target_price = target_result

    return stop_price, target_price


def get_target_type(strategy, trading_strategy=None):
    """Extract target type information for display purposes."""
    if trading_strategy:
        return trading_strategy.get_target_type(strategy)
    
    # Fallback to original logic
    try:
        target_formula = strategy.get("target", {}).get("formula", {})
        if target_formula.get("type") == "atr_multiple":
            atr_length = target_formula.get("atr_length", 5)
            multiplier = target_formula.get("multiplier", 1.0)
            return f"ATR{atr_length} x {multiplier}"
        elif target_formula.get("type") == "fixed_atr_target":
            atr_length = target_formula.get("atr_length", 5)
            return f"ATR{atr_length} x 0.6"  # fixed_atr_target uses 0.6 multiplier
        else:
            return target_formula.get("type", "Unknown")
    except:
        return "Unknown"

def simulate_trade(strategy, symbol, df, signal_date, strategy_name, direction=None, trading_strategy=None):
    # print(f'simulate_trade: direction {direction}')
    # print('Enter simulate_trade')

    formulas = {
        "entry": strategy["entry"]["formula"],
        "target": strategy["target"]["formula"]
    }

    # Handle stop formula only if defined
    if strategy.get("stop") and strategy["stop"].get("formula"):
        formulas["stop"] = strategy["stop"]["formula"]
        has_stop = True
    else:
        has_stop = False

    # If no trigger window is defined, assume open-ended trade
    max_days = strategy.get("trigger_window_days", len(df))
    # print(f'signal_date: {signal_date} in df.index: {df.index} (signal_date not in df.index: {(signal_date not in df.index)}')

    # Ensure df.index is datetime and normalized (no time component), just like signal_date
    df.index = pd.to_datetime(df.index).normalize()

    if signal_date not in df.index:
        return {"status": "Signal Date Missing in Data"}

    signal_idx = df.index.get_loc(signal_date)
    valid_dates = df.index[signal_idx: signal_idx + max_days]
    # print('valid_dates:',valid_dates)

    for test_date in valid_dates:
        triggered, entry_price = evaluate_entry(
            strategy["entry"],
            df,
            test_date,
            symbol,
            strategy_name,
            signal_date
        )

        if triggered:
            stop_price, target_price = evaluate_exit(
                strategy, formulas, df, test_date, symbol, entry_price, strategy_name, direction, trading_strategy
            )

            post_entry = df[df.index > test_date]

            for i, row in post_entry.iterrows():
                # Check target hit using trading strategy
                if trading_strategy and trading_strategy.check_target_hit(row, target_price, direction):
                    return {
                        "status": "Target Hit",
                        "entry": round(entry_price, 5),
                        "stop": round(stop_price, 5) if isinstance(stop_price, (float, int)) else ("Expired" if isinstance(stop_price, date) else ""),
                        "target": round(target_price, 5),
                        "entry_date": test_date.strftime('%Y-%m-%d'),
                        "exit_date": i.strftime('%Y-%m-%d')
                    }
                # Fallback to original logic
                elif not trading_strategy:
                    if direction == "buy" and row["High"] >= target_price:
                        return {
                            "status": "Target Hit",
                            "entry": round(entry_price, 5),
                            "stop": round(stop_price, 5) if has_stop else "",
                            "target": round(target_price, 5),
                            "entry_date": test_date.strftime('%Y-%m-%d'),
                            "exit_date": i.strftime('%Y-%m-%d')
                        }
                    elif direction == "sell" and row["Low"] <= target_price:
                        return {
                            "status": "Target Hit",
                            "entry": round(entry_price, 5),
                            "stop": round(stop_price, 5) if has_stop else "",
                            "target": round(target_price, 5),
                            "entry_date": test_date.strftime('%Y-%m-%d'),
                            "exit_date": i.strftime('%Y-%m-%d')
                        }

                # Check stop hit using trading strategy
                if trading_strategy:
                    stop_result = trading_strategy.check_stop_hit(row, stop_price, direction, has_stop, i)
                    if stop_result["hit"]:
                        return {
                            "status": stop_result["status"],
                            "entry": round(entry_price, 5),
                            "stop": stop_result["stop_display"],
                            "target": round(target_price, 5),
                            "entry_date": test_date.strftime('%Y-%m-%d'),
                            "exit_date": i.strftime('%Y-%m-%d')
                        }
                # Fallback to original stop logic
                elif isinstance(stop_price, date):
                    if i.date() >= stop_price:
                        return {
                            "status": "Expired",
                            "entry": round(entry_price, 5),
                            "stop": "Expired",
                            "target": round(target_price, 5),
                            "entry_date": test_date.strftime('%Y-%m-%d'),
                            "exit_date": i.strftime('%Y-%m-%d')
                        }
                elif has_stop:
                    if direction == "buy" and row["Low"] <= stop_price:
                        return {
                            "status": "Stopped out",
                            "entry": round(entry_price, 5),
                            "stop": round(stop_price, 5),
                            "target": round(target_price, 5),
                            "entry_date": test_date.strftime('%Y-%m-%d'),
                            "exit_date": i.strftime('%Y-%m-%d')
                        }
                    elif direction == "sell" and row["High"] >= stop_price:
                        return {
                            "status": "Stopped out",
                            "entry": round(entry_price, 5),
                            "stop": round(stop_price, 5),
                            "target": round(target_price, 5),
                            "entry_date": test_date.strftime('%Y-%m-%d'),
                            "exit_date": i.strftime('%Y-%m-%d')
                        }
            return {
                "status": "Open",
                "entry": round(entry_price, 5),
                "stop": (
                    round(stop_price, 5) if isinstance(stop_price, (float, int))
                    else "Active" if isinstance(stop_price, date)
                    else ""
                ),
                "target": round(target_price, 5),
                "entry_date": test_date.strftime('%Y-%m-%d'),
                "exit_date": "",
            }


    # If not triggered, the for ETF that means there is no next day yet and the return dict values will be empty
    return {
        "status": "Expired - Entry Not Triggered",
        "entry": "",
        "stop": "",
        "target": "",
        "entry_date": "",
        "exit_date": ""
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Futures Strategy Backtester")
    parser.add_argument("--mode", choices=["futures", "etfs"], required=True, help="Which type of trades to backtest")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for matched strategy names")
    parser.add_argument("--etf-source", choices=["polygon", "insightsentry"], help="Data source for ETF data (overrides config)")
    parser.add_argument("--test-auth", action="store_true", help="Test authentication for the selected data source and exit")
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    
    # Determine data source to use
    if args.mode == "etfs":
        etf_source = args.etf_source or config["data_sources"]["etf_data_source"]
        print(f"📊 Using {etf_source} for ETF data")
    else:
        etf_source = "polygon"  # Not used for futures mode
    
    # Test authentication if requested
    if args.test_auth:
        source_to_test = etf_source if args.mode == "etfs" else config["data_sources"]["futures_data_source"]
        print(f"🔐 Testing authentication for {source_to_test}...")
        auth_result = data_manager.test_authentication(source_to_test)
        if auth_result.get(source_to_test, False):
            print("✅ Authentication successful!")
            return
        else:
            print("❌ Authentication failed!")
            return

    # Choose file based on mode
    input_file = "trade_signals_futures.csv" if args.mode == "futures" else "trade_signals_ETFs.csv"
    strategy_path = "strategies_complete.json"
    output_file = "trade_results_futures.csv" if args.mode == "futures" else "trade_results_ETFs.csv"

    strategies = load_strategies(strategy_path)
    df_signals = pd.read_csv(input_file, encoding="ISO-8859-1")
    
    # Create appropriate trading strategy based on mode
    trading_strategy = StrategyFactory.create_strategy(
        mode=args.mode, 
        symbol_mappings=symbol_map, 
        tick_sizes=TICK_SIZE
    )
    
    # For futures, set the evaluate_formula function
    if args.mode == "futures":
        trading_strategy.set_evaluate_formula_function(evaluate_formula)
    
    # Filter out empty rows (Excel often adds millions of empty rows)
    df_signals = df_signals.dropna(how='all')  # Remove completely empty rows
    df_signals = df_signals.dropna(subset=['symbol', 'strategy'])  # Remove rows missing key columns

    results = []

    for _, row in df_signals.iterrows():
        symbol = str(row['symbol']).strip()

        # The "buy or sell" and "frequency" columns are only in the trade_signals_ETFs file, 
        # so this code sets values for when runing on futures   
        # Safely parse 'frequency'
        freq_raw = row.get("frequency", "Daily")
        freq = str(freq_raw).strip().capitalize() if pd.notna(freq_raw) else "Daily"

        # Safely parse 'buy or sell' as direction
        direction_raw = row.get("direction", "")
        if pd.isna(direction_raw) or not direction_raw:
            print(f"❌ Invalid or missing direction for row: {row.name}")
            continue
        direction = str(direction_raw).strip().lower()

        # For ETFs, try to use actual strategy name first, then fall back to generic
        raw_strategy_name = str(row.get("strategy", "")).strip()
        
        # List of known daily ETF strategy patterns that should map to generic daily strategies
        daily_etf_patterns = [
            "Donchian", "Ichimoku", "ETF Squeeze", "Squeeze Play", "Stochastics", 
            "20.8 Trigger", "Gap and Go", "Put/Call Buy"
        ]
        
        # Check if raw name matches any daily pattern and frequency is Daily
        is_daily_strategy = freq.lower() == "daily" and any(pattern in raw_strategy_name for pattern in daily_etf_patterns)
        
        if is_daily_strategy:
            resolved_name = f"{freq} ETF Options {direction.capitalize()}"
        else:
            # Use actual name for Weekly strategies or exact matches
            resolved_name = raw_strategy_name if raw_strategy_name else f"{freq} ETF Options {direction.capitalize()}"

        try:
            # signal_date = pd.to_datetime(row['date'], format='%m/%d/%y')
            # Make sure signal_date is in the exact same format as df.index
            signal_date = pd.to_datetime(row['date'], format='%m/%d/%y').normalize()
            # print('signal_date:',signal_date)
            
        except ValueError:
            print(f"❌ Invalid date format (must be MM/DD/YY): '{row['date']}'")
            results.append({
                "symbol": symbol,
                "strategy": row.get("strategy", ""),
                "signal_date": row['date'],
                "status": "Error - Invalid Date Format",
                "entry_date": "",
                "exit_date": ""
            })
            continue

        # Check the signal date is actually a trading day.  if not, it's a mistake in the input file.
        if not util.is_trading_day(signal_date):
            print(f"⚠️ {symbol} signal date on {signal_date} is not a trading day. Skipping.")
            continue

        # Convert alt futures symbol codes
        alternate_symbols = {"AD": "6A", "BP": "6B", "CD": "6C", "EC": "6E", "SF": "6S"}
        if args.mode == "futures" and symbol in alternate_symbols:
            symbol = alternate_symbols[symbol]

        # Resolve strategy name for futures mode
        if args.mode == "futures":
            raw_name = str(row["strategy"]).strip()
            resolved_name = resolve_strategy_name(raw_name, list(strategies.keys()))

        if not resolved_name or resolved_name not in strategies:
            print(f"⚠️ Could not resolve strategy name: '{resolved_name}'")
            results.append({
                "symbol": symbol,
                "strategy": resolved_name,
                "signal_date": signal_date.strftime('%m/%d/%y'),
                "status": "Error - Strategy Not Recognized",
                "entry_date": "",
                "exit_date": ""
            })
            continue

        if args.debug:
            print(f"→ Matched to strategy: '{resolved_name}'")

        # Retrieve the symbol data
        # - at the moment, this is just 1 symbol at a time
        if symbol in fetched_data_cache:
            df = fetched_data_cache[symbol]
        else:
            # Use unified data fetching for both ETFs and futures
            df = get_price_data(symbol, args.mode, fetched_data_cache, etf_source=etf_source)
            if df is None:
                print(f"⚠️ No price data available for {symbol}. Skipping.")
                continue

            fetched_data_cache[symbol] = df
        
        # print('sym data for',symbol)
        # print(df.tail)
        # print('Close:',df['Close'].iloc[-1])
        result = simulate_trade(strategies[resolved_name], symbol, df, signal_date, resolved_name, direction=direction, trading_strategy=trading_strategy)
        target_type = get_target_type(strategies[resolved_name], trading_strategy=trading_strategy)

        entry_date_str = result.get("entry_date", "")
        entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d').date() if entry_date_str else None
        if not entry_date and args.mode == "etfs":
            print(f"⚠️ {symbol} signal date on {signal_date} is the most recent date.  No next day open price yet. Skipping.")
            continue

        # Checks if exit_date is non-empty before parsing, falls back to 'open' if no exit has occurred, and 
        # adds a small safeguard: if for some reason entry_date is None, it won’t crash when trying to calculate num_days_open.
        exit_date_str = result.get("exit_date", "")
        if exit_date_str:
            exit_date = datetime.strptime(exit_date_str, '%Y-%m-%d').date()
            num_days_open = (exit_date - entry_date).days if entry_date else "?"
        else:
            exit_date = None
            num_days_open = 'open'
            # print('\'num_days_open\'', num_days_open)

        entry_price = result.get("entry", "")
        last_close_price = df['Close'].iloc[-1]

        # If current close is better than the next day open, then still can enter
        # for buy, close must be lower than next day open.  for sell, close must e high
        # a negative number means the trade is still valid to get in.
        if entry_price and isinstance(entry_price, (int, float)):
            if direction == 'buy':
                diff_from_entry = last_close_price - entry_price
            else:    
                diff_from_entry = entry_price - last_close_price
        else:
            diff_from_entry = "" 

        # Calculate expiration date for ETF options (2 full months out)
        expiration_date = util.get_final_expiration_date(signal_date, months_out=2)
        
        results.append({
            "symbol": symbol,
            # "strategy": resolved_name,
            "strategy": str(row["strategy"]).strip(), # just get the actual strstegy name, not the reduced
            "direction": direction,
            "signal_date": signal_date.strftime('%m/%d/%y'),
            "status": result["status"],
            "expiration": expiration_date.strftime('%m/%d/%y') if expiration_date else "",
            "target_type": target_type,
            "entry_price": result.get("entry", ""),
            "target_price": result.get("target", ""),
            "last_close_price": last_close_price,
            "diff_from_entry": round(diff_from_entry, 2) if isinstance(diff_from_entry, (int, float)) else diff_from_entry,
            "entry_date": result.get("entry_date", ""),
            "exit_date": result.get("exit_date", ""),
            "num_days_open": num_days_open
        })
        
    print(f"Total processed signals: {len(results)}")
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"✅ Backtest completed. Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
