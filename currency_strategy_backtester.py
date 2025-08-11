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

from polygon.rest import RESTClient
POLYGON_API_KEY = 'qFTWmhmyAj2pJqpx0Pwxp2AyShKFVPv9'
client = RESTClient(api_key = POLYGON_API_KEY)
print("Import succeeded!")
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
def get_price_data(symbol, mode, cache, start_date=datetime(2025, 1, 2)):
    if symbol in cache:
        return cache[symbol]

    if mode == "futures":
        df = load_futures_data(symbol)
        if df is None:
            print(f"❌ Could not load data for futures symbol {symbol}")
            return None
    else:  # ETF
        try:
            end_date = datetime.today() + timedelta(days=1)  # buffer in case you're running same-day
            all_data = fetch_price_data([symbol], start_date, end_date)

            if symbol not in all_data:
                raise ValueError(f"{symbol} was not returned by fetch_price_data")

            df = all_data[symbol]
            print(f"{symbol}: price range {df.index.min().date()} to {df.index.max().date()}")

        except Exception as e:
            print(f"❌ Failed to fetch ETF data for {symbol}: {e}")
            return None

    cache[symbol] = df
    return df

# Get the ticker data from Polygon 
def fetch_price_data(symbols, start_date, end_date):
    global fetched_data_cache
    data = {}

    # print('Start fetch_price_data. Symbols:', symbols)

    # At the moment, this array will only be 1 symbol
    for sym in symbols:
        clean_sym = sym.strip().upper()  # Normalize symbol for consistent caching

        if clean_sym in fetched_data_cache:
            print(f"✅ Using cached data for {clean_sym}")
            data[clean_sym] = fetched_data_cache[clean_sym]
            # print(data)
            continue
        
        print(f"📡 Fetching {sym} from Polygon (Attempt 1)")
        success = False
        df = None  # prevent unbound local error
        for attempt in range(10):
            if attempt > 0:
                print(f"📡 Fetching {sym} from Polygon (Attempt {attempt + 1})")
                time.sleep(2)

            try:
                aggs = client.get_aggs(
                    ticker=sym,
                    multiplier=1,
                    timespan="day",
                    from_=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    limit=50000
                )

                if not aggs:
                    raise ValueError("Empty response")

                df = pd.DataFrame([{
                    "Date": pd.to_datetime(bar.timestamp, unit='ms'),
                    "Open": bar.open,
                    "High": bar.high,
                    "Low": bar.low,
                    "Close": bar.close
                } for bar in aggs])

                df.set_index("Date", inplace=True)
                df.index = pd.to_datetime(df.index).normalize()
                df.sort_index(inplace=True)

                data[sym] = df
                fetched_data_cache[sym] = df
                print(f"{sym}: fetched {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
                success = True
                break

            except Exception as e:
                print(f"❌ Failed to fetch ETF data for {sym}: {e}")
                continue

            except Exception as e:
                print(f"❌ Error fetching {sym} from Polygon: {e}")

        if not success:
            print(f"⚠️ No price data available for {sym}. Skipping.")

    # print('Exit fetch_price_data')
    return data

# Load futures data from local files from ninja trader
def load_futures_data(symbol):
    file_path = f"./data/{symbol}.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        print(f"Error loading futures data from {file_path}: {e}")
        return None

 
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

    # Resolve offset_ticks safely
    offset_ticks = None
    needs_offset = formula["type"] in {
        "lookback_extreme_plus_offset",
        "high_offset_ticks",
        "low_offset_ticks",
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

    elif formula["type"] == "high_offset_ticks":
        days = formula.get("lookback_days", 1)
        include_signal = formula.get("include_signal_day", True)
        end_idx = df.index.get_loc(signal_date)
        start_idx = end_idx - days + (1 if include_signal else 0)

        sub_df = df.iloc[start_idx:end_idx + 1] if include_signal else df.iloc[start_idx:end_idx]
        base_high = sub_df["High"].max()

        threshold = base_high + offset_ticks * tick_size
        return threshold

    elif formula["type"] == "low_offset_ticks":
        days = formula.get("lookback_days", 1)
        include_signal = formula.get("include_signal_day", True)
        end_idx = df.index.get_loc(signal_date)
        start_idx = end_idx - days + (1 if include_signal else 0)

        sub_df = df.iloc[start_idx:end_idx + 1] if include_signal else df.iloc[start_idx:end_idx]
        base_low = sub_df["Low"].min()

        threshold = base_low - offset_ticks * tick_size
        return threshold
    
    elif formula["type"] == "max_of_atr_or_high_offset":
        if entry_price is None:
            raise ValueError("entry_price is required for max_of_atr_or_high_offset")

        atr_length = formula.get("atr_length", 5)
        atr_mult = formula.get("atr_multiplier", 1.4)

        base_high = df.loc[signal_date]["High"]
        offset_stop = base_high + offset_ticks * tick_size

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
        offset_stop = base_low - offset_ticks * tick_size

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


def evaluate_exit(strategy, formulas, df, signal_date, symbol, entry_price, strategy_name, direction):
    # Handle ETF logic using provided direction
    # print(f'evaluate_exit, direction {direction}')
    if direction in ("buy", "sell"):
        atr = wilders_atr(df, 5)
        atr_value = atr.loc[signal_date]  # ✅ ATR from the signal date only
        
        # Th target is 1x ATR(5)
        if direction == "sell":
            target_price = entry_price - atr_value
        else:
            target_price = entry_price + atr_value
        
        # print('atr_value:', atr_value)
        # print('direction:', direction)
        # print('entry_price:', entry_price)
        # print('target_price:', target_price)

        # stop_price = None  # ETF options have no stop
        # 🟡 Add expiration-based stop for ETFs
        stop_price = util.get_final_expiration_date(signal_date, months_out=2)
        return stop_price, round(target_price, 2)

    # Fallback for futures or other strategies
    stop_formula = strategy["stop"]["formula"]
    stop_price = evaluate_formula(stop_formula, df, signal_date, symbol, entry_price=entry_price)

    target_formula = strategy["target"]["formula"]
    target_result = evaluate_formula(target_formula, df, signal_date, symbol, entry_price=entry_price, stop_price=stop_price)

    if isinstance(target_result, list):
        target_price = max(target_result)
    else:
        target_price = target_result

    return stop_price, target_price


def get_target_type(strategy):
    """Extract target type information for display purposes."""
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

def simulate_trade(strategy, symbol, df, signal_date, strategy_name, direction=None):
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
                strategy, formulas, df, test_date, symbol, entry_price, strategy_name, direction
            )

            post_entry = df[df.index > test_date]

            for i, row in post_entry.iterrows():
                # Check target hit
                # Buy: price must rise to or above target
                if direction == "buy" and row["High"] >= target_price:
                    return {
                        "status": "Target Hit",
                        "entry": round(entry_price, 5),
                        "stop": round(stop_price, 5) if has_stop else "",
                        "target": round(target_price, 5),
                        "entry_date": test_date.strftime('%Y-%m-%d'),
                        "exit_date": i.strftime('%Y-%m-%d')
                    }
                # Sell: price must fall to or below target
                elif direction == "sell" and row["Low"] <= target_price:
                    return {
                        "status": "Target Hit",
                        "entry": round(entry_price, 5),
                        "stop": round(stop_price, 5) if has_stop else "",
                        "target": round(target_price, 5),
                        "entry_date": test_date.strftime('%Y-%m-%d'),
                        "exit_date": i.strftime('%Y-%m-%d')
                    }

                # Check stop_price hit for ETFs with stop as a date
                if isinstance(stop_price, date):
                    if i.date() >= stop_price:
                        return {
                            "status": "Expired",
                            "entry": round(entry_price, 5),
                            "stop": "Expired",
                            "target": round(target_price, 5),
                            "entry_date": test_date.strftime('%Y-%m-%d'),
                            "exit_date": i.strftime('%Y-%m-%d')
                        }
                    
                # Check stop_price hit for futures with stop as a price
                if has_stop:
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
    args = parser.parse_args()

    # Choose file based on mode
    input_file = "trade_signals_futures.csv" if args.mode == "futures" else "trade_signals_ETFs.csv"
    strategy_path = "strategies_complete.json"
    output_file = "trade_results_futures.csv" if args.mode == "futures" else "trade_results_ETFs.csv"

    strategies = load_strategies(strategy_path)
    df_signals = pd.read_csv(input_file, encoding="ISO-8859-1")
    
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
            if args.mode == "futures":
                df = util.get_price_data_from_file(symbol)
                if df is None:
                    print(f"❌ Could not load data for futures symbol {symbol}")
                    continue
            else:
                df = get_price_data(symbol, args.mode, fetched_data_cache)
                if df is None:
                    print(f"⚠️ No price data available for {symbol}. Skipping.")
                    continue

            fetched_data_cache[symbol] = df
        
        # print('sym data for',symbol)
        # print(df.tail)
        # print('Close:',df['Close'].iloc[-1])
        result = simulate_trade(strategies[resolved_name], symbol, df, signal_date, resolved_name, direction=direction)
        target_type = get_target_type(strategies[resolved_name])

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
        if direction == 'buy':
            diff_from_entry = last_close_price - entry_price
        else:    
            diff_from_entry = entry_price - last_close_price 

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
            "diff_from_entry": round(diff_from_entry, 2),
            "entry_date": result.get("entry_date", ""),
            "exit_date": result.get("exit_date", ""),
            "num_days_open": num_days_open
        })
        
    print(f"Total processed signals: {len(results)}")
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"✅ Backtest completed. Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
