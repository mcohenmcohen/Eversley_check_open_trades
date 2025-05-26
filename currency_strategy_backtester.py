import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import time
import os
from collections import defaultdict
from rapidfuzz import process
from alpha_vantage.timeseries import TimeSeries

from polygon.rest import RESTClient
POLYGON_API_KEY = 'qFTWmhmyAj2pJqpx0Pwxp2AyShKFVPv9'
client = RESTClient(api_key = POLYGON_API_KEY)
print("Import succeeded!")
fetched_data_cache = {}  # cache already-fetched symbols

AV_API_KEY = 'IN8A22IKNXHRN9P5'  # Replace with your Alpha Vantage key
TICK_SIZE = {
    # Currency futures
    "6A": 0.0001, "6B": 0.0001, "6C": 0.0001, "6E": 0.0001, "6S": 0.0001,

    # Common ETFs
    "DIA": 0.01, "EEM": 0.01, "EFA": 0.01, "EWG": 0.01, "EWH": 0.01,
    "FXI": 0.01, "GDX": 0.01, "GLD": 0.01, "IBB": 0.01, "IWM": 0.01,
    "IYR": 0.01, "QQQ": 0.01, "SLV": 0.01, "SPY": 0.01, "TLT": 0.01,
    "UNG": 0.01, "USO": 0.01, "VNQ": 0.01, "VWO": 0.01, "XHB": 0.01,
    "XLB": 0.01, "XLE": 0.01, "XLF": 0.01, "XLI": 0.01, "XLK": 0.01,
    "XLV": 0.01, "XME": 0.01, "XOP": 0.01, "XRT": 0.01
}

# # Maps internal CME futures symbols to 
# Yahoo Finance ticker symbols (for currencies, not needed now since i'm downloading them from ninjatrader)
# ETF tickers from Alpha Vantage
symbol_map = {
    # Currency futures
    "6A": "6A=F",
    "6B": "6B=F",
    "6C": "6C=F",
    "6E": "6E=F",
    "6S": "6S=F",

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

# Downloads historical OHLC data for all required symbols
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

def fetch_price_data(symbols, start_date, end_date):
    global fetched_data_cache
    data = {}

    for sym in symbols:
        clean_sym = sym.strip().upper()  # Normalize symbol for consistent caching

        if clean_sym in fetched_data_cache:
            print(f"✅ Using cached data for {clean_sym}")
            data[clean_sym] = fetched_data_cache[clean_sym]
            continue
        
        print(f"📡 Fetching {sym} from Polygon (Attempt 1)")
        success = False
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
                df.index = pd.to_datetime(df.index).normalize()  # 👈 Normalize to remove time component
                df.sort_index(inplace=True)
                data[sym] = df
                fetched_data_cache[sym] = df
                success = True
                break

            except Exception as e:
                print(f"❌ Error fetching {sym} from Polygon: {e}")

        if not success:
            print(f"⚠️ No price data available for {sym}. Skipping.")

    return data

 
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
    symbol_code_map = {"6A": "A", "6B": "B", "6C": "C", "6E": "E", "6S": "S"}
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

    # type is in the json.  "open_or_better" is only for ETFs
    if formula_type == "open_or_better":
        try:
            next_day_idx = df.index.get_loc(signal_date) + 1
            next_day = df.index[next_day_idx]
        except IndexError:
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
            return triggered, price if triggered else None, None

    else:
        threshold = evaluate_formula(formula, df, signal_date, symbol)
        price = df.loc[test_date]["Low"] if is_sell else df.loc[test_date]["High"]
        triggered = price <= threshold if is_sell else price >= threshold

        print(f"[{symbol}] {strategy_name} on {test_date.strftime('%Y-%m-%d (%a)')} - Price: {price:.5f}, Threshold: {threshold:.5f}, Triggered: {triggered}")
        return triggered, price if triggered else None


def evaluate_exit(strategy, formulas, df, signal_date, symbol, entry_price, strategy_name, direction):
    # Handle ETF logic using provided direction
    if direction in ("buy", "sell"):
        atr = wilders_atr(df, 5)
        atr_value = atr.loc[signal_date]  # ✅ ATR from the signal date only
        print('atr_value:', atr_value)
        print('direction:', direction)
        print('entry_price:', entry_price)

        if direction == "sell":
            target_price = entry_price - atr_value
        else:
            target_price = entry_price + atr_value
        print('target_price:', target_price)

        stop_price = None  # ETF options have no stop
        return stop_price, round(target_price, 5)

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


# Simulates a trade over time until it's triggered and closed or expires
def simulate_trade(strategy, symbol, df, signal_date, strategy_name, direction=None):

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

    if signal_date not in df.index:
        return {"status": "Signal Date Missing in Data"}

    signal_idx = df.index.get_loc(signal_date)
    valid_dates = df.index[signal_idx: signal_idx + max_days]

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
                if row["High"] >= target_price:
                    return {
                        "status": "Closed - Target Hit",
                        "entry": round(entry_price, 5),
                        "stop": round(stop_price, 5) if stop_price is not None else "",
                        "target": round(target_price, 5),
                        "entry_date": test_date.strftime('%Y-%m-%d'),
                        "exit_date": i.strftime('%Y-%m-%d'),
                    }

                if stop_price is not None and row["Low"] <= stop_price:
                    return {
                        "status": "Closed - Stop Hit",
                        "entry": round(entry_price, 5),
                        "stop": round(stop_price, 5),
                        "target": round(target_price, 5),
                        "entry_date": test_date.strftime('%Y-%m-%d'),
                        "exit_date": i.strftime('%Y-%m-%d'),
                    }

            return {
                "status": "Open",
                "entry": round(entry_price, 5),
                "stop": round(stop_price, 5) if stop_price is not None else "",
                "target": round(target_price, 5),
                "entry_date": test_date.strftime('%Y-%m-%d'),
                "exit_date": "",
            }

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
    parser = argparse.ArgumentParser(description="Currency Futures Strategy Backtester")
    parser.add_argument("--mode", choices=["futures", "etfs"], required=True, help="Which type of trades to backtest")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for matched strategy names")
    args = parser.parse_args()

    # Choose file based on mode
    input_file = "trade_signals_futures.csv" if args.mode == "futures" else "trade_signals_ETFs.csv"
    strategy_path = "strategies_complete.json"
    output_file = "trade_results_futures.csv" if args.mode == "futures" else "trade_results_ETFs.csv"

    strategies = load_strategies(strategy_path)
    df_signals = pd.read_csv(input_file, encoding="ISO-8859-1")

    results = []

    for _, row in df_signals.iterrows():
        symbol = str(row['symbol']).strip()

        try:
            # signal_date = pd.to_datetime(row['date'], format='%m/%d/%y')
            # Make sure signal_date is in the exact same format as df.index
            signal_date = pd.to_datetime(row['date'], format='%m/%d/%y').normalize()

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

        # Convert alt futures symbol codes
        alternate_symbols = {"AD": "6A", "BP": "6B", "CD": "6C", "EC": "6E", "SF": "6S"}
        if args.mode == "futures" and symbol in alternate_symbols:
            symbol = alternate_symbols[symbol]

        # Resolve strategy name
        if args.mode == "futures":
            raw_name = str(row["strategy"]).strip()
            resolved_name = resolve_strategy_name(raw_name, list(strategies.keys()))
        else:
            # ETFs: infer from 'buy or sell' and 'frequency' fields
            action = row["buy or sell"].strip().capitalize()  # e.g., "Buy" or "Sell"
            direction = str(row.get("buy or sell", "")).strip().lower()
            freq = row["frequency"].strip().capitalize()      # e.g., "Daily" or "Weekly"
            frequency = str(row.get("frequency", "")).strip().lower()
            resolved_name = f"{freq} ETF Options {action}"

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

        # all_data = fetch_price_data([symbol], signal_date - timedelta(days=10), datetime.today())
        all_data = fetch_price_data([symbol], datetime(2025, 1, 2), datetime.today())
        df = all_data.get(symbol)

        if df is None or df.empty:
            print(f"⚠️ No price data available for {symbol}. Skipping.")
            continue


        if symbol not in all_data:
            print(f"⚠️ No price data available for {symbol}. Skipping.")
            continue
        df = all_data[symbol]
        direction = row.get("buy or sell", "").strip().lower()
        result = simulate_trade(strategies[resolved_name], symbol, df, signal_date, resolved_name, direction=direction)


        results.append({
            "symbol": symbol,
            "strategy": resolved_name,
            "direction": direction,
            "signal_date": signal_date.strftime('%m/%d/%y'),
            "status": result["status"],
            "entry_price": result.get("entry", ""),
            "stop_price": result.get("stop", ""),
            "target_price": result.get("target", ""),
            "entry_date": result.get("entry_date", ""),
            "exit_date": result.get("exit_date", "")
        })
        
    print(f"Total processed signals: {len(results)}")
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"✅ Backtest completed. Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
