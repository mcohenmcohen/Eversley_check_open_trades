from polygon import RESTClient
import argparse
from datetime import datetime
import pandas as pd

client = RESTClient("qFTWmhmyAj2pJqpx0Pwxp2AyShKFVPv9")  # Replace with your real key
print("✅ Polygon SDK import works!")

# Run like this:
# - python test_polygon_import.py --symbol QQQ --start_date 2025-05-01 --end_date 2025-05-29

def main():

    parser = argparse.ArgumentParser(description="Polygon Data Import Test")
    parser.add_argument('--symbol', type=str, required=True, help="e.g. QQQ")
    parser.add_argument('--start_date', type=str, required=True, help="YYYY-MM-DD format (e.g., 2023-10-27).")
    parser.add_argument('--end_date', type=str, required=True, help="YYYY-MM-DD format (e.g., 2023-10-27).")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    symbol = args.symbol

    data = {}

    try:
        aggs = client.get_aggs(
            ticker=symbol,
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
        data[symbol] = df
        print('df.tail(10)', df.tail(10))
        # fetched_data_cache[sym] = df
        success = True
        # break

    except Exception as e:
        print(f"❌ Error fetching {symbol} from Polygon: {e}")

if __name__ == "__main__":
    main()