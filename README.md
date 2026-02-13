# Currency Strategy Trade Evaluation Project

This project allows you to evaluate trading strategies on currency futures and ETFs using signal data and historical prices.

## Files Included

- `trade_evaluator.py`
  A script-based engine that loads a strategy definition, downloads OHLC data, evaluates trade triggers, and simulates open/closed positions.

- `currency_backtest_notebook.ipynb`
  A Jupyter notebook for running multiple trade simulations interactively based on a CSV file of signals.

- `strategies_complete.json`
  JSON file containing the full definition of 11 currency futures strategies, including rules for entry, stop, and target prices.

- `trade_signals.csv`
  A sample input file listing trade signals: each row contains a ticker symbol, strategy name, and signal date.

- `trade_results.csv` (optional output)
  Created by the notebook or script to show whether each trade is open, closed (by stop or target), or expired.

## How to Use

### 1. Run the Notebook
Open `currency_backtest_notebook.ipynb` in Jupyter. Make sure:
- You have internet access for data from Yahoo Finance
- The `strategies_complete.json` and `trade_signals.csv` files are in the same directory

After running all cells, a `trade_results.csv` will be generated showing each trade's outcome.

### 2. Run the Script
```bash
python trade_evaluator.py
```
Modify the script or call the `run_backtest()` function with:
- `strategy_path`: path to the strategy JSON file
- `symbol`: one of 6A, 6B, 6C, 6E, 6S
- `strategy_name`: must match one defined in the JSON
- `signal_date_str`: date the signal triggered (YYYY-MM-DD)

## Requirements

- Python 3.x
- pandas
- yfinance
- json
- datetime

## Notes

- All OHLC data is pulled using Yahoo Finance's continuous futures tickers (e.g., 6E=F for Euro futures).
- Entry must trigger within 2 or 3 days of the signal (defined per strategy).
- Trades simulate forward until target or stop is hit — or remain open.
