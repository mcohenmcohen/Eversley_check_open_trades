from datetime import datetime, timedelta
import calendar
import os
import json
import pandas as pd


def load_strategies(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_date_safe(date_str):
    """Safely parse a date string in '%Y-%m-%d' format. Return None if blank or invalid."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.strptime(date_str.strip(), '%Y-%m-%d').date()
    except ValueError:
        return None


def resolve_strategy_name(strategy_name, direction, frequency):
    """
    Create a standardized strategy name format.
    E.g., 'Ichimoku Cross Currency Futures Buy' or 'Daily ETF Options Sell'
    """
    direction = direction.capitalize()
    frequency = frequency.capitalize()
    return f"{strategy_name.strip()} {direction} {frequency}"


def get_price_data_from_file(symbol):
    """
    Loads historical price data for a symbol from CSV (used for futures).
    """
    path = os.path.join("data", f"{symbol.upper()}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local price file not found: {path}")
    df = pd.read_csv(path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df


def get_option_expiration_date(year, month):
    """
    Calculates the options expiration date for a given month and year.

    If the 3rd Friday is a holiday, it returns the Thursday before it.

    Args:
        year: The year (e.g., 2025).
        month: The month (e.g., 1 for January, 12 for December).

    Returns:
        A datetime.date object representing the options expiration date.
    """

    # Calculate the 3rd Friday of the month
    third_friday_date = get_third_friday(year, month)

    # Check if the 3rd Friday is a holiday (replace with your actual holiday checking logic)
    if is_holiday(third_friday_date):
        # If it is a holiday, return the Thursday before
        return third_friday_date - timedelta(days=1)
    else:
        return third_friday_date

def get_third_friday(year, month):
    """
    Calculates the date of the 3rd Friday of a given month.

    Args:
        year: The year.
        month: The month.

    Returns:
        A datetime.date object representing the 3rd Friday.
    """

    # Get the first day of the month
    first_day = datetime(year, month, 1).date()

    # Find the first Friday of the month
    first_friday = first_day + timedelta((4 - first_day.weekday()) % 7)

    # Calculate the 3rd Friday
    third_friday = first_friday + timedelta(weeks=2)
    return third_friday

def is_holiday(date):
    """
    Checks if a given date is a trading holiday.

    Args:
        date: A datetime.date object.

    Returns:
        True if the date is a holiday, False otherwise.
    """
    # Replace this with your actual holiday checking logic.
    # This example assumes a specific list of holidays.
    holidays = [
        datetime(2025, 1, 1).date(),  # New Year's Day
        datetime(2025, 7, 4).date(),  # Independence Day
        datetime(2025, 9, 1).date(),  # Labor Day
        datetime(2025, 11, 27).date(),  # Thanksgiving
        datetime(2025, 12, 25).date()   # Christmas Day
    ]

    return date in holidays

# Example usage:
year = 2025
month = 6  # June

expiration_date = get_option_expiration_date(year, month)
print(f"The options expiration date for June {year} is: {expiration_date}")
