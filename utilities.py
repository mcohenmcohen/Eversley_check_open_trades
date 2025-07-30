from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import holidays
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
    third_friday = get_third_friday(year, month)
    return third_friday - timedelta(days=1) if is_holiday(third_friday) else third_friday


    # Check if the 3rd Friday is a holiday (replace with your actual holiday checking logic)
    if is_holiday(third_friday_date):
        # If it is a holiday, return the Thursday before
        return third_friday_date - timedelta(days=1)
    else:
        return third_friday_date
    

# def get_final_expiration_date(signal_date):
#     """
#     Given a trade signal date, calculates the options expiration date two full cycles later.
#     This is typically the 3rd Friday of the month two *after* the month of the first expiration.

#     Example:
#         Signal: May 1 2025 → May expiration = May 17 2025
#         Two full expirations = June + July → Final expiry = July 18 2025

#     Returns:
#         A datetime.date object representing the final expiration.
#     """
#     # Get first expiration date
#     first_exp = get_option_expiration_date(signal_date.year, signal_date.month)

#     # Advance by two months
#     second_month = first_exp + relativedelta(months=1)
#     third_month = second_month + relativedelta(months=1)

#     # Get final expiration
#     return get_option_expiration_date(third_month.year, third_month.month)

from dateutil.relativedelta import relativedelta

def get_final_expiration_date(signal_date, months_out=2):
    """
    Given a trade signal date, calculates the options expiration date a specified number of
    full expiration months after the first expiration month.

    Example:
        Signal: May 1, 2025 → First expiration: May 16, 2025
        Two full expirations (months_out=2): June + July → Final: July 18, 2025

    Args:
        signal_date (datetime.date): The trade entry/signal date.
        months_out (int): Number of full expiration months after the first.

    Returns:
        datetime.date: The final expiration date.
    """
    # Get first expiration date
    first_exp = get_option_expiration_date(signal_date.year, signal_date.month)

    # Step forward N full months after first expiration
    future_month = first_exp + relativedelta(months=months_out)
    final_exp = get_option_expiration_date(future_month.year, future_month.month)

    return final_exp



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

def is_holiday(date_in):
    """
    Checks if a given date is a US financial market holiday (NYSE calendar).

    Args:
        date_in (datetime.date): The date to check.

    Returns:
        bool: True if the date is a recognized NYSE trading holiday, False otherwise.
    """
    return date_in in holidays.financial_holidays("NYSE", years=date_in.year)


# Example usage:
year = 2025
month = 6  # June

expiration_date = get_option_expiration_date(year, month)
print(f"The options expiration date for June {year} is: {expiration_date}")
