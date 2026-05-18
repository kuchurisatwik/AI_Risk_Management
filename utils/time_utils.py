"""
Time Utilities
==============
Epoch conversion, timeframe intervals, and gap detection.
All timestamps stored as UTC. Never use local time.
"""

import pandas as pd
from datetime import datetime, timedelta, timezone

# Timeframe -> interval in seconds
TIMEFRAME_SECONDS = {
    "1m":  60,
    "3m":  180,
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "2h":  7200,
    "4h":  14400,
    "6h":  21600,
    "8h":  28800,
    "12h": 43200,
    "1d":  86400,
    "3d":  259200,
    "1w":  604800,
}

# Timeframe -> interval in milliseconds
TIMEFRAME_MS = {k: v * 1000 for k, v in TIMEFRAME_SECONDS.items()}


def epoch_ms_to_utc(epoch_ms: int) -> pd.Timestamp:
    """Convert epoch milliseconds to UTC pandas Timestamp."""
    return pd.Timestamp(epoch_ms, unit="ms", tz="UTC")


def utc_to_epoch_ms(dt: datetime) -> int:
    """Convert a UTC datetime to epoch milliseconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def get_expected_candle_count(start: pd.Timestamp, end: pd.Timestamp,
                               timeframe: str) -> int:
    """Calculate expected number of candles between start and end (inclusive)."""
    interval_sec = TIMEFRAME_SECONDS[timeframe]
    total_seconds = (end - start).total_seconds()
    return int(total_seconds / interval_sec) + 1


def find_missing_timestamps(df: pd.DataFrame, timeframe: str,
                             time_col: str = "open_time") -> pd.DatetimeIndex:
    """
    Find missing candle timestamps in a DataFrame.

    Returns a DatetimeIndex of expected but missing timestamps.
    """
    if df.empty:
        return pd.DatetimeIndex([])

    interval = pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    start = df[time_col].min()
    end = df[time_col].max()

    # Generate expected timestamps
    expected = pd.date_range(start=start, end=end, freq=interval)

    # Find missing
    actual = pd.DatetimeIndex(df[time_col])
    missing = expected.difference(actual)

    return missing


def generate_date_range(start_date: str = None, end_date: str = None,
                         days_back: int = 365) -> list:
    """
    Generate a list of date strings (YYYY-MM-DD) for the download range.

    If start_date is None, defaults to (today - days_back).
    If end_date is None, defaults to yesterday.
    """
    today = datetime.now(timezone.utc).date()

    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end = today - timedelta(days=1)  # yesterday (today's data may be incomplete)

    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        start = end - timedelta(days=days_back)

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates
