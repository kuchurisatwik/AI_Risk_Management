"""
Spike Remover — ATR-based bad tick rejection (Section 3.3).
Flags and removes candles where price deviates > 5 ATR from rolling mean.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def detect_spikes(df, atr_threshold=5.0, rolling_window=50):
    """
    Detect price spikes > N ATR from rolling mean.

    Args:
        df: DataFrame with high, low, close columns
        atr_threshold: multiplier for ATR-based threshold (default: 5.0)
        rolling_window: window for rolling ATR and mean

    Returns:
        Boolean Series (True = spike detected)
    """
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    rolling_atr = tr.rolling(window=rolling_window, min_periods=10).mean()
    rolling_mean = close.rolling(window=rolling_window, min_periods=10).mean()

    deviation = (close - rolling_mean).abs()
    spikes = deviation > (atr_threshold * rolling_atr)
    spikes.iloc[:rolling_window] = False

    return spikes


def remove_spikes(df, atr_threshold=5.0, rolling_window=50):
    """
    Remove spike rows from DataFrame. Logs removed ticks.

    Returns:
        Cleaned DataFrame with spikes removed.
    """
    spikes = detect_spikes(df, atr_threshold, rolling_window)
    n_spikes = spikes.sum()

    if n_spikes > 0:
        spike_rows = df[spikes]
        logger.info(f"Removing {n_spikes} spike candles:")
        for _, row in spike_rows.head(5).iterrows():
            logger.info(f"  {row.get('open_time', 'N/A')}: close={row['close']}")
        if n_spikes > 5:
            logger.info(f"  ... and {n_spikes - 5} more")

    return df[~spikes].reset_index(drop=True)
