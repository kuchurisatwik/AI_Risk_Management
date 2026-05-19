"""
Volatility Labeler — Realized volatility regression target.

Regression target: realized volatility (std of returns * sqrt(horizon))
over the next `horizon` candles.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def label_volatility(df, horizon=8):
    """
    Generate volatility regression labels.
    
    For each candle i:
      - Compute returns for candles i+1 to i+horizon
      - Realized vol = std(returns) * sqrt(horizon)
    
    Last `horizon` rows get NaN.
    
    Args:
        df: DataFrame with 'close' column
        horizon: Number of future candles to measure vol over
    
    Returns:
        numpy array of realized volatility values
    """
    closes = df['close'].values
    returns = np.diff(closes) / closes[:-1]  # pct_change equivalent
    
    n = len(df)
    labels = np.full(n, np.nan)
    
    for i in range(n - horizon):
        future_returns = returns[i:i + horizon]
        realized_vol = np.std(future_returns) * np.sqrt(horizon)
        labels[i] = realized_vol
    
    logger.info(
        f"Volatility labels: mean={np.nanmean(labels):.6f}, "
        f"std={np.nanstd(labels):.6f}, "
        f"unlabeled={int(np.isnan(labels).sum())}"
    )
    
    return labels
