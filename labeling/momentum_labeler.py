"""
Momentum Labeler — Forward-looking ATR-multiple TP/SL label.

Binary classification target:
  1 = TP hit before SL within N candles (favorable momentum)
  0 = SL hit first or neither hit (unfavorable)

Uses HIGH/LOW of future candles for realistic SL/TP simulation.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def label_momentum(df, n_candles=8, atr_mult_tp=1.5, atr_mult_sl=1.0):
    """
    Generate momentum labels for the entire DataFrame.
    
    For each candle i:
      - Determine direction from ema_20_slope (>0 = LONG, <0 = SHORT)
      - Set SL = entry +/- atr_14 * atr_mult_sl
      - Set TP = entry +/- atr_14 * atr_mult_tp
      - Look at next n_candles HIGH/LOW to check if TP or SL hit first
      - Label = 1 if TP hit before SL, else 0
    
    Last n_candles rows get NaN (no future data available).
    
    Args:
        df: DataFrame with columns: close, high, low, atr_14, ema_20_slope
        n_candles: Forward-looking horizon (default 8 = 2 hours on 15m)
        atr_mult_tp: ATR multiplier for take-profit
        atr_mult_sl: ATR multiplier for stop-loss
    
    Returns:
        List of labels (1/0/None)
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr_14'].values
    slopes = df['ema_20_slope'].values
    
    n = len(df)
    labels = np.full(n, np.nan)
    
    for i in range(n - n_candles):
        entry = closes[i]
        atr = atrs[i]
        direction = 1 if slopes[i] > 0 else -1
        
        sl = entry - direction * atr * atr_mult_sl
        tp = entry + direction * atr * atr_mult_tp
        
        tp_hit = False
        sl_hit = False
        
        for j in range(i + 1, i + 1 + n_candles):
            if direction == 1:  # LONG
                if lows[j] <= sl:
                    sl_hit = True
                    break
                if highs[j] >= tp:
                    tp_hit = True
                    break
            else:  # SHORT
                if highs[j] >= sl:
                    sl_hit = True
                    break
                if lows[j] <= tp:
                    tp_hit = True
                    break
        
        labels[i] = 1.0 if tp_hit and not sl_hit else 0.0
    
    logger.info(
        f"Momentum labels: {int(np.nansum(labels))} positive / "
        f"{int(np.nansum(labels == 0))} negative / "
        f"{int(np.isnan(labels).sum())} unlabeled (tail)"
    )
    
    return labels
