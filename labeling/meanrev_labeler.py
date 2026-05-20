"""
Mean-Reversion Labeler — VWAP Reversion Target.

Labels for mean-reversion trades in sideways regimes:
  1 = Price reverts to VWAP within time barrier
  0 = Price extends further from VWAP or times out without reversion

Uses VWAP distance as the primary signal instead of ATR-based TP/SL.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def label_meanrev(df, sl_extension_mult=1.5, t1=24):
    """
    Generate mean-reversion labels.

    For each candle i where price is far from VWAP:
      - Direction: LONG if close < VWAP (buy the dip), SHORT if close > VWAP (fade the rally)
      - TP: Price touches or crosses VWAP
      - SL: Price extends further from VWAP by sl_extension_mult × current distance
      - Time barrier: t1 candles

    Args:
        df: DataFrame with columns: close, high, low, vwap (rolling VWAP)
        sl_extension_mult: How much further price must extend to trigger SL
        t1: Forward-looking time barrier in candles

    Returns:
        numpy array of labels (1.0 / 0.0 / NaN)
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # Use rolling VWAP if available, else compute from typical price
    if 'vwap' in df.columns:
        vwaps = df['vwap'].values
    else:
        typical = (df['high'] + df['low'] + df['close']) / 3
        tp_vol = typical * df['volume']
        vwaps = (tp_vol.rolling(100).sum() / df['volume'].rolling(100).sum()).values

    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - t1):
        entry = closes[i]
        vwap = vwaps[i]

        if np.isnan(vwap) or vwap <= 0:
            continue

        distance = entry - vwap  # positive = above VWAP, negative = below

        # Only label candles with meaningful VWAP deviation
        # Require at least 0.1% distance from VWAP
        if abs(distance) / entry < 0.001:
            labels[i] = 0.0  # Too close to VWAP, no reversion setup
            continue

        if distance > 0:
            # Price is ABOVE VWAP → SHORT reversion
            tp_price = vwap  # Target: revert to VWAP
            sl_price = entry + abs(distance) * sl_extension_mult  # Extends further up
        else:
            # Price is BELOW VWAP → LONG reversion
            tp_price = vwap  # Target: revert to VWAP
            sl_price = entry - abs(distance) * sl_extension_mult  # Extends further down

        tp_hit = False
        sl_hit = False

        for j in range(i + 1, i + 1 + t1):
            if distance > 0:  # SHORT reversion
                # SL hit: price goes higher
                if highs[j] >= sl_price:
                    sl_hit = True
                    break
                # TP hit: price touches VWAP
                if lows[j] <= tp_price:
                    tp_hit = True
                    break
            else:  # LONG reversion
                # SL hit: price goes lower
                if lows[j] <= sl_price:
                    sl_hit = True
                    break
                # TP hit: price touches VWAP
                if highs[j] >= tp_price:
                    tp_hit = True
                    break

        labels[i] = 1.0 if tp_hit and not sl_hit else 0.0

    pos = int(np.nansum(labels))
    neg = int(np.nansum(labels == 0))
    unlabeled = int(np.isnan(labels).sum())
    logger.info(
        f"MeanRev labels: {pos} reversion_success / "
        f"{neg} reversion_fail / {unlabeled} unlabeled (tail)"
    )

    return labels
