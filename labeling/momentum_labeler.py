"""
Momentum Labeler — Forward-looking ATR-multiple TP/SL label.

Binary classification target:
  1 = TP hit before SL within N candles (favorable momentum)
  0 = SL hit first or neither hit (unfavorable)

Uses HIGH/LOW of future candles for realistic SL/TP simulation.
Supports regime-specific Triple-Barrier parameters.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Regime-specific Triple-Barrier configurations
# IMPORTANT: These must match the execution-layer multipliers in policy_engine.py
# Scaled for 5-minute candles (n_candles ~3x the 15m equivalents)
REGIME_BARRIER_CONFIG = {
    'trending_low_vol':  {'atr_mult_tp': 4.0, 'atr_mult_sl': 2.5, 'n_candles': 48},
    'trending_high_vol': {'atr_mult_tp': 3.0, 'atr_mult_sl': 3.0, 'n_candles': 36},
    'sideways_low_vol':  {'atr_mult_tp': 2.0, 'atr_mult_sl': 2.0, 'n_candles': 24},
    'choppy_high_vol':   {'atr_mult_tp': 2.0, 'atr_mult_sl': 4.0, 'n_candles': 18},
    'crash_mode':        {'atr_mult_tp': 2.0, 'atr_mult_sl': 2.0, 'n_candles': 24},
}

# Default fallback for rows without regime labels
DEFAULT_BARRIER_CONFIG = {'atr_mult_tp': 1.5, 'atr_mult_sl': 1.0, 'n_candles': 8}


def label_momentum(df, n_candles=None, atr_mult_tp=None, atr_mult_sl=None,
                    regime_aware=True):
    """
    Generate momentum labels for the entire DataFrame.

    If regime_aware=True and 'regime_label' column exists, uses
    regime-specific Triple-Barrier parameters from REGIME_BARRIER_CONFIG.
    Otherwise falls back to the provided fixed parameters.

    Args:
        df: DataFrame with columns: close, high, low, atr_14, ema_20_slope
        n_candles: Forward-looking horizon (ignored if regime_aware=True)
        atr_mult_tp: ATR multiplier for take-profit (ignored if regime_aware)
        atr_mult_sl: ATR multiplier for stop-loss (ignored if regime_aware)
        regime_aware: If True, use regime-specific barrier params

    Returns:
        List of labels (1/0/NaN)
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr_14'].values
    slopes = df['ema_20_slope'].values

    has_regime = regime_aware and 'regime_label' in df.columns
    if has_regime:
        regimes = df['regime_label'].values
    else:
        regimes = None

    n = len(df)
    labels = np.full(n, np.nan)

    # Determine max possible lookahead for the tail NaN region
    max_horizon = max(c['n_candles'] for c in REGIME_BARRIER_CONFIG.values()) if has_regime else (n_candles or DEFAULT_BARRIER_CONFIG['n_candles'])

    for i in range(n - max_horizon):
        # Get regime-specific or default config
        if has_regime and regimes[i] in REGIME_BARRIER_CONFIG:
            cfg = REGIME_BARRIER_CONFIG[regimes[i]]
        else:
            cfg = {
                'atr_mult_tp': atr_mult_tp or DEFAULT_BARRIER_CONFIG['atr_mult_tp'],
                'atr_mult_sl': atr_mult_sl or DEFAULT_BARRIER_CONFIG['atr_mult_sl'],
                'n_candles': n_candles or DEFAULT_BARRIER_CONFIG['n_candles'],
            }

        curr_n = cfg['n_candles']
        curr_tp = cfg['atr_mult_tp']
        curr_sl = cfg['atr_mult_sl']

        # Ensure we don't read past the end
        if i + 1 + curr_n > n:
            continue

        entry = closes[i]
        atr = atrs[i]
        direction = 1 if slopes[i] > 0 else -1

        sl = entry - direction * atr * curr_sl
        tp = entry + direction * atr * curr_tp

        tp_hit = False
        sl_hit = False

        for j in range(i + 1, i + 1 + curr_n):
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

