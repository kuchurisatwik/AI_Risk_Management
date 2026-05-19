"""
Triple-Barrier Meta-Labeler.

Applies Lopez de Prado's Triple-Barrier Method.
Instead of an arbitrary fixed target, this splits labeling into:
1. Primary Label: Direction (sign of forward return)
2. Meta Label: Did the trade hit the dynamic Volatility-adjusted Take Profit 
   before hitting the Stop Loss or Time barrier?
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def apply_triple_barrier(df, vol_span=20, pt_sl=[1.5, 1.0], t1=12):
    """
    Applies the triple-barrier method.
    
    Args:
        df: DataFrame containing 'close', 'high', 'low'
        vol_span: Span for volatility calculation (e.g., atr_14)
        pt_sl: Multipliers for [Profit_Taking, Stop_Loss] based on volatility
        t1: Vertical barrier (time out in periods)
    
    Returns:
        DataFrame with 'primary_label' (direction) and 'meta_label' (success)
    """
    df = df.copy()
    
    # We need volatility for dynamic barriers. Use ATR if available, else standard deviation of returns.
    if 'atr_14' in df.columns:
        vol = df['atr_14']
    else:
        log_ret = np.log(df['close'] / df['close'].shift(1))
        vol = log_ret.rolling(vol_span).std() * df['close']
    
    # 1. Primary Direction Label (Forward Return over t1 periods)
    # 1 = Long, -1 = Short, 0 = Flat
    forward_return = df['close'].shift(-t1) - df['close']
    # Use a small threshold to avoid labeling pure noise as direction
    threshold = vol * 0.1
    primary_label = np.where(forward_return > threshold, 1, np.where(forward_return < -threshold, -1, 0))
    df['primary_label'] = primary_label
    
    # 2. Meta-Labeling (Triple Barrier)
    meta_labels = np.full(len(df), np.nan)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    vols = vol.values
    dirs = primary_label
    
    for i in range(len(df) - t1):
        if dirs[i] == 0:
            meta_labels[i] = 0 # No trade, so not successful
            continue
            
        entry = closes[i]
        curr_vol = vols[i]
        
        # Dynamic barriers based on current volatility
        tp_dist = curr_vol * pt_sl[0]
        sl_dist = curr_vol * pt_sl[1]
        
        # If volatility is too low or NaN, skip
        if np.isnan(curr_vol) or curr_vol <= 0:
            meta_labels[i] = 0
            continue
            
        if dirs[i] == 1: # Long
            tp_price = entry + tp_dist
            sl_price = entry - sl_dist
            
            success = 0
            for j in range(i + 1, i + 1 + t1):
                if lows[j] <= sl_price:
                    success = 0
                    break
                elif highs[j] >= tp_price:
                    success = 1
                    break
            meta_labels[i] = success
            
        elif dirs[i] == -1: # Short
            tp_price = entry - tp_dist
            sl_price = entry + sl_dist
            
            success = 0
            for j in range(i + 1, i + 1 + t1):
                if highs[j] >= sl_price:
                    success = 0
                    break
                elif lows[j] <= tp_price:
                    success = 1
                    break
            meta_labels[i] = success

    df['meta_label'] = meta_labels
    
    logger.info(f"Triple-Barrier Meta-Labeling Complete.")
    logger.info(f"Primary (Direction): Long={np.sum(dirs==1)}, Short={np.sum(dirs==-1)}, Flat={np.sum(dirs==0)}")
    logger.info(f"Meta (Success): {np.nansum(meta_labels)} successful trades out of {np.sum(~np.isnan(meta_labels))} valid setups")
    
    return df

def process_datasets(data_dir: Path):
    logger.info(f"Processing datasets in {data_dir}")
    
    for split in ['train', 'val', 'test']:
        file_path = data_dir / f"{split}.parquet"
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        df = pd.read_parquet(file_path)
        
        # Apply triple barrier (dynamic ATR thresholds, 12 bars = 3 hours)
        df = apply_triple_barrier(df, pt_sl=[1.5, 1.5], t1=12)
        
        # Save back
        df.to_parquet(file_path)
        logger.info(f"Updated {split}.parquet with primary_label and meta_label")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "labeled" / "BTCUSDT"
    process_datasets(data_dir)
