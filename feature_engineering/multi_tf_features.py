"""
Multi-Timeframe Alignment and Features.
Aligns micro (5m) and macro (1h) features to the base (15m) timeframe
using causal backward filling to prevent forward-looking bias.
Computes cross-timeframe alignment scores.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def align_and_compute_multi_tf(symbol="BTCUSDT", base_tf="15m"):
    features_dir = Path(__file__).resolve().parent.parent / "data" / "features" / symbol
    
    base_path = features_dir / f"base_features_{base_tf}.parquet"
    if not base_path.exists():
        logger.error(f"Base {base_tf} features not found.")
        return None
        
    df_base = pd.read_parquet(base_path).sort_values("close_time")
    
    # Load and align 5m
    path_5m = features_dir / "base_features_5m.parquet"
    if path_5m.exists():
        df_5m = pd.read_parquet(path_5m).sort_values("close_time")
        # Keep only required columns
        cols_to_keep = ["close_time", "ema_20_slope", "atr_expansion_ratio", "atr_14", "rsi_14", "rsi_velocity"]
        df_5m = df_5m[cols_to_keep]
        # Merge exactly or backward (causal)
        df_base = pd.merge_asof(df_base, df_5m, on="close_time", direction="backward", suffixes=("", "_5m"))
        logger.info("Aligned 5m features to base timeframe.")
    else:
        logger.warning("5m features missing, cannot align.")
        return df_base
        
    # Load and align 1h
    path_1h = features_dir / "base_features_1h.parquet"
    if path_1h.exists():
        df_1h = pd.read_parquet(path_1h).sort_values("close_time")
        cols_to_keep = ["close_time", "ema_20_slope", "atr_expansion_ratio", "atr_14", "rsi_14"]
        df_1h = df_1h[cols_to_keep]
        df_base = pd.merge_asof(df_base, df_1h, on="close_time", direction="backward", suffixes=("", "_1h"))
        logger.info("Aligned 1h features to base timeframe.")
    else:
        logger.warning("1h features missing, cannot align.")
        return df_base
        
    # Compute Multi-TF Features
    # Trend Alignment Score (How many timeframes agree on direction)
    dir_15 = (df_base["ema_20_slope"] > 0).astype(int) - (df_base["ema_20_slope"] < 0).astype(int)
    dir_5 = (df_base["ema_20_slope_5m"] > 0).astype(int) - (df_base["ema_20_slope_5m"] < 0).astype(int)
    dir_1h = (df_base["ema_20_slope_1h"] > 0).astype(int) - (df_base["ema_20_slope_1h"] < 0).astype(int)
    df_base["trend_alignment_score"] = dir_15 + dir_5 + dir_1h
    
    # Volatility Alignment Score
    vol_15 = (df_base["atr_expansion_ratio"] > 1.0).astype(int)
    vol_5 = (df_base["atr_expansion_ratio_5m"] > 1.0).astype(int)
    vol_1h = (df_base["atr_expansion_ratio_1h"] > 1.0).astype(int)
    df_base["volatility_alignment_score"] = vol_15 + vol_5 + vol_1h
    
    # Micro/Macro Volatility Ratio
    df_base["micro_macro_volatility_ratio"] = df_base["atr_14_5m"] / df_base["atr_14_1h"].replace(0, np.nan)
    
    # Multi-TF Momentum Score (Weighted RSI)
    df_base["multi_tf_momentum_score"] = (df_base["rsi_14"] * 0.5) + (df_base["rsi_14_1h"] * 0.3) + (df_base["rsi_14_5m"] * 0.2)
    
    logger.info("Computed Multi-TF alignment features.")
    return df_base

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = align_and_compute_multi_tf()
    if df is not None:
        print(f"Computed Multi-TF features. Total columns: {len(df.columns)}")
