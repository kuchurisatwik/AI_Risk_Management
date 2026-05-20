"""
Multi-Timeframe Alignment and Features.
Aligns macro (15m, 1h) features to the base (5m) timeframe
using causal backward filling to prevent forward-looking bias.
Computes cross-timeframe alignment scores.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def align_and_compute_multi_tf(symbol="BTCUSDT", base_tf="5m"):
    features_dir = Path(__file__).resolve().parent.parent / "data" / "features" / symbol
    
    base_path = features_dir / f"base_features_{base_tf}.parquet"
    if not base_path.exists():
        logger.error(f"Base {base_tf} features not found for {symbol}.")
        return None
        
    df_base = pd.read_parquet(base_path).sort_values("close_time")
    
    # Load and align 15m
    path_15m = features_dir / "base_features_15m.parquet"
    if path_15m.exists():
        df_15m = pd.read_parquet(path_15m).sort_values("close_time")
        # Keep only required columns
        cols_to_keep = ["close_time", "ema_20_slope", "atr_expansion_ratio", "atr_14", "rsi_14", "rsi_velocity"]
        df_15m = df_15m[[c for c in cols_to_keep if c in df_15m.columns]]
        # Merge exactly or backward (causal)
        df_base = pd.merge_asof(df_base, df_15m, on="close_time", direction="backward", suffixes=("", "_15m"))
        logger.info(f"Aligned 15m features to {base_tf} for {symbol}.")
    else:
        logger.warning(f"15m features missing for {symbol}, cannot align.")
        
    # Load and align 1h
    path_1h = features_dir / "base_features_1h.parquet"
    if path_1h.exists():
        df_1h = pd.read_parquet(path_1h).sort_values("close_time")
        cols_to_keep = ["close_time", "ema_20_slope", "atr_expansion_ratio", "atr_14", "rsi_14"]
        df_1h = df_1h[[c for c in cols_to_keep if c in df_1h.columns]]
        df_base = pd.merge_asof(df_base, df_1h, on="close_time", direction="backward", suffixes=("", "_1h"))
        logger.info(f"Aligned 1h features to {base_tf} for {symbol}.")
    else:
        logger.warning(f"1h features missing for {symbol}, cannot align.")
        
    # Compute Multi-TF Features
    dir_base = (df_base["ema_20_slope"] > 0).astype(int) - (df_base["ema_20_slope"] < 0).astype(int)
    
    if "ema_20_slope_15m" in df_base.columns and "ema_20_slope_1h" in df_base.columns:
        dir_15 = (df_base["ema_20_slope_15m"] > 0).astype(int) - (df_base["ema_20_slope_15m"] < 0).astype(int)
        dir_1h = (df_base["ema_20_slope_1h"] > 0).astype(int) - (df_base["ema_20_slope_1h"] < 0).astype(int)
        df_base["trend_alignment_score"] = dir_base + dir_15 + dir_1h
        
        vol_base = (df_base["atr_expansion_ratio"] > 1.0).astype(int)
        vol_15 = (df_base["atr_expansion_ratio_15m"] > 1.0).astype(int)
        vol_1h = (df_base["atr_expansion_ratio_1h"] > 1.0).astype(int)
        df_base["volatility_alignment_score"] = vol_base + vol_15 + vol_1h
        
        df_base["micro_macro_volatility_ratio"] = df_base["atr_14"] / df_base["atr_14_1h"].replace(0, np.nan)
        
        df_base["multi_tf_momentum_score"] = (df_base["rsi_14"] * 0.5) + (df_base["rsi_14_15m"] * 0.3) + (df_base["rsi_14_1h"] * 0.2)
    else:
        # Fallback if higher TFs missing
        df_base["trend_alignment_score"] = dir_base
        df_base["volatility_alignment_score"] = (df_base["atr_expansion_ratio"] > 1.0).astype(int)
        df_base["micro_macro_volatility_ratio"] = 1.0
        df_base["multi_tf_momentum_score"] = df_base["rsi_14"]
    
    logger.info(f"Computed Multi-TF alignment features for {symbol}.")
    return df_base

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = align_and_compute_multi_tf()
    if df is not None:
        print(f"Computed Multi-TF features. Total columns: {len(df.columns)}")
