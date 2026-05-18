"""
Liquidity Features Module (Efficient Proxy Approach).
Extracts highly efficient proxy features directly from base klines.
No tick data required!
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def compute_liquidity_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes liquidity and imbalance proxies directly from klines.
    
    1. Volume Delta = taker_buy_volume - aggressive_sell_volume
    2. Trade Imbalance = Volume Delta / Total Volume
    3. Amihud Illiquidity = abs(Close - Open) / Volume (Proxy for Spread/Liquidity)
    4. Volatility/Liquidity Ratio = ATR_14 / Volume
    """
    df = df.copy()
    
    # 1 & 2: Imbalance Proxies
    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        buy_vol = df["taker_buy_volume"]
        sell_vol = df["volume"] - df["taker_buy_volume"]
        
        df["volume_delta"] = buy_vol - sell_vol
        df["trade_imbalance"] = df["volume_delta"] / df["volume"].replace(0, pd.NA)
        
        df["delta_velocity"] = df["volume_delta"] - df["volume_delta"].shift(1)
        df["aggressive_buy_ratio"] = buy_vol / df["volume"].replace(0, pd.NA)
        
        logger.info("Computed advanced imbalance proxies (delta_velocity, aggressive_buy_ratio)")
    else:
        logger.warning("Missing taker_buy_volume. Cannot compute imbalance proxies.")
        
    # 3 & 4: Spread / Illiquidity Proxies
    if "volume" in df.columns:
        # Avoid division by zero
        safe_volume = df["volume"].replace(0, pd.NA)
        
        # Amihud Illiquidity proxy (absolute return / volume)
        df["amihud_illiquidity"] = (df["close"] - df["open"]).abs() / safe_volume
        
        # ATR/Volume ratio proxy
        if "atr_14" in df.columns:
            df["volatility_liquidity_ratio"] = df["atr_14"] / safe_volume
            
        if "volume_ratio" in df.columns:
            df["volume_spike_score"] = df["volume_ratio"].rolling(200).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            
        # Composite score
        if "trade_imbalance" in df.columns and "delta_velocity" in df.columns and "volatility_liquidity_ratio" in df.columns:
            # liquidity pressure = imbalance * delta * (1 / spread_ratio)
            # Since our proxy for spread_ratio is volatility_liquidity_ratio, we use its inverse (volume/atr)
            df["liquidity_pressure_score"] = df["trade_imbalance"] * df["delta_velocity"] * (1.0 / df["volatility_liquidity_ratio"].replace(0, np.nan))
            
        logger.info("Computed amihud_illiquidity and advanced liquidity pressure scores")
            
    return df

def build_liquidity_features(symbol="BTCUSDT", base_tf="15m"):
    """
    Loads base features, computes the efficient liquidity proxies,
    and returns the merged DataFrame ready for trade simulation.
    """
    project_root = Path(__file__).resolve().parent.parent
    features_dir = project_root / "data" / "features" / symbol
    
    base_path = features_dir / f"multi_tf_merged_{base_tf}.parquet"
    
    if not base_path.exists():
        logger.error(f"Multi-TF merged features not found at {base_path}")
        return None
        
    df = pd.read_parquet(base_path)
    logger.info(f"Loaded {len(df)} base features for proxy computation.")
    
    df = compute_liquidity_proxies(df)
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    merged_df = build_liquidity_features()
    if merged_df is not None:
        print(f"Columns: {list(merged_df.columns)}")
