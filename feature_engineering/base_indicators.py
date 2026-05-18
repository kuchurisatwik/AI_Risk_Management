"""
Base Indicators — EMA, ATR, RSI computation.
All computations are strictly causal (no forward-looking).
Uses only pandas/numpy — no TA-Lib dependency.
"""

import numpy as np
import pandas as pd


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = EMA(TR, period)
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.
    Output range: 0-100. Normalize to 0-1 before model input.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(series: pd.Series, period: int = 20,
                             std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: middle, upper, lower, width."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle
    return pd.DataFrame({
        "bb_middle": middle, "bb_upper": upper,
        "bb_lower": lower, "bb_width": width
    })


def compute_vwap(df: pd.DataFrame, rolling_window: int = None) -> pd.Series:
    """
    Volume-Weighted Average Price.
    For crypto (24/7), uses rolling window instead of session reset.
    If rolling_window is None, computes cumulative VWAP.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    if rolling_window:
        vwap = (tp_vol.rolling(rolling_window).sum() /
                df["volume"].rolling(rolling_window).sum())
    else:
        vwap = tp_vol.cumsum() / df["volume"].cumsum()
    return vwap


def compute_all_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Phase 1 base indicators on a DataFrame.
    Input must have: open, high, low, close, volume columns.
    Returns DataFrame with original + indicator columns.
    """
    result = df.copy()

    # EMAs (Section 4.2.1)
    for period in [5, 9, 20, 50, 100, 200]:
        result[f"ema_{period}"] = compute_ema(df["close"], period)

    # EMA slopes normalized by ATR (Section 4.2.1)
    atr_14 = compute_atr(df, 14)
    result["atr_14"] = atr_14
    result["atr_50"] = compute_atr(df, 50)

    for period in [5, 9, 20, 50]:
        ema_col = f"ema_{period}"
        slope = result[ema_col] - result[ema_col].shift(1)
        result[f"ema_{period}_slope"] = slope / atr_14

    # Trend Crossovers & Strength (Section 4.2.1)
    result["ema_cross_5_20"] = (result["ema_5"] > result["ema_20"]).astype(int) - (result["ema_5"] < result["ema_20"]).astype(int)
    result["ema_cross_20_50"] = (result["ema_20"] > result["ema_50"]).astype(int) - (result["ema_20"] < result["ema_50"]).astype(int)
    result["trend_strength_score"] = result["ema_5_slope"] + result["ema_9_slope"] + result["ema_20_slope"] + result["ema_50_slope"]
    result["trend_acceleration"] = result["trend_strength_score"] - result["trend_strength_score"].shift(1)
    
    # Momentum (Price velocity)
    result["price_velocity"] = (df["close"] - df["close"].shift(10)) / atr_14

    # ATR features (Section 4.2.2)
    atr_rolling_mean = atr_14.rolling(50).mean()
    result["atr_expansion_ratio"] = atr_14 / atr_rolling_mean
    result["atr_velocity"] = atr_14 - atr_14.shift(1)
    result["volatility_percentile"] = atr_14.rolling(200).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    # Realized Volatility (annualized)
    # Assuming 15m candles: 365 * 24 * 4 = 35040 periods in a year
    # We will use rolling 20 period std of returns annualized
    pct_returns = df["close"].pct_change()
    result["realized_volatility"] = pct_returns.rolling(20).std() * np.sqrt(35040)
    result["volatility_regime_score"] = result["atr_expansion_ratio"] + result["volatility_percentile"]

    # RSI (Section 4.2.3)
    rsi = compute_rsi(df["close"], 14)
    result["rsi_14"] = rsi
    result["rsi_velocity"] = rsi - rsi.shift(1)
    result["rsi_acceleration"] = result["rsi_velocity"] - result["rsi_velocity"].shift(1)
    
    # RSI Divergence (simplified): price makes higher high, RSI makes lower high over 10 periods
    price_hh = df["close"] > df["close"].shift(10)
    rsi_lh = rsi < rsi.shift(10)
    price_ll = df["close"] < df["close"].shift(10)
    rsi_hl = rsi > rsi.shift(10)
    result["rsi_divergence_score"] = (price_hh & rsi_lh).astype(int) - (price_ll & rsi_hl).astype(int)

    # Bollinger Bands (Section 4.2.4)
    bb = compute_bollinger_bands(df["close"], 20, 2.0)
    result = pd.concat([result, bb], axis=1)
    result["bb_width_percentile"] = result["bb_width"].rolling(200).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    result["compression_score"] = ((result["bb_width_percentile"] < 0.2) & (result["bb_width"] < result["bb_width"].shift(1))).astype(int)
    result["expansion_score"] = ((result["bb_width_percentile"] > 0.8) & (result["bb_width"] > result["bb_width"].shift(1))).astype(int)

    # VWAP (Section 4.2.5)
    result["vwap"] = compute_vwap(df, rolling_window=100)
    result["vwap_distance"] = (df["close"] - result["vwap"]) / atr_14
    result["normalized_vwap_distance"] = result["vwap_distance"] / result["vwap_distance"].rolling(50).std().replace(0, np.nan)
    result["vwap_trend_alignment"] = (df["close"] > result["vwap"]).astype(int) - (df["close"] < result["vwap"]).astype(int)

    # Volume features
    vol_mean = df["volume"].rolling(20).mean()
    result["volume_ratio"] = df["volume"] / vol_mean

    # Rolling std (Section 4.2.2)
    pct_change = df["close"].pct_change()
    result["rolling_std_20"] = pct_change.rolling(20).std()
    result["rolling_std_50"] = pct_change.rolling(50).std()

    # Price action (Section 4.2.7)
    result["candle_body_size"] = (df["close"] - df["open"]).abs() / atr_14
    candle_range = df["high"] - df["low"]
    result["body_to_range_ratio"] = (
        (df["close"] - df["open"]).abs() / candle_range.replace(0, np.nan)
    )
    result["upper_wick_size"] = (df["high"] - df[["open", "close"]].max(axis=1)) / atr_14
    result["lower_wick_size"] = (df[["open", "close"]].min(axis=1) - df["low"]) / atr_14
    result["wick_imbalance_ratio"] = result["upper_wick_size"] / result["lower_wick_size"].replace(0, np.nan)
    
    # Engulfing Flags
    prev_body = (df["close"].shift(1) - df["open"].shift(1))
    curr_body = (df["close"] - df["open"])
    result["bullish_engulfing_flag"] = ((prev_body < 0) & (curr_body > 0) & (df["close"] > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))).astype(int)
    result["bearish_engulfing_flag"] = ((prev_body > 0) & (curr_body < 0) & (df["close"] < df["open"].shift(1)) & (df["open"] > df["close"].shift(1))).astype(int)
    
    # Inside / Outside bars
    result["inside_bar_flag"] = ((df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))).astype(int)
    result["outside_bar_flag"] = ((df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))).astype(int)
    
    # Breakout & Fake Breakout (using rolling 20 high/low as resistance/support)
    rolling_high = df["high"].shift(1).rolling(20).max()
    rolling_low = df["low"].shift(1).rolling(20).min()
    breakout_up = df["close"] - rolling_high
    breakout_down = rolling_low - df["close"]
    
    # breakout distance = how far past the resistance
    result["breakout_distance"] = np.where(breakout_up > 0, breakout_up / atr_14, 
                                    np.where(breakout_down > 0, -breakout_down / atr_14, 0))
                                    
    # Composite scores
    result["momentum_score"] = result["rsi_velocity"] + result["trend_strength_score"]
    result["momentum_exhaustion_score"] = (result["rsi_divergence_score"].abs() * result["volatility_percentile"])

    return result


if __name__ == "__main__":
    # Quick test with sample data
    print("Base indicators module loaded successfully.")
    print("Available functions: compute_ema, compute_atr, compute_rsi,")
    print("  compute_bollinger_bands, compute_vwap, compute_all_base_indicators")
