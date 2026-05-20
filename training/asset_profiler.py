"""
Asset Profiler — Computes per-asset microstructure statistics.

Analyzes the TRAINING SET ONLY to build a statistical fingerprint
of each asset's volatility, liquidity, trend persistence, and
optimal execution parameters.

Output: YAML profile saved to config/asset_profiles/{SYMBOL}.yaml
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)


def compute_asset_profile(train_df: pd.DataFrame, symbol: str) -> dict:
    """
    Compute comprehensive microstructure profile from training data.
    
    Args:
        train_df: Training set DataFrame with all features
        symbol: Asset symbol (e.g., 'BTCUSDT')
    
    Returns:
        dict: Asset profile with all computed statistics
    """
    logger.info(f"Computing asset profile for {symbol} ({len(train_df):,} rows)...")
    
    profile = {
        'symbol': symbol,
        'n_training_rows': len(train_df),
    }
    
    # ================================================================
    # 1. VOLATILITY STATISTICS
    # ================================================================
    if 'atr_14' in train_df.columns:
        atr = train_df['atr_14'].dropna()
        close = train_df['close'].dropna()
        
        # Normalize ATR as percentage of price for cross-asset comparison
        atr_pct = (atr / close.iloc[:len(atr)]).dropna() * 100
        
        profile['volatility'] = {
            'median_atr_14': float(atr.median()),
            'atr_iqr': float(atr.quantile(0.75) - atr.quantile(0.25)),
            'atr_pct_median': float(atr_pct.median()),
            'atr_pct_p95': float(atr_pct.quantile(0.95)),
            'atr_pct_p5': float(atr_pct.quantile(0.05)),
        }
    
    if 'realized_volatility' in train_df.columns:
        rv = train_df['realized_volatility'].dropna()
        profile['volatility']['realized_vol_median'] = float(rv.median())
        profile['volatility']['realized_vol_p95'] = float(rv.quantile(0.95))
        
        # Volatility clustering: autocorrelation at lag 1
        if len(rv) > 100:
            vol_autocorr = rv.autocorr(lag=1)
            profile['volatility']['vol_clustering_autocorr'] = float(vol_autocorr) if not np.isnan(vol_autocorr) else 0.0
    
    if 'volatility_percentile' in train_df.columns:
        vp = train_df['volatility_percentile'].dropna()
        # Time spent in high-vol regime
        profile['volatility']['pct_time_high_vol'] = float((vp > 0.75).mean() * 100)
        profile['volatility']['pct_time_low_vol'] = float((vp < 0.25).mean() * 100)
    
    # ================================================================
    # 2. LIQUIDITY BEHAVIOR
    # ================================================================
    profile['liquidity'] = {}
    
    if 'amihud_illiquidity' in train_df.columns:
        ami = train_df['amihud_illiquidity'].dropna()
        profile['liquidity']['amihud_median'] = float(ami.median())
        profile['liquidity']['amihud_p95'] = float(ami.quantile(0.95))
        profile['liquidity']['amihud_p99'] = float(ami.quantile(0.99))
    
    if 'volume_ratio' in train_df.columns:
        vr = train_df['volume_ratio'].dropna()
        profile['liquidity']['volume_ratio_median'] = float(vr.median())
        profile['liquidity']['volume_ratio_std'] = float(vr.std())
    
    if 'spread_ratio' in train_df.columns:
        sr = train_df['spread_ratio'].dropna()
        profile['liquidity']['spread_ratio_median'] = float(sr.median())
        profile['liquidity']['spread_ratio_p95'] = float(sr.quantile(0.95))
    
    # ================================================================
    # 3. TREND PERSISTENCE
    # ================================================================
    profile['trend'] = {}
    
    if 'ema_20_slope' in train_df.columns:
        slope = train_df['ema_20_slope'].dropna()
        slope_sign = np.sign(slope)
        
        # Autocorrelation of slope sign at lags 1-12
        # Higher = stronger trend persistence
        autocorrs = []
        for lag in [1, 3, 6, 12]:
            ac = pd.Series(slope_sign.values).autocorr(lag=lag)
            if not np.isnan(ac):
                autocorrs.append(float(ac))
        
        if autocorrs:
            profile['trend']['slope_sign_autocorr_lag1'] = autocorrs[0] if len(autocorrs) > 0 else 0.0
            profile['trend']['slope_sign_autocorr_lag12'] = autocorrs[-1] if len(autocorrs) > 3 else 0.0
        
        # Trend persistence: average consecutive bars with same slope sign
        sign_changes = (slope_sign.diff().abs() > 0).sum()
        if sign_changes > 0:
            avg_trend_duration = len(slope_sign) / sign_changes
            profile['trend']['avg_trend_duration_bars'] = float(avg_trend_duration)
        
    if 'trend_strength_score' in train_df.columns:
        ts = train_df['trend_strength_score'].dropna()
        profile['trend']['strength_median'] = float(ts.median())
        profile['trend']['strength_p75'] = float(ts.quantile(0.75))
    
    # ================================================================
    # 4. REGIME PERSISTENCE
    # ================================================================
    if 'regime_label' in train_df.columns:
        regimes = train_df['regime_label'].dropna()
        regime_stats = {}
        
        # Calculate per-regime duration statistics
        current_regime = regimes.iloc[0]
        current_count = 1
        regime_runs = []
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_count += 1
            else:
                regime_runs.append({'regime': current_regime, 'duration': current_count})
                current_regime = regimes.iloc[i]
                current_count = 1
        regime_runs.append({'regime': current_regime, 'duration': current_count})
        
        runs_df = pd.DataFrame(regime_runs)
        for regime_name in runs_df['regime'].unique():
            regime_durations = runs_df[runs_df['regime'] == regime_name]['duration']
            regime_stats[regime_name] = {
                'mean_duration_bars': float(regime_durations.mean()),
                'median_duration_bars': float(regime_durations.median()),
                'pct_time': float((regimes == regime_name).mean() * 100),
                'n_episodes': int(len(regime_durations)),
            }
        
        profile['regime_persistence'] = regime_stats
    
    # ================================================================
    # 5. MOMENTUM DECAY
    # ================================================================
    profile['momentum'] = {}
    
    if 'momentum_score' in train_df.columns:
        ms = train_df['momentum_score'].dropna()
        
        # Momentum decay halflife: how quickly momentum autocorrelation drops
        autocorrs = []
        for lag in range(1, 25):
            ac = ms.autocorr(lag=lag)
            if not np.isnan(ac):
                autocorrs.append(float(ac))
        
        if len(autocorrs) > 5:
            # Find lag where autocorrelation drops below 0.5 of lag-1
            halflife = None
            if autocorrs[0] > 0:
                target = autocorrs[0] * 0.5
                for i, ac in enumerate(autocorrs):
                    if ac < target:
                        halflife = i + 1
                        break
            profile['momentum']['decay_halflife_bars'] = halflife or len(autocorrs)
    
    # ================================================================
    # 6. OPTIMAL SL/TP EMPIRICAL ANALYSIS
    # ================================================================
    if 'atr_14' in train_df.columns and 'close' in train_df.columns:
        profile['optimal_execution'] = _compute_optimal_barriers(train_df)
    
    # ================================================================
    # 7. DRAWDOWN BEHAVIOR
    # ================================================================
    if 'close' in train_df.columns:
        close = train_df['close'].dropna()
        cum_max = close.cummax()
        drawdown = (close - cum_max) / cum_max
        
        profile['drawdown'] = {
            'max_drawdown_pct': float(drawdown.min() * 100),
            'avg_drawdown_pct': float(drawdown.mean() * 100),
            'drawdown_duration_median_bars': float(_avg_drawdown_duration(close)),
        }
    
    logger.info(f"  Asset profile computed for {symbol}")
    return profile


def _compute_optimal_barriers(df: pd.DataFrame, sample_size: int = 20000) -> dict:
    """
    Empirically find optimal ATR SL/TP multipliers that maximize expectancy.
    
    Tests a grid of SL/TP multipliers on a sample of the training data
    and returns the combination with the highest expectancy.
    """
    # Sample to keep computation fast
    if len(df) > sample_size:
        sample_idx = np.random.choice(len(df) - 50, size=sample_size, replace=False)
    else:
        sample_idx = np.arange(len(df) - 50)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr_14'].values
    slopes = df['ema_20_slope'].values if 'ema_20_slope' in df.columns else np.ones(len(df))
    
    best_expectancy = -np.inf
    best_params = {'sl_mult': 2.5, 'tp_mult': 4.0, 'time_barrier': 36}
    
    # Grid search over SL/TP multipliers and time barriers
    sl_range = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    tp_range = [2.0, 3.0, 4.0, 5.0, 6.0]
    tb_range = [24, 36, 48]
    
    for sl_mult in sl_range:
        for tp_mult in tp_range:
            for tb in tb_range:
                wins = 0
                losses = 0
                total_win_pnl = 0.0
                total_loss_pnl = 0.0
                
                for i in sample_idx:
                    if i + tb >= len(df):
                        continue
                    
                    entry = closes[i]
                    atr = atrs[i]
                    if atr <= 0 or entry <= 0:
                        continue
                    
                    direction = 1 if slopes[i] > 0 else -1
                    sl = entry - direction * atr * sl_mult
                    tp = entry + direction * atr * tp_mult
                    
                    hit_tp = False
                    hit_sl = False
                    
                    for j in range(i + 1, min(i + 1 + tb, len(df))):
                        if direction == 1:
                            if lows[j] <= sl:
                                hit_sl = True
                                break
                            if highs[j] >= tp:
                                hit_tp = True
                                break
                        else:
                            if highs[j] >= sl:
                                hit_sl = True
                                break
                            if lows[j] <= tp:
                                hit_tp = True
                                break
                    
                    if hit_tp:
                        wins += 1
                        total_win_pnl += atr * tp_mult
                    elif hit_sl:
                        losses += 1
                        total_loss_pnl += atr * sl_mult
                
                total = wins + losses
                if total > 100:
                    win_rate = wins / total
                    avg_win = total_win_pnl / wins if wins > 0 else 0
                    avg_loss = total_loss_pnl / losses if losses > 0 else 1
                    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                    
                    if expectancy > best_expectancy:
                        best_expectancy = expectancy
                        best_params = {
                            'sl_mult': float(sl_mult),
                            'tp_mult': float(tp_mult),
                            'time_barrier': int(tb),
                            'win_rate': float(win_rate),
                            'expectancy': float(expectancy),
                            'n_trades': int(total),
                        }
    
    return best_params


def _avg_drawdown_duration(close: pd.Series) -> float:
    """Calculate average duration (in bars) of drawdown periods."""
    cum_max = close.cummax()
    in_drawdown = close < cum_max
    
    # Find contiguous drawdown segments
    segments = []
    current_length = 0
    for val in in_drawdown:
        if val:
            current_length += 1
        else:
            if current_length > 0:
                segments.append(current_length)
            current_length = 0
    if current_length > 0:
        segments.append(current_length)
    
    return np.median(segments) if segments else 0.0


def save_asset_profile(profile: dict, output_dir: str = 'config/asset_profiles'):
    """Save asset profile to YAML."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    symbol = profile['symbol']
    filepath = output_dir / f'{symbol}.yaml'
    
    with open(filepath, 'w') as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"  Saved asset profile to {filepath}")
    return filepath


def load_asset_profile(symbol: str, profiles_dir: str = 'config/asset_profiles') -> dict:
    """Load asset profile from YAML."""
    filepath = Path(profiles_dir) / f'{symbol}.yaml'
    if not filepath.exists():
        logger.warning(f"No asset profile found for {symbol}")
        return {}
    
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)
