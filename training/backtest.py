"""
Backtest Validation.

Simulates a basic walk-forward backtest on the Test set to validate
regime robustness, Sharpe ratio, and drawdown.
"""

import logging
import numpy as np
import pandas as pd

from training.config import THRESHOLDS

logger = logging.getLogger(__name__)


def run_backtest_validation(test_df):
    """
    Run a simplified strategy simulation on the test set.
    
    Logic:
      - If Risk == NO_TRADE or Behavioral == Anomaly: skip
      - Else if Momentum > 0.6:
          - LONG if ema_20_slope > 0
          - SHORT if ema_20_slope < 0
          - PnL based on forward 1-period close-to-close return
    """
    logger.info("=" * 50)
    logger.info("BACKTEST VALIDATION (TEST SET)")
    logger.info("=" * 50)
    
    df = test_df.copy()
    
    # 1. Trade Filter
    # In live, we'd use model predictions. Here we use labels to see the *theoretical maximum*
    # given our rules, or we can use the injected model outputs if available.
    
    # Check if we have model outputs
    has_models = 'momentum_probability' in df.columns
    if not has_models:
        logger.warning("No model outputs found in test set. Skipping backtest.")
        return {}
    
    # Make trade decisions
    trade_mask = (
        (df['label_risk'] != 'NO_TRADE') & 
        (df['label_behavioral'] == 0) & 
        (df['momentum_probability'] >= THRESHOLDS['momentum_action_threshold'])
    )
    
    direction = np.where(df['ema_20_slope'] > 0, 1, -1)
    returns = df['close'].pct_change().shift(-1).fillna(0)
    
    pnl = np.where(trade_mask, direction * returns, 0)
    df['strategy_return'] = pnl
    df['equity'] = (1 + df['strategy_return']).cumprod()
    
    # Metrics
    total_return = df['equity'].iloc[-1] - 1
    n_trades = trade_mask.sum()
    
    # Annualized Sharpe (assuming 15m candles: 4 * 24 * 365 = 35040 per year)
    mean_ret = df['strategy_return'].mean()
    std_ret = df['strategy_return'].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(35040) if std_ret > 0 else 0
    
    # Max Drawdown
    peak = df['equity'].expanding(min_periods=1).max()
    drawdown = (df['equity'] / peak) - 1
    max_dd = drawdown.min()
    
    logger.info(f"  Total Trades: {n_trades} ({n_trades/len(df)*100:.1f}% of bars)")
    logger.info(f"  Total Return: {total_return*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {sharpe:.2f} (Min Required: {THRESHOLDS['backtest_sharpe_min']})")
    logger.info(f"  Max Drawdown: {max_dd*100:.2f}%")
    
    # Regime breakdown
    logger.info("\n  Performance by Regime:")
    for regime in df['regime_label'].unique():
        r_df = df[df['regime_label'] == regime]
        r_trades = trade_mask[r_df.index].sum()
        r_ret = r_df['strategy_return'].sum()
        logger.info(f"    {regime:18s}: {r_trades:4d} trades | Ret: {r_ret*100:6.2f}%")
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': n_trades
    }
