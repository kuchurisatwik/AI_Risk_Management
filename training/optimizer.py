"""
Optuna Hyperparameter Optimizer — Asset-Specific Walk-Forward Optimization.

Uses Optuna's TPE sampler to find the optimal hyperparameters for each asset
independently. The objective function is walk-forward Sharpe ratio computed
via a lightweight backtest on the validation set.

Saves optimized configs to config/optimized/{SYMBOL}/ for downstream use.
"""

import logging
import numpy as np
import pandas as pd
import yaml
import json
import optuna
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# SEARCH SPACE DEFINITION
# ============================================================

def _suggest_model_params(trial: optuna.Trial) -> dict:
    """Suggest XGBoost model hyperparameters."""
    return {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500, step=50),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 3, 20),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 0.95),
        'random_state': 42,
    }


def _suggest_execution_params(trial: optuna.Trial) -> dict:
    """Suggest execution-layer hyperparameters."""
    return {
        # SL/TP multipliers per regime
        'sl_mult_trending_low': trial.suggest_float('sl_mult_trending_low', 1.5, 5.0),
        'tp_mult_trending_low': trial.suggest_float('tp_mult_trending_low', 2.0, 8.0),
        'sl_mult_trending_high': trial.suggest_float('sl_mult_trending_high', 2.0, 6.0),
        'tp_mult_trending_high': trial.suggest_float('tp_mult_trending_high', 1.5, 5.0),
        'sl_mult_sideways': trial.suggest_float('sl_mult_sideways', 1.0, 4.0),
        'tp_mult_sideways': trial.suggest_float('tp_mult_sideways', 1.0, 4.0),
        'sl_mult_choppy': trial.suggest_float('sl_mult_choppy', 2.0, 6.0),
        'tp_mult_choppy': trial.suggest_float('tp_mult_choppy', 1.0, 4.0),
        
        # Time barrier
        'time_barrier_bars': trial.suggest_int('time_barrier_bars', 18, 72, step=6),
        
        # Threshold floors
        'prob_floor_trending_low': trial.suggest_float('prob_floor_trending_low', 0.50, 0.62),
        'prob_floor_trending_high': trial.suggest_float('prob_floor_trending_high', 0.52, 0.65),
        'prob_floor_sideways': trial.suggest_float('prob_floor_sideways', 0.50, 0.65),
        'prob_floor_choppy': trial.suggest_float('prob_floor_choppy', 0.55, 0.70),
        
        # Risk sizing
        'kelly_frac_trending_low': trial.suggest_float('kelly_frac_trending_low', 0.2, 0.6),
        'kelly_frac_trending_high': trial.suggest_float('kelly_frac_trending_high', 0.15, 0.45),
        'kelly_frac_sideways': trial.suggest_float('kelly_frac_sideways', 0.15, 0.4),
        'target_vol_trending_low': trial.suggest_float('target_vol_trending_low', 0.06, 0.18),
        'target_vol_trending_high': trial.suggest_float('target_vol_trending_high', 0.04, 0.12),
        'max_risk_pct': trial.suggest_float('max_risk_pct', 0.5, 2.5),
        
        # Policy
        'min_inter_trade_bars': trial.suggest_int('min_inter_trade_bars', 6, 24),
        'max_daily_trades': trial.suggest_int('max_daily_trades', 1, 5),
        'soft_reduction_floor': trial.suggest_float('soft_reduction_floor', 0.15, 0.50),
    }


# ============================================================
# LIGHTWEIGHT BACKTEST (for objective evaluation)
# ============================================================

def _lightweight_backtest(
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    exec_params: dict,
    initial_equity: float = 10000.0,
) -> dict:
    """
    Run a lightweight backtest using the specified execution parameters.
    
    This is a simplified version of the full PaperTrader that accepts
    dynamic execution parameters instead of reading from global config.
    Uses precomputed ML predictions to speed up 100x.
    
    Returns dict with sharpe_ratio, total_return, max_drawdown, win_rate, n_trades
    """
    from inference.threshold_engine import REGIME_PROBABILITY_FLOORS
    from training.config import REGIME_RISK_CONFIG
    
    # Override probability floors
    floor_map = {
        'trending_low_vol': exec_params.get('prob_floor_trending_low', 0.52),
        'trending_high_vol': exec_params.get('prob_floor_trending_high', 0.55),
        'sideways_low_vol': exec_params.get('prob_floor_sideways', 0.54),
        'choppy_high_vol': exec_params.get('prob_floor_choppy', 0.60),
        'crash_mode': 1.00,
        'unknown': 0.60,
    }
    
    # Override regime risk config
    regime_cfg = deepcopy(REGIME_RISK_CONFIG)
    regime_cfg['trending_low_vol']['kelly_frac'] = exec_params.get('kelly_frac_trending_low', 0.5)
    regime_cfg['trending_low_vol']['target_vol'] = exec_params.get('target_vol_trending_low', 0.10)
    regime_cfg['trending_high_vol']['kelly_frac'] = exec_params.get('kelly_frac_trending_high', 0.35)
    regime_cfg['trending_high_vol']['target_vol'] = exec_params.get('target_vol_trending_high', 0.08)
    regime_cfg['sideways_low_vol']['kelly_frac'] = exec_params.get('kelly_frac_sideways', 0.3)
    regime_cfg['trending_low_vol']['max_risk_pct'] = exec_params.get('max_risk_pct', 1.5)
    regime_cfg['trending_high_vol']['max_risk_pct'] = exec_params.get('max_risk_pct', 1.0)
    
    # SL/TP multiplier map
    sl_tp_map = {
        'trending_low_vol': (exec_params['sl_mult_trending_low'], exec_params['tp_mult_trending_low']),
        'trending_high_vol': (exec_params['sl_mult_trending_high'], exec_params['tp_mult_trending_high']),
        'sideways_low_vol': (exec_params['sl_mult_sideways'], exec_params['tp_mult_sideways']),
        'choppy_high_vol': (exec_params['sl_mult_choppy'], exec_params['tp_mult_choppy']),
    }
    
    time_barrier = exec_params.get('time_barrier_bars', 36)
    min_spacing = exec_params.get('min_inter_trade_bars', 12)
    max_daily = exec_params.get('max_daily_trades', 3)
    soft_floor = exec_params.get('soft_reduction_floor', 0.25)
    
    # Simulation state
    equity = initial_equity
    peak_equity = equity
    max_drawdown = 0.0
    all_returns = []
    trades = []
    open_trade = None
    bars_held = 0
    bars_since_trade = 999
    daily_count = 0
    current_day = None
    
    SLIPPAGE_PCT = 0.0001
    FEE_PCT = 0.0004
    
    columns = list(test_df.columns)
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        features = {col: float(row[col]) if isinstance(row[col], (int, float, np.integer, np.floating)) else row[col] for col in columns}
        current_close = row['close']
        
        # Day tracking
        open_time = row.get('open_time', None)
        if open_time is not None:
            try:
                day = pd.Timestamp(open_time).date()
                if day != current_day:
                    current_day = day
                    daily_count = 0
            except Exception:
                pass
        
        bars_since_trade += 1
        
        # Check open trade
        if open_trade is not None:
            hit = False
            direction = open_trade['direction']
            
            if direction == 1:
                if row['low'] <= open_trade['sl']:
                    exit_price = open_trade['sl']
                    hit = True
                elif row['high'] >= open_trade['tp']:
                    exit_price = open_trade['tp']
                    hit = True
            else:
                if row['high'] >= open_trade['sl']:
                    exit_price = open_trade['sl']
                    hit = True
                elif row['low'] <= open_trade['tp']:
                    exit_price = open_trade['tp']
                    hit = True
            
            if hit:
                raw_pnl = direction * (exit_price - open_trade['entry'])
                fees = (open_trade['entry'] + exit_price) * FEE_PCT
                slip = exit_price * SLIPPAGE_PCT
                units = open_trade['size_usd'] / open_trade['entry'] if open_trade['entry'] > 0 else 0
                pnl = (raw_pnl - fees - slip) * units
                
                equity += pnl
                all_returns.append(pnl / peak_equity if peak_equity > 0 else 0)
                trades.append(pnl)
                open_trade = None
                bars_held = 0
                bars_since_trade = 0
            else:
                bars_held += 1
                if bars_held >= time_barrier:
                    raw_pnl = direction * (current_close - open_trade['entry'])
                    fees = (open_trade['entry'] + current_close) * FEE_PCT
                    slip = current_close * SLIPPAGE_PCT
                    units = open_trade['size_usd'] / open_trade['entry'] if open_trade['entry'] > 0 else 0
                    pnl = (raw_pnl - fees - slip) * units
                    
                    equity += pnl
                    all_returns.append(pnl / peak_equity if peak_equity > 0 else 0)
                    trades.append(pnl)
                    open_trade = None
                    bars_held = 0
                    bars_since_trade = 0
        
        # New trade decision
        if open_trade is None and bars_since_trade >= min_spacing and daily_count < max_daily:
            pred_row = predictions.iloc[i]
            
            # Probability floor check
            floor = floor_map.get(pred_row['regime_label'], 0.60)
            if pred_row['meta_probability'] >= floor and pred_row['active_branch'] != 'none':
                direction = pred_row['predicted_direction']
                if direction != 0:
                    atr = pred_row['atr_14']
                    entry_price = current_close
                    if atr > 0 and entry_price > 0:
                        sl_mult, tp_mult = sl_tp_map.get(pred_row['regime_label'], (2.5, 4.0))
                        
                        sl_dist = atr * sl_mult
                        tp_dist = atr * tp_mult
                        
                        if direction == 1:
                            sl_price = entry_price - sl_dist
                            tp_price = entry_price + tp_dist
                        else:
                            sl_price = entry_price + sl_dist
                            tp_price = entry_price - tp_dist
                        
                        # Simplified sizing
                        cfg = regime_cfg.get(pred_row['regime_label'], {})
                        risk_pct = max(soft_floor, min(cfg.get('max_risk_pct', 1.5), 1.5))
                        risk_amount = equity * (risk_pct / 100.0)
                        sl_pct = sl_dist / entry_price
                        size_usd = risk_amount / sl_pct if sl_pct > 0 else 0
                        
                        if size_usd > 0:
                            entry_with_slip = entry_price + direction * entry_price * SLIPPAGE_PCT
                            open_trade = {
                                'entry': entry_with_slip,
                                'direction': direction,
                                'sl': sl_price,
                                'tp': tp_price,
                                'size_usd': size_usd,
                            }
                            bars_held = 0
                            daily_count += 1
        
        if open_trade is None:
            all_returns.append(0.0)
        
        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        if dd < max_drawdown:
            max_drawdown = dd
    
    # Compute metrics
    returns = np.array(all_returns)
    n_trades = len(trades)
    
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(105120)
    else:
        sharpe = -100.0
    
    total_return = (equity - initial_equity) / initial_equity
    win_rate = sum(1 for t in trades if t > 0) / n_trades if n_trades > 0 else 0
    
    return {
        'sharpe_ratio': float(sharpe),
        'total_return': float(total_return),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'n_trades': n_trades,
    }


# ============================================================
# OPTIMIZER CLASS
# ============================================================

class AssetOptimizer:
    """
    Asset-specific hyperparameter optimizer using Optuna TPE.
    
    Optimizes execution parameters (SL/TP, thresholds, sizing) against
    walk-forward Sharpe ratio on the validation set.
    """
    
    def __init__(
        self,
        symbol: str,
        models_dir: str,
        val_df: pd.DataFrame,
        n_trials: int = 100,
        output_dir: str = 'config/optimized',
    ):
        self.symbol = symbol
        self.models_dir = models_dir
        self.val_df = val_df
        self.n_trials = n_trials
        self.output_dir = Path(output_dir) / symbol
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._best_params = None
        self._best_sharpe = -np.inf
        
        # Precompute ML predictions for the entire val_df to speed up trials
        logger.info(f"Precomputing ML predictions for {symbol} optimization...")
        from inference.model_ensemble import ModelEnsemble
        ensemble = ModelEnsemble(models_dir)
        ensemble.load()
        
        preds = {
            'regime_label': [],
            'meta_probability': [],
            'active_branch': [],
            'predicted_direction': [],
            'atr_14': []
        }
        columns = list(val_df.columns)
        for i in range(len(val_df)):
            row = val_df.iloc[i]
            features = {col: float(row[col]) if isinstance(row[col], (int, float, np.integer, np.floating)) else row[col] for col in columns}
            out = ensemble.predict(features)
            preds['regime_label'].append(out.regime_label)
            preds['meta_probability'].append(out.meta_probability)
            preds['active_branch'].append(out.active_branch)
            preds['predicted_direction'].append(out.predicted_direction)
            preds['atr_14'].append(features.get('atr_14', 0.0))
            
        self.predictions = pd.DataFrame(preds)
        logger.info(f"Precomputation complete.")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: maximize walk-forward Sharpe ratio."""
        exec_params = _suggest_execution_params(trial)
        
        try:
            result = _lightweight_backtest(
                self.val_df,
                self.predictions,
                exec_params,
            )
            
            sharpe = result['sharpe_ratio']
            n_trades = result['n_trades']
            
            # Penalize if too few trades (< 20)
            if n_trades < 20:
                sharpe -= (20 - n_trades) * 0.5
            
            # Penalize extreme drawdowns
            if result['max_drawdown'] < -0.10:
                sharpe -= abs(result['max_drawdown']) * 10
            
            trial.set_user_attr('total_return', result['total_return'])
            trial.set_user_attr('max_drawdown', result['max_drawdown'])
            trial.set_user_attr('win_rate', result['win_rate'])
            trial.set_user_attr('n_trades', result['n_trades'])
            
            return sharpe
            
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return -100.0
    
    def optimize(self) -> dict:
        """
        Run the optimization.
        
        Returns:
            dict: Best parameters found
        """
        logger.info(f"=" * 60)
        logger.info(f"OPTUNA OPTIMIZATION: {self.symbol}")
        logger.info(f"  Trials: {self.n_trials}")
        logger.info(f"  Val set: {len(self.val_df):,} rows")
        logger.info(f"=" * 60)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)
        
        best = study.best_trial
        self._best_params = best.params
        self._best_sharpe = best.value
        
        logger.info(f"\n  BEST TRIAL #{best.number}:")
        logger.info(f"    Sharpe Ratio:  {best.value:.4f}")
        logger.info(f"    Total Return:  {best.user_attrs.get('total_return', 0):.4%}")
        logger.info(f"    Max Drawdown:  {best.user_attrs.get('max_drawdown', 0):.4%}")
        logger.info(f"    Win Rate:      {best.user_attrs.get('win_rate', 0):.2%}")
        logger.info(f"    N Trades:      {best.user_attrs.get('n_trades', 0)}")
        
        # Save results
        self._save_optimized_config(study)
        
        return self._best_params
    
    def _save_optimized_config(self, study: optuna.Study):
        """Save optimized parameters and metadata."""
        best = study.best_trial
        
        # Execution params
        exec_params = {k: v for k, v in best.params.items()}
        exec_path = self.output_dir / 'execution_params.yaml'
        with open(exec_path, 'w') as f:
            yaml.dump(exec_params, f, default_flow_style=False, sort_keys=False)
        
        # Optimization metadata
        meta = {
            'symbol': self.symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'n_trials': len(study.trials),
            'best_trial': best.number,
            'best_sharpe': float(best.value),
            'best_return': float(best.user_attrs.get('total_return', 0)),
            'best_drawdown': float(best.user_attrs.get('max_drawdown', 0)),
            'best_win_rate': float(best.user_attrs.get('win_rate', 0)),
            'best_n_trades': int(best.user_attrs.get('n_trades', 0)),
        }
        meta_path = self.output_dir / 'optimization_meta.yaml'
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"  Saved optimized config to {self.output_dir}")


# ============================================================
# CONFIG REGISTRY HELPERS
# ============================================================

def config_exists(symbol: str, config_dir: str = 'config/optimized') -> bool:
    """Check if optimized config exists for this symbol."""
    meta_path = Path(config_dir) / symbol / 'optimization_meta.yaml'
    return meta_path.exists()


def config_is_stale(symbol: str, max_age_days: int = 30, config_dir: str = 'config/optimized') -> bool:
    """Check if existing config is too old."""
    meta_path = Path(config_dir) / symbol / 'optimization_meta.yaml'
    if not meta_path.exists():
        return True
    
    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    ts = datetime.fromisoformat(meta.get('timestamp', '2000-01-01'))
    age_days = (datetime.utcnow() - ts).days
    return age_days > max_age_days


def load_optimized_config(symbol: str, config_dir: str = 'config/optimized') -> dict:
    """Load optimized execution params for a symbol."""
    exec_path = Path(config_dir) / symbol / 'execution_params.yaml'
    if not exec_path.exists():
        logger.warning(f"No optimized config found for {symbol}, using defaults")
        return {}
    
    with open(exec_path, 'r') as f:
        return yaml.safe_load(f)


def should_optimize(symbol: str, config_dir: str = 'config/optimized') -> bool:
    """Determine if this asset needs (re)optimization."""
    if not config_exists(symbol, config_dir):
        logger.info(f"  {symbol}: No config found → optimization needed")
        return True
    if config_is_stale(symbol, config_dir=config_dir):
        logger.info(f"  {symbol}: Config is stale → re-optimization needed")
        return True
    logger.info(f"  {symbol}: Fresh config found → skipping optimization")
    return False
