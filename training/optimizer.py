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
# OPTIMIZER CLASS
# ==============================================================

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
        
        self.precomputed_list = []
        columns = list(val_df.columns)
        for i in range(len(val_df)):
            row = val_df.iloc[i]
            features = {col: float(row[col]) if isinstance(row[col], (int, float, np.integer, np.floating)) else row[col] for col in columns}
            self.precomputed_list.append(ensemble.predict(features))
            
        logger.info(f"Precomputation complete.")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: maximize walk-forward Sharpe ratio."""
        exec_params = _suggest_execution_params(trial)
        
        try:
            from execution.paper_trader import PaperTrader
            trader = PaperTrader(models_dir=self.models_dir, initial_equity=10000.0, exec_params=exec_params)
            # Skip engine loading because we are passing precomputed predictions
            trader.engine.ensemble._loaded = True
            
            pt_result = trader.run(self.val_df, precomputed_outputs=self.precomputed_list)
            
            sharpe = pt_result.sharpe_ratio
            n_trades = pt_result.total_trades
            
            # Penalize if too few trades (< 20)
            if n_trades < 20:
                sharpe -= (20 - n_trades) * 0.5
            
            # Penalize extreme drawdowns
            if pt_result.max_drawdown < -0.10:
                sharpe -= abs(pt_result.max_drawdown) * 10
            
            trial.set_user_attr('total_return', pt_result.total_return)
            trial.set_user_attr('max_drawdown', pt_result.max_drawdown)
            trial.set_user_attr('win_rate', pt_result.win_rate)
            trial.set_user_attr('n_trades', pt_result.total_trades)
            
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
