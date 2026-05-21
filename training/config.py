"""
Training Configuration — Feature lists, hyperparameters, thresholds.

Central configuration for all Phase 4 model training.
All feature lists map to actual column names in our labeled dataset.
"""

# ============================================================
# FEATURE SETS (Dual-Branch Architecture)
# ============================================================

# --- Trend-Following Branch ---
TREND_FEATURES = [
    'ema_20_slope', 'ema_50_slope', 'trend_strength_score',
    'momentum_score', 'breakout_distance', 'volume_ratio',
    'atr_expansion_ratio', 'rsi_velocity', 'vwap_distance',
    'regime_cluster', 'trend_alignment_score'
]

TREND_META_FEATURES = TREND_FEATURES + [
    'atr_14', 'realized_volatility', 'volatility_percentile',
    'amihud_illiquidity', 'regime_confidence',
    'momentum_exhaustion_score'
]

# --- Mean-Reversion Branch ---
MEANREV_FEATURES = [
    'vwap_zscore', 'normalized_vwap_distance', 'rsi_extremity',
    'range_position', 'upper_rejection_score', 'lower_rejection_score',
    'trade_imbalance', 'bb_width_percentile', 'compression_score',
    'regime_cluster', 'trend_alignment_score'
]

MEANREV_META_FEATURES = MEANREV_FEATURES + [
    'atr_14', 'realized_volatility', 'volatility_percentile',
    'amihud_illiquidity', 'regime_confidence',
    'volume_ratio'
]

# --- Legacy: kept for backward compatibility with existing models ---
MOMENTUM_FEATURES = [
    'ema_20_slope', 'ema_20_slope_5m',
    'rsi_velocity', 'rsi_velocity_5m',
    'volume_delta', 'atr_expansion_ratio',
    'vwap_distance', 'regime_cluster'
]

# Meta-Model features: MOMENTUM_FEATURES + orthogonal volatility/liquidity signals
META_FEATURES = MOMENTUM_FEATURES + [
    'atr_14', 'realized_volatility', 'volatility_percentile',
    'volume_ratio', 'amihud_illiquidity', 'regime_confidence'
]

VOLATILITY_FEATURES = [
    'atr_14', 'atr_expansion_ratio', 'atr_velocity',
    'bb_width', 'bb_width_percentile',
    'volume_ratio', 'amihud_illiquidity', 'realized_volatility'
]

RISK_FEATURES = [
    'regime_cluster', 'regime_confidence',
    'momentum_probability',       # injected from Model 1
    'predicted_volatility',       # injected from Model 2
    'atr_expansion_ratio', 'volatility_percentile',
    'amihud_illiquidity',
    'strategy_health_score', 'strategy_avg_rr',
    'last_5_trade_winrate', 'consecutive_losses', 'recent_drawdown',
    'emotional_risk_score'
]

BEHAVIORAL_FEATURES = [
    'oversized_trade_score',
    'overtrading_score', 'emotional_risk_score',
    'consecutive_losses', 'recent_drawdown',
    'fomo_score', 'loss_recovery_aggression',
    'time_since_last_loss'
]

# ============================================================
# LABEL COLUMNS
# ============================================================

LABEL_MOMENTUM = 'label_momentum'
LABEL_MEANREV = 'label_meanrev'
LABEL_VOLATILITY = 'label_volatility'
LABEL_RISK = 'label_risk'
LABEL_BEHAVIORAL = 'label_behavioral'

# Risk label encoding
RISK_CLASSES = ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK', 'NO_TRADE']

# ============================================================
# REGIME RISK CONFIGURATION
# ============================================================

REGIME_RISK_CONFIG = {
    'trending_low_vol':  {'target_vol': 0.10, 'max_risk_pct': 1.5, 'kelly_frac': 0.5,  'max_daily_trades': 3, 'branch': 'trend'},
    'trending_high_vol': {'target_vol': 0.08, 'max_risk_pct': 1.0, 'kelly_frac': 0.35, 'max_daily_trades': 2, 'branch': 'trend'},
    'sideways_low_vol':  {'target_vol': 0.06, 'max_risk_pct': 0.8, 'kelly_frac': 0.3,  'max_daily_trades': 2, 'branch': 'meanrev'},
    'choppy_high_vol':   {'target_vol': 0.04, 'max_risk_pct': 0.5, 'kelly_frac': 0.2,  'max_daily_trades': 1, 'branch': 'best'},
    'crash_mode':        {'target_vol': 0.00, 'max_risk_pct': 0.0, 'kelly_frac': 0.0,  'max_daily_trades': 0, 'branch': 'block'},
}

# ============================================================
# HYPERPARAMETERS
# ============================================================

MOMENTUM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
}

VOLATILITY_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 15,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
}

RISK_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'objective': 'multi:softprob',
    'num_class': 4,
}

# ============================================================
# VALIDATION THRESHOLDS (Minimum Acceptable)
# ============================================================

THRESHOLDS = {
    'momentum_auc_min': 0.56,
    'momentum_action_threshold': 0.60,
    'volatility_rmse_vs_baseline': 1.0,   # must be < 1.0 (beat EWMA)
    'risk_no_trade_precision_min': 0.70,
    'behavioral_f1_min': 0.65,
    'regime_silhouette_min': 0.30,
    'backtest_sharpe_min': 0.5,
}

# ============================================================
# WALK-FORWARD CV
# ============================================================

WF_N_SPLITS = 5

# ============================================================
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = 'mlruns'
MLFLOW_EXPERIMENT_MOMENTUM = 'momentum_v1'
MLFLOW_EXPERIMENT_VOLATILITY = 'volatility_v1'
MLFLOW_EXPERIMENT_RISK = 'risk_v1'
MLFLOW_EXPERIMENT_BEHAVIORAL = 'behavioral_v1'
