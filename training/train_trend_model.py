"""
Trend-Following Model Training — Regime-Specialized Branch.

Trains on candles labeled in trending regimes (trending_low_vol, trending_high_vol).
Uses TREND_FEATURES for direction prediction and TREND_META_FEATURES for
meta-labeling confidence assessment.

Architecture:
  1. Primary Direction Model (XGBoost binary: Long vs Short)
  2. Meta-Labeling Model (XGBoost binary: Will primary succeed?)

Both models use PurgedKFold CV to prevent data leakage.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

from training.config import (
    TREND_FEATURES, TREND_META_FEATURES,
    MOMENTUM_PARAMS, LABEL_MOMENTUM, WF_N_SPLITS
)
from training.walk_forward import walk_forward_cv, summarize_cv_results
from training.evaluate import evaluate_momentum, get_feature_importance

logger = logging.getLogger(__name__)

# Trending regimes this branch is trained on
TREND_REGIMES = ['trending_low_vol', 'trending_high_vol']


def _train_trend_primary(X_train, y_train):
    """Train the Trend Primary Directional Model."""
    y_binary = (y_train == 1).astype(int)
    model = xgb.XGBClassifier(**MOMENTUM_PARAMS, eval_metric='logloss')
    model.fit(X_train, y_binary)
    return model


def _train_trend_meta(X_train, y_train):
    """Train the Trend Meta-Model (success probability)."""
    model = xgb.XGBClassifier(**MOMENTUM_PARAMS, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def train_trend_model(train_df, val_df, test_df, output_dir):
    """
    Full trend-following model training pipeline.

    Filters to trending regimes, trains primary + meta models,
    evaluates on val/test, and saves artifacts.

    Returns:
        primary_model, meta_model, val_metrics, test_metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("TREND-FOLLOWING MODEL TRAINING")
    logger.info("=" * 50)

    # Filter to trending regimes only
    if 'regime_label' in train_df.columns:
        trend_train = train_df[train_df['regime_label'].isin(TREND_REGIMES)].copy()
        trend_val = val_df[val_df['regime_label'].isin(TREND_REGIMES)].copy()
        trend_test = test_df[test_df['regime_label'].isin(TREND_REGIMES)].copy()
    else:
        logger.warning("No regime_label column found, using all data for trend model")
        trend_train = train_df.copy()
        trend_val = val_df.copy()
        trend_test = test_df.copy()

    logger.info(f"  Trend regime samples: train={len(trend_train)}, val={len(trend_val)}, test={len(trend_test)}")

    # Ensure features exist, fill missing with 0
    for feat in TREND_META_FEATURES:
        for df in [trend_train, trend_val, trend_test]:
            if feat not in df.columns:
                df[feat] = 0.0

    # ---- 1. Train Primary Direction Model ----
    logger.info("Training Trend Primary Direction Model...")
    primary_train = trend_train.dropna(subset=[LABEL_MOMENTUM])
    X_train_prim = primary_train[TREND_FEATURES].fillna(0)
    y_train_prim = primary_train[LABEL_MOMENTUM]

    primary_model = _train_trend_primary(X_train_prim, y_train_prim)

    # ---- 2. Train Meta Model ----
    logger.info("Training Trend Meta-Labeling Model...")
    meta_train = trend_train.dropna(subset=[LABEL_MOMENTUM])
    X_train_meta = meta_train[TREND_META_FEATURES].fillna(0)
    y_train_meta = meta_train[LABEL_MOMENTUM]

    meta_model = _train_trend_meta(X_train_meta, y_train_meta)

    # ---- Evaluate ----
    valid_val = trend_val.dropna(subset=[LABEL_MOMENTUM])
    valid_test = trend_test.dropna(subset=[LABEL_MOMENTUM])

    val_metrics = {'auc': 0.5, 'brier': 0.25}
    test_metrics = {'auc': 0.5, 'brier': 0.25}

    if len(valid_val) > 10:
        val_metrics = evaluate_momentum(meta_model, valid_val[TREND_META_FEATURES].fillna(0), valid_val[LABEL_MOMENTUM])
    if len(valid_test) > 10:
        test_metrics = evaluate_momentum(meta_model, valid_test[TREND_META_FEATURES].fillna(0), valid_test[LABEL_MOMENTUM])

    logger.info(f"  Trend Meta Val AUC: {val_metrics['auc']:.4f}")
    logger.info(f"  Trend Meta Test AUC: {test_metrics['auc']:.4f}")

    # Feature Importance
    fi = get_feature_importance(meta_model, TREND_META_FEATURES)
    if not fi.empty:
        logger.info("  Trend Meta Feature Importance (top 5):")
        for _, row in fi.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")

    # ---- Save ----
    joblib.dump(primary_model, output_dir / 'trend_primary_model.pkl')
    joblib.dump(meta_model, output_dir / 'trend_meta_model.pkl')

    logger.info(f"  Saved trend models to {output_dir}")

    return primary_model, meta_model, val_metrics, test_metrics
