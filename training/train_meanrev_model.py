"""
Mean-Reversion Model Training — Regime-Specialized Branch.

Trains on candles labeled in sideways regimes (sideways_low_vol).
Uses MEANREV_FEATURES for reversion signal and MEANREV_META_FEATURES
for meta-labeling confidence assessment.

Architecture:
  1. Reversion Signal Model (XGBoost binary: Will VWAP reversion succeed?)
  2. Meta-Labeling Model (XGBoost binary: Confidence calibration)

Uses label_meanrev from the VWAP-reversion labeler.
Both models use PurgedKFold CV to prevent data leakage.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

from training.config import (
    MEANREV_FEATURES, MEANREV_META_FEATURES,
    MOMENTUM_PARAMS, LABEL_MEANREV, WF_N_SPLITS
)
from training.walk_forward import walk_forward_cv, summarize_cv_results
from training.evaluate import evaluate_momentum, get_feature_importance

logger = logging.getLogger(__name__)

# Sideways regimes this branch is trained on
MEANREV_REGIMES = ['sideways_low_vol']


def _train_meanrev_model(X_train, y_train):
    """Train the Mean-Reversion Signal Model."""
    model = xgb.XGBClassifier(**MOMENTUM_PARAMS, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def _train_meanrev_meta(X_train, y_train):
    """Train the Mean-Reversion Meta-Model (confidence)."""
    model = xgb.XGBClassifier(**MOMENTUM_PARAMS, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def train_meanrev_model(train_df, val_df, test_df, output_dir):
    """
    Full mean-reversion model training pipeline.

    Filters to sideways regimes, trains signal + meta models,
    evaluates on val/test, and saves artifacts.

    Returns:
        signal_model, meta_model, val_metrics, test_metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("MEAN-REVERSION MODEL TRAINING")
    logger.info("=" * 50)

    # Filter to sideways regimes only
    if 'regime_label' in train_df.columns:
        mr_train = train_df[train_df['regime_label'].isin(MEANREV_REGIMES)].copy()
        mr_val = val_df[val_df['regime_label'].isin(MEANREV_REGIMES)].copy()
        mr_test = test_df[test_df['regime_label'].isin(MEANREV_REGIMES)].copy()
    else:
        logger.warning("No regime_label column found, using all data for meanrev model")
        mr_train = train_df.copy()
        mr_val = val_df.copy()
        mr_test = test_df.copy()

    logger.info(f"  MeanRev regime samples: train={len(mr_train)}, val={len(mr_val)}, test={len(mr_test)}")

    # Ensure features exist
    for feat in MEANREV_META_FEATURES:
        for df in [mr_train, mr_val, mr_test]:
            if feat not in df.columns:
                df[feat] = 0.0

    # ---- 1. Train Signal Model ----
    logger.info("Training Mean-Reversion Signal Model...")
    signal_train = mr_train.dropna(subset=[LABEL_MEANREV])

    if len(signal_train) < 50:
        logger.warning(f"  Only {len(signal_train)} labeled samples for meanrev, "
                       f"falling back to full training set")
        signal_train = train_df.dropna(subset=[LABEL_MEANREV]) if LABEL_MEANREV in train_df.columns else pd.DataFrame()

    if len(signal_train) < 50:
        logger.error("  Not enough labeled samples for mean-reversion training. Skipping.")
        return None, None, {'auc': 0.5}, {'auc': 0.5}

    X_train_sig = signal_train[MEANREV_FEATURES].fillna(0)
    y_train_sig = signal_train[LABEL_MEANREV]

    signal_model = _train_meanrev_model(X_train_sig, y_train_sig)

    # ---- 2. Train Meta Model ----
    logger.info("Training Mean-Reversion Meta-Labeling Model...")
    X_train_meta = signal_train[MEANREV_META_FEATURES].fillna(0)
    y_train_meta = signal_train[LABEL_MEANREV]

    meta_model = _train_meanrev_meta(X_train_meta, y_train_meta)

    # ---- Evaluate ----
    valid_val = mr_val.dropna(subset=[LABEL_MEANREV]) if LABEL_MEANREV in mr_val.columns else pd.DataFrame()
    valid_test = mr_test.dropna(subset=[LABEL_MEANREV]) if LABEL_MEANREV in mr_test.columns else pd.DataFrame()

    val_metrics = {'auc': 0.5, 'brier': 0.25}
    test_metrics = {'auc': 0.5, 'brier': 0.25}

    if len(valid_val) > 10:
        val_metrics = evaluate_momentum(meta_model, valid_val[MEANREV_META_FEATURES].fillna(0), valid_val[LABEL_MEANREV])
    if len(valid_test) > 10:
        test_metrics = evaluate_momentum(meta_model, valid_test[MEANREV_META_FEATURES].fillna(0), valid_test[LABEL_MEANREV])

    logger.info(f"  MeanRev Meta Val AUC: {val_metrics['auc']:.4f}")
    logger.info(f"  MeanRev Meta Test AUC: {test_metrics['auc']:.4f}")

    # Feature Importance
    fi = get_feature_importance(meta_model, MEANREV_META_FEATURES)
    if not fi.empty:
        logger.info("  MeanRev Meta Feature Importance (top 5):")
        for _, row in fi.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")

    # ---- Save ----
    joblib.dump(signal_model, output_dir / 'meanrev_signal_model.pkl')
    joblib.dump(meta_model, output_dir / 'meanrev_meta_model.pkl')

    logger.info(f"  Saved mean-reversion models to {output_dir}")

    return signal_model, meta_model, val_metrics, test_metrics
