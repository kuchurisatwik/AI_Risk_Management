"""
Hierarchical Meta-Ensemble Training.

Phase 6 Redesign:
1. Primary Model: Predicts Direction (LONG vs SHORT)
2. Meta Model: Predicts Confidence (Will the primary model succeed?)

Drops Isotonic calibration. Outputs raw log-odds (margin) for 
cross-sectional Z-score ranking in the execution engine.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

from training.config import MOMENTUM_FEATURES, META_FEATURES, MOMENTUM_PARAMS, WF_N_SPLITS
from training.walk_forward import walk_forward_cv, summarize_cv_results
from training.evaluate import evaluate_momentum, get_feature_importance

logger = logging.getLogger(__name__)


def _train_primary_model(X_train, y_train):
    """Train the Primary Directional Model (0=Short/Flat, 1=Long)."""
    # y_train is -1, 0, 1. We map to 0 (Short/Flat) and 1 (Long) for XGBoost Binary
    y_binary = (y_train == 1).astype(int)
    
    model = xgb.XGBClassifier(**MOMENTUM_PARAMS, eval_metric='logloss')
    model.fit(X_train, y_binary)
    return model


def _train_meta_model(X_train, y_train):
    """Train the Secondary Meta-Model (0=Fail, 1=Success)."""
    # Output raw log-odds (margins) for ranking, not just probability
    model = xgb.XGBClassifier(**MOMENTUM_PARAMS, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def train_meta_ensemble(train_df, val_df, test_df, output_dir):
    """
    Full Meta-Labeling training pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("META-ENSEMBLE TRAINING (Primary + Meta)")
    logger.info("=" * 50)
    
    # ---- Filter Data ----
    # Primary model trains on all data where a directional move occurred
    primary_train = train_df.dropna(subset=['primary_label'])
    
    # Meta model ONLY trains on cases where the primary model would have traded
    # (i.e. where the primary model predicted Long and true label was Long, OR
    # primary predicted Short and true label was Short. But since we don't have
    # the out-of-sample primary predictions for the training set easily available
    # without a full K-Fold, we train the Meta Model on all valid setups).
    meta_train = train_df.dropna(subset=['meta_label'])
    
    # ---- 1. Train Primary Model ----
    logger.info("Training Primary Directional Model...")
    X_train_prim = primary_train[MOMENTUM_FEATURES]
    y_train_prim = primary_train['primary_label']
    primary_model = _train_primary_model(X_train_prim, y_train_prim)
    
    # ---- 2. Train Meta Model (uses expanded META_FEATURES for orthogonal info) ----
    logger.info("Training Secondary Meta-Labeling Model...")
    X_train_meta = meta_train[META_FEATURES]
    y_train_meta = meta_train['meta_label']
    meta_model = _train_meta_model(X_train_meta, y_train_meta)
    
    # ---- Evaluate Meta Model ----
    valid_val = val_df.dropna(subset=['meta_label'])
    valid_test = test_df.dropna(subset=['meta_label'])
    
    val_metrics = evaluate_momentum(meta_model, valid_val[META_FEATURES], valid_val['meta_label'])
    test_metrics = evaluate_momentum(meta_model, valid_test[META_FEATURES], valid_test['meta_label'])
    
    logger.info(f"  Meta-Model Val AUC: {val_metrics['auc']:.4f}")
    logger.info(f"  Meta-Model Test AUC: {test_metrics['auc']:.4f}")
    
    # Feature Importance (Meta Model)
    fi = get_feature_importance(meta_model, META_FEATURES)
    if not fi.empty:
        logger.info("  Meta-Model Feature Importance (top 5):")
        for _, row in fi.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")
            
    # ---- Generate Predictions for downstream models ----
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X_prim = df[MOMENTUM_FEATURES]
        X_meta = df[META_FEATURES]
        
        # Primary Prediction (Direction)
        primary_preds = primary_model.predict(X_prim)
        df['predicted_direction'] = np.where(primary_preds == 1, 1, -1)
        
        # Meta Prediction (Confidence) — uses expanded META_FEATURES
        meta_probs = meta_model.predict_proba(X_meta)[:, 1]
        meta_margins = meta_model.predict(X_meta, output_margin=True)
        
        df['momentum_probability'] = meta_probs
        df['meta_margin'] = meta_margins

    # ---- Save ----
    joblib.dump(primary_model, output_dir / 'primary_direction_model.pkl')
    joblib.dump(meta_model, output_dir / 'meta_confidence_model.pkl')
    
    logger.info(f"  Saved Meta-Ensemble models to {output_dir}")
    
    return primary_model, meta_model, val_metrics, test_metrics
