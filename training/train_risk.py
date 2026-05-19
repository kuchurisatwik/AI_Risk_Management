"""
Risk Model Training — LightGBM Multi-class.

Outputs predicted risk level (LOW_RISK, MEDIUM_RISK, HIGH_RISK, NO_TRADE).
Focus: Precision on NO_TRADE class.
Requires momentum and volatility outputs to be present in the dataset.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from training.config import RISK_FEATURES, RISK_PARAMS, LABEL_RISK, RISK_CLASSES, WF_N_SPLITS
from training.walk_forward import walk_forward_cv, summarize_cv_results
from training.evaluate import evaluate_risk, get_feature_importance

logger = logging.getLogger(__name__)


def _train_risk_model(X_train, y_train):
    """Train XGBoost Multi-class model."""
    # Ensure y_train is integer encoded for XGBoost
    if y_train.dtype == 'object':
        le = LabelEncoder()
        le.fit(RISK_CLASSES)
        y_train = le.transform(y_train)
        
    model = xgb.XGBClassifier(**RISK_PARAMS)
    model.fit(X_train, y_train)
    return model


class RiskModelWrapper:
    """Wraps the XGBoost model to handle label encoding internally."""
    def __init__(self, model, classes):
        self.model = model
        self.le = LabelEncoder()
        self.le.fit(classes)
        
    def predict(self, X):
        preds_int = self.model.predict(X)
        return self.le.inverse_transform(preds_int)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _train_risk_model_wrapped(X_train, y_train):
    """Train and wrap to return string labels."""
    raw_model = _train_risk_model(X_train, y_train)
    return RiskModelWrapper(raw_model, RISK_CLASSES)


def train_risk(train_df, val_df, test_df, output_dir):
    """
    Full risk training pipeline.
    
    1. Verify upstream features (momentum_proba, predicted_volatility) exist
    2. Walk-forward CV on training set
    3. Train final model
    4. Evaluate (focus on NO_TRADE precision)
    5. Save model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("RISK MODEL TRAINING")
    logger.info("=" * 50)
    
    # ---- Verify Dependencies ----
    for col in ['momentum_probability', 'predicted_volatility']:
        if col not in train_df.columns:
            raise ValueError(f"Missing upstream feature: {col}. Run momentum/vol training first.")
            
    # ---- Walk-Forward CV ----
    logger.info("Walk-Forward Cross-Validation...")
    cv_results = walk_forward_cv(
        train_df, RISK_FEATURES, LABEL_RISK,
        train_fn=_train_risk_model_wrapped,
        eval_fn=lambda m, X, y: evaluate_risk(m, X, y, RISK_CLASSES),
        n_splits=WF_N_SPLITS
    )
    cv_summary = summarize_cv_results(cv_results)
    
    logger.info(f"  CV NO_TRADE Precision: {cv_summary['no_trade_precision']['mean']:.4f}")
    logger.info(f"  CV Weighted F1:        {cv_summary['weighted_f1']['mean']:.4f}")
    
    # ---- Train Final Model ----
    logger.info("Training final model on full training set...")
    valid_train = train_df.dropna(subset=[LABEL_RISK])
    X_train = valid_train[RISK_FEATURES]
    y_train = valid_train[LABEL_RISK]
    
    final_model = _train_risk_model_wrapped(X_train, y_train)
    
    # ---- Evaluate on Val/Test ----
    valid_val = val_df.dropna(subset=[LABEL_RISK])
    valid_test = test_df.dropna(subset=[LABEL_RISK])
    
    val_metrics = evaluate_risk(final_model, valid_val[RISK_FEATURES], valid_val[LABEL_RISK], RISK_CLASSES)
    test_metrics = evaluate_risk(final_model, valid_test[RISK_FEATURES], valid_test[LABEL_RISK], RISK_CLASSES)
    
    logger.info(f"  Val NO_TRADE Precision: {val_metrics['no_trade_precision']:.4f} | F1: {val_metrics['weighted_f1']:.4f}")
    logger.info(f"  Test NO_TRADE Precision: {test_metrics['no_trade_precision']:.4f} | F1: {test_metrics['weighted_f1']:.4f}")
    
    # ---- Feature Importance ----
    fi = get_feature_importance(final_model.model, RISK_FEATURES)
    if not fi.empty:
        logger.info("  Feature Importance (top 5):")
        for _, row in fi.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")
    
    # ---- Save ----
    joblib.dump(final_model, output_dir / 'risk_model.pkl')
    if not fi.empty:
        fi.to_csv(output_dir / 'risk_feature_importance.csv', index=False)
        
    logger.info(f"  Saved risk model to {output_dir}")
    
    return final_model, cv_summary, val_metrics, test_metrics
