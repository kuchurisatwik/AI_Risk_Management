"""
Evaluation Module — Metrics, calibration, feature importance.

Provides standardized evaluation functions for all models.
Focuses on calibration quality, not just accuracy.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)

logger = logging.getLogger(__name__)


# ============================================================
# MOMENTUM EVALUATION (Binary Classification)
# ============================================================

def evaluate_momentum(model, X, y):
    """
    Evaluate momentum model with calibration-focused metrics.
    
    Returns:
        dict with auc, brier, precision_at_06, recall, f1
    """
    y = np.array(y, dtype=float)
    valid_mask = ~np.isnan(y)
    X, y = X[valid_mask], y[valid_mask]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X)
    
    auc = roc_auc_score(y, proba)
    brier = brier_score_loss(y, proba)
    
    # Precision at the action threshold (0.60)
    preds_at_06 = (proba >= 0.60).astype(int)
    prec_06 = precision_score(y, preds_at_06, zero_division=0)
    rec_06 = recall_score(y, preds_at_06, zero_division=0)
    f1_06 = f1_score(y, preds_at_06, zero_division=0)
    
    return {
        'auc': auc,
        'brier': brier,
        'precision_at_06': prec_06,
        'recall_at_06': rec_06,
        'f1_at_06': f1_06,
    }


# ============================================================
# VOLATILITY EVALUATION (Regression)
# ============================================================

def evaluate_volatility(model, X, y):
    """
    Evaluate volatility model with regression metrics + directional accuracy.
    """
    y = np.array(y, dtype=float)
    valid_mask = ~np.isnan(y)
    X, y = X[valid_mask], y[valid_mask]
    
    preds = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    
    # Directional accuracy: did we predict expanding vs contracting correctly?
    if len(y) > 1:
        y_diff = np.diff(y)
        p_diff = np.diff(preds)
        dir_acc = np.mean(np.sign(y_diff) == np.sign(p_diff))
    else:
        dir_acc = 0.0
    
    return {
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': dir_acc,
    }


# ============================================================
# RISK EVALUATION (Multi-class Classification)
# ============================================================

def evaluate_risk(model, X, y, class_names=None):
    """
    Evaluate risk model with NO_TRADE precision focus.
    """
    if class_names is None:
        class_names = ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK', 'NO_TRADE']
    
    preds = model.predict(X)
    
    # Weighted F1
    weighted_f1 = f1_score(y, preds, average='weighted', zero_division=0)
    
    # Precision on NO_TRADE specifically
    no_trade_prec = precision_score(
        y == 'NO_TRADE', preds == 'NO_TRADE', zero_division=0
    )
    
    # Per-class precision
    per_class_prec = precision_score(
        y, preds, average=None, labels=class_names, zero_division=0
    )
    
    return {
        'weighted_f1': weighted_f1,
        'no_trade_precision': no_trade_prec,
        'low_risk_precision': per_class_prec[0] if len(per_class_prec) > 0 else 0,
        'high_risk_precision': per_class_prec[2] if len(per_class_prec) > 2 else 0,
    }


# ============================================================
# BEHAVIORAL EVALUATION (Binary Anomaly Detection)
# ============================================================

def evaluate_behavioral(model, X, y):
    """
    Evaluate behavioral anomaly model.
    """
    y = np.array(y, dtype=float)
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        preds = (proba[:, 1] >= 0.5).astype(int) if proba.ndim > 1 else (proba >= 0.5).astype(int)
    elif hasattr(model, 'decision_function'):
        # Isolation Forest: -1 = anomaly, 1 = normal
        raw = model.predict(X)
        preds = (raw == -1).astype(int)
    else:
        preds = model.predict(X)
    
    f1_anomaly = f1_score(y, preds, zero_division=0)
    prec_anomaly = precision_score(y, preds, zero_division=0)
    rec_anomaly = recall_score(y, preds, zero_division=0)
    
    # False positive rate
    tn = ((y == 0) & (preds == 0)).sum()
    fp = ((y == 0) & (preds == 1)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'anomaly_f1': f1_anomaly,
        'anomaly_precision': prec_anomaly,
        'anomaly_recall': rec_anomaly,
        'false_positive_rate': fpr,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from tree-based models.
    Returns sorted DataFrame.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'calibrated_classifiers_'):
        # CalibratedClassifierCV wraps the base model
        base = model.calibrated_classifiers_[0].estimator
        importances = base.feature_importances_
    else:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df


# ============================================================
# EWMA BASELINE
# ============================================================

class EWMABaseline:
    """
    Simple EWMA baseline for volatility prediction.
    Predicts future vol = exponentially weighted recent vol.
    """
    def __init__(self, span=14):
        self.span = span
        self._last_ewma = None
    
    def fit(self, X, y):
        """Fit stores the EWMA of the training labels."""
        y = np.array(y, dtype=float)
        valid = y[~np.isnan(y)]
        series = pd.Series(valid)
        self._last_ewma = series.ewm(span=self.span).mean().iloc[-1]
        return self
    
    def predict(self, X):
        """Predict = last EWMA value (constant baseline)."""
        return np.full(len(X), self._last_ewma)
