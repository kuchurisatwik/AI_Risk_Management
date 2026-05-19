"""
Behavioral Intelligence Training — Rules + Isolation Forest.

Outputs binary anomaly detection (1 = compromised, 0 = normal).
Focus: Detect emotional breakdown.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

from training.config import BEHAVIORAL_FEATURES, LABEL_BEHAVIORAL
from training.evaluate import evaluate_behavioral

logger = logging.getLogger(__name__)


def train_behavioral(train_df, val_df, test_df, output_dir):
    """
    Behavioral training pipeline.
    
    1. Train Isolation Forest (unsupervised anomaly detection)
    2. Evaluate against our rule-based label (label_behavioral)
    3. Save model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("BEHAVIORAL MODEL TRAINING")
    logger.info("=" * 50)
    
    valid_train = train_df.dropna(subset=[LABEL_BEHAVIORAL])
    X_train = valid_train[BEHAVIORAL_FEATURES].fillna(0)
    y_train = valid_train[LABEL_BEHAVIORAL]
    
    # Train Isolation Forest
    # Contamination set to 0.01 since anomaly rate is ~0.8%
    model = IsolationForest(
        n_estimators=100, 
        contamination=0.01, 
        random_state=42
    )
    
    logger.info("Training Isolation Forest...")
    model.fit(X_train)
    
    # ---- Evaluate on Val/Test ----
    valid_val = val_df.dropna(subset=[LABEL_BEHAVIORAL])
    valid_test = test_df.dropna(subset=[LABEL_BEHAVIORAL])
    
    val_metrics = evaluate_behavioral(model, valid_val[BEHAVIORAL_FEATURES].fillna(0), valid_val[LABEL_BEHAVIORAL])
    test_metrics = evaluate_behavioral(model, valid_test[BEHAVIORAL_FEATURES].fillna(0), valid_test[LABEL_BEHAVIORAL])
    
    logger.info(f"  Val Anomaly F1:  {val_metrics['anomaly_f1']:.4f} | FPR: {val_metrics['false_positive_rate']:.4f}")
    logger.info(f"  Test Anomaly F1: {test_metrics['anomaly_f1']:.4f} | FPR: {test_metrics['false_positive_rate']:.4f}")
    
    # ---- Save ----
    joblib.dump(model, output_dir / 'behavioral_iforest.pkl')
    logger.info(f"  Saved behavioral model to {output_dir}")
    
    return model, val_metrics, test_metrics
