"""
Volatility Model Training — EWMA Baseline + XGBoost.

Outputs predicted_volatility (future realized vol).
Must beat the EWMA baseline to be considered useful.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

from training.config import VOLATILITY_FEATURES, VOLATILITY_PARAMS, LABEL_VOLATILITY, WF_N_SPLITS
from training.walk_forward import walk_forward_cv, summarize_cv_results
from training.evaluate import evaluate_volatility, get_feature_importance, EWMABaseline

logger = logging.getLogger(__name__)


def _train_xgb_model(X_train, y_train):
    """Train XGBoost regression model."""
    model = xgb.XGBRegressor(**VOLATILITY_PARAMS)
    model.fit(X_train, y_train)
    return model


def _train_ewma_model(X_train, y_train):
    """Train EWMA baseline."""
    model = EWMABaseline(span=14)
    model.fit(X_train, y_train)
    return model


def train_volatility(train_df, val_df, test_df, output_dir):
    """
    Full volatility training pipeline.
    
    1. Evaluate EWMA Baseline (walk-forward + final)
    2. Evaluate XGBoost (walk-forward + final)
    3. Compare — if XGB beats EWMA, use XGB, else EWMA
    4. Generate predicted_volatility for downstream models
    5. Save model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("VOLATILITY MODEL TRAINING")
    logger.info("=" * 50)
    
    # ---- 1. EWMA Baseline ----
    logger.info("Evaluating EWMA Baseline...")
    base_cv_results = walk_forward_cv(
        train_df, VOLATILITY_FEATURES, LABEL_VOLATILITY,
        train_fn=_train_ewma_model,
        eval_fn=evaluate_volatility,
        n_splits=WF_N_SPLITS
    )
    base_cv_summary = summarize_cv_results(base_cv_results)
    
    logger.info(f"  Baseline CV RMSE: {base_cv_summary['rmse']['mean']:.6f} +/- {base_cv_summary['rmse']['std']:.6f}")
    
    # ---- 2. XGBoost Model ----
    logger.info("Evaluating XGBoost ML Model...")
    xgb_cv_results = walk_forward_cv(
        train_df, VOLATILITY_FEATURES, LABEL_VOLATILITY,
        train_fn=_train_xgb_model,
        eval_fn=evaluate_volatility,
        n_splits=WF_N_SPLITS
    )
    xgb_cv_summary = summarize_cv_results(xgb_cv_results)
    
    logger.info(f"  XGBoost CV RMSE:  {xgb_cv_summary['rmse']['mean']:.6f} +/- {xgb_cv_summary['rmse']['std']:.6f}")
    
    # ---- 3. Train Final Model ----
    valid_train = train_df.dropna(subset=[LABEL_VOLATILITY])
    X_train = valid_train[VOLATILITY_FEATURES]
    y_train = valid_train[LABEL_VOLATILITY]
    
    # Decide which to use
    xgb_beats_baseline = xgb_cv_summary['rmse']['mean'] < base_cv_summary['rmse']['mean']
    
    if xgb_beats_baseline:
        logger.info("  => XGBoost beat EWMA baseline. Using XGBoost.")
        final_model = _train_xgb_model(X_train, y_train)
        fi = get_feature_importance(final_model, VOLATILITY_FEATURES)
        if not fi.empty:
            logger.info("  Feature Importance (top 5):")
            for _, row in fi.head(5).iterrows():
                logger.info(f"    {row['feature']}: {row['importance']:.4f}")
    else:
        logger.info("  => XGBoost failed to beat EWMA baseline. Using EWMA.")
        final_model = _train_ewma_model(X_train, y_train)
    
    # ---- Evaluate on Val/Test ----
    valid_val = val_df.dropna(subset=[LABEL_VOLATILITY])
    valid_test = test_df.dropna(subset=[LABEL_VOLATILITY])
    
    val_metrics = evaluate_volatility(final_model, valid_val[VOLATILITY_FEATURES], valid_val[LABEL_VOLATILITY])
    test_metrics = evaluate_volatility(final_model, valid_test[VOLATILITY_FEATURES], valid_test[LABEL_VOLATILITY])
    
    logger.info(f"  Val RMSE: {val_metrics['rmse']:.6f} | Dir Acc: {val_metrics['directional_accuracy']:.2f}")
    logger.info(f"  Test RMSE: {test_metrics['rmse']:.6f} | Dir Acc: {test_metrics['directional_accuracy']:.2f}")
    
    # ---- Generate predicted_volatility for all splits ----
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        preds = final_model.predict(df[VOLATILITY_FEATURES])
        df['predicted_volatility'] = preds
    
    # ---- Save ----
    joblib.dump(final_model, output_dir / 'volatility_model.pkl')
    logger.info(f"  Saved volatility model to {output_dir}")
    
    return final_model, xgb_cv_summary, val_metrics, test_metrics
