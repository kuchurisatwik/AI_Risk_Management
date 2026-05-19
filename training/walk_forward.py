"""
Walk-Forward Cross-Validation Framework.

Implements time-respecting CV within the training set.
Never shuffles. Never leaks future data.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def walk_forward_cv(train_df, feature_cols, label_col, train_fn, eval_fn, n_splits=5):
    """
    Walk-forward cross-validation on the training set.
    
    Args:
        train_df: Training DataFrame (time-ordered)
        feature_cols: List of feature column names
        label_col: Target label column name
        train_fn: Callable(X_train, y_train) -> model
        eval_fn: Callable(model, X_val, y_val) -> dict of metrics
        n_splits: Number of CV folds
    
    Returns:
        list of dicts: [{fold, metrics, model}]
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Drop rows where label is NaN
    valid_df = train_df.dropna(subset=[label_col]).reset_index(drop=True)
    
    X = valid_df[feature_cols]
    y = valid_df[label_col]
    
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        logger.info(
            f"  Fold {fold+1}/{n_splits}: "
            f"train={len(X_train):,} val={len(X_val):,}"
        )
        
        model = train_fn(X_train, y_train)
        metrics = eval_fn(model, X_val, y_val)
        
        results.append({
            'fold': fold,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'metrics': metrics,
            'model': model
        })
        
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        logger.info(f"    {metric_str}")
    
    return results


def summarize_cv_results(results, metric_keys=None):
    """
    Aggregate walk-forward CV results into mean/std per metric.
    
    Args:
        results: Output from walk_forward_cv
        metric_keys: List of metric names to summarize (None = all)
    
    Returns:
        dict: {metric_name: {'mean': float, 'std': float, 'values': list}}
    """
    if metric_keys is None:
        metric_keys = list(results[0]['metrics'].keys())
    
    summary = {}
    for key in metric_keys:
        values = []
        for r in results:
            val = r['metrics'].get(key)
            if val is not None and isinstance(val, (int, float)):
                values.append(val)
        
        if values:
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
    
    return summary
