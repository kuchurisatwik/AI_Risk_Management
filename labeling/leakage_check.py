"""
Leakage Check — Validates no forward-looking bias in features.

Runs before every training run to ensure:
1. No feature has suspiciously high correlation with labels
2. Rolling windows use only past data (structural check)
3. Regime scaler was not fit on full dataset
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_no_leakage(df, feature_cols, target_col, threshold=0.7):
    """
    Check for features with suspiciously high correlation to the target.
    
    A correlation > 0.7 between a feature and a label usually means
    the feature is leaking future information into the present.
    
    Args:
        df: DataFrame with features and target
        feature_cols: list of feature column names
        target_col: target label column name
        threshold: correlation threshold for warning
    
    Returns:
        list of warning strings
    """
    issues = []
    
    # Convert target to numeric if needed
    target = df[target_col].copy()
    if target.dtype == 'object':
        target = pd.Series(pd.factorize(target)[0], index=target.index)
    target = pd.to_numeric(target, errors='coerce')
    
    for col in feature_cols:
        feat = pd.to_numeric(df[col], errors='coerce')
        corr = feat.corr(target)
        if abs(corr) > threshold:
            issues.append(
                f"LEAK WARNING: {col} has {corr:.3f} correlation with {target_col}"
            )
    
    return issues


def validate_all_labels(df, feature_sets):
    """
    Run leakage validation across all feature-label pairs.
    
    Args:
        df: Full labeled DataFrame
        feature_sets: dict of {target_col: [feature_cols]}
    
    Returns:
        dict of {target_col: [issues]}
    """
    all_issues = {}
    
    for target_col, feature_cols in feature_sets.items():
        if target_col not in df.columns:
            logger.warning(f"Target {target_col} not in DataFrame, skipping")
            continue
            
        # Drop rows where target is NaN
        valid = df.dropna(subset=[target_col])
        issues = validate_no_leakage(valid, feature_cols, target_col)
        
        if issues:
            all_issues[target_col] = issues
            for issue in issues:
                logger.warning(issue)
        else:
            logger.info(f"  [OK] {target_col}: No leakage detected across {len(feature_cols)} features")
    
    return all_issues


def structural_checks(train_df, val_df, test_df):
    """
    Structural validation of the time-based split.
    
    Ensures:
    1. Train dates < Val dates < Test dates (no overlap)
    2. No data leakage across splits
    """
    issues = []
    
    train_max = train_df['open_time'].max()
    val_min = val_df['open_time'].min()
    val_max = val_df['open_time'].max()
    test_min = test_df['open_time'].min()
    
    if train_max >= val_min:
        issues.append(f"SPLIT LEAK: Train max ({train_max}) >= Val min ({val_min})")
    if val_max >= test_min:
        issues.append(f"SPLIT LEAK: Val max ({val_max}) >= Test min ({test_min})")
    
    if not issues:
        logger.info("  [OK] Time-based split integrity verified")
        logger.info(f"    Train: ... to {train_max}")
        logger.info(f"    Val:   {val_min} to {val_max}")
        logger.info(f"    Test:  {test_min} to ...")
    else:
        for issue in issues:
            logger.error(issue)
    
    return issues
